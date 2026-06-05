"""Sync the local run state with a Hugging Face **bucket** (S3-like Xet object
storage on the Hub — https://huggingface.co/docs/huggingface_hub/en/guides/buckets).

The harness keeps three kinds of run state under the data dir:

- ``results/<commit>/<harness>/<model_id>/<variant>__<task>__runN.jsonl`` (+ ``.meta.json``)
  — the canonical run transcripts.
- ``traces/<commit>/<harness>/<model_id>/...`` — native agent sessions (one per run).
- ``results/MANIFEST.json`` — a generated record of *which* configs/commits were run
  (commit → git subject/date + the set of harness/model/variant/task/run cells present).

``isth sync`` mirrors ``results/`` and ``traces/`` (each under its own prefix) to/from
a bucket via ``hf buckets sync``, which only transfers files that changed.

**Safety:** ``push``/``pull`` are *dry-run by default*. Nothing leaves or
overwrites the machine unless ``--push`` / ``--pull`` is passed explicitly.
Traces and transcripts can contain prompts, command output, local paths, and
secrets — review them (or keep the bucket private) before publishing.
"""

from __future__ import annotations

import json
import subprocess

from .log import log
from .paths import state_root, transformers_src
from .upload import _have_hf_cli


def _bucket_uri(bucket_id: str, prefix: str | None = None) -> str:
    """``lysandre/foo`` → ``hf://buckets/lysandre/foo[/prefix]``."""
    base = bucket_id
    if base.startswith("hf://buckets/"):
        base = base[len("hf://buckets/"):]
    uri = f"hf://buckets/{base}"
    return f"{uri}/{prefix}" if prefix else uri


def _git_meta(sha: str) -> tuple[str, str]:
    """Return ``(subject, date)`` for a commit, or ``("?", "?")`` if unknown."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(transformers_src()), "show", "-s",
             "--date=short", "--format=%s|%ad", sha],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        subject, date = out.split("|", 1)
        return subject, date
    except (Exception, SystemExit):
        # transformers_src() raises SystemExit when the repo isn't located; the
        # manifest should still build (just without git subjects/dates).
        return "?", "?"


def build_manifest() -> dict:
    """Scan ``results/`` and summarize which configs/commits were run.

    Layout assumed: ``results/<commit>/<harness>/<model_id>/<variant>__<task>__runN.meta.json``.
    """
    root = state_root() / "results"
    commits: dict[str, dict] = {}
    for meta in sorted(root.glob("*/*/*/*.meta.json")):
        commit, harness, model_id, fname = meta.relative_to(root).parts
        stem = fname[: -len(".meta.json")]
        try:
            variant, task, runtok = stem.split("__")
        except ValueError:
            continue  # unexpected filename; skip
        try:
            data = json.loads(meta.read_text())
        except Exception:
            data = {}
        entry = commits.setdefault(commit, {"runs": []})
        entry["runs"].append(
            {
                "harness": harness,
                "model_id": model_id,
                "variant": variant,
                "task": task,
                "run": int(runtok.replace("run", "") or 0),
                "status": data.get("status"),
                "runner": data.get("runner"),
                "model": data.get("model"),
                "sha": data.get("sha"),
            }
        )

    out_commits: dict[str, dict] = {}
    for sha in sorted(commits):
        entry = commits[sha]
        subject, date = _git_meta(entry["runs"][0].get("sha") or sha)
        namespaces = sorted({f"{r['harness']}/{r['model_id']}" for r in entry["runs"]})
        try:
            ref_info = json.loads((root / sha / "ref.json").read_text())
        except Exception:
            ref_info = {}
        out_commits[sha] = {
            "subject": subject,
            "date": date,
            "name": ref_info.get("name"),
            "ref": ref_info.get("ref"),
            "kind": ref_info.get("kind"),
            "n_runs": len(entry["runs"]),
            "namespaces": namespaces,
            "runs": sorted(
                entry["runs"],
                key=lambda r: (r["harness"], r["model_id"], r["variant"], r["task"], r["run"]),
            ),
        }
    return {"commits": out_commits}


def write_manifest() -> tuple[dict, int]:
    """Build the manifest and write it to ``results/MANIFEST.json``. Returns
    ``(manifest, total_runs)``."""
    manifest = build_manifest()
    results = state_root() / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(c["n_runs"] for c in manifest["commits"].values())
    return manifest, total


def _summarize(manifest: dict, total: int) -> None:
    commits = manifest["commits"]
    log(f"manifest: {len(commits)} commit(s), {total} run(s)")
    for sha, c in commits.items():
        log(f"  {sha}  {c['date']}  {c['n_runs']:>3} runs  [{', '.join(c['namespaces'])}]  {c['subject'][:60]}")


def _hf_missing() -> bool:
    if _have_hf_cli():
        return False
    log("The `hf` CLI is not installed. Install it with "
        "`curl -LsSf https://hf.co/cli/install.sh | bash` and `hf auth login`.")
    return True


def sync(
    bucket_id: str,
    *,
    push: bool = False,
    pull: bool = False,
    private: bool = True,
    delete: bool = False,
) -> int:
    """Mirror ``results/`` + ``traces/`` to/from a HF bucket via ``hf buckets sync``.

    Default (neither ``push`` nor ``pull``): regenerate the manifest and print a
    DRY-RUN plan of the ``hf buckets sync`` commands. ``push`` uploads (and
    creates the bucket if needed); ``pull`` downloads.
    """
    root = state_root()
    trees = [("results", root / "results"), ("traces", root / "traces")]

    if pull:
        cmds = []
        for prefix, local in trees:
            cmd = ["hf", "buckets", "sync", _bucket_uri(bucket_id, prefix), str(local)]
            if delete:
                cmd.append("--delete")
            cmds.append(cmd)
        log(f"pull plan ← {_bucket_uri(bucket_id)} (results/ + traces/)")
        for cmd in cmds:
            log("  " + " ".join(cmd))
        if _hf_missing():
            return 1
        for cmd in cmds:
            log("▶ " + " ".join(cmd))
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                return rc
        manifest, total = write_manifest()
        _summarize(manifest, total)
        return 0

    # push (or dry-run): refresh the manifest first so the bucket records what ran.
    manifest, total = write_manifest()
    _summarize(manifest, total)

    create_cmd = ["hf", "buckets", "create", bucket_id, "--exist-ok"]
    if private:
        create_cmd.append("--private")

    cmds: list[list[str]] = []
    for prefix, local in trees:
        if not local.exists() or not any(local.iterdir()):
            log(f"  (nothing under {prefix}/ to sync)")
            continue
        cmd = ["hf", "buckets", "sync", str(local), _bucket_uri(bucket_id, prefix)]
        if delete:
            cmd.append("--delete")
        cmds.append(cmd)

    log(f"push plan → {_bucket_uri(bucket_id)}  (https://huggingface.co/buckets/{bucket_id})")
    log("  " + " ".join(create_cmd))
    for cmd in cmds:
        log("  " + " ".join(cmd))

    if not push:
        log("DRY RUN — nothing uploaded. Re-run with `--push` to sync to the bucket.")
        return 0

    if _hf_missing():
        return 1

    log("▶ " + " ".join(create_cmd))
    rc = subprocess.run(create_cmd).returncode
    if rc != 0:
        return rc
    for cmd in cmds:
        log("▶ " + " ".join(cmd))
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            return rc
    return 0
