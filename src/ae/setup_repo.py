"""Per-binding sandbox management, generalized across target repos.

For a given ref (sha, branch, tag) of any target repo (``transformers``,
``diffusers``, …) creates::

    <cfg_dir>/worktree/   git worktree of the repo @ sha
    <cfg_dir>/.venv/      uv venv with ``pip install -e worktree`` + pinned deps
    <cfg_dir>/plugin/     Agent-Skills plugin dir (if the profile's skill
                          builder reports one available at this binding)
    <cfg_dir>/.ready      sentinel

The profile supplies what differs between libraries: the repo root, the
importable package name, extra pinned deps, and a ``skill_builder`` that
places the skill for the binding under ``<cfg_dir>/plugin/skills/<name>/``
(transformers derives one from a manifest; diffusers ships one in-repo).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from .log import log
from .paths import configs_dir, results_dir


def _looks_like_sha(ref: str) -> bool:
    return len(ref) >= 7 and all(c in "0123456789abcdef" for c in ref.lower())


def _git_ref_exists(refname: str, src: Path) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(src), "show-ref", "--verify", "--quiet", refname],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def classify_ref(ref: str, src: Path) -> dict:
    """What kind of ref the user asked to test: ``branch`` | ``tag`` | ``commit``.

    Tags are checked before branches (a release tag is the more meaningful
    label if both exist); raw SHAs, ``HEAD``, and ``HEAD~2``-style expressions
    are plain ``commit``s.
    """
    if _looks_like_sha(ref) or ref == "HEAD" or any(t in ref for t in ("~", "^", "@{")):
        kind = "commit"
    elif _git_ref_exists(f"refs/tags/{ref}", src):
        kind = "tag"
    elif (
        _git_ref_exists(f"refs/heads/{ref}", src)
        or _git_ref_exists(f"refs/remotes/origin/{ref}", src)
        or _git_ref_exists(f"refs/remotes/{ref}", src)
    ):
        kind = "branch"
    else:
        kind = "commit"
    return {"ref": ref, "kind": kind}


def record_ref(ref: str, sha: str, name: str | None = None, profile: str = "transformers",
               src: Path | None = None) -> None:
    """Persist what the commit was tested *as* to ``results/<short>/ref.json``
    so the label travels with the results (and into the bucket / report).

    Merge semantics — labels only ever get richer:
    - a branch/tag label is never downgraded by a later raw-SHA re-run;
    - an explicit ``--name`` updates the experiment title; without one, an
      existing title is kept.
    - ``profile`` records which profile produced the binding so the report can
      scope to one profile (e.g. keep mock runs out of the transformers report).

    ``src`` is the target repo used to classify the ref; when omitted the
    existing marker (if any) is trusted.
    """
    import json

    if src is not None:
        info = classify_ref(ref, src)
    else:
        info = {"ref": ref, "kind": "commit"}
    path = results_dir(sha[:10]) / "ref.json"
    try:
        existing = json.loads(path.read_text())
    except Exception:
        existing = {}
    if existing.get("kind") in ("branch", "tag") and info["kind"] == "commit":
        info = {"ref": existing["ref"], "kind": existing["kind"]}
    out = {**info, "sha": sha, "profile": profile}
    if name:
        out["name"] = name
    elif existing.get("name"):
        out["name"] = existing["name"]
    path.write_text(json.dumps(out) + "\n")
    # Jobs mirror each artefact to the bucket the moment it's written, so a
    # crash/eviction keeps the binding's label too — not just the run shards.
    # Without this, report labels (name/ref/kind) never leave the container.
    mdir = os.environ.get("AE_MIRROR_DIR")
    if mdir:
        try:
            dst = Path(mdir) / "results" / sha[:10] / "ref.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dst)
        except Exception:  # noqa: BLE001
            pass


def suggest_refs(ref: str, names: list[str], n: int = 3) -> list[str]:
    """Close/containing matches for an unknown ref, best first (`5.9.0` → `v5.9.0`)."""
    import difflib

    close = difflib.get_close_matches(ref, names, n=n, cutoff=0.6)
    contains = [c for c in names if ref.lower() in c.lower() and c not in close]
    return (close + contains)[:n]


def _local_ref_names(src: Path) -> list[str]:
    """Tag + branch names known to the local checkout of ``src``."""
    names: list[str] = []
    for args in (["tag", "--list"], ["branch", "-a", "--format=%(refname:short)"]):
        try:
            out = subprocess.check_output(["git", "-C", str(src), *args], text=True)
        except subprocess.CalledProcessError:
            continue
        for line in out.splitlines():
            line = line.strip().removeprefix("origin/")
            if line and line != "HEAD":
                names.append(line)
    return sorted(set(names))


def resolve_sha(ref: str, src: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(src), "rev-parse", ref],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        sugg = suggest_refs(ref, _local_ref_names(src))
        hint = (
            f" Did you mean: {', '.join(sugg)}?"
            if sugg
            else f" (If it's a brand-new branch, `git fetch` the {src.name} checkout first.)"
        )
        raise SystemExit(f"`{ref}` is not a known commit, branch, or tag in {src}.{hint}")
    out = proc.stdout.strip()
    # `git rev-parse` on a range (A..B) returns multiple lines — that's not a single
    # commit. Reject it with a clear message rather than corrupting callers.
    if "\n" in out:
        raise SystemExit(
            f"`{ref}` did not resolve to a single commit (got:\n{out}\n). "
            "If you want a range, use `agent-eval compare` or `agent-eval diff`."
        )
    return out


def _ensure_worktree(sha: str, dest: Path, src: Path) -> None:
    if dest.exists():
        return
    log(f"git worktree add {dest.name} @ {sha[:10]}")
    subprocess.check_call(
        ["git", "-C", str(src), "worktree", "add", "--detach", str(dest), sha],
    )


def _ensure_venv(cfg_dir: Path) -> Path:
    venv = cfg_dir / ".venv"
    if not (venv / "bin" / "python").exists():
        log(f"uv venv {venv}")
        subprocess.check_call(["uv", "venv", "--python", "3.13", str(venv)])
    return venv / "bin" / "python"


def _ensure_install(py: Path, worktree: Path, package: str, pinned_deps: list[str]) -> None:
    try:
        out = subprocess.check_output(
            [str(py), "-c",
             f"import {package}, pathlib; print(pathlib.Path({package}.__file__).parent)"],
            text=True,
        ).strip()
        if out.startswith(str(worktree)):
            return
    except subprocess.CalledProcessError:
        pass

    log(f"pip install -e {worktree.name}  (may take a minute)")
    subprocess.check_call(["uv", "pip", "install", "--python", str(py), "-e", str(worktree)])
    if pinned_deps:
        log(f"pip install deps {pinned_deps}")
        subprocess.check_call(["uv", "pip", "install", "--python", str(py), *pinned_deps])


def setup(
    ref: str,
    *,
    src: Path,
    profile: str,
    package: str,
    pinned_deps: list[str] | None = None,
    skill_builder: Callable[[Path, Path, Path], bool] | None = None,
    cfg_dir: Path | None = None,
) -> dict:
    """Prepare (or reuse) the sandbox for one binding of ``src`` at ``ref``.

    ``skill_builder(venv_python, worktree, plugin_dir) -> bool`` places the
    binding's Agent Skill under ``<plugin_dir>/skills/<name>/`` and returns
    whether the skill tier is usable at this binding. ``None`` means the
    profile has no skill tier at all.
    """
    sha = resolve_sha(ref, src)
    short = sha[:10]
    if cfg_dir is None:
        cfg_dir = configs_dir() / short
    cfg_dir.mkdir(parents=True, exist_ok=True)

    worktree = cfg_dir / "worktree"
    _ensure_worktree(sha, worktree, src)

    py = _ensure_venv(cfg_dir)
    _ensure_install(py, worktree, package, pinned_deps or [])

    if skill_builder is None:
        skill_available = False
    else:
        plugin_dir = cfg_dir / "plugin"
        skill_available = skill_builder(py, worktree, plugin_dir)

    (cfg_dir / ".ready").write_text(f"{sha}\n")
    info = {
        "sha": sha,
        "short": short,
        "worktree": str(worktree),
        "venv_python": str(py),
        "plugin_dir": str(cfg_dir / "plugin"),
        "skill_available": skill_available,
    }
    log(f"✓ setup {short}   skill={'yes' if skill_available else 'no'}")
    return info


def cleanup(ref: str, src: Path, cfg_dir: Path | None = None) -> None:
    sha = resolve_sha(ref, src)
    short = sha[:10]
    if cfg_dir is None:
        cfg_dir = configs_dir() / short
    if not cfg_dir.exists():
        return
    worktree = cfg_dir / "worktree"
    if worktree.exists():
        subprocess.check_call(
            ["git", "-C", str(src), "worktree", "remove", "--force", str(worktree)],
        )
    shutil.rmtree(cfg_dir, ignore_errors=True)
    print(f"[cleanup] removed {short}")
