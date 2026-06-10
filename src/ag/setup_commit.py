"""Per-commit cache management.

For a given ref (sha, branch, tag) creates:
    configs/<short-sha>/worktree/   git worktree of transformers @ sha
    configs/<short-sha>/.venv/      uv venv with ``pip install -e worktree``
    configs/<short-sha>/plugin/     Claude Code plugin dir (if skill derivable)
    configs/<short-sha>/.ready      sentinel
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .build_skill import build as build_skill_plugin
from .log import log
from .paths import configs_dir, results_dir, transformers_src


PINNED_DEPS = [
    "torch",
    "torchaudio",
    "pillow",
    "librosa",
    "scipy",
    "accelerate",
    "huggingface_hub",
]


def _looks_like_sha(ref: str) -> bool:
    return len(ref) >= 7 and all(c in "0123456789abcdef" for c in ref.lower())


def _git_ref_exists(refname: str) -> bool:
    try:
        src = str(transformers_src())
    except SystemExit:
        return False
    return (
        subprocess.run(
            ["git", "-C", src, "show-ref", "--verify", "--quiet", refname],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def classify_ref(ref: str) -> dict:
    """What kind of ref the user asked to test: ``branch`` | ``tag`` | ``commit``.

    Tags are checked before branches (a release tag is the more meaningful
    label if both exist); raw SHAs, ``HEAD``, and ``HEAD~2``-style expressions
    are plain ``commit``s.
    """
    if _looks_like_sha(ref) or ref == "HEAD" or any(t in ref for t in ("~", "^", "@{")):
        kind = "commit"
    elif _git_ref_exists(f"refs/tags/{ref}"):
        kind = "tag"
    elif (
        _git_ref_exists(f"refs/heads/{ref}")
        or _git_ref_exists(f"refs/remotes/origin/{ref}")
        or _git_ref_exists(f"refs/remotes/{ref}")
    ):
        kind = "branch"
    else:
        kind = "commit"
    return {"ref": ref, "kind": kind}


def record_ref(ref: str, sha: str, name: str | None = None, profile: str = "transformers") -> None:
    """Persist what the commit was tested *as* to ``results/<short>/ref.json``
    so the label travels with the results (and into the bucket / report).

    Merge semantics — labels only ever get richer:
    - a branch/tag label is never downgraded by a later raw-SHA re-run;
    - an explicit ``--name`` updates the experiment title; without one, an
      existing title is kept.
    - ``profile`` records which profile produced the binding so the report can
      scope to one profile (e.g. keep mock runs out of the transformers report).
    """
    import json

    info = classify_ref(ref)
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


def suggest_refs(ref: str, names: list[str], n: int = 3) -> list[str]:
    """Close/containing matches for an unknown ref, best first (`5.9.0` → `v5.9.0`)."""
    import difflib

    close = difflib.get_close_matches(ref, names, n=n, cutoff=0.6)
    contains = [c for c in names if ref.lower() in c.lower() and c not in close]
    return (close + contains)[:n]


def _local_ref_names() -> list[str]:
    """Tag + branch names known to the local transformers checkout."""
    src = str(transformers_src())
    names: list[str] = []
    for args in (["tag", "--list"], ["branch", "-a", "--format=%(refname:short)"]):
        try:
            out = subprocess.check_output(["git", "-C", src, *args], text=True)
        except subprocess.CalledProcessError:
            continue
        for line in out.splitlines():
            line = line.strip().removeprefix("origin/")
            if line and line != "HEAD":
                names.append(line)
    return sorted(set(names))


def resolve_sha(ref: str) -> str:
    src = str(transformers_src())
    proc = subprocess.run(
        ["git", "-C", src, "rev-parse", ref],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        sugg = suggest_refs(ref, _local_ref_names())
        hint = (
            f" Did you mean: {', '.join(sugg)}?"
            if sugg
            else " (If it's a brand-new branch, `git fetch` the transformers checkout first.)"
        )
        raise SystemExit(f"`{ref}` is not a known commit, branch, or tag in {src}.{hint}")
    out = proc.stdout.strip()
    # `git rev-parse` on a range (A..B) returns multiple lines — that's not a single
    # commit. Reject it with a clear message rather than corrupting callers.
    if "\n" in out:
        raise SystemExit(
            f"`{ref}` did not resolve to a single commit (got:\n{out}\n). "
            "If you want a range, use `ag compare` or `ag diff`."
        )
    return out


def _ensure_worktree(sha: str, dest: Path) -> None:
    if dest.exists():
        return
    log(f"git worktree add {dest.name} @ {sha[:10]}")
    subprocess.check_call(
        ["git", "-C", str(transformers_src()), "worktree", "add", "--detach", str(dest), sha],
    )


def _ensure_venv(cfg_dir: Path) -> Path:
    venv = cfg_dir / ".venv"
    if not (venv / "bin" / "python").exists():
        log(f"uv venv {venv}")
        subprocess.check_call(["uv", "venv", "--python", "3.13", str(venv)])
    return venv / "bin" / "python"


def _ensure_install(py: Path, worktree: Path) -> None:
    try:
        out = subprocess.check_output(
            [str(py), "-c", "import transformers, pathlib; print(pathlib.Path(transformers.__file__).parent)"],
            text=True,
        ).strip()
        if out.startswith(str(worktree)):
            return
    except subprocess.CalledProcessError:
        pass

    log(f"pip install -e {worktree.name}  (may take a minute)")
    subprocess.check_call(["uv", "pip", "install", "--python", str(py), "-e", str(worktree)])
    log(f"pip install deps {PINNED_DEPS}")
    subprocess.check_call(["uv", "pip", "install", "--python", str(py), *PINNED_DEPS])


def setup(ref: str) -> dict:
    sha = resolve_sha(ref)
    short = sha[:10]
    cfg_dir = configs_dir() / short
    cfg_dir.mkdir(parents=True, exist_ok=True)

    worktree = cfg_dir / "worktree"
    _ensure_worktree(sha, worktree)

    py = _ensure_venv(cfg_dir)
    _ensure_install(py, worktree)

    plugin_dir = cfg_dir / "plugin"
    already_built = (plugin_dir / "skills" / "transformers" / "SKILL.md").exists()
    if already_built:
        skill_available = True
    else:
        log("building SKILL.md from derived manifest")
        skill_available = build_skill_plugin(py, plugin_dir)
        if not skill_available:
            log("  (skill-derivation unavailable at this commit; skipping)")

    (cfg_dir / ".ready").write_text(f"{sha}\n")
    info = {
        "sha": sha,
        "short": short,
        "worktree": str(worktree),
        "venv_python": str(py),
        "plugin_dir": str(plugin_dir),
        "skill_available": skill_available,
    }
    log(f"✓ setup {short}   skill={'yes' if skill_available else 'no'}")
    return info


def cleanup(ref: str) -> None:
    sha = resolve_sha(ref)
    short = sha[:10]
    cfg_dir = configs_dir() / short
    if not cfg_dir.exists():
        return
    worktree = cfg_dir / "worktree"
    if worktree.exists():
        subprocess.check_call(
            ["git", "-C", str(transformers_src()), "worktree", "remove", "--force", str(worktree)],
        )
    shutil.rmtree(cfg_dir, ignore_errors=True)
    print(f"[cleanup] removed {short}")
