"""The ``diffusers`` profile: is the agentic CLI helping diffusers?

Binding = a git revision of ``diffusers``. The comparison axis is the commit
that introduced the agentic surface — ``diffusers-cli run/schema/skills`` and
the in-repo ``.ai/skills/`` registry (PR #13966, shipped in v0.40.0):
``v0.39.0`` (before) vs that commit / ``v0.40.0`` (after).

Tiers mirror the transformers study: ``bare`` / ``clone`` / ``skill``. Unlike
transformers (whose skill is *derived* from an install manifest), diffusers
*ships* its skill in-repo at ``.ai/skills/diffusers-cli/`` — the skill tier
copies that bundle from the binding's worktree, and is simply unavailable on
revisions that predate it.
"""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from ..log import log
from ..markers import Marker
from ..paths import configs_dir, package_data_path, repo_src, workspaces_dir
from ..profile import BuiltEnv, Profile, register

TIERS = ("bare", "clone", "skill")

# The in-repo skill bundle the skill tier is built from.
SKILL_NAME = "diffusers-cli"

# Extra deps the task code needs on top of the editable diffusers install.
# Mirrors diffusers-cli run's remote defaults minus the video-only extras.
PINNED_DEPS = [
    "torch",
    "transformers",
    "accelerate",
    "safetensors",
    "pillow",
    "huggingface_hub",
]


@lru_cache(maxsize=1)
def tasks() -> dict[str, dict]:
    """The diffusers task suite, ``{id: task}``, from ``data/diffusers.yaml``."""
    import yaml

    with open(package_data_path("diffusers.yaml")) as f:
        return {t["id"]: t for t in yaml.safe_load(f)["tasks"]}


def _copy_skill(py: Path, worktree: Path, plugin_dir: Path) -> bool:
    """The skill tier's asset: the in-repo ``.ai/skills/diffusers-cli/`` bundle,
    copied from the binding's worktree so it matches the revision under test.
    Unavailable on revisions that predate the skill registry."""
    skill_src = worktree / ".ai" / "skills" / SKILL_NAME
    if not (skill_src / "SKILL.md").exists():
        log("  (no .ai/skills/diffusers-cli at this revision; skipping skill tier)")
        return False
    dest = plugin_dir / "skills" / SKILL_NAME
    if dest.exists():
        return True
    shutil.copytree(skill_src, dest)
    return True


# Behavior markers for the diffusers study. Independent (a run may fire
# several): adoption of each is tracked across revisions.
MARKERS = [
    # Invoked the `diffusers-cli` tool — with a subcommand or a flag — as
    # opposed to writing Python. The leading anchor avoids matching
    # `diffusers-cli` inside `pip install diffusers-cli`-style text.
    Marker("cli", r"(?:^|[|&;]|/)\s*diffusers-cli\s+\S",
           "Ran the `diffusers-cli` command-line tool instead of writing Python.",
           "commands"),
    Marker("schema", r"diffusers-cli\s+(?:--format\s+\S+\s+)?schema\b",
           "Consulted `diffusers-cli schema` to introspect a pipeline's input signature.",
           "commands"),
    Marker("python", r"from_pretrained\s*\(",
           "Loaded the pipeline via the Python `from_pretrained(...)` API.",
           "any"),
    Marker("skill-read", r"/\.ai/skills/",
           "Read an in-repo `.ai/skills/*` skill file (clone tier).",
           "reads"),
]


class DiffusersProfile(Profile):
    name = "diffusers"
    # Git URL of the repo under test — what HF Jobs clone in their bootstrap.
    repo_git = "https://github.com/huggingface/diffusers"

    def expand_bindings(self, spec: list[str]) -> list[str]:
        """Expand ``A..B..C`` ranges and resolve branch/tag/SHA tokens to unique
        10-char short SHAs (the canonical diffusers binding id)."""
        from ..profile import expand_spec
        from ..setup_repo import resolve_sha

        src = repo_src("diffusers")

        def short_sha(ref: str) -> str:
            is_sha = len(ref) >= 10 and all(c in "0123456789abcdef" for c in ref.lower())
            return ref[:10] if is_sha else resolve_sha(ref, src)[:10]

        return expand_spec(spec, short_sha)

    def build(self, ref: str, *, name: str | None = None) -> BuiltEnv:
        from ..setup_repo import record_ref, resolve_sha, setup

        src = repo_src("diffusers")
        sha = resolve_sha(ref, src)
        record_ref(ref, sha, name, profile="diffusers", src=src)
        info = setup(
            ref,
            src=src,
            profile="diffusers",
            package="diffusers",
            pinned_deps=PINNED_DEPS,
            skill_builder=_copy_skill,
            # Namespaced under configs/diffusers/ so bindings from different
            # target repos can never share a cache dir.
            cfg_dir=self._cfg_dir(sha[:10]),
        )
        short = info["short"]
        tiers = ["bare", "clone"] + (["skill"] if info["skill_available"] else [])
        return BuiltEnv(
            binding=short,
            python=Path(info["venv_python"]),
            available_tiers=tiers,
            cfg_dir=configs_dir() / "diffusers" / short,
            label=name or ref,
            extra={"sha": sha},
        )

    @staticmethod
    def _cfg_dir(short: str) -> Path:
        return configs_dir() / "diffusers" / short

    def all_tiers(self) -> list[str]:
        return list(TIERS)

    def prepare_workspace(self, built: BuiltEnv, tier: str, task_id: str, run_idx: int) -> Path:
        """Fresh cwd for one run. For ``clone`` it IS a git worktree of
        diffusers @ the binding's SHA (so .ai/skills and the CLI source
        auto-discover from cwd); other tiers get an empty dir. Both seed the
        task ``inputs/`` (cat.jpg, sample.wav, …)."""
        ws = workspaces_dir() / f"{built.binding}__{tier}__{task_id}__run{run_idx}"
        if ws.exists():
            self.remove_workspace(ws)
        if tier == "clone":
            subprocess.check_call(
                ["git", "-C", str(repo_src("diffusers")), "worktree", "add", "--detach",
                 str(ws), built.extra["sha"]],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            ws.mkdir(parents=True)
        shutil.copytree(package_data_path("inputs"), ws / "inputs")
        return ws

    def remove_workspace(self, ws: Path) -> None:
        """Best-effort teardown: ``git worktree remove`` for clone worktrees,
        then rmtree."""
        if (ws / ".git").exists():
            subprocess.run(
                ["git", "-C", str(repo_src("diffusers")), "worktree", "remove", "--force", str(ws)],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        if ws.exists():
            shutil.rmtree(ws, ignore_errors=True)

    def agent_assets(self, built: BuiltEnv, tier: str) -> dict:
        if tier != "skill" or built.cfg_dir is None:
            return {}
        # The skill lives at <cfg_dir>/plugin/skills/diffusers-cli/ (an
        # Agent-Skills layout); the pi runner is handed that leaf via --skill.
        return {"skill_dir": built.cfg_dir / "plugin" / "skills" / SKILL_NAME}

    def markers(self) -> list:
        return list(MARKERS)

    def tasks(self) -> dict[str, dict]:
        return tasks()

    def cleanup(self, ref: str) -> None:
        """Remove this binding's cached sandbox (worktree + venv + skill copy)."""
        from ..setup_repo import cleanup, resolve_sha

        src = repo_src("diffusers")
        cleanup(ref, src, cfg_dir=self._cfg_dir(resolve_sha(ref, src)[:10]))


register(DiffusersProfile())
