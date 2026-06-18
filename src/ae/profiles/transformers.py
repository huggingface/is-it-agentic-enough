"""The ``transformers`` profile: the original study, expressed as a profile.

Binding = a git revision of ``transformers``. Tiers = the historical
``bare`` / ``clone`` / ``skill`` discovery conditions. Delegates to the existing
machinery (`setup_commit`, `run_task` workspace helpers) so there is one
implementation of each behavior.
"""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from ..markers import Marker
from ..paths import configs_dir, package_data_path, transformers_src, workspaces_dir
from ..profile import BuiltEnv, Profile, register

TIERS = ("bare", "clone", "skill")


@lru_cache(maxsize=1)
def tasks() -> dict[str, dict]:
    """The transformers task suite, ``{id: task}``, from ``data/transformers.yaml``.
    Module-level + cached so the ``mock`` profile and the mock runner can reuse it."""
    import yaml

    with open(package_data_path("transformers.yaml")) as f:
        return {t["id"]: t for t in yaml.safe_load(f)["tasks"]}

# Behavior markers for the transformers study. Independent (a run may fire
# several): adoption of each is tracked across revisions. These replace the old
# hard-wired CLI-vs-Python bucketing + read_agentic/ran_help signals.
MARKERS = [
    # Invoked the `transformers` CLI (start of command, after a pipe/&&/;, or a
    # path-prefixed binary) — with a subcommand or a flag like `--format` — as
    # opposed to writing Python. The leading anchor avoids matching `transformers`
    # inside `pip install transformers` or `import transformers`.
    Marker("cli", r"(?:^|[|&;]|/)\s*transformers\s+\S",
           "Ran the `transformers` command-line tool instead of writing Python.",
           "commands"),
    Marker("pipeline", r"\bpipeline\s*\(",
           "Used the high-level `pipeline(...)` Python API.",
           "any"),
    Marker("ran-help", r"transformers\b[^\n]*--help",
           "Consulted the CLI's built-in help (`transformers ... --help`).",
           "commands"),
    Marker("agentic-exemplar", r"/cli/agentic/\w+\.py",
           "Read an in-repo `cli/agentic/*.py` example to learn the agentic interface (clone tier).",
           "reads"),
]


class TransformersProfile(Profile):
    name = "transformers"

    def expand_bindings(self, spec: list[str]) -> list[str]:
        """Expand ``A..B..C`` ranges and resolve branch/tag/SHA tokens to unique
        10-char short SHAs (the canonical transformers binding id)."""
        from ..profile import expand_spec
        from ..setup_commit import resolve_sha

        def short_sha(ref: str) -> str:
            is_sha = len(ref) >= 10 and all(c in "0123456789abcdef" for c in ref.lower())
            return ref[:10] if is_sha else resolve_sha(ref)[:10]

        return expand_spec(spec, short_sha)

    def build(self, ref: str, *, name: str | None = None) -> BuiltEnv:
        from ..setup_commit import record_ref, resolve_sha, setup

        sha = resolve_sha(ref)
        record_ref(ref, sha, name, profile="transformers")  # label the binding: branch/tag/commit + optional title
        info = setup(ref)
        short = info["short"]
        tiers = ["bare", "clone"] + (["skill"] if info["skill_available"] else [])
        return BuiltEnv(
            binding=short,
            python=Path(info["venv_python"]),
            available_tiers=tiers,
            cfg_dir=configs_dir() / short,
            label=name or ref,
            extra={"sha": sha},
        )

    def all_tiers(self) -> list[str]:
        return list(TIERS)

    def prepare_workspace(self, built: BuiltEnv, tier: str, task_id: str, run_idx: int) -> Path:
        """Fresh cwd for one run. For ``clone`` it IS a git worktree of
        transformers @ the binding's SHA (so AGENTS.md/CLAUDE.md/cli/agentic
        auto-discover from cwd); other tiers get an empty dir. Both seed the
        task ``inputs/`` (cat.jpg, sample.wav, …)."""
        ws = workspaces_dir() / f"{built.binding}__{tier}__{task_id}__run{run_idx}"
        if ws.exists():
            self.remove_workspace(ws)
        if tier == "clone":
            subprocess.check_call(
                ["git", "-C", str(transformers_src()), "worktree", "add", "--detach",
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
                ["git", "-C", str(transformers_src()), "worktree", "remove", "--force", str(ws)],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        if ws.exists():
            shutil.rmtree(ws, ignore_errors=True)

    def agent_assets(self, built: BuiltEnv, tier: str) -> dict:
        if tier != "skill" or built.cfg_dir is None:
            return {}
        # The skill lives at <cfg_dir>/plugin/skills/transformers/ (an Agent-Skills
        # layout); the pi runner is handed that leaf via --skill.
        return {"skill_dir": built.cfg_dir / "plugin" / "skills" / "transformers"}

    def markers(self) -> list:
        return list(MARKERS)

    def tasks(self) -> dict[str, dict]:
        return tasks()


register(TransformersProfile())
