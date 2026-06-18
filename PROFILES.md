# Profiles

A **profile** is the pluggable seam that makes `agent-eval` general. The harness
core only knows about tasks, tiers, runners, and answer matching. Everything
environment-specific lives behind a profile: how to build a sandbox for a revision,
what "assistance tiers" exist, how to seed the agent's workspace, which task suite
to run, and which behaviors to track.

Two profiles ship: `transformers` (the reference study) and `mock` (a fast fake for
trying the UI). This document describes the contract and walks through the
`transformers` profile as a worked example.

## Vocabulary

- **binding**: one point on the comparison axis the suite sweeps. For `transformers`
  a binding is a git revision (resolved to a 10-char short SHA). A `batch` matrix
  runs the suite at each binding and the report compares them.
- **tier**: how much help the agent gets. A profile declares its own tiers and
  decides how each seeds the workspace and what assets the agent is handed.
  `transformers` uses `bare` (nothing), `clone` (the repo in the working directory),
  and `skill` (a packaged Skill).
- **task**: one instruction the agent must carry out, with an optional `expected`
  answer to score against. The suite is profile-defined.
- **marker**: a regex that flags whether a run exhibited some behavior (for example,
  "ran the CLI" vs. "wrote Python"), so adoption can be tracked across bindings.
- **assets**: per-tier extras passed to the runner, normalized so runners stay
  profile-agnostic. Currently `skill_dir` (handed to the `pi` runner as `--skill`).

## The contract

A profile is any object that satisfies the `Profile` protocol in
[`src/ae/profile.py`](./src/ae/profile.py) and registers itself. The methods:

| Method | Responsibility |
| --- | --- |
| `name: str` | The profile id used on the CLI and in result paths. |
| `expand_bindings(spec) -> list[str]` | Turn a CLI/YAML spec (e.g. an `A..B` range) into canonical binding ids. |
| `build(ref, *, name=None) -> BuiltEnv` | Prepare (or reuse a cached) sandbox for one binding. |
| `all_tiers() -> list[str]` | Every tier this profile defines, most-bare first. |
| `tasks() -> dict[str, dict]` | The task suite as `{id: task}`. |
| `prepare_workspace(built, tier, task_id, run_idx) -> Path` | A fresh working directory for one run. |
| `remove_workspace(ws)` | Tear down a workspace (best-effort). |
| `agent_assets(built, tier) -> dict` | Per-tier extras for the runner (e.g. `{"skill_dir": ...}`). |
| `markers() -> list[Marker]` | Behavior markers whose adoption the report tracks. |

`build()` returns a `BuiltEnv`:

```python
@dataclass
class BuiltEnv:
    binding: str                  # canonical id (transformers: 10-char sha)
    python: Path                  # interpreter the agent's task code runs under
    available_tiers: list[str]    # tiers usable at THIS binding (skill may be absent)
    cfg_dir: Path | None = None   # profile-internal cache/scratch root
    label: str | None = None      # display label (branch/tag/title)
    extra: dict = field(default_factory=dict)
```

Note the difference between `all_tiers()` (everything the profile defines) and
`BuiltEnv.available_tiers` (what a *specific* binding can run; for `transformers`,
`skill` is dropped on revisions where the Skill can't be derived).

A profile registers itself at import time and is resolved by name:

```python
from ..profile import register
register(MyProfile())
```

```python
from ae.profile import get_profile
profile = get_profile("transformers")   # or "mock"; None gives the default (transformers)
```

## Worked example: the `transformers` profile

Full source: [`src/ae/profiles/transformers.py`](./src/ae/profiles/transformers.py).
A binding is a git revision; tiers are `bare` / `clone` / `skill`.

### Bindings: resolve refs to short SHAs

`expand_spec` (from `ae.profile`) does the shared work of splitting `A..B..C` ranges
and de-duplicating; the profile only supplies how to map one ref to a canonical id.

```python
TIERS = ("bare", "clone", "skill")

class TransformersProfile(Profile):
    name = "transformers"

    def expand_bindings(self, spec: list[str]) -> list[str]:
        from ..profile import expand_spec
        from ..setup_commit import resolve_sha

        def short_sha(ref: str) -> str:
            is_sha = len(ref) >= 10 and all(c in "0123456789abcdef" for c in ref.lower())
            return ref[:10] if is_sha else resolve_sha(ref)[:10]

        return expand_spec(spec, short_sha)

    def all_tiers(self) -> list[str]:
        return list(TIERS)
```

### Build: prepare a sandbox per revision

`build()` resolves the ref, records what it was tested as (for report labels), and
sets up a per-revision cache (a git worktree at the SHA plus a `uv venv` with the
repo installed). It reports which tiers that revision can actually run.

```python
    def build(self, ref: str, *, name: str | None = None) -> BuiltEnv:
        from ..setup_commit import record_ref, resolve_sha, setup

        sha = resolve_sha(ref)
        record_ref(ref, sha, name, profile="transformers")
        info = setup(ref)
        tiers = ["bare", "clone"] + (["skill"] if info["skill_available"] else [])
        return BuiltEnv(
            binding=info["short"],
            python=Path(info["venv_python"]),
            available_tiers=tiers,
            cfg_dir=configs_dir() / info["short"],
            label=name or ref,
            extra={"sha": sha},
        )
```

### Workspace: a fresh cwd per run

Each run gets its own directory. For `clone`, that directory *is* a git worktree of
transformers at the binding's SHA, so the repo's `AGENTS.md` / `CLAUDE.md` /
`cli/agentic/` auto-discover from the cwd; other tiers get an empty directory. Both
seed the task `inputs/` (the image and audio files tasks reference).

```python
    def prepare_workspace(self, built, tier, task_id, run_idx) -> Path:
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
        if (ws / ".git").exists():
            subprocess.run(
                ["git", "-C", str(transformers_src()), "worktree", "remove", "--force", str(ws)],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        if ws.exists():
            shutil.rmtree(ws, ignore_errors=True)
```

### Assets: hand the Skill to the agent

Only the `skill` tier has an asset. The Skill lives at
`<cfg_dir>/plugin/skills/transformers/` in an Agent-Skills layout; the `pi` runner
is handed that leaf via `--skill`. Other tiers return `{}`.

```python
    def agent_assets(self, built, tier) -> dict:
        if tier != "skill" or built.cfg_dir is None:
            return {}
        return {"skill_dir": built.cfg_dir / "plugin" / "skills" / "transformers"}
```

### Tasks: the suite the profile owns

Tasks are loaded from a packaged YAML and cached. The `mock` profile reuses this
same loader, which is why it lives at module level.

```python
@lru_cache(maxsize=1)
def tasks() -> dict[str, dict]:
    import yaml
    with open(package_data_path("transformers.yaml")) as f:
        return {t["id"]: t for t in yaml.safe_load(f)["tasks"]}

class TransformersProfile(Profile):
    def tasks(self) -> dict[str, dict]:
        return tasks()
```

A task in [`src/ae/data/transformers.yaml`](./src/ae/data/transformers.yaml):

```yaml
- id: classify-sentiment        # required: unique id
  category: atomic              # atomic | compositional (for grouping in the report)
  prompt: |                     # required: the developer-style instruction
    Using distilbert/…-sst-2-english, classify the sentiment of "…".
  expected: positive            # optional: scored against the final answer
  match: substring              # optional: substring (default) | exact | regex
  runs: 5                       # optional: samples per cell (default 3)
```

Prompts that reference `./inputs/<file>` get that file copied into the workspace by
`prepare_workspace`, and the report inlines it for display.

### Markers: track behavior across revisions

A `Marker` is a regex over one **scope** of the run. Scopes:

- `commands`: the shell commands the agent ran (Bash inputs)
- `wrote`: contents and paths the agent wrote
- `reads`: paths the agent read / grepped / globbed
- `final`: the agent's final answer
- `any`: all of the above plus tool-result text

Each marker is independent (a run can fire several or none). The report shows, per
cell and per binding, `fired/total` for each.

```python
from ..markers import Marker

MARKERS = [
    Marker("cli", r"(?:^|[|&;]|/)\s*transformers\s+\S",
           "Ran the `transformers` command-line tool instead of writing Python.",
           "commands"),
    Marker("pipeline", r"\bpipeline\s*\(",
           "Used the high-level `pipeline(...)` Python API.", "any"),
    Marker("ran-help", r"transformers\b[^\n]*--help",
           "Consulted the CLI's built-in help.", "commands"),
    Marker("agentic-exemplar", r"/cli/agentic/\w+\.py",
           "Read an in-repo `cli/agentic/*.py` example (clone tier).", "reads"),
]

class TransformersProfile(Profile):
    def markers(self) -> list:
        return list(MARKERS)
```

A profile that declares no markers (`return []`) simply gets no marker columns in
the report.

## A minimal profile: `mock`

[`src/ae/profiles/mock.py`](./src/ae/profiles/mock.py) is the smallest viable
profile and a good template. It does no git, no venv, and no real install:

- `expand_bindings` sanitizes arbitrary labels (no resolution).
- `build` writes only the `ref.json` label marker and points `python` at the current
  interpreter.
- `prepare_workspace` returns a `tempfile.mkdtemp()`; `remove_workspace` deletes it.
- `agent_assets` returns `{}`.
- `tiers`, `markers`, and `tasks` are reused from the `transformers` profile so the
  report has the same shape.

Paired with the `mock` runner, a whole suite finishes in seconds.

## Adding your own profile

1. Create `src/ae/profiles/<yourname>.py` with a class implementing the eight
   methods above, then `register(YourProfile())` at the bottom of the module.
2. Register it for import in
   [`src/ae/profiles/__init__.py`](./src/ae/profiles/__init__.py) (importing the
   package is what triggers `register`):

   ```python
   from . import yourname  # noqa: F401  (registration side-effect)
   ```
3. Define its task suite. The simplest path is a packaged YAML
   (`src/ae/data/<yourname>.yaml`, included in the wheel) loaded by an `@lru_cache`d
   `tasks()` helper, mirroring `transformers`.
4. Use it: put `profile: yourname` in a batch YAML, or pass it as the positional to
   `agent-eval setup` / `agent-eval report`.

### Tips

- Reuse `expand_spec(spec, resolve)` for `expand_bindings`; you only supply the
  per-ref mapping.
- Cache expensive loaders (`@lru_cache`) so `tasks()` and friends are cheap to call
  repeatedly across the run loop.
- `prepare_workspace` should be safe to re-enter: remove any stale directory for the
  same cell before creating a new one.
- Keep `agent_assets` keys to the normalized set the runners understand
  (`skill_dir`); unknown keys are ignored.
- Results are namespaced by `name`, so a new profile never collides with existing
  data under `results/`.
