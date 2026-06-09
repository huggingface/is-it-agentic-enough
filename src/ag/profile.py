"""Environment profiles: the pluggable seam that makes ``ag`` general.

A **profile** defines the environment a task suite runs inside, and the comparison
axis. The harness core only knows tasks, tiers, runners, and expected-response
matching; everything environment-specific (how to build a sandbox, what
"assistance tiers" exist, how to seed the agent's workspace) lives behind a
:class:`Profile`.

Vocabulary:

- **binding** — one point on the comparison axis the suite sweeps. For the
  ``transformers`` profile a binding is a git revision (``transformers@<sha>``);
  ``ag diff A..B`` runs the suite at bindings A and B and compares them.
- **tier** — "how much help the agent gets" (the generic form of the old
  ``bare``/``clone``/``skill`` variants). A profile declares its own tiers and
  decides how each seeds the workspace and what assets the agent is handed.
- **assets** — per-tier extras passed to the runner, normalized so runners stay
  profile-agnostic: ``plugin_dir`` (Claude Code ``--plugin-dir``) and
  ``skill_dir`` (Pi ``--skill``); both point at an Agent-Skills layout. Empty
  for tiers/profiles with no agent assets.

Profiles register themselves in :data:`_REGISTRY`; resolve one with
:func:`get_profile`. The default profile is ``transformers`` so existing
invocations are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class BuiltEnv:
    """The result of preparing a profile's sandbox for one binding."""

    binding: str                       # canonical id (transformers: 10-char sha)
    python: Path                       # interpreter the agent's task code runs under
    available_tiers: list[str]         # tiers usable at this binding (e.g. skill may be absent)
    cfg_dir: Path | None = None        # profile-internal cache/scratch root
    label: str | None = None           # display label (branch/tag/title)
    extra: dict = field(default_factory=dict)


@runtime_checkable
class Profile(Protocol):
    """A pluggable environment + comparison axis. See module docstring."""

    name: str

    def expand_bindings(self, spec: list[str]) -> list[str]:
        """Turn a CLI spec (e.g. a ``A..B`` ref range) into canonical binding ids."""
        ...

    def build(self, ref: str, *, name: str | None = None) -> BuiltEnv:
        """Prepare (or reuse a cached) sandbox for ``ref``; returns its :class:`BuiltEnv`."""
        ...

    def all_tiers(self) -> list[str]:
        """All tiers this profile defines, most-bare first."""
        ...

    def prepare_workspace(self, built: BuiltEnv, tier: str, task_id: str, run_idx: int) -> Path:
        """Create and return a fresh cwd for one run under ``tier``."""
        ...

    def remove_workspace(self, ws: Path) -> None:
        """Tear down a workspace created by :meth:`prepare_workspace` (best-effort)."""
        ...

    def agent_assets(self, built: BuiltEnv, tier: str) -> dict:
        """Normalized per-tier assets for the runner (``plugin_dir``/``skill_dir``)."""
        ...

    def markers(self) -> list:
        """Behavior markers (:class:`ag.markers.Marker`) whose adoption the report
        tracks across bindings. Empty for profiles that don't classify behavior."""
        ...


_REGISTRY: dict[str, Profile] = {}


def register(profile: Profile) -> Profile:
    _REGISTRY[profile.name] = profile
    return profile


def get_profile(name: str | None) -> Profile:
    key = name or "transformers"
    if key not in _REGISTRY:
        # Import side-effect registers built-in profiles on first use.
        from . import profiles  # noqa: F401
    if key not in _REGISTRY:
        raise SystemExit(f"Unknown profile: {key!r} (have {sorted(_REGISTRY)})")
    return _REGISTRY[key]
