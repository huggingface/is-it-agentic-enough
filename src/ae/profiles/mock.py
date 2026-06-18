"""A fast, fake profile for exercising the UI / reports end-to-end.

No git, no venv, no installation, no real agent. Paired with the ``mock`` runner,
a whole suite finishes in seconds and produces realistic-looking (randomized)
results, traces, and reports, so you can iterate on the report / sync UI and
smoke-test the full pipeline without touching transformers.

Launch it from a batch YAML with ``{model: m, runner: mock}`` (mock cells run
locally), then ``agent-eval report mock`` to eyeball the HTML.

Bindings are arbitrary labels (no resolution); tiers, markers, and tasks mirror
the transformers profile so the report looks the same shape.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from ..paths import results_dir
from ..profile import BuiltEnv, Profile, expand_spec, register
from .transformers import MARKERS, TIERS
from .transformers import tasks as _transformers_tasks


def _safe(s: str) -> str:
    out = "".join(c if (c.isalnum() or c in "-._") else "-" for c in s)[:24]
    return out or "mock"


class MockProfile(Profile):
    name = "mock"

    def expand_bindings(self, spec: list[str]) -> list[str]:
        return expand_spec(spec, _safe)

    def build(self, ref: str, *, name: str | None = None) -> BuiltEnv:
        b = _safe(ref)
        # Record the binding label so `--name` maps a display name in the report
        # (same ref.json marker the report reads; mock bindings have no git kind).
        (results_dir(b) / "ref.json").write_text(
            json.dumps({"ref": ref, "name": name, "kind": "commit", "profile": "mock"})
        )
        return BuiltEnv(
            binding=b,
            python=Path(sys.executable),
            available_tiers=list(TIERS),
            cfg_dir=None,
            label=name or ref,
            extra={"sha": b},
        )

    def all_tiers(self) -> list[str]:
        return list(TIERS)

    def prepare_workspace(self, built: BuiltEnv, tier: str, task_id: str, run_idx: int) -> Path:
        # Encode the cell in the dir name so the mock runner can read it back.
        return Path(tempfile.mkdtemp(prefix=f"{built.binding}__{tier}__{task_id}__run{run_idx}__"))

    def remove_workspace(self, ws: Path) -> None:
        shutil.rmtree(ws, ignore_errors=True)

    def agent_assets(self, built: BuiltEnv, tier: str) -> dict:
        return {}  # the mock runner ignores assets

    def markers(self) -> list:
        return list(MARKERS)

    def tasks(self) -> dict[str, dict]:
        return _transformers_tasks()  # reuse the reference suite, like TIERS/MARKERS


register(MockProfile())
