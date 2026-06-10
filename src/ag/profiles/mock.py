"""A fast, fake profile for exercising the UI / reports end-to-end.

No git, no venv, no installation, no real agent. Paired with ``--runner mock``,
a whole suite finishes in seconds and produces realistic-looking (randomized)
results, traces, and reports — so you can iterate on the dashboard / report /
sync UI, and smoke-test the full pipeline, without touching transformers.

    ag suite dev --profile mock --runner mock
    ag diff v1..v2 --profile mock --runner mock
    ag report --profile mock           # then eyeball the HTML

Bindings are arbitrary labels (no resolution); tiers and markers mirror the
transformers profile so the report looks the same shape.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from ..paths import results_dir
from ..profile import BuiltEnv, Profile, register
from .transformers import MARKERS, TIERS


def _safe(s: str) -> str:
    out = "".join(c if (c.isalnum() or c in "-._") else "-" for c in s)[:24]
    return out or "mock"


class MockProfile(Profile):
    name = "mock"

    def expand_bindings(self, spec: list[str]) -> list[str]:
        toks: list[str] = []
        for s in spec:
            toks += [p for p in s.split("..") if p] if ".." in s else [s]
        seen: set[str] = set()
        out: list[str] = []
        for t in toks:
            b = _safe(t)
            if b not in seen:
                seen.add(b)
                out.append(b)
        return out

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


register(MockProfile())
