"""On-disk run store: one JSONL file per (binding, harness, model) cell.

Layout (the canonical format)::

    results/<binding>/<harness>/<model_id>.jsonl   # one line per run
    traces/<binding>/<harness>/<model_id>.jsonl    # one line per run (native session)
    results/<binding>/ref.json                     # per-binding label marker
    results/MANIFEST.json                          # generated index

Each results line is a complete run::

    {"tier": "...", "task": "...", "run": 1, "meta": {...}, "events": [ ...canonical transcript events... ]}

Each traces line bundles the agent's *native* session verbatim::

    {"tier": "...", "task": "...", "run": 1, "raw": "<native session file text>"}

This replaces the previous one-file-per-run layout, which made bucket sync slow
(thousands of tiny objects to upload/download). Bundling to one file per
model-per-revision keeps the object count low while staying append/merge
friendly: a binding×model cell is produced by a single suite run (one HF Job),
so writes to a given cell never race across processes, and within a suite runs
execute sequentially.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .paths import state_root


def _mirror(tree: str, binding: str, ns: str, src: Path) -> None:
    """If ``AG_MIRROR_DIR`` is set, copy a just-written cell file there too.

    Used by HF Jobs (``AG_MIRROR_DIR=/bucket``) so each run is persisted to the
    bucket the moment it finishes — a crash/eviction mid-suite then keeps every
    completed run instead of losing the whole job. Best-effort: a mirror failure
    never breaks the run."""
    mdir = os.environ.get("AG_MIRROR_DIR")
    if not mdir:
        return
    try:
        dst = Path(mdir) / tree / binding / f"{ns}.jsonl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    except Exception:  # noqa: BLE001
        pass


@dataclass
class RunRecord:
    """One run: its identity, meta.json payload, and canonical transcript events."""

    tier: str
    task: str
    run: int
    meta: dict = field(default_factory=dict)
    events: list = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.tier, self.task, int(self.run))


def _cell_file(tree: str, binding: str, ns: str) -> Path:
    """``<tree>/<binding>/<harness>/<model_id>.jsonl`` (``ns`` == ``harness/model_id``)."""
    return state_root() / tree / binding / f"{ns}.jsonl"


def results_path(binding: str, ns: str) -> Path:
    return _cell_file("results", binding, ns)


def traces_path(binding: str, ns: str) -> Path:
    return _cell_file("traces", binding, ns)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a partial trailing line
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# --------- results ---------


def list_runs(binding: str, ns: str) -> list[RunRecord]:
    """All runs in a cell, ordered by (tier, task, run)."""
    recs = [
        RunRecord(tier=o.get("tier"), task=o.get("task"), run=int(o.get("run") or 0),
                  meta=o.get("meta") or {}, events=o.get("events") or [])
        for o in _read_jsonl(results_path(binding, ns))
    ]
    return sorted(recs, key=lambda r: (r.tier or "", r.task or "", r.run))


def read_cell(binding: str, ns: str) -> dict[tuple[str, str, int], RunRecord]:
    return {r.key: r for r in list_runs(binding, ns)}


def get_run(binding: str, ns: str, tier: str, task: str, run: int) -> RunRecord | None:
    return read_cell(binding, ns).get((tier, task, int(run)))


def run_exists(binding: str, ns: str, tier: str, task: str, run: int) -> bool:
    return (tier, task, int(run)) in read_cell(binding, ns)


def upsert_run(binding: str, ns: str, rec: RunRecord) -> Path:
    """Add or replace a run's line in its cell file; return the cell path."""
    cell = read_cell(binding, ns)
    cell[rec.key] = rec
    rows = [
        {"tier": r.tier, "task": r.task, "run": r.run, "meta": r.meta, "events": r.events}
        for r in sorted(cell.values(), key=lambda r: (r.tier, r.task, r.run))
    ]
    path = results_path(binding, ns)
    _write_jsonl(path, rows)
    _mirror("results", binding, ns, path)
    return path


# --------- traces ---------


def upsert_trace(binding: str, ns: str, tier: str, task: str, run: int, raw: str) -> Path:
    """Add or replace a run's native session (stored verbatim as ``raw``)."""
    path = traces_path(binding, ns)
    rows = {(o.get("tier"), o.get("task"), int(o.get("run") or 0)): o for o in _read_jsonl(path)}
    rows[(tier, task, int(run))] = {"tier": tier, "task": task, "run": int(run), "raw": raw}
    _write_jsonl(path, [rows[k] for k in sorted(rows)])
    _mirror("traces", binding, ns, path)
    return path


def list_traces(binding: str, ns: str) -> list[dict]:
    """Native-session rows for a cell: ``[{tier, task, run, raw}]``."""
    return sorted(_read_jsonl(traces_path(binding, ns)),
                  key=lambda o: (o.get("tier") or "", o.get("task") or "", int(o.get("run") or 0)))


# --------- discovery ---------


def iter_cells(refs: list[str] | None = None) -> Iterator[tuple[str, str]]:
    """Yield ``(binding, ns)`` for every results cell. ``ns`` == ``harness/model_id``.

    ``ref.json`` (``results/<binding>/ref.json``, 2 levels) and ``MANIFEST.json``
    (1 level) are naturally excluded by the 3-level glob."""
    root = state_root() / "results"
    if not root.exists():
        return
    for path in sorted(root.glob("*/*/*.jsonl")):
        binding, harness, fname = path.relative_to(root).parts
        if refs and binding not in refs:
            continue
        yield binding, f"{harness}/{fname[: -len('.jsonl')]}"


def cells_for_binding(binding: str) -> list[str]:
    """Namespaces (``harness/model_id``) present for one binding."""
    return [ns for b, ns in iter_cells([binding]) if b == binding]


def discover_tasks(binding: str, ns: str) -> list[str]:
    return sorted({r.task for r in list_runs(binding, ns)})
