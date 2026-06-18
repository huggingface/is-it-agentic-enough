"""On-disk run store: one JSONL **shard per (binding, harness, model, task)**.

Layout (the canonical format)::

    results/<binding>/<harness>/<model_id>/<task>.jsonl              # one line per run
    traces/<binding>/<harness>/<model_id>/<tier>__<task>__run<N>.jsonl  # one native session per file
    results/<binding>/ref.json                                      # per-binding label marker
    results/MANIFEST.json                                           # generated index

Each results line is a complete run::

    {"tier": "...", "task": "...", "run": 1, "meta": {...}, "events": [ ...canonical transcript events... ]}

Each **traces** file is one run's *native* agent session, stored **verbatim** —
i.e. the file's contents are exactly the session JSONL the agent emitted. One
session per file (not bundled) so the Hub's agent-traces viewer auto-detects and
renders each one in place, in a dataset *or* in the bucket.

**Why shard by task.** A whole-file overwrite is the only write object storage
offers, so a single ``<model>.jsonl`` per cell is safe only when one process owns
it. ``agent-eval batch --per-task`` breaks that: it launches one job per (model,
revision, task), and concurrent jobs writing the same object would clobber each
other (no atomic compare-and-swap). Sharding by task gives every per-task job its
**own** object, so concurrent writers never collide; no locks, eviction-safe. A
non-per-task job owns all of a model's task shards but runs as a single process,
so it still never races. The read side (:func:`list_runs`) merges a model's
shards back together, and the report pull (``hf buckets sync``) is incremental,
so only newly-written shards come down.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .paths import state_root


def _mirror(tree: str, binding: str, ns: str, filename: str, src: Path) -> None:
    """If ``AE_MIRROR_DIR`` is set, copy a just-written file there too.

    Used by HF Jobs (``AE_MIRROR_DIR=/bucket``) so each run is persisted to the
    bucket the moment it finishes — a crash/eviction mid-suite then keeps every
    completed run instead of losing the whole job. ``filename`` is the leaf name
    (e.g. ``classify-sentiment.jsonl`` or ``bare__classify-sentiment__run1.jsonl``);
    each file is owned by exactly one job, so this copy never races a peer.
    Best-effort: a mirror failure never breaks the run."""
    mdir = os.environ.get("AE_MIRROR_DIR")
    if not mdir:
        return
    try:
        dst = Path(mdir) / tree / binding / ns / filename
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


def _model_dir(tree: str, binding: str, ns: str) -> Path:
    """``<tree>/<binding>/<harness>/<model_id>/`` (``ns`` == ``harness/model_id``)."""
    return state_root() / tree / binding / ns


def _shard_path(tree: str, binding: str, ns: str, task: str) -> Path:
    """The task shard ``<tree>/<binding>/<harness>/<model_id>/<task>.jsonl``."""
    return _model_dir(tree, binding, ns) / f"{task}.jsonl"


def results_path(binding: str, ns: str, task: str | None = None) -> Path:
    """A model's results shard for ``task``, or its shard directory when ``task`` is None."""
    return _shard_path("results", binding, ns, task) if task else _model_dir("results", binding, ns)


def _native_trace_name(tier: str, task: str, run: int) -> str:
    """Leaf filename for a run's native session: ``<tier>__<task>__run<N>.jsonl``.
    (``tier`` and ``run`` carry no ``__``, and task ids carry no ``__``, so this
    round-trips cleanly via ``rsplit('__', 2)``.)"""
    return f"{tier}__{task}__run{int(run)}.jsonl"


def traces_path(binding: str, ns: str, tier: str | None = None, task: str | None = None,
                run: int | None = None) -> Path:
    """A run's native-session file, or the model's trace directory when unspecified."""
    if tier and task and run is not None:
        return _model_dir("traces", binding, ns) / _native_trace_name(tier, task, run)
    return _model_dir("traces", binding, ns)


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


def _read_cell_rows(tree: str, binding: str, ns: str) -> list[dict]:
    """All raw rows for a model, merged across its task shards."""
    mdir = _model_dir(tree, binding, ns)
    if not mdir.is_dir():
        return []
    rows: list[dict] = []
    for shard in sorted(mdir.glob("*.jsonl")):
        rows += _read_jsonl(shard)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# --------- results ---------


def list_runs(binding: str, ns: str) -> list[RunRecord]:
    """All runs in a model cell (merged across task shards), ordered by (tier, task, run)."""
    by_key: dict[tuple[str, str, int], RunRecord] = {}
    for o in _read_cell_rows("results", binding, ns):
        rec = RunRecord(tier=o.get("tier"), task=o.get("task"), run=int(o.get("run") or 0),
                        meta=o.get("meta") or {}, events=o.get("events") or [])
        by_key[rec.key] = rec  # last write of a (tier, task, run) wins
    return sorted(by_key.values(), key=lambda r: (r.tier or "", r.task or "", r.run))


def read_cell(binding: str, ns: str) -> dict[tuple[str, str, int], RunRecord]:
    return {r.key: r for r in list_runs(binding, ns)}


def get_run(binding: str, ns: str, tier: str, task: str, run: int) -> RunRecord | None:
    return read_cell(binding, ns).get((tier, task, int(run)))


def run_exists(binding: str, ns: str, tier: str, task: str, run: int) -> bool:
    return (tier, task, int(run)) in read_cell(binding, ns)


def upsert_run(binding: str, ns: str, rec: RunRecord) -> Path:
    """Add or replace a run's line in its **task shard**; return the shard path."""
    path = _shard_path("results", binding, ns, rec.task)
    rows = {(o.get("tier"), o.get("task"), int(o.get("run") or 0)): o for o in _read_jsonl(path)}
    rows[rec.key] = {"tier": rec.tier, "task": rec.task, "run": rec.run, "meta": rec.meta, "events": rec.events}
    _write_jsonl(path, [rows[k] for k in sorted(rows)])
    _mirror("results", binding, ns, f"{rec.task}.jsonl", path)
    return path


# --------- traces ---------


def upsert_trace(binding: str, ns: str, tier: str, task: str, run: int, raw: str) -> Path:
    """Write a run's native session **verbatim** to its own file.

    One file per run (``<tier>__<task>__run<N>.jsonl``, content = the raw native
    agent session), so the Hub's agent-traces viewer auto-detects and renders each
    one in place — both in a dataset and in the bucket. (The previous bundled
    ``{tier,task,run,raw}`` wrapper was valid JSONL but each *line* was a wrapper
    object, not a session, so nothing recognized it as a trace.)"""
    path = _model_dir("traces", binding, ns) / _native_trace_name(tier, task, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw)
    _mirror("traces", binding, ns, path.name, path)
    return path


def list_traces(binding: str, ns: str) -> list[dict]:
    """Native sessions for a model cell as ``[{tier, task, run, raw}]``.

    Reads the per-run native files (``<tier>__<task>__run<N>.jsonl``)."""
    d = _model_dir("traces", binding, ns)
    if not d.is_dir():
        return []
    out: list[dict] = []
    for p in sorted(d.glob("*__run*.jsonl")):
        tier, task, runtok = p.name[: -len(".jsonl")].rsplit("__", 2)
        out.append({"tier": tier, "task": task, "run": int(runtok[3:]), "raw": p.read_text()})
    return sorted(out, key=lambda o: (o["tier"], o["task"], o["run"]))


# --------- discovery ---------


def iter_cells(refs: list[str] | None = None) -> Iterator[tuple[str, str]]:
    """Yield each distinct ``(binding, ns)`` model cell once. ``ns`` == ``harness/model_id``.

    The shard layout is ``<binding>/<harness>/<model>/<task>.jsonl`` (4 levels);
    ``ref.json`` (2 levels) and ``MANIFEST.json`` (1 level) are naturally excluded
    by the glob."""
    root = state_root() / "results"
    if not root.exists():
        return
    seen: set[tuple[str, str]] = set()
    for path in root.glob("*/*/*/*.jsonl"):
        binding, harness, model, _fname = path.relative_to(root).parts
        if refs and binding not in refs:
            continue
        seen.add((binding, f"{harness}/{model}"))
    yield from sorted(seen)


def cells_for_binding(binding: str) -> list[str]:
    """Namespaces (``harness/model_id``) present for one binding."""
    return [ns for b, ns in iter_cells([binding]) if b == binding]


def discover_tasks(binding: str, ns: str) -> list[str]:
    return sorted({r.task for r in list_runs(binding, ns)})
