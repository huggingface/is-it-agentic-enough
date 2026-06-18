"""Shared test fixtures.

The harness keys its data root off ``AE_DATA_DIR`` (read live in
``ae.paths.state_root``), so each test gets an isolated tmp root via the
``data_root`` fixture. ``write_run`` lays down a synthetic run (transcript +
meta) in the real on-disk layout so the read-side can be tested without running
an agent.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def data_root(tmp_path, monkeypatch) -> Path:
    """Isolate all harness state (results/, traces/, …) under a tmp dir."""
    monkeypatch.setenv("AE_DATA_DIR", str(tmp_path))
    return tmp_path


def _events(*, tool_calls=(), final="done", final_error=False):
    """Build a canonical stream-json event list.

    ``tool_calls`` is a list of ``(name, input, result, is_error)`` tuples.
    """
    evs: list[dict] = [{"type": "system", "subtype": "init", "session_id": "abc"}]
    for i, (name, inp, result, is_error) in enumerate(tool_calls, 1):
        evs.append({
            "type": "assistant",
            "message": {
                "content": [{"type": "tool_use", "id": f"t{i}", "name": name, "input": inp}],
                "usage": {"input_tokens": 100, "output_tokens": 20,
                          "cache_read_input_tokens": 5, "cache_creation_input_tokens": 0},
            },
        })
        evs.append({
            "type": "user",
            "message": {"content": [{"type": "tool_result", "tool_use_id": f"t{i}",
                                     "content": result, "is_error": is_error}]},
        })
    evs.append({"type": "result", "result": final, "is_error": final_error})
    return evs


@pytest.fixture
def write_run(data_root):
    """Factory: write a synthetic run as one line in its task shard
    ``results/<binding>/<harness>/<model_id>/<task>.jsonl`` via the store, and return
    the :class:`ae.store.RunRecord` (the read-side now parses records, not files)."""
    from ae import store

    def _write(binding, tier, task, run=1, *, ns="pi/default",
               tool_calls=(), final="done", final_error=False,
               status="ok", exit_code=0, elapsed=12.0,
               tokens=None):
        evs = _events(tool_calls=tool_calls, final=final, final_error=final_error)
        meta = {
            "sha": binding + "0" * 30, "short_sha": binding, "ref": binding,
            "variant": tier, "task_id": task, "run_index": run,
            "runner": ns.split("/", 1)[0], "model": None, "status": status,
            "tool_call_count": len(tool_calls), "exit_code": exit_code,
            "elapsed_sec": elapsed,
            "tokens": tokens or {"in": 100, "out": 20, "cache_read": 5, "cache_creation": 0},
        }
        rec = store.RunRecord(tier=tier, task=task, run=run, meta=meta, events=evs)
        store.upsert_run(binding, ns, rec)
        return rec

    return _write
