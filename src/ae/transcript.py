"""Shared parsing of a normalized stream-json run transcript.

Both the per-commit report (:mod:`ae.analyze`) and the per-cell drill-down
(:mod:`ae.explain`) need the same thing from a run's ``.jsonl``: the ordered
sequence of tool calls, each paired with its ``tool_result`` (content +
``is_error``), plus the final answer. This module is the single place that walks
the file and pairs ``tool_use`` ↔ ``tool_result`` by id; the two readers build
their own views on top of :class:`Transcript`.

The walk tolerates in-flight writes (a partial final line) and missing files, so
it is safe to call while ``agent-eval diff`` is still producing the file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolStep:
    """One ``tool_use`` and the ``tool_result`` it produced."""

    name: str            # tool name (Bash, Read, Write, ...)
    input: dict          # tool-call arguments
    result: str = ""     # paired tool_result content ("" until/unless paired)
    is_error: bool = False


@dataclass
class Transcript:
    steps: list[ToolStep] = field(default_factory=list)
    final: str | None = None     # text of the final `result` event, if any
    broken: bool = False         # a line failed to decode (partial in-flight write)
    missing: bool = False        # the file did not exist


def tool_result_content(block: dict) -> str:
    """Coerce a ``tool_result`` block's ``content`` to a plain string (it may be
    a string or a list of ``{type:text,text}`` blocks)."""
    c = block.get("content", "")
    if isinstance(c, list):
        return "\n".join(
            str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in c
        )
    return str(c)


def parse_events(events: list | None) -> Transcript:
    """Walk an in-memory list of canonical events (the per-run format the store
    holds) and return its ordered tool steps + final answer. This is the primary
    entry now that runs are stored as event lists rather than per-run files."""
    tx = Transcript()
    by_id: dict[str, ToolStep] = {}
    last_text: str | None = None
    for e in events or []:
        if not isinstance(e, dict):
            continue
        t = e.get("type")
        if t == "assistant":
            for b in e.get("message", {}).get("content", []) or []:
                if b.get("type") == "tool_use":
                    step = ToolStep(name=b.get("name", "?"), input=b.get("input") or {})
                    tx.steps.append(step)
                    tid = b.get("id")
                    if tid:
                        by_id[tid] = step
                elif b.get("type") == "text" and (b.get("text") or "").strip():
                    last_text = b["text"]
        elif t == "user":
            for b in e.get("message", {}).get("content", []) or []:
                if b.get("type") == "tool_result":
                    tid = b.get("tool_use_id")
                    step = by_id.get(tid) if tid else None
                    if step is not None:
                        step.is_error = bool(b.get("is_error"))
                        step.result = tool_result_content(b)
        elif t == "result":
            tx.final = e.get("result") or ""
    # Some runners (Pi) never emit a `result` event; their final answer is the
    # last assistant text.
    if tx.final is None:
        tx.final = last_text
    return tx


def parse_transcript(path: Path) -> Transcript:
    """Compatibility shim: read a run's ``.jsonl`` file and parse it. Retained for
    callers/tests that still hold a per-run file; the store path uses
    :func:`parse_events` directly."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        tx = Transcript()
        tx.missing = True
        return tx
    events = []
    broken = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            broken = True  # partial in-flight write; keep what parsed so far
            break
    tx = parse_events(events)
    tx.broken = broken
    return tx
