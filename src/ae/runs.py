"""Shared run-parsing primitives: turn a stored ``RunRecord`` into a scored ``Run``.

Profile-agnostic: correctness comes from a task's ``match`` mode via
:mod:`ae.matcher`, and behavior classification from the profile's
:class:`~ae.markers.Marker`s. The report builder (:mod:`ae.report`) is the sole
consumer — results are viewed only in the web UI it generates.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import matcher
from .transcript import parse_events


@dataclass
class Run:
    tool_calls: list[tuple[str, dict]]
    tool_results: list[str]
    final: str | None

    # From meta.json (fall back to zeros/None if meta missing or partial):
    elapsed: float
    tokens_in: int
    tokens_out: int
    tokens_cache_read: int
    tokens_cache_creation: int
    exit_code: int
    status: str                    # "ok" | "empty" | "budget_tool_calls" | "timeout" | "error"

    # Derived:
    errored_calls: int             # count of tool_results with is_error=true
    error_details: list[str]       # first-line snippet of each errored result
    matched_expected: bool | None  # None if the task defines no expected response
    first_success_turn: int | None  # tool-call index where expected first appeared in a result


def parse(record, task: dict) -> Run:
    """Build a :class:`Run` from a :class:`ae.store.RunRecord` (events + meta),
    scored against ``task`` (its ``expected`` / ``match`` fields; pass ``{}`` for
    a task with no expected answer)."""
    tx = parse_events(record.events)
    expected = (task.get("expected") or "").strip() or None
    mode = task.get("match") or matcher.DEFAULT_MODE

    tool_calls = [(s.name, s.input) for s in tx.steps]
    tool_results = [s.result for s in tx.steps]
    errored_calls = sum(1 for s in tx.steps if s.is_error)
    error_details = [
        (s.result.strip().splitlines() or [""])[0][:140] for s in tx.steps if s.is_error
    ]

    first_success_turn: int | None = None
    if expected:
        needle = expected.lower()
        for i, s in enumerate(tx.steps, 1):
            if needle in s.result.lower():
                first_success_turn = i
                break

    final = tx.final
    meta = record.meta or {}
    tokens = meta.get("tokens") or {}

    matched = matcher.check(expected, final, mode) if (expected and final is not None) else None

    # Silent-failure guard: a run that generated nothing — no output tokens, no
    # tool calls, no final answer — exited "ok" but did no work (e.g. an unknown
    # model id that the provider accepted then returned an empty completion).
    # Reclassify it as "empty" so it surfaces as an error instead of a silent 0.
    status = str(meta.get("status") or "ok")
    if status == "ok" and not tool_calls and not (final and final.strip()) \
            and int(tokens.get("out") or 0) == 0:
        status = "empty"

    return Run(
        tool_calls=tool_calls,
        tool_results=tool_results,
        final=final,
        elapsed=float(meta.get("elapsed_sec") or 0.0),
        tokens_in=int(tokens.get("in") or 0),
        tokens_out=int(tokens.get("out") or 0),
        tokens_cache_read=int(tokens.get("cache_read") or 0),
        tokens_cache_creation=int(tokens.get("cache_creation") or 0),
        exit_code=int(meta.get("exit_code") or 0),
        status=status,
        errored_calls=errored_calls,
        error_details=error_details,
        matched_expected=matched,
        first_success_turn=first_success_turn,
    )


def step_kind(name: str, inp: dict) -> str:
    """A neutral, profile-agnostic label for one tool call (for human display)."""
    if name == "Bash":
        cmd = (inp.get("command") or "").strip()
        head = cmd.split()[0] if cmd else ""
        return f"bash:{head.split('/')[-1]}" if head else "bash"
    return name.lower()
