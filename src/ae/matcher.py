"""Score an agent's final answer against a task's expected response.

A task in ``tasks.yaml`` may declare ``match:`` to choose how its ``expected``
string is compared to the agent's final answer:

- ``substring`` (default) — case-insensitive containment (the historical behavior).
- ``exact`` — case-insensitive full-string equality (whitespace-trimmed).
- ``regex`` — ``re.search`` of ``expected`` (a pattern) against the answer.
- ``judge`` — semantic grading by an LLM. Not implemented yet (raises).

Kept deliberately tiny and dependency-free so the run-scoring layer
(:mod:`ae.runs`) and any future scorer can share one definition of "matched".
"""

from __future__ import annotations

import re

MODES = ("substring", "exact", "regex", "judge")
DEFAULT_MODE = "substring"


def check(expected: str, final: str | None, mode: str = DEFAULT_MODE) -> bool:
    """Return whether ``final`` matches ``expected`` under ``mode``."""
    if final is None:
        return False
    if mode == "substring":
        return expected.strip().lower() in final.lower()
    if mode == "exact":
        return expected.strip().lower() == final.strip().lower()
    if mode == "regex":
        return re.search(expected, final, re.IGNORECASE) is not None
    if mode == "judge":
        raise NotImplementedError("`match: judge` (LLM grading) is not implemented yet")
    raise ValueError(f"unknown match mode: {mode!r} (expected one of {MODES})")
