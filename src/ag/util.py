"""Small dependency-free helpers shared across the read-side modules."""

from __future__ import annotations

import json
from pathlib import Path


def read_meta(path: str | Path) -> dict | None:
    """Load a run's ``.meta.json`` sidecar, or ``None`` if absent/unparseable."""
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def median(xs, default=0.0):
    """Median of ``xs`` (averaging the two middle values for even length), or
    ``default`` when empty."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return default
    return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def fmt_tokens(n: int) -> str:
    """Compact token count: ``950`` → ``950``, ``12345`` → ``12k``, ``2_500_000`` → ``2.5M``."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n // 1000}k"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)
