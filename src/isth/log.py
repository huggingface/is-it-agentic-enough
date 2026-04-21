"""Small helpers for live, timestamped progress output.

Everything goes to stderr so stdout stays clean for tools that pipe the
report (``isth analyze ... > report.md``). Timestamps are seconds since
the current process started, which is what you actually want to know
when watching a run in progress.

All output goes through a shared :class:`rich.console.Console` so that a
live dashboard (see ``dashboard.py``) can be active without log lines
garbling its frame — Rich routes prints above the live region
automatically.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.text import Text


_START = time.time()
_VERBOSE = False
_CONSOLE = Console(stderr=True, highlight=False, soft_wrap=True)


def set_verbose(value: bool) -> None:
    """Enable/disable per-tool-call event logging for the current process."""
    global _VERBOSE
    _VERBOSE = value


def is_verbose() -> bool:
    return _VERBOSE


def get_console() -> Console:
    """Shared stderr console; reuse this from the dashboard for clean output."""
    return _CONSOLE


def log(msg: str) -> None:
    elapsed = time.time() - _START
    line = Text()
    line.append(f"[{elapsed:7.1f}s] ", style="dim")
    line.append(msg)
    _CONSOLE.print(line)


def vlog(msg: str) -> None:
    """Verbose-only log: prints only when ``set_verbose(True)`` has been called."""
    if _VERBOSE:
        log(msg)


def _trunc(s: str, n: int = 140) -> str:
    s = s.replace("\n", " ⏎ ")
    return s if len(s) <= n else s[: n - 1] + "…"


TokenDelta = dict[str, int]  # keys: in, out, cache_read, cache_creation


def extract_usage(event: dict[str, Any]) -> TokenDelta:
    """Pull token counts from an assistant event. Returns an empty dict if absent."""
    usage = event.get("message", {}).get("usage") or {}
    return {
        "in": int(usage.get("input_tokens") or 0),
        "out": int(usage.get("output_tokens") or 0),
        "cache_read": int(usage.get("cache_read_input_tokens") or 0),
        "cache_creation": int(usage.get("cache_creation_input_tokens") or 0),
    }


def _fmt_tokens(d: TokenDelta) -> str:
    if not d or (d["in"] == 0 and d["out"] == 0 and d["cache_read"] == 0 and d["cache_creation"] == 0):
        return ""
    parts = []
    if d["in"]:
        parts.append(f"in:{d['in']}")
    if d["out"]:
        parts.append(f"out:{d['out']}")
    if d["cache_read"]:
        parts.append(f"cache:{d['cache_read']}")
    return f"[{' '.join(parts)}] " if parts else ""


def summarize_event(event: dict[str, Any]) -> str | None:
    """Turn one stream-json event into a short human summary, or None to skip.

    Assistant events are annotated with per-turn token usage (from
    ``message.usage``). Tokens are counted once per assistant turn — they
    appear on the first content block we render from that turn.
    """
    t = event.get("type")
    if t == "assistant":
        tok = _fmt_tokens(extract_usage(event))
        contents = event.get("message", {}).get("content", []) or []
        for b in contents:
            btype = b.get("type")
            if btype == "tool_use":
                name = b.get("name", "?")
                inp = b.get("input", {}) or {}
                if name == "Bash":
                    return f"  ⏵ Bash         {tok}{_trunc(inp.get('command') or '')}"
                if name == "Read":
                    return f"  ⏵ Read         {tok}{inp.get('file_path', '')}"
                if name == "Write":
                    return f"  ⏵ Write        {tok}{inp.get('file_path', '')}"
                if name == "Edit":
                    return f"  ⏵ Edit         {tok}{inp.get('file_path', '')}"
                if name == "Grep":
                    return f"  ⏵ Grep         {tok}{_trunc(inp.get('pattern') or '')}"
                if name == "Glob":
                    return f"  ⏵ Glob         {tok}{inp.get('pattern', '')}"
                return f"  ⏵ {name:<12} {tok}{_trunc(str(inp))}"
            if btype == "text":
                txt = (b.get("text") or "").strip()
                if txt:
                    return f"  ✎ text         {tok}{_trunc(txt)}"
    elif t == "user":
        for b in event.get("message", {}).get("content", []) or []:
            if b.get("type") == "tool_result":
                content = b.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for c in content:
                        if isinstance(c, dict) and "text" in c:
                            parts.append(c["text"])
                    content = "\n".join(parts) if parts else str(content)
                return f"  ⇦ result       {_trunc(str(content))}"
    elif t == "system" and event.get("subtype") == "init":
        return f"  session        {event.get('session_id', '')[:8]}"
    return None
