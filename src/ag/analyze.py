"""Per-binding analysis + shared run-loading primitives used by compare.

Profile-agnostic: behavior classification comes from the profile's
:class:`~ag.markers.Marker`s (adoption flags), not a hard-wired CLI-vs-Python
bucket. Correctness comes from the task's ``match`` mode via :mod:`ag.matcher`.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import matcher
from .markers import fired
from .paths import package_data_path, results_dir
from .transcript import parse_transcript
from .util import fmt_tokens, median

# Fallback tiers when a caller doesn't pass any (keeps direct/legacy callers
# working). The CLI passes the active profile's tiers instead.
_DEFAULT_TIERS = ("bare", "clone", "skill")


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
    status: str                    # "ok" | "budget_tool_calls" | "timeout" | "error"

    # Derived:
    errored_calls: int             # count of tool_results with is_error=true
    error_details: list[str]       # first-line snippet of each errored result
    matched_expected: bool | None  # None if the task defines no expected response
    first_success_turn: int | None  # tool-call index where expected first appeared in a result


# --------- task metadata ---------


@lru_cache(maxsize=1)
def _tasks_data() -> list[dict]:
    import yaml

    with open(package_data_path("tasks.yaml")) as f:
        return yaml.safe_load(f).get("tasks", []) or []


def _task_expectations() -> dict[str, str]:
    """``{task_id: expected}`` (stripped) for tasks that define one."""
    out: dict[str, str] = {}
    for task in _tasks_data():
        exp = task.get("expected")
        if isinstance(exp, str) and exp.strip():
            out[task["id"]] = exp.strip()
    return out


def _task_match_modes() -> dict[str, str]:
    """``{task_id: match_mode}`` (default ``substring``)."""
    return {t["id"]: t.get("match", matcher.DEFAULT_MODE) for t in _tasks_data()}


# --------- parsing ---------


def parse(path: Path, task_id: str) -> Run:
    tx = parse_transcript(path)
    expected = _task_expectations().get(task_id)
    mode = _task_match_modes().get(task_id, matcher.DEFAULT_MODE)

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
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    tokens = meta.get("tokens") or {}

    matched = matcher.check(expected, final, mode) if (expected and final is not None) else None

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
        status=str(meta.get("status") or "ok"),
        errored_calls=errored_calls,
        error_details=error_details,
        matched_expected=matched,
        first_success_turn=first_success_turn,
    )


# --------- classification (generic) ---------


def step_kind(name: str, inp: dict) -> str:
    """A neutral, profile-agnostic label for one tool call (for human display)."""
    if name == "Bash":
        cmd = (inp.get("command") or "").strip()
        head = cmd.split()[0] if cmd else ""
        return f"bash:{head.split('/')[-1]}" if head else "bash"
    return name.lower()


def marker_fired_counts(runs: list[Run], markers: list) -> dict[str, int]:
    """``{marker_name: number_of_runs_that_fired_it}`` over ``runs``."""
    counts: dict[str, int] = {m.name: 0 for m in markers}
    for r in runs:
        for name, hit in fired(markers, r).items():
            if hit:
                counts[name] += 1
    return counts


# --------- aggregates for table cells ---------


def cell(runs: list[Run], markers: list | None = None) -> str:
    """Compact cell:

        ✓match · !failed/total · ⇢first-success · 🏷markers · ⏻abort · time · new · repeat · out

    Fields that are zero / not applicable are omitted.

    - ``✓match`` — runs whose final answer matched the task's ``expected`` (under
      its ``match`` mode), over runs where an expected was defined.
    - ``!failed/total`` — errored tool calls / total tool calls.
    - ``⇢first-success`` — median tool-call index at which the expected first
      appeared in a tool result. Lower is better.
    - ``🏷`` — per-profile behavior markers that fired (``name=k/n``); independent
      and possibly overlapping. Empty when the profile defines no markers.
    - ``new`` / ``repeat`` / ``out`` — token accounting (see report glossary).
    """
    markers = markers or []
    if not runs:
        return "—"
    n = len(runs)
    parts: list[str] = []

    matched = [r.matched_expected for r in runs if r.matched_expected is not None]
    if matched:
        parts.append(f"✓{sum(matched)}/{len(matched)}")

    total_calls = sum(len(r.tool_calls) for r in runs)
    failed_calls = sum(r.errored_calls for r in runs)
    if failed_calls:
        parts.append(f"!{failed_calls}/{total_calls}")

    fs_turns = [r.first_success_turn for r in runs if r.first_success_turn is not None]
    if fs_turns:
        parts.append(f"⇢{int(median([float(t) for t in fs_turns]))}")

    fired_parts = [f"{name}={c}/{n}" for name, c in marker_fired_counts(runs, markers).items() if c]
    if fired_parts:
        parts.append("🏷" + " ".join(fired_parts))

    bad: Counter[str] = Counter()
    for r in runs:
        if r.status != "ok":
            bad[r.status] += 1
        elif r.exit_code != 0:
            # Finished but exited nonzero (e.g. invalid model) — would otherwise
            # masquerade as a clean run.
            bad["error"] += 1
    if bad:
        parts.append("⏻" + ",".join(f"{k}:{v}" for k, v in bad.items()))

    parts.append(f"{median([r.elapsed for r in runs]):.0f}s")
    parts.append(f"new:{fmt_tokens(int(median([r.tokens_in + r.tokens_cache_creation for r in runs])))}")
    parts.append(f"repeat:{fmt_tokens(int(median([r.tokens_cache_read for r in runs])))}")
    parts.append(f"out:{fmt_tokens(int(median([r.tokens_out for r in runs])))}")
    return " · ".join(parts)


# --------- loading ---------


def load_runs(short_sha: str, variant: str, task_id: str, ns: str | None = None) -> list[Run]:
    return [
        parse(p, task_id)
        for p in sorted(results_dir(short_sha, ns).glob(f"{variant}__{task_id}__run*.jsonl"))
    ]


# --------- rendering ---------


def _render_run(run: Run, idx: int) -> list[str]:
    err_suffix = f", errors:{run.errored_calls}" if run.errored_calls else ""
    new_tok = run.tokens_in + run.tokens_cache_creation
    lines = [f"Run {idx} — {len(run.tool_calls)} tool calls, {run.elapsed:.0f}s, "
             f"new:{new_tok} repeat:{run.tokens_cache_read} out:{run.tokens_out}{err_suffix}"]
    if not run.tool_calls:
        lines.append("  (answered from model knowledge)")
    for i, (n, inp) in enumerate(run.tool_calls, 1):
        extra = ""
        if n == "Bash":
            cmd = (inp.get("command") or "").replace("\n", " ⏎ ")
            extra = f" `{cmd[:140]}{'…' if len(cmd) > 140 else ''}`"
        elif n in ("Write", "Read"):
            extra = f" `{inp.get('file_path', '')}`"
        lines.append(f"  {i}. {step_kind(n, inp)}{extra}")
    for detail in run.error_details:
        lines.append(f"    ✗ {detail}")
    if run.final:
        lines.append(f"  → {run.final.replace(chr(10), ' ')[:180]}")
    return lines


def _task_section(
    short_sha: str,
    task_id: str,
    ns: str | None = None,
    tiers: tuple[str, ...] | list[str] = _DEFAULT_TIERS,
    markers: list | None = None,
) -> str:
    lines: list[str] = [f"## {task_id}", ""]
    any_runs = False
    for variant in tiers:
        runs = load_runs(short_sha, variant, task_id, ns)
        if not runs:
            continue
        any_runs = True
        lines.append(f"### {variant}  — {cell(runs, markers)}")
        for i, r in enumerate(runs, 1):
            lines.extend(_render_run(r, i))
        lines.append("")
    return "\n".join(lines) if any_runs else ""


def discover_task_ids(short_sha: str, ns: str | None = None) -> list[str]:
    ids: set[str] = set()
    for path in results_dir(short_sha, ns).glob("*.jsonl"):
        parts = path.stem.split("__")
        if len(parts) == 3:
            ids.add(parts[1])
    return sorted(ids)


def analyze(
    short_sha: str,
    task_id: str | None = None,
    ns: str | None = None,
    tiers: tuple[str, ...] | list[str] = _DEFAULT_TIERS,
    markers: list | None = None,
) -> str:
    tasks = [task_id] if task_id else discover_task_ids(short_sha, ns)
    if not tasks:
        loc = f" ({ns})" if ns else ""
        return f"No results for {short_sha}{loc}"
    header = f"# Agent behavior @ {short_sha}"
    if ns:
        header += f"  [{ns}]"
    out = [header, ""]
    for tid in tasks:
        section = _task_section(short_sha, tid, ns, tiers, markers)
        if section:
            out.append(section)
    return "\n".join(out)
