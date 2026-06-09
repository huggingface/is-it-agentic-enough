"""``ag explain`` — focused, per-cell breakdown for one (variant, task) cell
across one or more refs.

Designed to be safe to run **while ``ag diff`` is still working**: it only
reads ``results/<model>/`` files that already exist on disk, tolerates
in-flight ``.jsonl`` files whose final line might be a partial write, and
treats missing ``.meta.json`` sidecars as "run still in progress".

The output is meant for human eyeballs, not LLM consumption — for the
latter, the trace paths printed at the end can be wrapped in
``BEGIN UNTRUSTED TRACE`` / ``END UNTRUSTED TRACE`` markers (see
SECURITY.md) and handed to a reviewer model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .analyze import _task_expectations, step_kind
from .markers import fired as _markers_fired
from .log import get_console
from .paths import results_dir, state_root
from .transcript import parse_transcript
from .util import median


# --- per-step parse (keeps tool_call <-> tool_result pairing) ---------------


@dataclass
class Step:
    idx: int  # 1-based tool call index
    name: str  # Tool name (Bash, Read, Write, ...)
    inp: dict
    kind: str  # neutral per-step label (analyze.step_kind)
    is_error: bool
    result_snippet: str  # first non-empty line of the tool_result, truncated


@dataclass
class CellRun:
    run_index: int
    jsonl_path: Path
    meta_path: Path
    steps: list[Step]
    final: str | None
    elapsed_sec: float | None
    tool_call_count: int
    status: str  # "ok" | "budget_tool_calls" | "timeout" | "in-flight" | "broken-trace"
    exit_code: int | None
    matched_expected: bool | None
    expected: str | None
    errored_calls: int
    tokens_in: int
    tokens_out: int


def _trunc(s: str, n: int = 100) -> str:
    s = s.replace("\n", " ⏎ ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _parse_run(jsonl_path: Path, task_id: str, run_index: int) -> CellRun:
    meta_path = jsonl_path.with_suffix(".meta.json")
    expected = _task_expectations().get(task_id)

    tx = parse_transcript(jsonl_path)
    final = tx.final
    broken = tx.broken
    ordered_steps: list[Step] = []
    for i, s in enumerate(tx.steps, 1):
        snippet_lines = [ln for ln in s.result.splitlines() if ln.strip()]
        snippet = snippet_lines[0] if snippet_lines else ""
        ordered_steps.append(
            Step(
                idx=i,
                name=s.name,
                inp=s.input,
                kind=step_kind(s.name, s.input),
                is_error=s.is_error,
                result_snippet=_trunc(snippet, 140),
            )
        )

    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = {}

    if not meta:
        # Still running (or post-mortem death between jsonl write and meta).
        status = "broken-trace" if broken else "in-flight"
    else:
        status = str(meta.get("status") or "ok")

    matched = (
        expected in (final or "").lower() if (expected and final is not None) else None
    )
    errored = sum(1 for s in ordered_steps if s.is_error)

    return CellRun(
        run_index=run_index,
        jsonl_path=jsonl_path,
        meta_path=meta_path,
        steps=ordered_steps,
        final=final,
        elapsed_sec=float(meta["elapsed_sec"]) if "elapsed_sec" in meta else None,
        tool_call_count=int(meta.get("tool_call_count") or len(ordered_steps)),
        status=status,
        exit_code=int(meta["exit_code"]) if "exit_code" in meta else None,
        matched_expected=matched,
        expected=expected,
        errored_calls=errored,
        tokens_in=int((meta.get("tokens") or {}).get("in") or 0),
        tokens_out=int((meta.get("tokens") or {}).get("out") or 0),
    )


def _runs_for(ref: str, variant: str, task: str, ns: str | None) -> list[CellRun]:
    rdir = results_dir(ref, ns)
    paths = sorted(rdir.glob(f"{variant}__{task}__run*.jsonl"))
    runs: list[CellRun] = []
    for p in paths:
        # Extract the run index from the filename suffix `runN.jsonl`.
        try:
            idx = int(p.stem.rsplit("run", 1)[-1])
        except ValueError:
            idx = len(runs) + 1
        runs.append(_parse_run(p, task, idx))
    return runs


# --- rendering --------------------------------------------------------------


def _short_input(name: str, inp: dict) -> str:
    if name == "Bash":
        return _trunc(inp.get("command", ""), 110)
    if name in ("Read", "Write", "Edit"):
        return inp.get("file_path", "")
    if name == "Grep":
        pat = inp.get("pattern", "")
        path = inp.get("path") or inp.get("glob") or ""
        return f"{pat!r}" + (f"  in {path}" if path else "")
    if name == "Glob":
        return inp.get("pattern", "")
    if name == "WebFetch":
        return inp.get("url", "")
    return _trunc(str(inp), 110)


def _print_run(console, run: CellRun) -> None:
    head = Text()
    head.append(f"  run{run.run_index}  ", style="bold")
    if run.matched_expected is True:
        head.append("✓match  ", style="green")
    elif run.matched_expected is False:
        head.append("✗no-match  ", style="red")
    if run.elapsed_sec is not None:
        head.append(f"{run.elapsed_sec:.0f}s  ")
    if run.status == "in-flight":
        head.append("⏵in-flight  ", style="bold yellow")
    elif run.status == "broken-trace":
        head.append("⚠ broken-trace  ", style="bold red")
    elif run.status != "ok":
        head.append(f"{run.status}  ", style="bold red")
    if run.exit_code is not None:
        style = "white" if run.exit_code == 0 else "red"
        head.append(f"exit={run.exit_code}  ", style=style)
    err_style = "red" if run.errored_calls else "white"
    head.append(
        f"{run.tool_call_count} tools  errors={run.errored_calls}", style=err_style
    )
    head.append(f"  tokens in:{run.tokens_in:,} out:{run.tokens_out:,}", style="dim")
    console.print(head)

    if not run.steps:
        console.print(Text("    (no tool calls — answered from model knowledge)", style="dim"))
    for step in run.steps:
        line = Text()
        marker = "❗" if step.is_error else " "
        line.append(f"   {marker} {step.idx:2}  ")
        line.append(f"{step.name:<8} ", style="cyan")
        line.append(_short_input(step.name, step.inp))
        line.append(f"   [{step.kind}]", style="dim")
        console.print(line)
        if step.is_error and step.result_snippet:
            console.print(Text(f"          ↳ {step.result_snippet}", style="red"))

    if run.final:
        final_line = _trunc(run.final, 240)
        marker = ""
        marker_style = "white"
        if run.expected and run.matched_expected is True:
            marker = f"   [contains {run.expected!r}]"
            marker_style = "green"
        elif run.expected and run.matched_expected is False:
            marker = f"   [missing {run.expected!r}]"
            marker_style = "red"
        out = Text("    final  ", style="bold")
        out.append(f"{final_line}", style=marker_style)
        if marker:
            out.append(marker, style=marker_style)
        console.print(out)
    console.print("")


def _cellrun_markers(run: CellRun, markers: list) -> dict[str, bool]:
    """Fire ``markers`` against a CellRun (adapts its steps to the Run shape
    markers.fired expects). Result snippets are truncated, but the markers we
    ship key off tool *inputs* (commands/paths), which are kept in full."""
    adapter = SimpleNamespace(
        tool_calls=[(s.name, s.inp) for s in run.steps],
        tool_results=[s.result_snippet for s in run.steps],
        final=run.final,
    )
    return _markers_fired(markers, adapter)


def _markers_summary(runs: list[CellRun], markers: list | None) -> str:
    """Compact per-marker adoption (``cli=2/3 pipeline=1/3``) for finished runs."""
    from collections import Counter

    markers = markers or []
    finished = [r for r in runs if r.status not in ("in-flight", "broken-trace")]
    n = len(finished)
    if n == 0:
        return "(no finished runs yet)"
    if not markers:
        return f"{n} finished"
    counts: Counter[str] = Counter()
    for r in finished:
        for name, hit in _cellrun_markers(r, markers).items():
            if hit:
                counts[name] += 1
    fired_parts = [f"{name}={c}/{n}" for name, c in counts.items() if c]
    return "  ".join(fired_parts) if fired_parts else f"{n} finished (no markers fired)"


def _diff_table(refs: list[str], by_ref: dict[str, list[CellRun]], markers: list | None = None) -> Table:
    """Side-by-side metric diff for two refs."""
    a, b = refs[0], refs[-1]
    runs_a, runs_b = by_ref[a], by_ref[b]

    def _stats(runs: list[CellRun]) -> dict[str, str]:
        finished = [r for r in runs if r.status not in ("in-flight", "broken-trace")]
        n = len(finished)
        med_t = median([r.elapsed_sec for r in finished if r.elapsed_sec is not None], None)
        med_tc = median([float(r.tool_call_count) for r in finished], None)
        med_in = median([float(r.tokens_in) for r in finished], None)
        med_out = median([float(r.tokens_out) for r in finished], None)
        total_calls = sum(r.tool_call_count for r in finished)
        total_err = sum(r.errored_calls for r in finished)
        match_total = sum(1 for r in finished if r.matched_expected is not None)
        match_ok = sum(1 for r in finished if r.matched_expected is True)
        return {
            "markers": _markers_summary(runs, markers),
            "errors": f"{total_err}/{total_calls}" if total_calls else "—",
            "median time": f"{med_t:.0f}s" if med_t is not None else "—",
            "median tools": f"{med_tc:.0f}" if med_tc is not None else "—",
            "median tokens in": f"{med_in:,.0f}" if med_in is not None else "—",
            "median tokens out": f"{med_out:,.0f}" if med_out is not None else "—",
            "matched": f"{match_ok}/{match_total}" if match_total else "—",
            "runs available": f"{n} finished" + (
                f" (+{len(runs) - n} in-flight)" if len(runs) > n else ""
            ),
        }

    sa, sb = _stats(runs_a), _stats(runs_b)

    table = Table(box=None, header_style="bold", show_lines=False, pad_edge=False)
    table.add_column("metric", style="bold")
    table.add_column(a, justify="right")
    table.add_column("→", justify="center", style="dim")
    table.add_column(b, justify="right")

    for key in (
        "runs available",
        "markers",
        "errors",
        "median time",
        "median tools",
        "median tokens in",
        "median tokens out",
        "matched",
    ):
        va, vb = sa[key], sb[key]
        # Color-grade the simple numeric rows.
        cell_a = Text(va)
        cell_b = Text(vb)
        if key == "median time" and "—" not in (va, vb):
            ta = float(va.rstrip("s"))
            tb = float(vb.rstrip("s"))
            if ta != tb:
                cell_a.stylize("green" if ta < tb else "red")
                cell_b.stylize("green" if tb < ta else "red")
        elif key == "errors" and "—" not in (va, vb):
            ea = int(va.split("/")[0])
            eb = int(vb.split("/")[0])
            if ea != eb:
                cell_a.stylize("green" if ea < eb else "red")
                cell_b.stylize("green" if eb < ea else "red")
        elif key == "median tools" and "—" not in (va, vb):
            ta = float(va)
            tb = float(vb)
            if ta != tb:
                cell_a.stylize("green" if ta < tb else "red")
                cell_b.stylize("green" if tb < ta else "red")
        table.add_row(key, cell_a, "→", cell_b)

    return table


# --- public entry -----------------------------------------------------------


def _discover_namespaces(
    refs: list[str], variant: str, task: str
) -> dict[str, int]:
    """Scan ``results/<ref>/<harness>/<model_id>/`` for files matching this cell.

    Returns ``{"<harness>/<model_id>": file_count}`` for every namespace that
    has at least one matching ``.jsonl`` across the given refs.
    """
    root = state_root() / "results"
    if not root.exists():
        return {}
    found: dict[str, int] = {}
    for ref in refs:
        ref_dir = root / ref
        if not ref_dir.is_dir():
            continue
        for harness_dir in ref_dir.iterdir():
            if not harness_dir.is_dir():
                continue
            for model_dir in harness_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                n = sum(1 for _ in model_dir.glob(f"{variant}__{task}__run*.jsonl"))
                if n:
                    ns = f"{harness_dir.name}/{model_dir.name}"
                    found[ns] = found.get(ns, 0) + n
    return found


def explain(
    refs: list[str],
    variant: str,
    task: str,
    *,
    ns: str | None = None,
    markers: list | None = None,
) -> None:
    console = get_console()
    refs_resolved = list(dict.fromkeys(refs))  # already-expanded bindings (CLI expands via the profile)

    # `ns` here is the <harness>/<model_id> namespace. If it has nothing for
    # this cell, look for a single namespace that does and silently switch to it
    # (with a one-line note). If several have data, refuse to guess.
    avail = _discover_namespaces(refs_resolved, variant, task)
    if not avail.get(ns or "", 0) and avail:
        if len(avail) == 1:
            only = next(iter(avail))
            console.print(
                Text(
                    f"ℹ no results under {ns}; using "
                    f"--runner/--model namespace {only} ({avail[only]} matching files).",
                    style="dim",
                )
            )
            ns = only
        else:
            listing = ", ".join(
                f"{k} ({v} files)" for k, v in sorted(avail.items())
            )
            console.print(
                Text(
                    f"⚠ no results under {ns}, but matching files exist under "
                    f"multiple namespaces: {listing}. Pass --runner/--model "
                    f"to pick one.",
                    style="yellow",
                )
            )
            return

    # Header panel.
    title = Text()
    title.append("Cell:  ", style="bold")
    title.append(f"{variant} / {task}\n")
    if len(refs_resolved) == 1:
        title.append(f"Ref:   {refs_resolved[0]}")
    else:
        title.append("Refs:  " + " → ".join(refs_resolved))
    if ns:
        title.append(f"   [{ns}]", style="dim")
    expected = _task_expectations().get(task)
    if expected:
        title.append(f"\nExpected substring: {expected!r}", style="dim")
    console.print(Panel(title, border_style="cyan", padding=(0, 1)))

    by_ref: dict[str, list[CellRun]] = {}
    for ref in refs_resolved:
        runs = _runs_for(ref, variant, task, ns)
        by_ref[ref] = runs
        console.print(Rule(f"[bold]{ref}[/bold]  ({len(runs)} run(s) on disk)", style="cyan"))
        if not runs:
            console.print(Text("  (no completed runs yet)", style="dim"))
            console.print("")
            continue
        summary = _markers_summary(runs, markers)
        console.print(Text(f"  summary  {summary}", style="bold"))
        console.print("")
        for run in runs:
            _print_run(console, run)

    if len(refs_resolved) >= 2:
        console.print(Rule("[bold]Diff[/bold]", style="cyan"))
        console.print(_diff_table(refs_resolved, by_ref, markers))
        console.print("")

    # Trace paths for hand-off.
    console.print(Rule("[dim]trace files[/dim]", style="dim"))
    for ref in refs_resolved:
        for run in by_ref[ref]:
            console.print(Text(f"  {run.jsonl_path}", style="dim"))
    console.print(
        Text(
            "\n  ↑ wrap each .jsonl in BEGIN/END UNTRUSTED TRACE markers before "
            "feeding to an LLM (see SECURITY.md).",
            style="dim",
        )
    )
