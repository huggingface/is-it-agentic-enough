"""``isth explain`` — focused, per-cell breakdown for one (variant, task) cell
across one or more refs.

Designed to be safe to run **while ``isth diff`` is still working**: it only
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

from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .analyze import _task_expectations, bucket
from .compare import expand_refs
from .log import get_console
from .paths import results_dir, state_root


# --- per-step parse (keeps tool_call <-> tool_result pairing) ---------------


@dataclass
class Step:
    idx: int  # 1-based tool call index
    name: str  # Tool name (Bash, Read, Write, ...)
    inp: dict
    bucket: str  # output of analyze.bucket(name, inp)
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


def _trunc(s: str, n: int = 100) -> str:
    s = s.replace("\n", " ⏎ ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _result_text(block: dict) -> str:
    c = block.get("content", "")
    if isinstance(c, list):
        return "\n".join(
            str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in c
        )
    return str(c)


def _parse_run(jsonl_path: Path, task_id: str, run_index: int) -> CellRun:
    meta_path = jsonl_path.with_suffix(".meta.json")
    expected = _task_expectations().get(task_id)

    steps_by_id: dict[str, Step] = {}
    ordered_steps: list[Step] = []
    final: str | None = None
    broken = False

    try:
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    # In-flight write: last line may be partial. Stop here but
                    # don't lose what we already parsed.
                    broken = True
                    break
                t = e.get("type")
                if t == "assistant":
                    for b in e.get("message", {}).get("content", []) or []:
                        if b.get("type") == "tool_use":
                            tid = b.get("id") or f"_{len(ordered_steps)}"
                            name = b.get("name", "?")
                            inp = b.get("input") or {}
                            step = Step(
                                idx=len(ordered_steps) + 1,
                                name=name,
                                inp=inp,
                                bucket=bucket(name, inp),
                                is_error=False,
                                result_snippet="",
                            )
                            ordered_steps.append(step)
                            steps_by_id[tid] = step
                elif t == "user":
                    for b in e.get("message", {}).get("content", []) or []:
                        if b.get("type") == "tool_result":
                            tid = b.get("tool_use_id")
                            content = _result_text(b)
                            snippet_lines = [
                                ln for ln in content.splitlines() if ln.strip()
                            ]
                            snippet = snippet_lines[0] if snippet_lines else ""
                            step = steps_by_id.get(tid) if tid else None
                            if step is not None:
                                step.is_error = bool(b.get("is_error"))
                                step.result_snippet = _trunc(snippet, 140)
                elif t == "result":
                    final = e.get("result") or ""
    except FileNotFoundError:
        pass

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
    )


def _runs_for(ref: str, variant: str, task: str, model: str | None) -> list[CellRun]:
    rdir = results_dir(model)
    paths = sorted(rdir.glob(f"{ref}__{variant}__{task}__run*.jsonl"))
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
    console.print(head)

    if not run.steps:
        console.print(Text("    (no tool calls — answered from model knowledge)", style="dim"))
    for step in run.steps:
        line = Text()
        marker = "❗" if step.is_error else " "
        line.append(f"   {marker} {step.idx:2}  ")
        line.append(f"{step.name:<8} ", style="cyan")
        line.append(_short_input(step.name, step.inp))
        line.append(f"   [{step.bucket}]", style="dim")
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


def _approach_summary(runs: list[CellRun]) -> str:
    """Compact approach bucket string (CLI-clean=2/3 etc.) for finished runs."""
    from collections import Counter

    if not runs:
        return "—"
    # Mirror analyze._run_path semantics on our Step list.
    counts: Counter[str] = Counter()
    finished = [r for r in runs if r.status not in ("in-flight", "broken-trace")]
    for r in finished:
        seen = {s.bucket for s in r.steps}
        if "CLI" in seen:
            path = "CLI"
        elif any(b.startswith(("python ", "write .py")) for b in seen):
            path = "Python"
        elif not r.steps:
            path = "no-tool"
        else:
            path = "other"
        if path in ("CLI", "Python"):
            counts[f"{path}-{'retry' if r.errored_calls else 'clean'}"] += 1
        else:
            counts[path] += 1
    n = len(finished)
    if n == 0:
        return "(no finished runs yet)"
    return "  ".join(f"{k}={v}/{n}" for k, v in counts.items())


def _diff_table(refs: list[str], by_ref: dict[str, list[CellRun]]) -> Table:
    """Side-by-side metric diff for two refs."""
    a, b = refs[0], refs[-1]
    runs_a, runs_b = by_ref[a], by_ref[b]

    def _med(xs: list[float]) -> float | None:
        if not xs:
            return None
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    def _stats(runs: list[CellRun]) -> dict[str, str]:
        finished = [r for r in runs if r.status not in ("in-flight", "broken-trace")]
        n = len(finished)
        med_t = _med([r.elapsed_sec for r in finished if r.elapsed_sec is not None])
        med_tc = _med([float(r.tool_call_count) for r in finished])
        total_calls = sum(r.tool_call_count for r in finished)
        total_err = sum(r.errored_calls for r in finished)
        match_total = sum(1 for r in finished if r.matched_expected is not None)
        match_ok = sum(1 for r in finished if r.matched_expected is True)
        return {
            "approach": _approach_summary(runs),
            "errors": f"{total_err}/{total_calls}" if total_calls else "—",
            "median time": f"{med_t:.0f}s" if med_t is not None else "—",
            "median tools": f"{med_tc:.0f}" if med_tc is not None else "—",
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
        "approach",
        "errors",
        "median time",
        "median tools",
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


def _discover_model_namespaces(
    refs: list[str], variant: str, task: str
) -> dict[str | None, int]:
    """Scan ``results/`` and ``results/<model>/`` for files matching this cell.

    Returns ``{model_or_None: file_count}`` for every namespace that has at
    least one matching ``.jsonl``. ``None`` means the default (un-namespaced)
    ``results/`` dir.
    """
    root = state_root() / "results"
    if not root.exists():
        return {}
    found: dict[str | None, int] = {}
    candidates: list[tuple[str | None, Path]] = [(None, root)]
    for child in root.iterdir():
        if child.is_dir():
            candidates.append((child.name, child))
    for ns, d in candidates:
        n = 0
        for ref in refs:
            n += sum(1 for _ in d.glob(f"{ref}__{variant}__{task}__run*.jsonl"))
        if n:
            found[ns] = n
    return found


def explain(
    refs: list[str],
    variant: str,
    task: str,
    *,
    model: str | None = None,
) -> None:
    console = get_console()
    refs_resolved = expand_refs(refs)

    # If the user didn't pass --model and the default results/ dir has
    # nothing for this cell, look for a single model namespace that does
    # and silently switch to it (with a one-line note). If multiple
    # namespaces have data, refuse to guess and tell the user.
    if model is None:
        ns_counts = _discover_model_namespaces(refs_resolved, variant, task)
        # Drop the default dir entry if it had nothing.
        ns_counts.pop(None, None) if ns_counts.get(None, 0) == 0 else None
        if not ns_counts.get(None):
            namespaced = {k: v for k, v in ns_counts.items() if k is not None}
            if not namespaced:
                pass  # nothing anywhere; let the normal flow show "0 runs"
            elif len(namespaced) == 1:
                only = next(iter(namespaced))
                console.print(
                    Text(
                        f"ℹ no results in default results/ dir; using "
                        f"--model {only} ({namespaced[only]} matching files).",
                        style="dim",
                    )
                )
                model = only
            else:
                listing = ", ".join(
                    f"--model {k} ({v} files)" for k, v in sorted(namespaced.items())
                )
                console.print(
                    Text(
                        f"⚠ no results in default results/ dir, but matching files "
                        f"exist under multiple model namespaces: {listing}. "
                        f"Pass --model to pick one.",
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
    if model:
        title.append(f"   [model: {model}]", style="dim")
    expected = _task_expectations().get(task)
    if expected:
        title.append(f"\nExpected substring: {expected!r}", style="dim")
    console.print(Panel(title, border_style="cyan", padding=(0, 1)))

    by_ref: dict[str, list[CellRun]] = {}
    for ref in refs_resolved:
        runs = _runs_for(ref, variant, task, model)
        by_ref[ref] = runs
        console.print(Rule(f"[bold]{ref}[/bold]  ({len(runs)} run(s) on disk)", style="cyan"))
        if not runs:
            console.print(Text("  (no completed runs yet)", style="dim"))
            console.print("")
            continue
        summary = _approach_summary(runs)
        console.print(Text(f"  summary  {summary}", style="bold"))
        console.print("")
        for run in runs:
            _print_run(console, run)

    if len(refs_resolved) >= 2:
        console.print(Rule("[bold]Diff[/bold]", style="cyan"))
        console.print(_diff_table(refs_resolved, by_ref))
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
