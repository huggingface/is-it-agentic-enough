"""Live, rich-rendered progress dashboard for `isth diff` / `isth suite`.

The dashboard maintains an in-memory matrix of cells indexed by
``(task, variant, ref)``. Each cell aggregates the runs scheduled for it:
how many are pending / running / done / failed, plus median wall-time and
median tool-call count once we have completed runs.

While `Live` is active, normal `log()` calls still go through the same
`rich.console.Console` and print *above* the live region, so verbose
per-tool-call streaming is not broken.

Coloring is **row-relative**: within a single (task, variant) row we look
across the ref columns and tint the best cell green, the worst red, the
middle yellow. That is the "which commit is doing a better job" signal.
Aborted runs (``⏻``) and failed runs (``!``) are always red.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text


# --- cell state -------------------------------------------------------------


@dataclass
class CellAgg:
    """Aggregate of all runs for one (task, variant, ref) cell."""

    total_planned: int = 0
    runs_done: int = 0
    runs_failed: int = 0
    elapsed_secs: list[float] = field(default_factory=list)
    tool_calls: list[int] = field(default_factory=list)
    errors: int = 0  # is_error tool results, summed across runs
    aborted: int = 0  # runs killed for budget / timeout
    running: bool = False

    @property
    def has_data(self) -> bool:
        return bool(self.elapsed_secs)

    @property
    def median_time(self) -> float:
        return _median(self.elapsed_secs) if self.elapsed_secs else 0.0

    @property
    def median_tool_calls(self) -> float:
        return _median(self.tool_calls) if self.tool_calls else 0.0


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# --- dashboard --------------------------------------------------------------


class Dashboard:
    """Live progress matrix for a `diff` / `suite` run.

    Construction is cheap; nothing renders until ``with dash.live():`` is
    entered. If `enabled=False`, the dashboard becomes a no-op shim — all
    `mark_*` methods still work but no Live region is created. Use that
    when stderr isn't a TTY (CI logs).
    """

    def __init__(
        self,
        refs: list[str],
        plan: list[tuple[str, str, str, int]],
        *,
        console: Console | None = None,
        enabled: bool = True,
        title: str | None = None,
    ) -> None:
        self.refs = list(refs)
        self.plan = list(plan)
        self.console = console or Console(stderr=True, highlight=False)
        self.enabled = enabled
        self.title = title or self._default_title()

        # cells[task][variant][ref] = CellAgg (only for (t,v,r) actually planned).
        self.cells: dict[str, dict[str, dict[str, CellAgg]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self.tasks_order: list[str] = []
        self.variants_order: list[str] = []
        for ref, variant, task, _run_idx in plan:
            if task not in self.tasks_order:
                self.tasks_order.append(task)
            if variant not in self.variants_order:
                self.variants_order.append(variant)
            cell = self.cells[task][variant].setdefault(ref, CellAgg())
            cell.total_planned += 1

        self.total = len(plan)
        self.completed = 0
        self.skipped = 0
        self.failed = 0
        self._current: tuple[str, str, str, int] | None = None
        self._live: Live | None = None

    # --- public hooks -------------------------------------------------------

    def mark_running(self, ref: str, variant: str, task: str, run_idx: int) -> None:
        cell = self._cell(task, variant, ref)
        if cell:
            cell.running = True
        self._current = (ref, variant, task, run_idx)
        self._refresh()

    def mark_done(
        self, ref: str, variant: str, task: str, run_idx: int, meta: dict
    ) -> None:
        cell = self._cell(task, variant, ref)
        if cell:
            cell.running = False
            cell.runs_done += 1
            cell.elapsed_secs.append(float(meta.get("elapsed_sec") or 0))
            cell.tool_calls.append(int(meta.get("tool_call_count") or 0))
            if meta.get("status") in ("budget_tool_calls", "timeout"):
                cell.aborted += 1
        self.completed += 1
        if self._current and self._current[:3] == (ref, variant, task):
            self._current = None
        self._refresh()

    def mark_skipped_existing(
        self, ref: str, variant: str, task: str, run_idx: int, meta: dict | None
    ) -> None:
        """A planned run was found already on disk and not re-executed."""
        cell = self._cell(task, variant, ref)
        if cell and meta is not None:
            cell.runs_done += 1
            cell.elapsed_secs.append(float(meta.get("elapsed_sec") or 0))
            cell.tool_calls.append(int(meta.get("tool_call_count") or 0))
            if meta.get("status") in ("budget_tool_calls", "timeout"):
                cell.aborted += 1
        elif cell:
            cell.runs_done += 1
        self.completed += 1
        self.skipped += 1
        self._refresh()

    def mark_failed(
        self, ref: str, variant: str, task: str, run_idx: int, err: str
    ) -> None:
        cell = self._cell(task, variant, ref)
        if cell:
            cell.running = False
            cell.runs_failed += 1
        self.completed += 1
        self.failed += 1
        if self._current and self._current[:3] == (ref, variant, task):
            self._current = None
            self._current_started_at = None
        self._refresh()

    # --- live context -------------------------------------------------------

    @contextmanager
    def live(self) -> Iterator["Dashboard"]:
        if not self.enabled:
            # Print a one-line header so non-TTY users see what's happening.
            self.console.print(f"[bold]{self.title}[/bold]")
            yield self
            return
        with Live(
            self._build_renderable(),
            console=self.console,
            # 12 fps keeps the spinners smooth without burning CPU.
            refresh_per_second=12,
            transient=False,
        ) as live:
            self._live = live
            try:
                yield self
            finally:
                # Final refresh so the last completed cell is visible.
                self._refresh()
                self._live = None

    # --- internals ----------------------------------------------------------

    def _cell(self, task: str, variant: str, ref: str) -> CellAgg | None:
        return self.cells.get(task, {}).get(variant, {}).get(ref)

    def _default_title(self) -> str:
        if len(self.refs) == 1:
            return f"Suite: {self.refs[0]}"
        return f"Comparing {len(self.refs)} commits: " + " → ".join(self.refs)

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._build_renderable())

    def _build_renderable(self) -> Group:
        header = Panel(
            Text(self.title, justify="center", style="bold"),
            border_style="cyan",
            padding=(0, 1),
        )

        counters = Text()
        counters.append(f"  {self.completed}/{self.total} runs")
        if self.skipped:
            counters.append(f"   ({self.skipped} cached)", style="dim")
        if self.failed:
            counters.append(f"   ({self.failed} failed)", style="bold red")

        if self._current:
            ref, variant, task, run_idx = self._current
            current_label = Text(
                f"{ref} {variant} {task} run{run_idx}", style="yellow"
            )
            current_line = Spinner(
                "dots", text=current_label, style="bold yellow"
            )
            progress: object = Group(counters, Text("  "), Columns([Text("  "), current_line]))
        else:
            progress = counters

        return Group(header, progress, self._build_table())

    def _build_table(self) -> Table:
        table = Table(
            expand=False,
            show_lines=False,
            header_style="bold",
            pad_edge=False,
            box=None,
        )
        table.add_column("task / variant", no_wrap=True, style="white")
        for ref in self.refs:
            table.add_column(ref, justify="right", no_wrap=True)

        for task in self.tasks_order:
            table.add_row(Text(task, style="bold cyan"), *[Text("") for _ in self.refs])
            for variant in self.variants_order:
                row_cells = [self._cell(task, variant, r) for r in self.refs]
                # If this (task, variant) row has no planned cells anywhere,
                # skip it to keep the table compact.
                if all(c is None or c.total_planned == 0 for c in row_cells):
                    continue
                rendered = [self._format_cell(c, row_cells) for c in row_cells]
                table.add_row(Text(f"  {variant}", style="dim"), *rendered)
        return table

    def _format_cell(
        self, cell: CellAgg | None, peers: list[CellAgg | None]
    ) -> Text:
        if cell is None or cell.total_planned == 0:
            return Text("—", style="dim")

        if not cell.has_data and not cell.running:
            if cell.runs_failed:
                return Text(f"! {cell.runs_failed} fail", style="bold red")
            return Text("· pending", style="dim")

        if cell.running and not cell.has_data:
            return Spinner(
                "dots", text=Text("running…", style="bold yellow"), style="bold yellow"
            )

        # Have at least one completed run.
        med_t = cell.median_time
        med_tc = cell.median_tool_calls
        out = Text()

        # --- time, colored row-relative ---
        t_style = self._row_relative_style(
            med_t,
            [c.median_time for c in peers if c and c.has_data],
            higher_is_worse=True,
        )
        out.append(f"{med_t:.0f}s", style=t_style)
        out.append("  ")

        # --- tool calls, colored row-relative ---
        tc_style = self._row_relative_style(
            med_tc,
            [c.median_tool_calls for c in peers if c and c.has_data],
            higher_is_worse=True,
        )
        out.append(f"{med_tc:.0f} tools", style=tc_style)

        # --- bad-news flags ---
        if cell.aborted:
            out.append(f"  ⏻{cell.aborted}", style="bold red")
        if cell.runs_failed:
            out.append(f"  !{cell.runs_failed}", style="bold red")

        # --- progress within this cell ---
        if cell.runs_done < cell.total_planned:
            out.append(f"  ({cell.runs_done}/{cell.total_planned})", style="dim")

        # If this cell still has more runs going, animate a spinner next to
        # the current partial stats so the user sees it's actively working.
        if cell.running:
            return Spinner("dots", text=out, style="bold yellow")

        return out

    @staticmethod
    def _row_relative_style(
        value: float, peer_values: list[float], *, higher_is_worse: bool
    ) -> str:
        """Pick green / yellow / red based on this value vs. its row peers."""
        if len(peer_values) < 2:
            return "white"
        best = min(peer_values) if higher_is_worse else max(peer_values)
        worst = max(peer_values) if higher_is_worse else min(peer_values)
        if best == worst:
            return "white"
        if value == best:
            return "green"
        if value == worst:
            return "red"
        return "yellow"


# --- helpers ----------------------------------------------------------------


def stderr_is_tty() -> bool:
    return sys.stderr.isatty()
