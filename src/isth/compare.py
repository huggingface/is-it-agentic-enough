"""Side-by-side comparison across two or more refs.

The output is designed to be self-contained — everything an LLM needs to
interpret the report (experiment setup, variant definitions, task
descriptions, commit metadata, metric glossary, raw stats) is included
in the markdown. No editorial judgment is added; the reader decides what
the numbers mean.
"""

from __future__ import annotations

import statistics
import subprocess
from pathlib import Path

import yaml

from .analyze import (
    VARIANTS,
    approach_counts,
    cell,
    discover_task_ids,
    load_runs,
)
from .paths import package_data_path, transformers_src
from .setup_commit import resolve_sha


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n // 1000}k"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


# --------- preamble sections ---------


def _context_section() -> str:
    return """## Context

This report was produced by the `is-transformers-agentic-enough` harness.
The harness runs headless Claude Code against a fixed set of tasks, using
a pinned build of the `transformers` library at each commit being compared.
The goal is to measure how an agent's *behaviour* changes across commits —
specifically, whether it uses the `transformers` CLI (the subject of the
9-commit "agent-first CLI" effort) vs. falling back to writing Python.

Each (commit × variant × task) cell is typically run N times (default 3)
to smooth out model non-determinism. The stats below report medians and
totals over those runs.
"""


def _variants_section() -> str:
    return """## Variants

Each run happens under one of three "variants" that differ only in how the
transformers CLI is surfaced to the agent. Installed transformers version
is identical across variants for a given commit; the difference is context.

- **bare** — `pip install transformers` only. Workspace contains only task
  inputs. No pointer to the CLI exists; the agent would need to discover it
  on its own (e.g. by running `transformers --help`).
- **clone** — same install, but the agent's cwd **is** a git worktree of the
  transformers repo at that commit. `AGENTS.md`, `CLAUDE.md`, and
  `src/transformers/cli/agentic/*.py` auto-discover via Claude Code's
  standard cwd scanning.
- **skill** — same install, plus a Claude Code plugin directory containing
  a generated `SKILL.md` is loaded with `--plugin-dir`. SKILL.md explicitly
  tells the agent "for atomic tasks, use the CLI". Skipped for commits
  where the skill can't be derived (the `_skill_derive` module doesn't
  exist yet).
"""


def _commit_metadata(shas: list[str]) -> str:
    """Retrieve `{subject, author date}` for each sha via git."""
    lines = ["## Commits compared", ""]
    lines.append("| Short SHA | Date | Subject |")
    lines.append("|---|---|---|")
    src = transformers_src()
    for sha in shas:
        try:
            out = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(src),
                    "show",
                    "-s",
                    "--date=short",
                    "--format=%ad|%s",
                    sha,
                ],
                text=True,
            ).strip()
            date, subject = out.split("|", 1)
        except Exception:
            date, subject = "?", "?"
        lines.append(f"| {sha} | {date} | {subject} |")
    lines.append("")
    lines.append(
        "Commits are displayed in the order given on the command line; when the "
        "user passed `A..B` this is chronological, but arbitrary ordering is also valid."
    )
    lines.append("")
    return "\n".join(lines)


def _tasks_section() -> str:
    with open(package_data_path("tasks.yaml")) as f:
        data = yaml.safe_load(f)
    tasks = data.get("tasks") or []

    lines = ["## Tasks", ""]
    lines.append(
        "Each task is a natural-language prompt handed to the agent. All prompts "
        "name a specific Hugging Face model so the agent must actually load and "
        "run the model (preventing it from answering purely from world knowledge)."
    )
    lines.append("")
    lines.append("| id | category | expected substring | prompt (one-line preview) |")
    lines.append("|---|---|---|---|")
    for t in tasks:
        prompt_preview = (t.get("prompt") or "").replace("\n", " ").strip()
        if len(prompt_preview) > 120:
            prompt_preview = prompt_preview[:119] + "…"
        expected = t.get("expected") or "—"
        lines.append(f"| `{t['id']}` | {t.get('category', '?')} | `{expected}` | {prompt_preview} |")
    lines.append("")
    lines.append(
        "**Category meaning.** `atomic` = one existing CLI command in the post-effort "
        "state covers the task; the expected behaviour shift is Python → CLI. "
        "`compositional` = no single CLI command fits; the agent must write Python "
        "(ideally modelled on the `cli/agentic/*.py` exemplars rather than "
        "`pipeline(...)`)."
    )
    lines.append("")
    lines.append(
        "**`expected substring`.** If set, each run's final output is checked for a "
        "case-insensitive substring match; this is the `✓match` signal in cells. "
        "Tasks without an `expected` field are not checked for correctness."
    )
    lines.append("")
    return "\n".join(lines)


def _metric_glossary() -> str:
    return """## Metrics and cell format

Each cell in the per-task tables uses the format:

```
**approach** · ✓match · !failed/total · ⇢first-success · 📖docs · ⏻abort · median-time · new · repeat · out
```

Fields that are zero or not applicable are omitted. This report is framed
around **ease of use** — whether the agent had an easy time using transformers
(fewer retries, less thrashing, earlier success), not just whether it used
the CLI. The CLI/Python split is retained but each is further split by
whether any tool call errored (``-retry``) or not (``-clean``).

### Approach — what path the agent took

Each run is bucketed by examining its tool-call sequence, then split by
whether any tool call in the run returned ``is_error=true``:

- `CLI-clean=k/n` / `CLI-retry=k/n` — ran the `transformers` CLI via Bash;
  retry variant means ≥1 errored tool call (a traceback, non-zero exit, etc.).
- `Python-clean=k/n` / `Python-retry=k/n` — executed Python (via `python -c`,
  `Write`+`python <file>`, or similar) without invoking the CLI; retry
  variant means ≥1 errored tool call.
- `no-tool=k/n` — answered with zero tool calls (from model knowledge).
- `other=k/n` — used tools that fit no bucket above (e.g. WebFetch).

Cells containing any CLI adoption are bolded.

### ✓match — correctness check

`✓k/m` where `m` is the number of runs with an `expected` substring
defined for the task, and `k` is the number of those runs whose final
output contained that substring (case-insensitive). Omitted when the task
has no `expected` field.

### !failed/total — tool-call errors

`!k/n` where `k` is the number of tool calls across all runs in the cell
that were flagged as errors (via `is_error: true` on the tool_result),
and `n` is the total tool calls in the cell. Omitted when `k=0`. A tool
error is not necessarily fatal — the agent often recovers and still
matches `expected`.

### ⇢first-success — how fast the agent got a useful answer

`⇢k` where `k` is the median (across runs in the cell) tool-call index at
which the agent's tool_result content first contained the task's
`expected` substring. Lower is better — a well-equipped agent finds the
answer earlier in its exploration. Omitted for tasks without an
`expected` field, and for runs where the expected substring never
appeared in a tool_result (e.g. the match came only from the final
narrative answer).

### 📖docs — which docs the agent consulted

`📖` followed by any non-zero combination of:

- ``agentic=k/n`` — runs that explicitly Read or Grepped a
  ``src/transformers/cli/agentic/*.py`` exemplar.
- ``help=k/n`` — runs that invoked ``transformers … --help``.
- ``AGENTS.md=k/n`` / ``CLAUDE.md=k/n`` / ``SKILL.md=k/n`` — runs that
  explicitly Read that file. These are usually loaded into the agent's
  context automatically by variant configuration (cwd scanning for the
  clone variant, plugin loader for the skill variant), so these fields
  typically show zero and are omitted — explicit Reads are a rare
  behaviour and worth flagging when they happen.

### ⏻ — runs aborted early

The harness kills a run that exceeds a tool-call budget (default 50,
configurable via `--max-tool-calls`) or a wall-clock timeout. When any
run in a cell was aborted, the cell shows ``⏻<reason>:<count>``:

- ``budget_tool_calls`` — killed because tool call count exceeded the budget.
  The run's JSONL stops mid-sequence; no final answer exists. A pattern of
  frequent abort-by-budget suggests the agent is thrashing (retries,
  exploration without converging).
- ``timeout`` — killed because wall-clock elapsed exceeded the internal
  limit (15 minutes). Rarely triggered unless the agent is genuinely stuck.

Aborted runs still contribute tool-call, token, and time numbers up to
the point of kill; they do not contribute to `✓match` (the `expected`
check needs a final answer).

### median-time — wall-clock seconds

Median across the runs in the cell, rounded to the nearest second.

### new / repeat / out — token accounting

Claude Code reports four token fields per assistant turn:
`input_tokens`, `output_tokens`, `cache_creation_input_tokens`,
`cache_read_input_tokens`. These are summed across turns and then split
into two input-side aggregates for reporting:

- `new` = `input_tokens + cache_creation_input_tokens`.
  The unique prompt content the run introduced (system prompt, any
  SKILL.md, files the agent `Read`, tool-result content). Grows when the
  agent needs more information per run; does NOT grow with turn count
  after the cache is warm.
- `repeat` = `cache_read_input_tokens`.
  Tokens re-read from the prompt cache on turns 2, 3, ... Grows with
  turn count (more tool calls → more API turns → more re-reads of the
  same cached prefix). A high `repeat` without a proportional `new`
  increase typically indicates "the agent took more turns", not "the
  prompt got bigger".
- `out` = `output_tokens`. Tokens the model generated (text + tool-call
  arguments). Larger `out` typically means the agent wrote more text
  or more detailed tool inputs.

Both `new` and `repeat` are input-side; they do not overlap. Values are
formatted with `k` (1000) and `M` (1_000_000) suffixes above 1000.

### Summary tables

The report carries two kinds of summary:

- **Headline summary** — split into *atomic tasks* and *compositional
  tasks*, because they have very different token/time profiles and
  aggregating them together lets compositional tasks dominate the
  headline numbers. Each ref gets one column.
- **Per-variant summary** — one sub-table per variant (`bare`, `clone`,
  `skill`) with refs as columns. Use these to ask "holding the variant
  constant, what changed between commits?" directly.

Ratios (`cli_runs/total_runs`, `errored_calls/total_calls`,
`matched_total/matched_seen`) preserve sample size so the reader can see
the denominators directly. Medians are reported with IQR
(``23s (IQR 19–28)``) when the cell has ≥4 runs; below that, the
min–max range is shown in place of IQR.

Above each summary table, a ``⚠ coverage:`` line is emitted whenever the
refs being compared have mismatched coverage (e.g. ``skill`` data
missing for one commit). Headline ratios compare uneven samples in that
case, so the reader is told before reading the numbers.

### What the report does NOT include

- No judgment about which numbers are "good" or "bad".
- No causal claims about what drove a difference between commits.
- Missing cells (`—`) reflect runs that were never executed for that
  (commit, variant, task), not failures. Partial coverage is common when
  a suite was interrupted or deliberately scoped with `--tasks` / `--variants`.
"""


# --------- per-task section (no judgment) ---------


def _compare_task(task_id: str, shas: list[str], model: str | None = None) -> str:
    lines = [f"### {task_id}", ""]
    header = "| Variant | " + " | ".join(shas) + " |"
    sep = "|" + "|".join(["---"] * (len(shas) + 1)) + "|"
    lines.extend([header, sep])

    any_row = False
    for variant in VARIANTS:
        cells = [variant]
        has_any = False
        for sha in shas:
            runs = load_runs(sha, variant, task_id, model)
            if runs:
                has_any = True
                cells.append(cell(runs))
            else:
                cells.append("—")
        if has_any:
            any_row = True
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    return "\n".join(lines) if any_row else ""


# --------- summary ---------


def expand_refs(refs: list[str]) -> list[str]:
    """Expand `ref1..ref2..refN` tokens and resolve everything to short SHAs."""
    expanded: list[str] = []
    for token in refs:
        if ".." in token:
            expanded.extend(part for part in token.split("..") if part)
        else:
            expanded.append(token)
    out: list[str] = []
    for ref in expanded:
        if len(ref) >= 10 and all(c in "0123456789abcdef" for c in ref.lower()):
            out.append(ref[:10])
        else:
            out.append(resolve_sha(ref)[:10])
    seen: set[str] = set()
    unique: list[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _task_categories() -> dict[str, str]:
    """``{task_id: category}`` from the packaged tasks.yaml."""
    with open(package_data_path("tasks.yaml")) as f:
        data = yaml.safe_load(f)
    return {t["id"]: t.get("category", "?") for t in data.get("tasks", []) or []}


def _iqr(xs: list[float]) -> tuple[float, float, float]:
    """Return (median, Q1, Q3). For n<4, Q1/Q3 fall back to min/max."""
    if not xs:
        return (0.0, 0.0, 0.0)
    med = float(statistics.median(xs))
    if len(xs) < 4:
        return (med, float(min(xs)), float(max(xs)))
    q = statistics.quantiles(xs, n=4)
    return (med, q[0], q[2])


def _fmt_sec(med: float, q1: float, q3: float) -> str:
    if q1 == q3:
        return f"{med:.0f}s"
    return f"{med:.0f}s (IQR {q1:.0f}–{q3:.0f})"


def _fmt_tok_iqr(med: float, q1: float, q3: float) -> str:
    if q1 == q3:
        return _fmt_tokens(int(med))
    return f"{_fmt_tokens(int(med))} (IQR {_fmt_tokens(int(q1))}–{_fmt_tokens(int(q3))})"


def _aggregate(
    sha: str,
    task_ids: list[str],
    variants: list[str],
    model: str | None,
) -> dict:
    """Aggregate every run for (sha, tasks ∈ task_ids, variant ∈ variants)."""
    total_runs = 0
    total_calls = 0
    errored_calls = 0
    runs_with_errors = 0
    aborted_budget = 0
    aborted_timeout = 0
    bucket_runs: dict[str, int] = {}
    matched_total = 0
    matched_seen = 0
    elapsed_all: list[float] = []
    new_all: list[int] = []
    repeat_all: list[int] = []
    out_all: list[int] = []

    for tid in task_ids:
        for variant in variants:
            runs = load_runs(sha, variant, tid, model)
            if not runs:
                continue
            total_runs += len(runs)
            for k, v in approach_counts(runs).items():
                bucket_runs[k] = bucket_runs.get(k, 0) + v
            for r in runs:
                total_calls += len(r.tool_calls)
                errored_calls += r.errored_calls
                if r.errored_calls or r.exit_code != 0:
                    runs_with_errors += 1
                if r.status == "budget_tool_calls":
                    aborted_budget += 1
                elif r.status == "timeout":
                    aborted_timeout += 1
                if r.matched_expected is not None:
                    matched_seen += 1
                    if r.matched_expected:
                        matched_total += 1
                elapsed_all.append(r.elapsed)
                new_all.append(r.tokens_in + r.tokens_cache_creation)
                repeat_all.append(r.tokens_cache_read)
                out_all.append(r.tokens_out)

    return {
        "total_runs": total_runs,
        "total_calls": total_calls,
        "errored_calls": errored_calls,
        "runs_with_errors": runs_with_errors,
        "aborted_budget": aborted_budget,
        "aborted_timeout": aborted_timeout,
        "bucket_runs": bucket_runs,
        "matched_total": matched_total,
        "matched_seen": matched_seen,
        "elapsed_iqr": _iqr(elapsed_all),
        "new_iqr": _iqr([float(x) for x in new_all]),
        "repeat_iqr": _iqr([float(x) for x in repeat_all]),
        "out_iqr": _iqr([float(x) for x in out_all]),
        "total_new": sum(new_all),
        "total_repeat": sum(repeat_all),
        "total_out": sum(out_all),
    }


def _render_summary(title: str, rows: list[tuple[str, dict]], note: str = "") -> str:
    """Render a summary table from pre-aggregated rows: [(column_label, agg_dict), ...]."""

    def _bucket_sum(d: dict, *keys: str) -> int:
        return sum(d["bucket_runs"].get(k, 0) for k in keys)

    def _ratio_col(num_key: str, den_key: str) -> list[str]:
        return [f"{d[num_key]}/{d[den_key]}" if d[den_key] else "—" for _, d in rows]

    def _bucket_ratio(d: dict, *keys: str) -> str:
        tot = d["total_runs"]
        if not tot:
            return "—"
        return f"{_bucket_sum(d, *keys)}/{tot}"

    lines = [f"## {title}", ""]
    if note:
        lines.append(note)
        lines.append("")
    lines.append("| Metric | " + " | ".join(label for label, _ in rows) + " |")
    lines.append("|" + "|".join(["---"] * (len(rows) + 1)) + "|")
    lines.append("| runs | " + " | ".join(str(d["total_runs"]) for _, d in rows) + " |")
    def _bucket_cell(d: dict, clean_key: str, retry_key: str) -> str:
        if not d["total_runs"]:
            return "—"
        return (
            f"{_bucket_ratio(d, clean_key, retry_key)} "
            f"({_bucket_sum(d, clean_key)} clean / {_bucket_sum(d, retry_key)} retry)"
        )

    lines.append(
        "| CLI adoption (clean + retry) | "
        + " | ".join(_bucket_cell(d, "CLI-clean", "CLI-retry") for _, d in rows) + " |"
    )
    lines.append(
        "| Python (clean + retry) | "
        + " | ".join(_bucket_cell(d, "Python-clean", "Python-retry") for _, d in rows) + " |"
    )
    lines.append(
        "| no-tool / other | "
        + " | ".join(
            "—" if not d["total_runs"]
            else f"{_bucket_sum(d, 'no-tool')} / {_bucket_sum(d, 'other')}"
            for _, d in rows
        )
        + " |"
    )
    lines.append(
        "| runs where final output matched `expected` | "
        + " | ".join(_ratio_col("matched_total", "matched_seen")) + " |"
    )
    lines.append(
        "| errored tool calls (is_error=true) | "
        + " | ".join(_ratio_col("errored_calls", "total_calls")) + " |"
    )
    lines.append(
        "| runs with any error | "
        + " | ".join(_ratio_col("runs_with_errors", "total_runs")) + " |"
    )
    lines.append(
        "| runs aborted (tool-call budget) | "
        + " | ".join(str(d["aborted_budget"]) for _, d in rows) + " |"
    )
    lines.append(
        "| runs aborted (wall-clock timeout) | "
        + " | ".join(str(d["aborted_timeout"]) for _, d in rows) + " |"
    )
    def _empty(d: dict) -> bool:
        return not d["total_runs"]

    def _sec_cell(d: dict) -> str:
        return "—" if _empty(d) else _fmt_sec(*d["elapsed_iqr"])

    def _tok_iqr_cell(d: dict, key: str) -> str:
        return "—" if _empty(d) else _fmt_tok_iqr(*d[key])

    def _tok_cell(d: dict, key: str) -> str:
        return "—" if _empty(d) else _fmt_tokens(d[key])

    lines.append("| median wall-time per run | " + " | ".join(_sec_cell(d) for _, d in rows) + " |")
    lines.append("| median `new` tokens per run | " + " | ".join(_tok_iqr_cell(d, "new_iqr") for _, d in rows) + " |")
    lines.append("| median `repeat` tokens per run | " + " | ".join(_tok_iqr_cell(d, "repeat_iqr") for _, d in rows) + " |")
    lines.append("| median output tokens per run | " + " | ".join(_tok_iqr_cell(d, "out_iqr") for _, d in rows) + " |")
    lines.append("| total `new` tokens (all runs) | " + " | ".join(_tok_cell(d, "total_new") for _, d in rows) + " |")
    lines.append("| total `repeat` tokens (all runs) | " + " | ".join(_tok_cell(d, "total_repeat") for _, d in rows) + " |")
    lines.append("| total output tokens (all runs) | " + " | ".join(_tok_cell(d, "total_out") for _, d in rows) + " |")
    lines.append("")
    return "\n".join(lines)


def _coverage_note(
    shas: list[str],
    task_ids: list[str],
    variants: list[str],
    model: str | None,
) -> str:
    """Return a ``⚠ coverage:`` note if any (variant × task) cell exists for some
    shas but not all of them, else ""."""
    missing: list[str] = []
    for variant in variants:
        for tid in task_ids:
            present = [s for s in shas if load_runs(s, variant, tid, model)]
            absent = [s for s in shas if s not in present]
            if present and absent:
                missing.append(f"{variant}/{tid} missing for {', '.join(absent)}")
    if not missing:
        return ""
    return "⚠ coverage: " + "; ".join(missing) + ". Headline ratios compare uneven samples."


def _headline_summary(
    shas: list[str], task_ids: list[str], model: str | None
) -> str:
    """Two tables: atomic-tasks-only, then compositional-tasks-only."""
    cats = _task_categories()
    atomic = [t for t in task_ids if cats.get(t) == "atomic"]
    compositional = [t for t in task_ids if cats.get(t) == "compositional"]

    parts: list[str] = []
    if atomic:
        rows = [(s, _aggregate(s, atomic, list(VARIANTS), model)) for s in shas]
        parts.append(
            _render_summary(
                "Summary — atomic tasks",
                rows,
                _coverage_note(shas, atomic, list(VARIANTS), model),
            )
        )
    if compositional:
        rows = [(s, _aggregate(s, compositional, list(VARIANTS), model)) for s in shas]
        parts.append(
            _render_summary(
                "Summary — compositional tasks",
                rows,
                _coverage_note(shas, compositional, list(VARIANTS), model),
            )
        )
    return "\n".join(parts)


def _per_variant_summary(
    shas: list[str], task_ids: list[str], model: str | None
) -> str:
    """One summary table per variant (bare / clone / skill), commits as columns."""
    parts = ["## Per-variant summary", ""]
    parts.append(
        "Holding the variant constant, how did behaviour change between commits? "
        "Each sub-table aggregates across all tasks for one variant."
    )
    parts.append("")
    for variant in VARIANTS:
        rows = [(s, _aggregate(s, task_ids, [variant], model)) for s in shas]
        if not any(d["total_runs"] for _, d in rows):
            continue
        parts.append(
            _render_summary(
                f"Variant: {variant}",
                rows,
                _coverage_note(shas, task_ids, [variant], model),
            )
        )
    return "\n".join(parts)


# --------- top-level ---------


def compare(refs: list[str], model: str | None = None) -> str:
    shas = expand_refs(refs)
    if len(shas) < 2:
        return "compare needs at least two distinct refs"
    task_ids = sorted(set().union(*(discover_task_ids(s, model) for s in shas)))
    if not task_ids:
        return f"No results for any of: {shas}"

    header = f"# transformers agent behavior: {' → '.join(shas)}"
    if model:
        header += f"  [model: {model}]"

    out = [header, ""]
    out.append(_context_section())
    out.append(_variants_section())
    out.append(_commit_metadata(shas))
    out.append(_tasks_section())
    out.append(_metric_glossary())
    out.append(_headline_summary(shas, task_ids, model))
    out.append(_per_variant_summary(shas, task_ids, model))
    out.append("## Per-task results")
    out.append("")
    for tid in task_ids:
        section = _compare_task(tid, shas, model)
        if section:
            out.append(section)
    return "\n".join(out)
