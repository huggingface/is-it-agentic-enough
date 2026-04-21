"""Per-commit analysis + shared run-loading primitives used by compare."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .paths import package_data_path, results_dir


VARIANTS = ("bare", "clone", "skill")


@dataclass
class Run:
    tool_calls: list[tuple[str, dict]]
    tool_results: list[str]
    final: str | None
    read_agentic: set[str]
    read_docs: set[str]            # subset of {"AGENTS.md","CLAUDE.md","SKILL.md"} explicitly Read
    ran_help: bool                 # ran `transformers --help` (or subcommand --help)

    # From meta.json (fall back to zeros/None if meta missing or partial):
    elapsed: float
    tokens_in: int
    tokens_out: int
    tokens_cache_read: int
    tokens_cache_creation: int
    exit_code: int
    status: str                    # "ok" | "budget_tool_calls" | "timeout" (default "ok")

    # Derived:
    errored_calls: int             # count of tool_results with is_error=true
    error_details: list[str]       # first-line snippet of each errored result
    matched_expected: bool | None  # None if no expected substring is defined for the task
    first_success_turn: int | None  # tool-call index where expected substring first appeared in a tool_result; None if no expected or never matched


# --------- task metadata (expected-substring lookup) ---------


@lru_cache(maxsize=1)
def _task_expectations() -> dict[str, str]:
    """Return ``{task_id: expected_substring_lowercase}`` from the packaged tasks.yaml."""
    import yaml

    with open(package_data_path("tasks.yaml")) as f:
        data = yaml.safe_load(f)
    out: dict[str, str] = {}
    for task in data.get("tasks", []) or []:
        exp = task.get("expected")
        if isinstance(exp, str) and exp.strip():
            out[task["id"]] = exp.strip().lower()
    return out


# --------- parsing ---------


def _tool_result_content(block: dict) -> str:
    c = block.get("content", "")
    if isinstance(c, list):
        return "\n".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in c)
    return str(c)


def parse(path: Path, task_id: str) -> Run:
    tool_calls: list[tuple[str, dict]] = []
    tool_results: list[str] = []
    errored_calls = 0
    error_details: list[str] = []
    final: str | None = None
    read_agentic: set[str] = set()
    read_docs: set[str] = set()
    ran_help = False
    expected = _task_expectations().get(task_id)
    first_success_turn: int | None = None
    # We need to pair each tool_result with the tool_call that generated it.
    # Track by tool_use_id, and keep the running 1-indexed call number.
    call_index_by_id: dict[str, int] = {}

    for line in path.open():
        e = json.loads(line)
        t = e.get("type")
        if t == "assistant":
            for b in e.get("message", {}).get("content", []) or []:
                if b.get("type") == "tool_use":
                    name, inp = b.get("name", "?"), b.get("input") or {}
                    tool_calls.append((name, inp))
                    tid = b.get("id")
                    if tid:
                        call_index_by_id[tid] = len(tool_calls)
                    if name in ("Read", "Grep", "Glob"):
                        fp = inp.get("file_path") or inp.get("pattern") or inp.get("path") or ""
                        if "/cli/agentic/" in fp and fp.endswith(".py"):
                            read_agentic.add(Path(fp).name)
                        for doc in ("AGENTS.md", "CLAUDE.md", "SKILL.md"):
                            if doc in fp:
                                read_docs.add(doc)
                    elif name == "Bash":
                        cmd = (inp.get("command") or "")
                        if "transformers" in cmd and "--help" in cmd:
                            ran_help = True
        elif t == "user":
            for b in e.get("message", {}).get("content", []) or []:
                if b.get("type") == "tool_result":
                    content = _tool_result_content(b)
                    tool_results.append(content)
                    if b.get("is_error"):
                        errored_calls += 1
                        snippet = content.strip().splitlines()
                        error_details.append(snippet[0][:140] if snippet else "")
                    if (
                        first_success_turn is None
                        and expected
                        and expected in content.lower()
                    ):
                        first_success_turn = call_index_by_id.get(b.get("tool_use_id"), 0)
        elif t == "result":
            final = e.get("result") or ""

    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    tokens = meta.get("tokens") or {}

    matched = None
    if expected and final:
        matched = expected in final.lower()

    return Run(
        tool_calls=tool_calls,
        tool_results=tool_results,
        final=final,
        read_agentic=read_agentic,
        read_docs=read_docs,
        ran_help=ran_help,
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


# --------- classification ---------


def bucket(name: str, inp: dict) -> str:
    if name == "Bash":
        cmd = (inp.get("command") or "").strip()
        head = cmd.split()[0] if cmd else ""
        tail = head.split("/")[-1] if head else ""
        if cmd.startswith("transformers ") or tail == "transformers":
            return "CLI"
        if "python -c" in cmd or "python3 -c" in cmd:
            return "python -c"
        if cmd.startswith(("python ", "python3 ")) or cmd.startswith("./"):
            return "python <file>"
        if "pip install" in cmd:
            return "pip install"
        return f"bash:{tail}"
    if name == "Write":
        return "write .py" if (inp.get("file_path") or "").endswith(".py") else "write"
    if name == "Read":
        fp = inp.get("file_path", "")
        if "/cli/agentic/" in fp:
            return f"read agentic/{Path(fp).name}"
        return "read"
    return name.lower()


def _run_path(r: Run) -> str:
    """Base approach bucket for one run: 'CLI' | 'Python' | 'no-tool' | 'other'."""
    seen = {bucket(n, i) for n, i in r.tool_calls}
    if "CLI" in seen:
        return "CLI"
    if any(b.startswith(("python ", "write .py")) for b in seen):
        return "Python"
    if not r.tool_calls:
        return "no-tool"
    return "other"


def approach_counts(runs: list[Run]) -> Counter[str]:
    """Bucket each run. CLI/Python are split into -clean (no errored calls) and
    -retry (≥1 errored call). no-tool/other are not split."""
    counts: Counter[str] = Counter()
    for r in runs:
        path = _run_path(r)
        if path in ("CLI", "Python"):
            counts[f"{path}-{'retry' if r.errored_calls else 'clean'}"] += 1
        else:
            counts[path] += 1
    return counts


def approach(runs: list[Run]) -> str:
    """Short human string like 'CLI-clean=2/3 Python-retry=1/3'."""
    counts = approach_counts(runs)
    if not runs:
        return "—"
    return " ".join(f"{k}={v}/{len(runs)}" for k, v in counts.items())


# --------- aggregates for table cells ---------


def _fmt_tokens(n: int) -> str:
    if n >= 10_000:
        return f"{n // 1000}k"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def cell(runs: list[Run]) -> str:
    """Compact cell:

        **approach** · ✓match · !failed/total · ⇢first-success · 📖docs · ⏻abort ·
        time · new · repeat · out

    Fields that are zero / not applicable are omitted.

    - ``approach`` — bucketed counts (CLI-clean, CLI-retry, Python-clean,
      Python-retry, no-tool, other). "-retry" means ≥1 tool call in that run
      returned is_error=true; "-clean" means no errored tool calls.
    - ``⇢first-success`` — median tool-call index at which the agent's
      output first contained the task's `expected` substring (tasks with an
      expected only). Lower is better.
    - ``📖`` — runs that explicitly consulted in-repo docs. Shows any of
      ``agentic=k/n`` (Read a cli/agentic/*.py exemplar), ``help=k/n``
      (invoked transformers --help), or a filename (``CLAUDE.md=k/n`` etc.)
      when non-zero. AGENTS.md / CLAUDE.md / SKILL.md are usually auto-loaded
      by variant configuration and so rarely surface here.
    - ``new`` / ``repeat`` / ``out`` — token accounting (see report glossary).
    """
    if not runs:
        return "—"
    appr = approach(runs)
    parts = [f"**{appr}**"] if "CLI-" in appr else [appr]

    matched = [r.matched_expected for r in runs if r.matched_expected is not None]
    if matched:
        parts.append(f"✓{sum(matched)}/{len(matched)}")

    total_calls = sum(len(r.tool_calls) for r in runs)
    failed_calls = sum(r.errored_calls for r in runs)
    if failed_calls:
        parts.append(f"!{failed_calls}/{total_calls}")

    fs_turns = [r.first_success_turn for r in runs if r.first_success_turn is not None]
    if fs_turns:
        parts.append(f"⇢{int(_median([float(t) for t in fs_turns]))}")

    docs_parts: list[str] = []
    n = len(runs)
    agentic_hits = sum(1 for r in runs if r.read_agentic)
    if agentic_hits:
        docs_parts.append(f"agentic={agentic_hits}/{n}")
    help_hits = sum(1 for r in runs if r.ran_help)
    if help_hits:
        docs_parts.append(f"help={help_hits}/{n}")
    for doc in ("AGENTS.md", "CLAUDE.md", "SKILL.md"):
        hits = sum(1 for r in runs if doc in r.read_docs)
        if hits:
            docs_parts.append(f"{doc}={hits}/{n}")
    if docs_parts:
        parts.append("📖" + " ".join(docs_parts))

    aborted = [r.status for r in runs if r.status != "ok"]
    if aborted:
        counter: Counter[str] = Counter(aborted)
        parts.append("⏻" + ",".join(f"{k}:{v}" for k, v in counter.items()))

    parts.append(f"{_median([r.elapsed for r in runs]):.0f}s")
    parts.append(f"new:{_fmt_tokens(int(_median([r.tokens_in + r.tokens_cache_creation for r in runs])))}")
    parts.append(f"repeat:{_fmt_tokens(int(_median([r.tokens_cache_read for r in runs])))}")
    parts.append(f"out:{_fmt_tokens(int(_median([r.tokens_out for r in runs])))}")
    return " · ".join(parts)


# --------- loading ---------


def load_runs(short_sha: str, variant: str, task_id: str, model: str | None = None) -> list[Run]:
    return [
        parse(p, task_id)
        for p in sorted(results_dir(model).glob(f"{short_sha}__{variant}__{task_id}__run*.jsonl"))
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
        b = bucket(n, inp)
        extra = ""
        if n == "Bash":
            cmd = (inp.get("command") or "").replace("\n", " ⏎ ")
            extra = f" `{cmd[:140]}{'…' if len(cmd) > 140 else ''}`"
        elif n in ("Write", "Read"):
            extra = f" `{inp.get('file_path', '')}`"
        lines.append(f"  {i}. {b}{extra}")
    for detail in run.error_details:
        lines.append(f"    ✗ {detail}")
    if run.final:
        lines.append(f"  → {run.final.replace(chr(10), ' ')[:180]}")
    return lines


def _task_section(short_sha: str, task_id: str, model: str | None = None) -> str:
    lines: list[str] = [f"## {task_id}", ""]
    any_runs = False
    for variant in VARIANTS:
        runs = load_runs(short_sha, variant, task_id, model)
        if not runs:
            continue
        any_runs = True
        lines.append(f"### {variant}  — {cell(runs)}")
        for i, r in enumerate(runs, 1):
            lines.extend(_render_run(r, i))
        if any(r.read_agentic for r in runs):
            c: Counter[str] = Counter()
            for r in runs:
                for n in r.read_agentic:
                    c[n] += 1
            lines.append(f"  *read from cli/agentic/: {dict(c)}*")
        lines.append("")
    return "\n".join(lines) if any_runs else ""


def discover_task_ids(short_sha: str, model: str | None = None) -> list[str]:
    ids: set[str] = set()
    for path in results_dir(model).glob(f"{short_sha}__*.jsonl"):
        parts = path.stem.split("__")
        if len(parts) == 4:
            ids.add(parts[2])
    return sorted(ids)


def analyze(short_sha: str, task_id: str | None = None, model: str | None = None) -> str:
    tasks = [task_id] if task_id else discover_task_ids(short_sha, model)
    if not tasks:
        loc = f" ({model})" if model else ""
        return f"No results for {short_sha}{loc}"
    header = f"# Agent behavior — transformers @ {short_sha}"
    if model:
        header += f"  [model: {model}]"
    out = [header, ""]
    for tid in tasks:
        section = _task_section(short_sha, tid, model)
        if section:
            out.append(section)
    return "\n".join(out)
