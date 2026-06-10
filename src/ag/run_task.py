"""Run one (sha, variant, task, run) through headless Claude Code."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from . import store
from .log import extract_usage, log, summarize_event, vlog
from .paths import (
    package_data_path,
    results_label,
    workspaces_dir,
)
from .runners import get_runner


def load_tasks() -> dict:
    with open(package_data_path("tasks.yaml")) as f:
        return {t["id"]: t for t in yaml.safe_load(f)["tasks"]}


def _redact_cmd(cmd: list[str]) -> list[str]:
    """Mask secret-bearing flag values (e.g. ``--api-key``) before recording
    the command in meta.json, which may be shared."""
    out = list(cmd)
    for i, tok in enumerate(out):
        if tok == "--api-key" and i + 1 < len(out):
            out[i + 1] = "***"
    return out


def run(
    profile,
    ref: str,
    tier: str,
    task_id: str,
    run_idx: int,
    model: str | None = None,
    max_tool_calls: int = 50,
    runner: str = "claude",
    name: str | None = None,
) -> Path:
    if tier not in profile.all_tiers():
        raise SystemExit(f"Unknown tier {tier!r} for profile {profile.name!r}")

    tasks = load_tasks()
    if task_id not in tasks:
        raise SystemExit(f"Unknown task: {task_id}")
    prompt = tasks[task_id]["prompt"]

    built = profile.build(ref, name=name)  # idempotent: reuses the cached sandbox
    if tier not in built.available_tiers:
        raise SystemExit(f"tier {tier!r} not available for {built.binding}")

    runner_impl = get_runner(runner)
    ws = profile.prepare_workspace(built, tier, task_id, run_idx)
    assets = profile.agent_assets(built, tier)

    ns = results_label(runner, model)

    # Per-run staging dir for the agent's *native* session. Always captured —
    # sharing traces (Hub agent-traces viewer) is the point of the harness.
    session_dir = workspaces_dir() / f"{ws.name}__session"
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
    session_dir.mkdir(parents=True)

    try:
        return _run_body(
            runner_impl, ws, built, assets, session_dir,
            prompt, tier, task_id, run_idx, ref, ns,
            model, runner, max_tool_calls,
        )
    finally:
        # Workspaces (e.g. a full transformers worktree for the `clone` tier) and
        # the native-session staging dir are scratch — the run's transcript,
        # meta, and copied trace already live under results/ + traces/. Remove
        # them so a suite doesn't accumulate tens of GB (HF Jobs evict at 50G).
        profile.remove_workspace(ws)
        shutil.rmtree(session_dir, ignore_errors=True)


def _run_body(
    runner_impl,
    ws: Path,
    built,
    assets: dict,
    session_dir: Path,
    prompt: str,
    tier: str,
    task_id: str,
    run_idx: int,
    ref: str,
    ns: str | None,
    model: str | None,
    runner: str,
    max_tool_calls: int,
):
    # Aliases so the recording/logging below stays identical to the pre-profile code.
    variant = tier
    short = built.binding
    sha = built.extra.get("sha", built.binding)
    venv_python = built.python

    cmd = runner_impl.build_cmd(prompt, ws, assets, model, session_dir)
    model_tag = f" [{runner}:{model}]" if model else f" [{runner}]"
    log(f"▶ {short} {variant} {task_id} run{run_idx}{model_tag}   cwd={ws.name}")

    totals = {"in": 0, "out": 0, "cache_read": 0, "cache_creation": 0}
    tool_call_count = 0
    status = "ok"  # "ok" | "budget_tool_calls" | "timeout" | "error"
    result_is_error = False

    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=ws,
        env=runner_impl.env(venv_python, session_dir),
        stdin=subprocess.DEVNULL,  # pi blocks reading stdin when stdout isn't a TTY
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )
    budget_hit = False
    events: list = []  # canonical transcript events, bundled into the cell file at the end
    if True:
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Normalize the runner's native event(s) to the canonical schema,
            # then collect *those* so the read-side stays runner-agnostic.
            for event in runner_impl.normalize(raw):
                events.append(event)
                if event.get("type") == "assistant":
                    delta = extract_usage(event)
                    for k, v in delta.items():
                        totals[k] += v
                    for b in event.get("message", {}).get("content", []) or []:
                        if b.get("type") == "tool_use":
                            tool_call_count += 1
                    if tool_call_count > max_tool_calls:
                        status = "budget_tool_calls"
                        log(f"  ⏻ budget exceeded ({tool_call_count} tool calls > {max_tool_calls}); killing run")
                        budget_hit = True
                elif event.get("type") == "result" and event.get("is_error"):
                    # The agent terminated in an error state (e.g. invalid model,
                    # auth failure) — distinct from a clean run that used 0 tools.
                    result_is_error = True
                summary = summarize_event(event)
                if summary:
                    vlog(summary)
            if budget_hit:
                proc.kill()
                break
    try:
        proc.wait(timeout=15 * 60)
    except subprocess.TimeoutExpired:
        status = "timeout"
        proc.kill()
        proc.wait()
    stderr_output = proc.stderr.read() if proc.stderr else ""
    elapsed = time.time() - start

    # A nonzero exit or an is_error result means the run didn't actually do its
    # job — flag it so it isn't reported as a clean zero-tool run. (budget/timeout
    # kills also exit nonzero, but those statuses are already set above.)
    if status == "ok" and (result_is_error or proc.returncode not in (0, None)):
        status = "error"

    # Collect the agent's native session (bundled verbatim into the traces cell
    # file; upload.py unpacks it back into a per-run file for the Hub viewer).
    have_trace = False
    native = runner_impl.collect_session(session_dir, ws)
    if native and native.exists():
        store.upsert_trace(short, ns, variant, task_id, run_idx, native.read_text())
        have_trace = True
    else:
        log("  (no native session captured for this run)")

    meta = {
        "sha": sha,
        "short_sha": short,
        "ref": ref,
        "variant": variant,
        "task_id": task_id,
        "run_index": run_idx,
        "runner": runner,
        "model": model,
        "status": status,
        "tool_call_count": tool_call_count,
        "max_tool_calls": max_tool_calls,
        "elapsed_sec": round(elapsed, 1),
        "exit_code": proc.returncode,
        "tokens": totals,
        "stderr_tail": stderr_output[-2000:],
        "cmd": " ".join(shlex.quote(c) for c in _redact_cmd(cmd)),
        "workspace": str(ws),
        "has_trace": have_trace,
    }
    record = store.RunRecord(tier=variant, task=task_id, run=run_idx, meta=meta, events=events)
    cell_path = store.upsert_run(short, ns, record)

    tok_summary = (
        f"tokens in:{totals['in']} out:{totals['out']}"
        + (f" cache:{totals['cache_read']}" if totals["cache_read"] else "")
    )
    status_tag = "" if status == "ok" else f"  status={status}"
    log(
        f"■ {short} {variant} {task_id} run{run_idx}  {elapsed:.1f}s  "
        f"exit={proc.returncode}{status_tag}  {tok_summary}  → {cell_path.name}"
    )
    # Surface *why* a run failed inline (otherwise it's buried in meta.json).
    if status != "ok":
        rc = proc.returncode
        if rc is not None and rc < 0:
            log(f"  ↳ killed by signal {-rc} (often OOM / timeout / job eviction)")
        tail = [ln for ln in (stderr_output or "").splitlines() if ln.strip()]
        for ln in tail[-6:]:
            log(f"  ↳ stderr: {ln[:300]}")
        if not tail:
            log("  ↳ (no stderr captured)")
    return record
