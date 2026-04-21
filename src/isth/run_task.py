"""Run one (sha, variant, task, run) through headless Claude Code."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from .log import extract_usage, log, summarize_event, vlog
from .paths import configs_dir, package_data_path, results_dir, transformers_src, workspaces_dir
from .setup_commit import resolve_sha, setup


VARIANTS = ("bare", "clone", "skill")


def load_tasks() -> dict:
    with open(package_data_path("tasks.yaml")) as f:
        return {t["id"]: t for t in yaml.safe_load(f)["tasks"]}


def _remove_workspace(ws: Path) -> None:
    """Best-effort cleanup: git worktree remove, fall back to rmtree."""
    if (ws / ".git").exists():
        subprocess.run(
            ["git", "-C", str(transformers_src()), "worktree", "remove", "--force", str(ws)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)


def _prepare_workspace(short_sha: str, variant: str, task_id: str, run_idx: int, sha: str) -> Path:
    """Create a clean workspace. For ``clone`` the workspace IS a git worktree
    of transformers @ sha so CLAUDE.md / AGENTS.md auto-discover from cwd.
    Otherwise the workspace is empty-but-for-inputs/."""
    ws = workspaces_dir() / f"{short_sha}__{variant}__{task_id}__run{run_idx}"
    if ws.exists():
        _remove_workspace(ws)

    if variant == "clone":
        subprocess.check_call(
            ["git", "-C", str(transformers_src()), "worktree", "add", "--detach", str(ws), sha],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        ws.mkdir(parents=True)

    shutil.copytree(package_data_path("inputs"), ws / "inputs")
    return ws


def _claude_cmd(prompt: str, ws: Path, cfg_dir: Path, variant: str, model: str | None) -> list[str]:
    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "bypassPermissions",
        "--no-session-persistence",
        "--add-dir",
        str(ws),
    ]
    if model:
        cmd.extend(["--model", model])
    if variant == "skill":
        cmd.extend(["--plugin-dir", str(cfg_dir / "plugin")])
    return cmd


def _env_for(venv_python: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{venv_python.parent}:{env.get('PATH', '')}"
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        env.pop(k, None)
    return env


def run(
    ref: str,
    variant: str,
    task_id: str,
    run_idx: int,
    model: str | None = None,
    max_tool_calls: int = 50,
) -> Path:
    if variant not in VARIANTS:
        raise SystemExit(f"Unknown variant: {variant}")

    tasks = load_tasks()
    if task_id not in tasks:
        raise SystemExit(f"Unknown task: {task_id}")
    prompt = tasks[task_id]["prompt"]

    sha = resolve_sha(ref)
    short = sha[:10]
    cfg_dir = configs_dir() / short
    if not (cfg_dir / ".ready").exists():
        setup(ref)

    if variant == "skill" and not (cfg_dir / "plugin" / "skills" / "transformers" / "SKILL.md").exists():
        raise SystemExit(f"skill not available for {short}")

    venv_python = cfg_dir / ".venv" / "bin" / "python"
    ws = _prepare_workspace(short, variant, task_id, run_idx, sha)

    rdir = results_dir(model)
    out_path = rdir / f"{short}__{variant}__{task_id}__run{run_idx}.jsonl"
    meta_path = rdir / f"{short}__{variant}__{task_id}__run{run_idx}.meta.json"

    cmd = _claude_cmd(prompt, ws, cfg_dir, variant, model)
    model_tag = f" [{model}]" if model else ""
    log(f"▶ {short} {variant} {task_id} run{run_idx}{model_tag}   cwd={ws.name}")

    totals = {"in": 0, "out": 0, "cache_read": 0, "cache_creation": 0}
    tool_call_count = 0
    status = "ok"  # "ok" | "budget_tool_calls" | "timeout"

    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=ws,
        env=_env_for(venv_python),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )
    with open(out_path, "w") as f:
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
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
                    proc.kill()
                    break
            summary = summarize_event(event)
            if summary:
                vlog(summary)
    try:
        proc.wait(timeout=15 * 60)
    except subprocess.TimeoutExpired:
        status = "timeout"
        proc.kill()
        proc.wait()
    stderr_output = proc.stderr.read() if proc.stderr else ""
    elapsed = time.time() - start

    meta_path.write_text(
        json.dumps(
            {
                "sha": sha,
                "short_sha": short,
                "variant": variant,
                "task_id": task_id,
                "run_index": run_idx,
                "model": model,
                "status": status,
                "tool_call_count": tool_call_count,
                "max_tool_calls": max_tool_calls,
                "elapsed_sec": round(elapsed, 1),
                "exit_code": proc.returncode,
                "tokens": totals,
                "stderr_tail": stderr_output[-2000:],
                "cmd": " ".join(shlex.quote(c) for c in cmd),
                "workspace": str(ws),
            },
            indent=2,
        )
    )

    tok_summary = (
        f"tokens in:{totals['in']} out:{totals['out']}"
        + (f" cache:{totals['cache_read']}" if totals["cache_read"] else "")
    )
    status_tag = "" if status == "ok" else f"  status={status}"
    log(
        f"■ {short} {variant} {task_id} run{run_idx}  {elapsed:.1f}s  "
        f"exit={proc.returncode}{status_tag}  {tok_summary}  → {out_path.name}"
    )
    return out_path
