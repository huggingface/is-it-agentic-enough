"""Run the full task suite for a single transformers SHA."""

from __future__ import annotations

from .dashboard import Dashboard, stderr_is_tty
from .log import get_console, log
from .paths import results_dir, results_label
from .run_task import VARIANTS, load_tasks, run
from .setup_commit import setup
from .util import read_meta


def run_suite(
    ref: str,
    *,
    runs: int | None = None,
    tasks: list[str] | None = None,
    variants: list[str] | None = None,
    skip_existing: bool = False,
    model: str | None = None,
    max_tool_calls: int = 50,
    live: bool = True,
    runner: str = "claude",
) -> None:
    info = setup(ref)
    short = info["short"]
    skill_ok = info["skill_available"]

    all_tasks = load_tasks()
    selected = tasks or list(all_tasks.keys())
    unknown = [t for t in selected if t not in all_tasks]
    if unknown:
        raise SystemExit(f"Unknown task ids: {unknown}")

    chosen_variants = variants or list(VARIANTS)
    resolved_variants = [v for v in chosen_variants if v != "skill" or skill_ok]
    skipped = set(chosen_variants) - set(resolved_variants)
    if skipped:
        log(f"[{short}] skipping variants for this commit: {sorted(skipped)}")

    def _runs_for(tid: str) -> int:
        # Explicit --runs overrides every per-task `runs:`; otherwise fall back
        # to the task's own override, then to 3.
        if runs is not None:
            return runs
        return int(all_tasks[tid].get("runs") or 3)

    rdir = results_dir(short, results_label(runner, model))
    plan: list[tuple[str, str, str, int]] = []
    for tid in selected:
        for variant in resolved_variants:
            for run_idx in range(1, _runs_for(tid) + 1):
                plan.append((short, variant, tid, run_idx))

    total = len(plan)
    model_tag = f" [{runner}:{model}]" if model else f" [{runner}]"
    runs_desc = f"{runs} (--runs override)" if runs is not None else "per-task `runs:` or 3"
    log(
        f"suite {short}{model_tag}: {total} runs  "
        f"({len(selected)} tasks × {len(resolved_variants)} variants, "
        f"runs per task: {runs_desc})"
    )

    enabled = live and stderr_is_tty()
    dash = Dashboard(
        refs=[short],
        plan=plan,
        console=get_console(),
        enabled=enabled,
        title=f"isth suite{model_tag}: {short}",
    )

    with dash.live():
        for i, (_ref, variant, tid, run_idx) in enumerate(plan, 1):
            out = rdir / f"{variant}__{tid}__run{run_idx}.jsonl"
            meta_path = rdir / f"{variant}__{tid}__run{run_idx}.meta.json"
            if skip_existing and out.exists():
                log(f"[{i}/{total}] skip (exists) {variant} {tid} run{run_idx}")
                dash.mark_skipped_existing(
                    short, variant, tid, run_idx,
                    read_meta(meta_path),
                )
                continue
            log(f"[{i}/{total}] → {variant} {tid} run{run_idx}")
            dash.mark_running(short, variant, tid, run_idx)
            try:
                run(
                    ref, variant, tid, run_idx,
                    model=model, max_tool_calls=max_tool_calls, runner=runner,
                )
            except Exception as e:  # noqa: BLE001
                log(f"  ! failed: {e}")
                dash.mark_failed(short, variant, tid, run_idx, str(e))
                continue
            meta = read_meta(meta_path)
            if meta is None:
                dash.mark_failed(short, variant, tid, run_idx, "no meta")
            else:
                dash.mark_done(short, variant, tid, run_idx, meta)
