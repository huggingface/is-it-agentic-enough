"""Run the full task suite for one binding of a profile's environment."""

from __future__ import annotations

from . import store
from .dashboard import Dashboard, stderr_is_tty
from .log import get_console, log
from .paths import results_label
from .run_task import load_tasks, run


def run_suite(
    ref: str,
    *,
    profile,
    runs: int | None = None,
    tasks: list[str] | None = None,
    tiers: list[str] | None = None,
    skip_existing: bool = False,
    model: str | None = None,
    max_tool_calls: int = 50,
    live: bool = True,
    runner: str = "claude",
    name: str | None = None,
) -> None:
    built = profile.build(ref, name=name)
    short = built.binding

    all_tasks = load_tasks()
    selected = tasks or list(all_tasks.keys())
    unknown = [t for t in selected if t not in all_tasks]
    if unknown:
        raise SystemExit(f"Unknown task ids: {unknown}")

    chosen_tiers = tiers or profile.all_tiers()
    resolved_tiers = [t for t in chosen_tiers if t in built.available_tiers]
    skipped = set(chosen_tiers) - set(resolved_tiers)
    if skipped:
        log(f"[{short}] skipping tiers unavailable for this binding: {sorted(skipped)}")

    def _runs_for(tid: str) -> int:
        # Explicit --runs overrides every per-task `runs:`; otherwise fall back
        # to the task's own override, then to 3.
        if runs is not None:
            return runs
        return int(all_tasks[tid].get("runs") or 3)

    ns = results_label(runner, model)
    plan: list[tuple[str, str, str, int]] = []
    for tid in selected:
        for tier in resolved_tiers:
            for run_idx in range(1, _runs_for(tid) + 1):
                plan.append((short, tier, tid, run_idx))

    total = len(plan)
    model_tag = f" [{runner}:{model}]" if model else f" [{runner}]"
    runs_desc = f"{runs} (--runs override)" if runs is not None else "per-task `runs:` or 3"
    log(
        f"suite {short}{model_tag}: {total} runs  "
        f"({len(selected)} tasks × {len(resolved_tiers)} tiers, "
        f"runs per task: {runs_desc})"
    )

    enabled = live and stderr_is_tty()
    dash = Dashboard(
        refs=[short],
        plan=plan,
        console=get_console(),
        enabled=enabled,
        title=f"ag suite{model_tag}: {short}",
    )

    with dash.live():
        for i, (_binding, tier, tid, run_idx) in enumerate(plan, 1):
            if skip_existing:
                existing = store.get_run(short, ns, tier, tid, run_idx)
                if existing is not None:
                    log(f"[{i}/{total}] skip (exists) {tier} {tid} run{run_idx}")
                    dash.mark_skipped_existing(short, tier, tid, run_idx, existing.meta)
                    continue
            log(f"[{i}/{total}] → {tier} {tid} run{run_idx}")
            dash.mark_running(short, tier, tid, run_idx)
            try:
                record = run(
                    profile, ref, tier, tid, run_idx,
                    model=model, max_tool_calls=max_tool_calls, runner=runner, name=name,
                )
            except Exception as e:  # noqa: BLE001
                log(f"  ! failed: {e}")
                dash.mark_failed(short, tier, tid, run_idx, str(e))
                continue
            if record is None or not record.meta:
                dash.mark_failed(short, tier, tid, run_idx, "no meta")
            else:
                dash.mark_done(short, tier, tid, run_idx, record.meta)
