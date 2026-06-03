"""``isth`` command-line interface."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _cmd_setup(args: argparse.Namespace) -> int:
    from .setup_commit import cleanup, setup

    if args.remove:
        cleanup(args.ref)
    else:
        setup(args.ref)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .compare import expand_refs
    from .log import log
    from .paths import results_dir, results_label
    from .run_task import VARIANTS, run

    refs = expand_refs([args.ref])  # each already resolved to a 10-char short sha
    variants = args.variants or list(VARIANTS)
    total = len(refs) * len(variants)
    ns = results_label(args.runner, args.model)
    done = 0
    for ref in refs:
        rdir = results_dir(ref, ns)
        for variant in variants:
            done += 1
            out = rdir / f"{variant}__{args.task}__run{args.run_index}.jsonl"
            if out.exists() and not args.force_rerun:
                log(f"[{done}/{total}] skip (exists) {ref} {variant} {args.task} run{args.run_index}")
                continue
            log(f"[{done}/{total}] → {ref} {variant} {args.task} run{args.run_index}")
            try:
                run(
                    ref,
                    variant,
                    args.task,
                    args.run_index,
                    model=args.model,
                    max_tool_calls=args.max_tool_calls,
                    runner=args.runner,
                )
            except SystemExit as e:
                # e.g. "skill not available for <sha>" — don't abort the loop
                print(f"  ! {ref} {variant}: {e}", file=sys.stderr)
    return 0


def _cmd_suite(args: argparse.Namespace) -> int:
    from .run_suite import run_suite

    run_suite(
        args.ref,
        runs=args.runs,
        tasks=args.tasks,
        variants=args.variants,
        skip_existing=not args.force_rerun,
        model=args.model,
        max_tool_calls=args.max_tool_calls,
        live=not args.no_live,
        runner=args.runner,
    )
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .analyze import analyze
    from .paths import results_label

    label = results_label(args.runner, args.model)
    print(analyze(args.sha, args.task, ns=label))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from .compare import compare
    from .paths import results_label

    label = results_label(args.runner, args.model)
    print(compare(args.refs, ns=label))
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    from .explain import explain
    from .paths import results_label

    label = results_label(args.runner, args.model)
    explain(args.refs, args.variant, args.task, ns=label)
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    """Run the full matrix across refs in `ref1..ref2[..refN]`, then print the comparison.

    Iteration order is **task-first**: each task completes on all refs × variants
    × runs before moving to the next task. If the process is interrupted mid-way,
    the partial results are still comparable — earlier tasks have equal samples
    across refs.
    """
    from .compare import compare, expand_refs
    from .dashboard import Dashboard, stderr_is_tty
    from .log import get_console, log
    from .paths import results_dir, results_label
    from .run_task import VARIANTS, load_tasks, run
    from .setup_commit import setup
    from .util import read_meta

    label = results_label(args.runner, args.model)

    refs = expand_refs(args.spec if isinstance(args.spec, list) else [args.spec])
    if len(refs) < 2:
        print("diff needs a spec with at least two refs (e.g. ref1..ref2)", file=sys.stderr)
        return 2

    tag = f" [{args.runner}:{args.model}]" if args.model else f" [{args.runner}]"
    log(f"diff{tag}: {' → '.join(refs)}")

    # Ensure each ref is set up before the first run touches it (and record which
    # refs can actually execute the `skill` variant).
    skill_available: dict[str, bool] = {}
    for ref in refs:
        skill_available[ref] = setup(ref)["skill_available"]

    all_tasks = load_tasks()
    selected_tasks = args.tasks or list(all_tasks.keys())
    unknown = [t for t in selected_tasks if t not in all_tasks]
    if unknown:
        raise SystemExit(f"Unknown task ids: {unknown}")
    chosen_variants = args.variants or list(VARIANTS)

    # Plan order: task → run_idx → variant → ref. Rationale:
    #  - task-first: an interrupted diff leaves earlier tasks fully comparable
    #    across refs and variants (your original ask).
    #  - variant-before-ref inside a task: if *this* task is also cut short,
    #    each completed (task, variant) cell still has equal samples across
    #    refs — so e.g. `bare` is comparable across refs even if `clone` hasn't
    #    started yet.
    plan: list[tuple[str, str, str, int]] = []
    for tid in selected_tasks:
        # Explicit --runs overrides every per-task `runs:`; otherwise fall back
        # to the task's own override, then to 3.
        task_runs = args.runs if args.runs is not None else int(all_tasks[tid].get("runs") or 3)
        for run_idx in range(1, task_runs + 1):
            for variant in chosen_variants:
                for ref in refs:
                    if variant == "skill" and not skill_available[ref]:
                        continue
                    plan.append((ref, variant, tid, run_idx))

    total = len(plan)
    enabled = (not args.no_live) and stderr_is_tty()
    title = (
        f"isth diff{tag}: " + " → ".join(refs)
        if len(refs) > 1
        else f"isth diff{tag}: {refs[0]}"
    )
    dash = Dashboard(
        refs=refs, plan=plan, console=get_console(), enabled=enabled, title=title
    )

    with dash.live():
        for i, (ref, variant, tid, run_idx) in enumerate(plan, 1):
            rdir = results_dir(ref, label)
            out_path = rdir / f"{variant}__{tid}__run{run_idx}.jsonl"
            meta_path = rdir / f"{variant}__{tid}__run{run_idx}.meta.json"
            if out_path.exists() and not args.force_rerun:
                log(f"[{i}/{total}] skip (exists) {ref} {variant} {tid} run{run_idx}")
                dash.mark_skipped_existing(
                    ref, variant, tid, run_idx, read_meta(meta_path),
                )
                continue
            log(f"[{i}/{total}] → {ref} {variant} {tid} run{run_idx}")
            dash.mark_running(ref, variant, tid, run_idx)
            try:
                run(
                    ref,
                    variant,
                    tid,
                    run_idx,
                    model=args.model,
                    max_tool_calls=args.max_tool_calls,
                    runner=args.runner,
                )
            except Exception as e:  # noqa: BLE001
                log(f"  ! failed: {e}")
                dash.mark_failed(ref, variant, tid, run_idx, str(e))
                continue
            meta = read_meta(meta_path)
            if meta is None:
                dash.mark_failed(ref, variant, tid, run_idx, "no meta")
            else:
                dash.mark_done(ref, variant, tid, run_idx, meta)

    print(compare(refs, ns=label))
    return 0


def _cmd_upload(args: argparse.Namespace) -> int:
    from .paths import results_label
    from .upload import upload

    label = results_label(args.runner, args.model)
    return upload(args.repo, label, push=args.push, private=not args.public)


def _cmd_sync(args: argparse.Namespace) -> int:
    from .sync import sync

    return sync(
        args.bucket,
        push=args.push,
        pull=args.pull,
        private=not args.public,
        delete=args.delete,
    )


def _cmd_tasks(args: argparse.Namespace) -> int:  # noqa: ARG001
    from .run_task import load_tasks

    for tid, task in load_tasks().items():
        print(f"{tid}  [{task.get('category', '?')}]")
    return 0


_MODEL_HELP = (
    "Model the runner uses. For `--runner claude`: a Claude alias/id (`sonnet`, "
    "`opus`, `claude-sonnet-4-6`), passed to `claude --model`. For `--runner pi`: "
    "an HF model id (`Qwen/Qwen3-Coder-480B-A35B-Instruct`), served via HF "
    "inference providers. Results are namespaced under "
    "results/<commit>/<harness>/<model_id>/ (model_id is the model name, or "
    "`default` when omitted) so they don't collide with other runs."
)


def _add_model_flag(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--model", default=None, help=_MODEL_HELP)


_RUNNER_HELP = (
    "Coding agent that drives each run. `claude` (default) shells out to the "
    "`claude` CLI (your configured Claude model); `pi` shells out to the `pi` "
    "CLI, which serves the `--model` via Hugging Face inference providers "
    "(needs HF_TOKEN). Results are laid out as "
    "results/<commit>/<harness>/<model_id>/, e.g. claude/opus or "
    "pi/Qwen--Qwen3-Coder-480B-A35B-Instruct."
)


def _add_runner_flags(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--runner", default="claude", choices=["claude", "pi"], help=_RUNNER_HELP)


_RUNS_HELP = (
    "Number of runs per (variant, task) cell. When given, this overrides ALL "
    "per-task `runs:` values in tasks.yaml. When omitted, each task uses its own "
    "`runs:` override if it has one, otherwise 3."
)


_VERBOSE_HELP = "Emit per-tool-call events from each run (default is run-level summaries only)."


def _add_verbose_flag(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("-v", "--verbose", action="store_true", help=_VERBOSE_HELP)


_FORCE_RERUN_HELP = (
    "Re-run cells whose JSONL already exists in results/. "
    "Default is to skip them (writes are expensive)."
)


def _add_force_rerun_flag(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--force-rerun", action="store_true", help=_FORCE_RERUN_HELP)


_MAX_TOOL_CALLS_HELP = (
    "Kill a run after this many tool calls; meta records status=budget_tool_calls. "
    "Protects against pathological agents that loop forever. Default: 50."
)


def _add_max_tool_calls_flag(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--max-tool-calls", type=int, default=50, help=_MAX_TOOL_CALLS_HELP)


_NO_LIVE_HELP = (
    "Disable the live progress dashboard (auto-disabled when stderr isn't a TTY). "
    "Plain timestamped log lines are still printed."
)


def _add_no_live_flag(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--no-live", action="store_true", help=_NO_LIVE_HELP)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="isth",
        description="Measure how agents use the transformers CLI across commits.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    _add_verbose_flag(p)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("setup", help="Create/refresh the per-commit cache (venv, worktree, skill).")
    sp.add_argument("ref", help="Git ref (SHA / branch / tag).")
    sp.add_argument("--remove", action="store_true", help="Remove the cache instead.")
    sp.set_defaults(func=_cmd_setup)

    rp = sub.add_parser(
        "run",
        help="Run (ref, task, run-index) across one or more variants.",
        description=(
            "Trailing variant tokens select which variants to run. Omit them to "
            "run all three (bare + clone + skill). Examples:\n"
            "  isth run HEAD classify-sentiment 1              # all variants\n"
            "  isth run HEAD classify-sentiment 1 bare         # just bare\n"
            "  isth run HEAD classify-sentiment 1 bare clone"
        ),
    )
    rp.add_argument("ref")
    rp.add_argument("task")
    rp.add_argument("run_index", type=int)
    rp.add_argument("variants", nargs="*", default=None, choices=["bare", "clone", "skill"])
    _add_model_flag(rp)
    _add_runner_flags(rp)
    _add_verbose_flag(rp)
    _add_force_rerun_flag(rp)
    _add_max_tool_calls_flag(rp)
    rp.set_defaults(func=_cmd_run)

    suite = sub.add_parser("suite", help="Run the full task suite for a ref.")
    suite.add_argument("ref")
    suite.add_argument("--runs", type=int, default=None, help=_RUNS_HELP)
    suite.add_argument("--tasks", nargs="*", default=None)
    suite.add_argument("--variants", nargs="*", default=None, choices=["bare", "clone", "skill"])
    _add_model_flag(suite)
    _add_runner_flags(suite)
    _add_verbose_flag(suite)
    _add_force_rerun_flag(suite)
    _add_max_tool_calls_flag(suite)
    _add_no_live_flag(suite)
    suite.set_defaults(func=_cmd_suite)

    ap = sub.add_parser("analyze", help="Per-sha markdown report.")
    ap.add_argument("sha", help="Short SHA (first 10 chars).")
    ap.add_argument("task", nargs="?", default=None)
    _add_model_flag(ap)
    _add_runner_flags(ap)
    ap.set_defaults(func=_cmd_analyze)

    cp = sub.add_parser(
        "compare",
        help="Side-by-side table across refs.",
        description=(
            "Accepts refs as individual tokens (`isth compare A B C`) "
            "or as a range (`isth compare A..B..C`). Branches / tags / SHAs / "
            "short SHAs are all valid. Results must already exist — use `isth diff` "
            "to build+compare in one shot."
        ),
    )
    cp.add_argument("refs", nargs="+", help="Refs or ref-range (e.g. `A..B` or `A B C`).")
    _add_model_flag(cp)
    _add_runner_flags(cp)
    cp.set_defaults(func=_cmd_compare)

    dp = sub.add_parser(
        "diff",
        help="Run the suite for each ref, then compare (github-style ref1..ref2).",
        description=(
            "End-to-end: takes a ref range (`A..B` or `A..B..C`), ensures each commit's "
            "cache is built, runs the suite for each (skipping runs that already exist "
            "by default), and prints the comparison table on stdout."
        ),
    )
    dp.add_argument("spec", help="Ref range like `ref1..ref2` or `A..B..C`.")
    dp.add_argument("--runs", type=int, default=None, help=_RUNS_HELP)
    dp.add_argument("--tasks", nargs="*", default=None)
    dp.add_argument("--variants", nargs="*", default=None, choices=["bare", "clone", "skill"])
    _add_model_flag(dp)
    _add_verbose_flag(dp)
    _add_force_rerun_flag(dp)
    _add_max_tool_calls_flag(dp)
    _add_no_live_flag(dp)
    _add_runner_flags(dp)
    dp.set_defaults(func=_cmd_diff)

    ep = sub.add_parser(
        "explain",
        help="Per-cell breakdown for one (variant, task) across one or more refs.",
        description=(
            "Print, for each ref, the tool-call timeline of every run that's "
            "already on disk for the given (variant, task) cell, plus a "
            "side-by-side metric diff when two refs are given. Safe to run "
            "while `isth diff` is still working — it only reads existing "
            "results files and tolerates in-flight `.jsonl` traces."
        ),
    )
    ep.add_argument("variant", choices=["bare", "clone", "skill"])
    ep.add_argument("task")
    ep.add_argument(
        "refs", nargs="+", help="Refs or ref-range (e.g. `A..B` or `A B C`)."
    )
    _add_model_flag(ep)
    _add_runner_flags(ep)
    ep.set_defaults(func=_cmd_explain)

    tp = sub.add_parser("tasks", help="List the available task ids.")
    tp.set_defaults(func=_cmd_tasks)

    up = sub.add_parser(
        "upload",
        help="Upload captured native agent traces to a Hugging Face Hub dataset.",
        description=(
            "Package the native session files captured under "
            "traces/<commit>/<harness>/<model_id>/ (every run captures one) into a "
            "dataset directory with a `traces`-tagged card, and upload via the `hf` CLI. "
            "DRY-RUN by default — nothing is pushed unless you pass --push. "
            "Datasets are created private unless you pass --public."
        ),
    )
    up.add_argument("repo", help="Target dataset repo, e.g. `username/transformers-agent-traces`.")
    _add_model_flag(up)
    _add_runner_flags(up)
    up.add_argument("--push", action="store_true", help="Actually upload (otherwise dry-run).")
    up.add_argument("--public", action="store_true", help="Create the dataset as public (default: private).")
    up.set_defaults(func=_cmd_upload)

    syp = sub.add_parser(
        "sync",
        help="Sync results/ + traces/ + a run manifest with a Hugging Face bucket.",
        description=(
            "Mirror local run state to/from a Hugging Face bucket (S3-like Xet "
            "storage) via `hf buckets sync`. Refreshes results/MANIFEST.json (the "
            "record of which configs/commits were run), then syncs results/ and "
            "traces/ (laid out as <commit>/<harness>/<model_id>/) under matching "
            "prefixes in the bucket. DRY-RUN by default — pass --push to upload "
            "(creating the bucket if needed) or --pull to download. Buckets are "
            "created private unless --public."
        ),
    )
    syp.add_argument(
        "bucket",
        nargs="?",
        default="lysandre/transformers-agentic-use",
        help="Target bucket id <namespace>/<name> (default: lysandre/transformers-agentic-use).",
    )
    syp.add_argument("--push", action="store_true", help="Upload results/ + traces/ to the bucket.")
    syp.add_argument("--pull", action="store_true", help="Download results/ + traces/ from the bucket.")
    syp.add_argument("--public", action="store_true", help="Create the bucket as public (default: private).")
    syp.add_argument(
        "--delete",
        action="store_true",
        help="Also remove files on the receiving side that no longer exist on the sending side "
        "(rsync --delete semantics). Off by default (sync only adds/updates).",
    )
    syp.set_defaults(func=_cmd_sync)

    return p


def main(argv: list[str] | None = None) -> int:
    from .log import set_verbose

    args = build_parser().parse_args(argv)
    set_verbose(getattr(args, "verbose", False))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
