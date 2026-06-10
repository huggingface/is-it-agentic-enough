"""``ag`` command-line interface."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _cmd_setup(args: argparse.Namespace) -> int:
    from .profile import get_profile

    p = get_profile(args.profile)
    if args.remove:
        from .setup_commit import cleanup  # transformers-specific teardown

        cleanup(args.ref)
    else:
        p.build(args.ref)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from . import store
    from .log import log
    from .paths import results_label
    from .profile import get_profile
    from .run_task import run

    profile = get_profile(args.profile)
    bindings = profile.expand_bindings([args.ref])  # canonical ids (transformers: short SHAs)
    tiers = args.tiers or profile.all_tiers()
    total = len(bindings) * len(tiers)
    ns = results_label(args.runner, args.model)
    done = 0
    for binding in bindings:
        for tier in tiers:
            done += 1
            if store.run_exists(binding, ns, tier, args.task, args.run_index) and not args.force_rerun:
                log(f"[{done}/{total}] skip (exists) {binding} {tier} {args.task} run{args.run_index}")
                continue
            log(f"[{done}/{total}] → {binding} {tier} {args.task} run{args.run_index}")
            try:
                run(
                    profile,
                    binding,
                    tier,
                    args.task,
                    args.run_index,
                    model=args.model,
                    max_tool_calls=args.max_tool_calls,
                    runner=args.runner,
                )
            except SystemExit as e:
                # e.g. "tier not available for <binding>" — don't abort the loop
                print(f"  ! {binding} {tier}: {e}", file=sys.stderr)
    return 0


def _cmd_suite(args: argparse.Namespace) -> int:
    if args.job:
        from .job import submit_suite

        return submit_suite(args)

    from .profile import get_profile
    from .run_suite import run_suite

    run_suite(
        args.ref,
        profile=get_profile(args.profile),
        runs=args.runs,
        tasks=args.tasks,
        tiers=args.tiers,
        skip_existing=not args.force_rerun,
        model=args.model,
        max_tool_calls=args.max_tool_calls,
        live=not args.no_live,
        runner=args.runner,
        name=args.name,
    )
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .analyze import analyze
    from .paths import results_label
    from .profile import get_profile

    p = get_profile(args.profile)
    label = results_label(args.runner, args.model)
    print(analyze(args.sha, args.task, ns=label, tiers=p.all_tiers(), markers=p.markers()))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from .compare import compare
    from .paths import results_label
    from .profile import get_profile

    p = get_profile(args.profile)
    label = results_label(args.runner, args.model)
    bindings = p.expand_bindings(args.refs)
    print(compare(bindings, ns=label, tiers=p.all_tiers(), markers=p.markers()))
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    from .explain import explain
    from .paths import results_label
    from .profile import get_profile

    p = get_profile(args.profile)
    label = results_label(args.runner, args.model)
    bindings = p.expand_bindings(args.refs)
    explain(bindings, args.tier, args.task, ns=label, markers=p.markers())
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    """Run the full matrix across refs in `ref1..ref2[..refN]`, then print the comparison.

    Iteration order is **task-first**: each task completes on all refs × variants
    × runs before moving to the next task. If the process is interrupted mid-way,
    the partial results are still comparable — earlier tasks have equal samples
    across refs.
    """
    from . import store
    from .compare import compare
    from .dashboard import Dashboard, stderr_is_tty
    from .log import get_console, log
    from .paths import results_label
    from .profile import get_profile
    from .run_task import load_tasks, run

    profile = get_profile(args.profile)
    label = results_label(args.runner, args.model)

    refs = profile.expand_bindings(args.spec if isinstance(args.spec, list) else [args.spec])
    if len(refs) < 2:
        print("diff needs a spec with at least two bindings (e.g. ref1..ref2)", file=sys.stderr)
        return 2

    tag = f" [{args.runner}:{args.model}]" if args.model else f" [{args.runner}]"
    log(f"diff{tag}: {' → '.join(refs)}")

    # Build each binding up front (and record which tiers it can actually run).
    available_tiers: dict[str, list[str]] = {}
    for ref in refs:
        available_tiers[ref] = profile.build(ref).available_tiers

    all_tasks = load_tasks()
    selected_tasks = args.tasks or list(all_tasks.keys())
    unknown = [t for t in selected_tasks if t not in all_tasks]
    if unknown:
        raise SystemExit(f"Unknown task ids: {unknown}")
    chosen_tiers = args.tiers or profile.all_tiers()

    # Plan order: task → run_idx → tier → binding. Rationale:
    #  - task-first: an interrupted diff leaves earlier tasks fully comparable
    #    across bindings and tiers (your original ask).
    #  - tier-before-binding inside a task: if *this* task is also cut short,
    #    each completed (task, tier) cell still has equal samples across
    #    bindings — so e.g. `bare` is comparable even if `clone` hasn't started.
    plan: list[tuple[str, str, str, int]] = []
    for tid in selected_tasks:
        # Explicit --runs overrides every per-task `runs:`; otherwise fall back
        # to the task's own override, then to 3.
        task_runs = args.runs if args.runs is not None else int(all_tasks[tid].get("runs") or 3)
        for run_idx in range(1, task_runs + 1):
            for tier in chosen_tiers:
                for ref in refs:
                    if tier not in available_tiers[ref]:
                        continue
                    plan.append((ref, tier, tid, run_idx))

    total = len(plan)
    enabled = (not args.no_live) and stderr_is_tty()
    title = (
        f"ag diff{tag}: " + " → ".join(refs)
        if len(refs) > 1
        else f"ag diff{tag}: {refs[0]}"
    )
    dash = Dashboard(
        refs=refs, plan=plan, console=get_console(), enabled=enabled, title=title
    )

    with dash.live():
        for i, (ref, tier, tid, run_idx) in enumerate(plan, 1):
            if not args.force_rerun:
                existing = store.get_run(ref, label, tier, tid, run_idx)
                if existing is not None:
                    log(f"[{i}/{total}] skip (exists) {ref} {tier} {tid} run{run_idx}")
                    dash.mark_skipped_existing(ref, tier, tid, run_idx, existing.meta)
                    continue
            log(f"[{i}/{total}] → {ref} {tier} {tid} run{run_idx}")
            dash.mark_running(ref, tier, tid, run_idx)
            try:
                record = run(
                    profile,
                    ref,
                    tier,
                    tid,
                    run_idx,
                    model=args.model,
                    max_tool_calls=args.max_tool_calls,
                    runner=args.runner,
                )
            except Exception as e:  # noqa: BLE001
                log(f"  ! failed: {e}")
                dash.mark_failed(ref, tier, tid, run_idx, str(e))
                continue
            if record is None or not record.meta:
                dash.mark_failed(ref, tier, tid, run_idx, "no meta")
            else:
                dash.mark_done(ref, tier, tid, run_idx, record.meta)

    print(compare(refs, ns=label))
    return 0


def _cmd_upload(args: argparse.Namespace) -> int:
    from .paths import results_label
    from .upload import upload

    label = results_label(args.runner, args.model)
    return upload(args.repo, label, push=args.push, private=not args.public)


def _cmd_report(args: argparse.Namespace) -> int:
    from .profile import get_profile
    from .report import report

    p = get_profile(args.profile)
    refs = p.expand_bindings(args.refs) if args.refs else None
    return report(
        refs,
        markers=p.markers(),
        profile_name=p.name,
        pull=args.pull,
        push=args.push,
        space_id=args.space,
        public=args.public,
        open_browser=args.open,
        bucket=args.bucket,
    )


def _cmd_sync(args: argparse.Namespace) -> int:
    from .sync import sync

    return sync(
        args.bucket,
        push=args.push,
        pull=args.pull,
        private=not args.public,
        delete=args.delete,
    )


def _cmd_batch(args: argparse.Namespace) -> int:
    from .batch import run_batch

    return run_batch(args.file, submit=args.submit, watch=args.watch, status=args.status,
                     poll=args.poll, force=args.force_rerun)


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
    sp.add_argument("--runner", default="claude", choices=["claude", "pi", "mock"], help=_RUNNER_HELP)


_PROFILE_HELP = (
    "Environment profile — what the agent runs inside and the comparison axis "
    "(it dictates everything; the revision only varies within it). `transformers` "
    "builds a git worktree of transformers at each revision (tiers bare/clone/skill); "
    "`mock` is a fast fake for UI / testing."
)


def _add_profile_arg(sp: argparse.ArgumentParser) -> None:
    """Add the leading required ``profile`` positional (first arg of every run /
    read command): ``ag <command> <profile> …``."""
    sp.add_argument("profile", help=_PROFILE_HELP)


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
        prog="ag",
        description="Run agentic-eval task suites inside a profile's environment and score them.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    _add_verbose_flag(p)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("setup", help="Build/refresh a profile's environment for a revision.")
    _add_profile_arg(sp)
    sp.add_argument("ref", help="Revision / ref (for transformers: SHA / branch / tag).")
    sp.add_argument("--remove", action="store_true", help="Remove the cache instead.")
    sp.set_defaults(func=_cmd_setup)

    rp = sub.add_parser(
        "run",
        help="Run (profile, ref, task, run-index) across one or more tiers.",
        description=(
            "Trailing tier tokens select which tiers to run. Omit them to run all "
            "of the profile's tiers (transformers: bare + clone + skill). Examples:\n"
            "  ag run transformers HEAD classify-sentiment 1              # all tiers\n"
            "  ag run transformers HEAD classify-sentiment 1 bare         # just bare\n"
            "  ag run transformers HEAD classify-sentiment 1 bare clone"
        ),
    )
    _add_profile_arg(rp)
    rp.add_argument("ref")
    rp.add_argument("task")
    rp.add_argument("run_index", type=int)
    rp.add_argument("tiers", nargs="*", default=None, help="Tiers to run (default: all of the profile's).")
    _add_model_flag(rp)
    _add_runner_flags(rp)
    _add_verbose_flag(rp)
    _add_force_rerun_flag(rp)
    _add_max_tool_calls_flag(rp)
    rp.set_defaults(func=_cmd_run)

    suite = sub.add_parser("suite", help="Run the full task suite for a profile + revision.")
    _add_profile_arg(suite)
    suite.add_argument("ref")
    suite.add_argument(
        "--name",
        default=None,
        help="Experiment title for this commit (e.g. \"kv-cache rewrite\"). Stored in "
        "results/<commit>/ref.json and used as the commit's display name throughout the "
        "report (the branch/release badge is kept). Re-running with --name updates it.",
    )
    suite.add_argument("--runs", type=int, default=None, help=_RUNS_HELP)
    suite.add_argument("--tasks", nargs="*", default=None)
    suite.add_argument("--tiers", nargs="*", default=None, help="Tiers to run (default: all of the profile's).")
    _add_model_flag(suite)
    _add_runner_flags(suite)
    _add_verbose_flag(suite)
    _add_force_rerun_flag(suite)
    _add_max_tool_calls_flag(suite)
    _add_no_live_flag(suite)
    job = suite.add_argument_group(
        "HF Jobs",
        "Submit the suite to Hugging Face Jobs instead of running locally. "
        "Requires --runner pi and --model; the bucket is volume-mounted so "
        "results land in it directly, and completed cells already in the "
        "bucket are skipped (resumable).",
    )
    job.add_argument(
        "--job",
        action="store_true",
        help="Submit this suite as a detached HF Job (`hf jobs run`) instead of running locally.",
    )
    job.add_argument("--flavor", default="t4-small", help="Job hardware flavor (see `hf jobs hardware`). Default: t4-small.")
    job.add_argument("--timeout", default="4h", help="Job max duration (e.g. 90m, 4h). HF's default is only 30m. Default: 4h.")
    job.add_argument("--image", default="node:22-bookworm", help="Docker image (needs apt or preinstalled git/node). Default: node:22-bookworm.")
    job.add_argument(
        "--bucket",
        default="lysandre/transformers-agentic-use",
        help="Bucket mounted read+write at /bucket for seeding + storing results "
        "(default: lysandre/transformers-agentic-use).",
    )
    suite.set_defaults(func=_cmd_suite)

    ap = sub.add_parser("analyze", help="Per-binding markdown report.")
    _add_profile_arg(ap)
    ap.add_argument("sha", help="Binding id (transformers: short SHA, first 10 chars).")
    ap.add_argument("task", nargs="?", default=None)
    _add_model_flag(ap)
    _add_runner_flags(ap)
    ap.set_defaults(func=_cmd_analyze)

    cp = sub.add_parser(
        "compare",
        help="Side-by-side table across refs.",
        description=(
            "Accepts refs as individual tokens (`ag compare A B C`) "
            "or as a range (`ag compare A..B..C`). Branches / tags / SHAs / "
            "short SHAs are all valid. Results must already exist — use `ag diff` "
            "to build+compare in one shot."
        ),
    )
    _add_profile_arg(cp)
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
    _add_profile_arg(dp)
    dp.add_argument("spec", help="Ref range like `ref1..ref2` or `A..B..C`.")
    dp.add_argument("--runs", type=int, default=None, help=_RUNS_HELP)
    dp.add_argument("--tasks", nargs="*", default=None)
    dp.add_argument("--tiers", nargs="*", default=None, help="Tiers to run (default: all of the profile's).")
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
            "while `ag diff` is still working — it only reads existing "
            "results files and tolerates in-flight `.jsonl` traces."
        ),
    )
    _add_profile_arg(ep)
    ep.add_argument("tier", help="Tier to drill into (e.g. bare/clone/skill for transformers).")
    ep.add_argument("task")
    ep.add_argument(
        "refs", nargs="+", help="Refs or ref-range (e.g. `A..B` or `A B C`)."
    )
    _add_model_flag(ep)
    _add_runner_flags(ep)
    ep.set_defaults(func=_cmd_explain)

    bp = sub.add_parser(
        "batch",
        help="Launch a model × revision matrix of suites from a YAML file (pi cells as HF Jobs).",
        description=(
            "Read a YAML file declaring `profile`, `models`, and `revisions` (with "
            "optional `name`s), expand the full model × revision matrix, and launch "
            "each cell: pi cells as detached HF Jobs (job ids recorded under "
            "batches/), claude cells locally. DRY-RUN by default — pass --submit to "
            "launch, and --watch to poll the jobs until they finish and report failures."
        ),
    )
    bp.add_argument("file", help="YAML batch config (profile, models, revisions, optional tasks/tiers/runs/flavor).")
    bp.add_argument("--submit", action="store_true", help="Actually launch (default: print the plan only).")
    bp.add_argument("--watch", action="store_true", help="Poll the jobs until done and report failures (with --submit, or alongside --status).")
    bp.add_argument("--status", action="store_true", help="Don't launch; report the current state of the batch's already-submitted jobs (from batches/<name>.json).")
    bp.add_argument("--poll", type=int, default=30, help="Watch poll interval in seconds (default 30).")
    _add_force_rerun_flag(bp)
    bp.set_defaults(func=_cmd_batch)

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

    rep = sub.add_parser(
        "report",
        help="Generate a self-contained static HTML report (charts + run drill-down).",
        description=(
            "Walk results/ and emit report/index.html — a single static page with "
            "interactive Plotly charts: cross-commit trends, model-vs-model "
            "comparison, a per-task heatmap with click-through run drill-down, and "
            "token/duration distributions. The run data is embedded as JSON and "
            "rendered client-side, so the page stays fully interactive with no "
            "server. The report/ dir (index.html + plotly.min.js + README.md with "
            "`sdk: static`) is complete HF static-Space content: publish with "
            "--push, or add the Space as a git remote and push it yourself. "
            "No flags = write locally and print the path."
        ),
    )
    _add_profile_arg(rep)
    rep.add_argument(
        "refs",
        nargs="*",
        default=None,
        help="Optional refs / ranges (e.g. `A..B`) to include. Default: every revision under results/.",
    )
    rep.add_argument("--pull", action="store_true", help="Sync results/ down from the bucket first.")
    rep.add_argument("--push", action="store_true", help="Upload the report as a static HF Space (otherwise just print the plan).")
    rep.add_argument(
        "--space",
        default="lysandre/transformers-agentic-use-report",
        help="Target Space id (default: lysandre/transformers-agentic-use-report).",
    )
    rep.add_argument(
        "--bucket",
        default="lysandre/transformers-agentic-use",
        help="Bucket used by --pull and for trace links (default: lysandre/transformers-agentic-use).",
    )
    rep.add_argument("--public", action="store_true", help="Create the Space as public (default: private).")
    rep.add_argument("--open", action="store_true", help="Open the generated report in a browser.")
    rep.set_defaults(func=_cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    from .log import set_verbose

    args = build_parser().parse_args(argv)
    set_verbose(getattr(args, "verbose", False))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
