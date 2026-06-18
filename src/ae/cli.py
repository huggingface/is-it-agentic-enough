"""``agent-eval`` command-line interface.

Launching is YAML-only: ``agent-eval batch <file.yaml>`` expands a model ×
revision matrix and submits each cell to Hugging Face Jobs (dry-run until
``--submit``). Results are viewed only in the web UI built by ``report``. The
``suite`` command is the per-revision worker the Job bootstrap runs inside the
container; it is intentionally hidden from ``--help``.
"""

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


def _cmd_suite(args: argparse.Namespace) -> int:
    """Worker: run the full task suite for one revision (invoked inside HF Jobs)."""
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


def _cmd_batch(args: argparse.Namespace) -> int:
    from .batch import run_batch

    return run_batch(args.file, submit=args.submit, watch=args.watch, status=args.status,
                     poll=args.poll, force=args.force_rerun, skip_complete=args.skip_complete,
                     per_task=args.per_task)


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


_MODEL_HELP = (
    "HF model id the `pi` runner serves via Hugging Face inference providers "
    "(e.g. `Qwen/Qwen3-Coder-480B-A35B-Instruct`). Results are namespaced under "
    "results/<commit>/<harness>/<model_id>/ (model_id is the model name, or "
    "`default` when omitted) so they don't collide with other runs."
)


def _add_model_flag(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--model", default=None, help=_MODEL_HELP)


_RUNNER_HELP = (
    "Coding agent that drives each run. `pi` (default) shells out to the `pi` "
    "CLI, which serves the `--model` via Hugging Face inference providers (needs "
    "HF_TOKEN); `mock` is a fast fake for UI / testing. Results are laid out as "
    "results/<commit>/<harness>/<model_id>/, e.g. pi/Qwen--Qwen3-Coder-480B-A35B-Instruct."
)


def _add_runner_flags(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--runner", default="pi", choices=["pi", "mock"], help=_RUNNER_HELP)


_PROFILE_HELP = (
    "Environment profile — what the agent runs inside and the comparison axis "
    "(it dictates everything, including the task set; the revision only varies "
    "within it). `transformers` builds a git worktree of transformers at each "
    "revision (tiers bare/clone/skill); `mock` is a fast fake for UI / testing."
)


def _add_profile_arg(sp: argparse.ArgumentParser) -> None:
    """Add the leading required ``profile`` positional: ``agent-eval <command> <profile> …``."""
    sp.add_argument("profile", help=_PROFILE_HELP)


_RUNS_HELP = (
    "Number of runs per (tier, task) cell. When given, this overrides ALL "
    "per-task `runs:` values the profile defines. When omitted, each task uses "
    "its own `runs:` override if it has one, otherwise 3."
)


def _add_verbose_flag(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("-v", "--verbose", action="store_true",
                    help="Emit per-tool-call events from each run (default: run-level summaries only).")


def _add_force_rerun_flag(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--force-rerun", action="store_true",
                    help="Re-run cells whose JSONL already exists in results/. "
                         "Default is to skip them (writes are expensive).")


def _add_max_tool_calls_flag(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--max-tool-calls", type=int, default=50,
                    help="Kill a run after this many tool calls; meta records "
                         "status=budget_tool_calls. Protects against agents that loop. Default: 50.")


def _add_no_live_flag(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--no-live", action="store_true",
                    help="Disable the live progress dashboard (auto-disabled when stderr isn't a "
                         "TTY). Plain timestamped log lines are still printed.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent-eval",
        description="Launch agentic-eval task suites on Hugging Face Jobs from a YAML matrix, "
                    "then publish the results as a static web report.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    _add_verbose_flag(p)
    # metavar omits the hidden `suite` worker from usage/help; it stays callable.
    sub = p.add_subparsers(dest="command", required=True,
                           metavar="{setup,batch,upload,sync,report}")

    sp = sub.add_parser("setup", help="Build/refresh a profile's environment for a revision.")
    _add_profile_arg(sp)
    sp.add_argument("ref", help="Revision / ref (for transformers: SHA / branch / tag).")
    sp.add_argument("--remove", action="store_true", help="Remove the cache instead.")
    sp.set_defaults(func=_cmd_setup)

    bp = sub.add_parser(
        "batch",
        help="Launch a model × revision matrix of suites from a YAML file (cells run as HF Jobs).",
        description=(
            "Read a YAML file declaring `profile`, `models`, and `revisions` (with "
            "optional `name`s), expand the full model × revision matrix, and launch "
            "each cell as a detached HF Job (job ids recorded under batches/). "
            "DRY-RUN by default — pass --submit to launch, and --watch to poll the "
            "jobs until they finish and report failures."
        ),
    )
    bp.add_argument("file", help="YAML batch config (profile, models, revisions, optional tasks/tiers/runs/flavor).")
    bp.add_argument("--submit", action="store_true", help="Actually launch (default: print the plan only).")
    bp.add_argument("--watch", action="store_true", help="Poll the jobs until done and report failures (with --submit, or alongside --status).")
    bp.add_argument("--status", action="store_true", help="Don't launch; report the current state of the batch's already-submitted jobs (from batches/<name>.json).")
    bp.add_argument("--poll", type=int, default=30, help="Watch poll interval in seconds (default 30).")
    bp.add_argument("--skip-complete", action=argparse.BooleanOptionalAction, default=True,
                    help="Check the bucket and skip cells whose runs are already fully present, and flag partially-"
                         "done cells (a prior job likely died). Default: on; pass --no-skip-complete to disable. "
                         "Per-cell resume still happens inside each launched job regardless.")
    bp.add_argument("--per-task", action="store_true",
                    help="Launch one job per (model × revision × task) instead of one per (model × revision). "
                         "Smaller, isolated, more parallel jobs — a failure on one task no longer blocks the rest "
                         "(at the cost of rebuilding the env per job). Can also be set as `per_task: true` in the YAML.")
    _add_force_rerun_flag(bp)
    bp.set_defaults(func=_cmd_batch)

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
            "Walk results/ and emit report/index.html — a static page with "
            "interactive Plotly charts: cross-commit trends, model-vs-model "
            "comparison, a per-task heatmap with click-through run drill-down, and "
            "token/duration distributions. The run data is written to report/data.js "
            "and loaded client-side, so the page stays fully interactive with no "
            "server. The report/ dir (index.html + data.js + plotly.min.js + "
            "README.md with `sdk: static`) is complete HF static-Space content: "
            "publish with --push, or add the Space as a git remote and push it "
            "yourself. No flags = write locally and print the path."
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

    # Hidden worker: the per-revision suite runner the HF Job bootstrap invokes
    # inside the container (`agent-eval suite <profile> <ref> --runner pi …`).
    # Not part of the user-facing surface — launching is YAML-only via `batch`.
    suite = sub.add_parser("suite")  # no help= → omitted from the help listing
    _add_profile_arg(suite)
    suite.add_argument("ref")
    suite.add_argument("--name", default=None,
                       help="Experiment title for this commit, stored in results/<commit>/ref.json.")
    suite.add_argument("--runs", type=int, default=None, help=_RUNS_HELP)
    suite.add_argument("--tasks", nargs="*", default=None)
    suite.add_argument("--tiers", nargs="*", default=None, help="Tiers to run (default: all of the profile's).")
    _add_model_flag(suite)
    _add_runner_flags(suite)
    _add_verbose_flag(suite)
    _add_force_rerun_flag(suite)
    _add_max_tool_calls_flag(suite)
    _add_no_live_flag(suite)
    suite.set_defaults(func=_cmd_suite)

    return p


def main(argv: list[str] | None = None) -> int:
    from .log import set_verbose

    args = build_parser().parse_args(argv)
    set_verbose(getattr(args, "verbose", False))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
