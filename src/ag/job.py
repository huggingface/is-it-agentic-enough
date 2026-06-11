"""Submit an ``ag suite`` as a Hugging Face Job (``ag suite … --job``).

Instead of executing the suite locally, build the equivalent ``ag`` command,
wrap it in a self-contained bootstrap (uv + the ``pi`` CLI + clones of
``transformers`` and this repo), and submit it with ``hf jobs run``:

- the bucket is **volume-mounted read+write** into the job at ``/bucket``, so
  results land in it directly — no upload step and no extra auth;
- before running, the job seeds its local ``results/`` from the bucket so
  already-completed cells are skipped (interrupted jobs are resumable);
- after the suite (even a partially failed one), ``results/`` and ``traces/``
  are merged back into the bucket by commit directory.

Only the ``pi`` runner works on Jobs: the ``claude`` CLI needs interactive
auth, while ``pi`` just needs the ``HF_TOKEN`` secret (which ``ag`` strips
from the agent's task environment, keeping runs comparable). Track a submitted
job with ``hf jobs ps`` / ``hf jobs logs <id>``, then pull the new runs down
with ``ag report --pull``.
"""

from __future__ import annotations

import shlex
import subprocess

from .log import log
from .upload import _have_hf_cli

DEFAULT_IMAGE = "node:22-bookworm"  # git + node/npm for the pi CLI out of the box
# t4-medium = same T4 GPU as t4-small but 100 GB ephemeral storage (vs 50 GB).
# A suite downloads many task models into the HF cache; 50 GB evicts the pod.
DEFAULT_FLAVOR = "t4-medium"
DEFAULT_TIMEOUT = "4h"
DEFAULT_BUCKET = "lysandre/transformers-agentic-use"

AG_GIT = "https://github.com/huggingface/is-transformers-agentic-enough"
TRANSFORMERS_GIT = "https://github.com/huggingface/transformers"

# The bootstrap is submitted as ONE argv token after a single `-c`. The Jobs
# backend does not exec the command array verbatim: a combined `-lc` flag plus
# a multiline script reached bash as `bash <script-as-filename>` (no -c). The
# documented-working shape is `<exe> -c "<one-liner>"`, so the steps below are
# joined with ` ; ` into a single line — which is also why there are no `#`
# comments inside the steps (a comment would swallow the rest of the line).
_BOOTSTRAP_STEPS = [
    "set -euo pipefail",
    "status=0",
    # If the container is told to stop (timeout / eviction / OOM-adjacent),
    # exit 143 is otherwise silent — leave a breadcrumb in the job log first.
    'trap \'echo "::job:: SIGTERM received — timeout, eviction, or out-of-resources. Last status=${status}." >&2; df -h /work /bucket 2>/dev/null | tail -3 >&2; cp -a /work/state/results/. /bucket/results/ 2>/dev/null || true; cp -a /work/state/traces/. /bucket/traces/ 2>/dev/null || true; exit 143\' TERM',
    # toolchain: git/node/npm only if the image lacks them, then uv + the pi CLI
    "command -v npm >/dev/null || (apt-get update -qq && apt-get install -y -qq git curl nodejs npm)",
    "curl -LsSf https://astral.sh/uv/install.sh | sh",
    'export PATH="$HOME/.local/bin:$PATH"',
    "npm i -g @mariozechner/pi-coding-agent",
    "git clone --filter=blob:none __TRANSFORMERS_GIT__ /work/transformers",
    "git clone __AG_GIT__ /work/ag",
    "cd /work/ag",
    "uv venv --python 3.13 .env",
    "uv pip install --python .env/bin/python -e .",
    "export AG_TRANSFORMERS_SRC=/work/transformers",
    "export AG_DATA_DIR=/work/state",
    # Persist each run to the mounted bucket the moment it finishes (the store
    # mirrors its cell file to AG_MIRROR_DIR after every upsert). So a crash or
    # eviction mid-suite keeps every completed run instead of losing the job.
    "export AG_MIRROR_DIR=/bucket",
    # resumability: seed local state from the bucket so completed cells are skipped
    "mkdir -p /work/state/results /work/state/traces /bucket/results /bucket/traces",
    "cp -a /bucket/results/. /work/state/results/ 2>/dev/null || true",
    "__AG_CMD__ || status=$?",
    # safety net: a final merge back in case any per-run mirror write was skipped
    "cp -a /work/state/results/. /bucket/results/ 2>/dev/null || true",
    "cp -a /work/state/traces/. /bucket/traces/ 2>/dev/null || true",
    'exit "$status"',
]
_BOOTSTRAP = " ; ".join(_BOOTSTRAP_STEPS)


def _validate_remote_ref(ref: str) -> None:
    """Fail at submit time (≈1s, free) if ``ref`` won't resolve in the job's
    fresh clone of ``transformers`` — instead of minutes into a paid job.

    Validates against the *remote* (exactly what the job clones), so it works
    for branches your local checkout hasn't fetched yet. Raw SHAs and
    ``HEAD``-style expressions can't be checked via ``ls-remote`` and are
    trusted (the job's full clone has all reachable history)."""
    from .setup_commit import _looks_like_sha, suggest_refs

    if _looks_like_sha(ref) or ref == "HEAD" or any(t in ref for t in ("~", "^", "@{")):
        return
    found = (
        subprocess.run(
            ["git", "ls-remote", "--exit-code", TRANSFORMERS_GIT,
             f"refs/heads/{ref}", f"refs/tags/{ref}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if found:
        return
    names: list[str] = []
    for flag in ("--heads", "--tags"):
        proc = subprocess.run(
            ["git", "ls-remote", flag, "--refs", TRANSFORMERS_GIT],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                parts = line.split("\t")
                if len(parts) == 2:
                    names.append(parts[1].split("/", 2)[-1])
    sugg = suggest_refs(ref, names)
    hint = f" Did you mean: {', '.join(sugg)}?" if sugg else ""
    raise SystemExit(
        f"`{ref}` is neither a branch nor a tag on {TRANSFORMERS_GIT} "
        f"(checked with ls-remote — this is what the job would clone).{hint}"
    )


def build_suite_cmd(args) -> list[str]:
    """Reconstruct the `ag suite` command line the job should execute."""
    cmd = [".env/bin/ag", "suite", args.profile, args.ref, "--runner", "pi",
           "--model", args.model, "--no-live"]
    if args.name:
        cmd += ["--name", args.name]
    if args.runs is not None:
        cmd += ["--runs", str(args.runs)]
    if args.tasks:
        cmd += ["--tasks", *args.tasks]
    if args.tiers:
        cmd += ["--tiers", *args.tiers]
    if args.max_tool_calls != 50:
        cmd += ["--max-tool-calls", str(args.max_tool_calls)]
    if args.force_rerun:
        cmd += ["--force-rerun"]
    return cmd


def bootstrap_script(args) -> str:
    """The one-line bootstrap that builds the env and runs the suite inside the job."""
    return (
        _BOOTSTRAP
        .replace("__TRANSFORMERS_GIT__", TRANSFORMERS_GIT)
        .replace("__AG_GIT__", AG_GIT)
        .replace("__AG_CMD__", shlex.join(build_suite_cmd(args)))
    )


def _check_job_args(args) -> None:
    if (args.runner or "claude") != "pi":
        raise SystemExit(
            "--job requires --runner pi: the claude CLI can't authenticate on HF Jobs; "
            "the pi runner only needs the HF_TOKEN secret."
        )
    if not args.model:
        raise SystemExit("--job requires --model (the HF model id the pi runner should serve).")
    _validate_remote_ref(args.ref)


def build_job_invocation(args) -> list[str]:
    """The full `hf jobs run …` argv (also the unit under test)."""
    _check_job_args(args)
    return [
        "hf", "jobs", "run",
        "--flavor", args.flavor,
        "--timeout", args.timeout,
        "--detach",
        "--secrets", "HF_TOKEN",
        "--volume", f"hf://buckets/{args.bucket}:/bucket",
        args.image,
        "bash", "-c", bootstrap_script(args),
    ]


def submit_job_api(args):
    """Submit one suite as an HF Job via the ``huggingface_hub`` Python API and
    return the ``JobInfo`` (carries ``.id`` / ``.status`` / ``.url`` for tracking).
    Used by ``ag batch``; ``ag suite --job`` keeps the shell path."""
    import os

    from huggingface_hub import HfApi
    from huggingface_hub.cli.jobs import parse_volumes

    _check_job_args(args)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) must be set to submit jobs.")
    return HfApi().run_job(
        image=args.image,
        command=["bash", "-c", bootstrap_script(args)],
        secrets={"HF_TOKEN": token},
        volumes=parse_volumes([f"hf://buckets/{args.bucket}:/bucket"]),
        flavor=args.flavor,
        timeout=args.timeout,
    )


def submit_suite(args) -> int:
    cmd = build_job_invocation(args)
    log(f"submitting suite for {args.ref} [pi:{args.model}] as an HF Job "
        f"(flavor={args.flavor}, timeout={args.timeout}, bucket={args.bucket})")
    log("  " + " ".join(cmd[:-1]) + " '<bootstrap script>'")
    if not _have_hf_cli():
        log("The `hf` CLI is not installed. Install it with "
            "`curl -LsSf https://hf.co/cli/install.sh | bash` and `hf auth login`.")
        return 1
    rc = subprocess.run(cmd).returncode
    if rc == 0:
        log("✓ submitted — track with `hf jobs ps` / `hf jobs logs <job-id>`; "
            "when it finishes, `ag report --pull` brings the new runs into the dashboard.")
    return rc
