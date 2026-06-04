"""Submit an ``isth suite`` as a Hugging Face Job (``isth suite … --job``).

Instead of executing the suite locally, build the equivalent ``isth`` command,
wrap it in a self-contained bootstrap (uv + the ``pi`` CLI + clones of
``transformers`` and this repo), and submit it with ``hf jobs run``:

- the bucket is **volume-mounted read+write** into the job at ``/bucket``, so
  results land in it directly — no upload step and no extra auth;
- before running, the job seeds its local ``results/`` from the bucket so
  already-completed cells are skipped (interrupted jobs are resumable);
- after the suite (even a partially failed one), ``results/`` and ``traces/``
  are merged back into the bucket by commit directory.

Only the ``pi`` runner works on Jobs: the ``claude`` CLI needs interactive
auth, while ``pi`` just needs the ``HF_TOKEN`` secret (which ``isth`` strips
from the agent's task environment, keeping runs comparable). Track a submitted
job with ``hf jobs ps`` / ``hf jobs logs <id>``, then pull the new runs down
with ``isth report --pull``.
"""

from __future__ import annotations

import shlex
import subprocess

from .log import log
from .upload import _have_hf_cli

DEFAULT_IMAGE = "node:22-bookworm"  # git + node/npm for the pi CLI out of the box
DEFAULT_FLAVOR = "t4-small"
DEFAULT_TIMEOUT = "4h"
DEFAULT_BUCKET = "lysandre/transformers-agentic-use"

ISTH_GIT = "https://github.com/huggingface/is-transformers-agentic-enough"
TRANSFORMERS_GIT = "https://github.com/huggingface/transformers"

# The bootstrap is submitted as ONE argv token after a single `-c`. The Jobs
# backend does not exec the command array verbatim: a combined `-lc` flag plus
# a multiline script reached bash as `bash <script-as-filename>` (no -c). The
# documented-working shape is `<exe> -c "<one-liner>"`, so the steps below are
# joined with ` ; ` into a single line — which is also why there are no `#`
# comments inside the steps (a comment would swallow the rest of the line).
_BOOTSTRAP_STEPS = [
    "set -euo pipefail",
    # toolchain: git/node/npm only if the image lacks them, then uv + the pi CLI
    "command -v npm >/dev/null || (apt-get update -qq && apt-get install -y -qq git curl nodejs npm)",
    "curl -LsSf https://astral.sh/uv/install.sh | sh",
    'export PATH="$HOME/.local/bin:$PATH"',
    "npm i -g @mariozechner/pi-coding-agent",
    "git clone --filter=blob:none __TRANSFORMERS_GIT__ /work/transformers",
    "git clone __ISTH_GIT__ /work/isth",
    "cd /work/isth",
    "uv venv --python 3.13 .env",
    "uv pip install --python .env/bin/python -e .",
    "export ISTH_TRANSFORMERS_SRC=/work/transformers",
    "export ISTH_DATA_DIR=/work/state",
    # resumability: seed local state from the bucket so completed cells are skipped
    "mkdir -p /work/state/results /work/state/traces /bucket/results /bucket/traces",
    "cp -a /bucket/results/. /work/state/results/ 2>/dev/null || true",
    "status=0",
    "__ISTH_CMD__ || status=$?",
    # land everything back in the mounted bucket (merge by commit dir), even if
    # the suite was interrupted — completed runs are still worth keeping
    "cp -a /work/state/results/. /bucket/results/ 2>/dev/null || true",
    "cp -a /work/state/traces/. /bucket/traces/ 2>/dev/null || true",
    'exit "$status"',
]
_BOOTSTRAP = " ; ".join(_BOOTSTRAP_STEPS)


def build_suite_cmd(args) -> list[str]:
    """Reconstruct the `isth suite` command line the job should execute."""
    cmd = [".env/bin/isth", "suite", args.ref, "--runner", "pi",
           "--model", args.model, "--no-live"]
    if args.runs is not None:
        cmd += ["--runs", str(args.runs)]
    if args.tasks:
        cmd += ["--tasks", *args.tasks]
    if args.variants:
        cmd += ["--variants", *args.variants]
    if args.max_tool_calls != 50:
        cmd += ["--max-tool-calls", str(args.max_tool_calls)]
    if args.force_rerun:
        cmd += ["--force-rerun"]
    return cmd


def build_job_invocation(args) -> list[str]:
    """The full `hf jobs run …` argv (also the unit under test)."""
    if (args.runner or "claude") != "pi":
        raise SystemExit(
            "--job requires --runner pi: the claude CLI can't authenticate on HF Jobs; "
            "the pi runner only needs the HF_TOKEN secret."
        )
    if not args.model:
        raise SystemExit("--job requires --model (the HF model id the pi runner should serve).")

    script = (
        _BOOTSTRAP
        .replace("__TRANSFORMERS_GIT__", TRANSFORMERS_GIT)
        .replace("__ISTH_GIT__", ISTH_GIT)
        .replace("__ISTH_CMD__", shlex.join(build_suite_cmd(args)))
    )
    return [
        "hf", "jobs", "run",
        "--flavor", args.flavor,
        "--timeout", args.timeout,
        "--detach",
        "--secrets", "HF_TOKEN",
        "--volume", f"hf://buckets/{args.bucket}:/bucket",
        args.image,
        "bash", "-c", script,
    ]


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
            "when it finishes, `isth report --pull` brings the new runs into the dashboard.")
    return rc
