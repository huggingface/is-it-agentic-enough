"""Upload captured agent traces to the Hugging Face Hub as a dataset.

The Hub natively renders raw agent session JSONL (Claude Code, Codex, Pi) in a
dedicated trace viewer — see https://huggingface.co/docs/hub/agent-traces. This
module packages the native session files the harness collects under
``traces/<label>/`` (when runs are executed with ``--keep-sessions``) into a
dataset directory with a ``traces``-tagged dataset card, and shells out to the
``hf`` CLI to upload it.

**Safety:** uploads are *dry-run by default*. Nothing is pushed to the Hub
unless ``push=True`` is passed explicitly (``isth upload ... --push``). Traces
can contain prompts, command output, local paths, and secrets — review them (or
keep the dataset private) before publishing.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .log import log
from .paths import traces_dir


_CARD_TEMPLATE = """---
tags:
- traces
- agent-traces
---

# {repo}

Agent traces collected by [`isth`](https://github.com/) — headless coding-agent
runs over the `transformers` library across commits and discovery variants.

- **Runner / provider / model:** `{label}`
- **Sessions:** {n} native session `.jsonl` files (one per run).

Each file is a raw agent session, natively viewable in the Hub
[agent-traces viewer](https://huggingface.co/docs/hub/agent-traces).

> ⚠️ Traces may contain prompts, tool output, local paths, and secrets.
> Review before making this dataset public.
"""


def _have_hf_cli() -> bool:
    return shutil.which("hf") is not None


def stage(label: str | None, repo: str, dest: Path | None = None) -> tuple[Path, list[Path]]:
    """Assemble a staging dir: copy the label's trace files + write a dataset
    card. Returns ``(staging_dir, trace_files)``."""
    src = traces_dir(label)
    trace_files = sorted(p for p in src.glob("*.jsonl") if p.is_file())
    staging = dest or (src.parent / f".upload__{(label or 'default').replace('/', '__')}")
    staging.mkdir(parents=True, exist_ok=True)
    for p in trace_files:
        shutil.copyfile(p, staging / p.name)
    (staging / "README.md").write_text(
        _CARD_TEMPLATE.format(repo=repo, label=label or "(default)", n=len(trace_files))
    )
    return staging, trace_files


def upload(repo: str, label: str | None = None, *, push: bool = False, private: bool = True) -> int:
    """Stage traces for ``label`` and (optionally) upload them to ``repo``.

    Dry-run unless ``push=True``. Returns a process-style exit code.
    """
    staging, trace_files = stage(label, repo)
    if not trace_files:
        log(
            f"No native session traces found under traces/{label or ''}. "
            "Re-run with `--keep-sessions` to capture them."
        )
        return 1

    cmd = ["hf", "upload", repo, str(staging), ".", "--repo-type", "dataset"]
    if private:
        cmd.append("--private")

    log(f"staged {len(trace_files)} trace file(s) → {staging}")
    log("upload command:")
    log("  " + " ".join(cmd))

    if not push:
        log(
            "DRY RUN — nothing uploaded. Review the staged files above, then "
            "re-run with `--push` to upload to the Hub."
        )
        return 0

    if not _have_hf_cli():
        log(
            "The `hf` CLI is not installed. Install it with "
            "`curl -LsSf https://hf.co/cli/install.sh | bash` and `hf auth login`."
        )
        return 1

    log(f"▶ pushing to https://huggingface.co/datasets/{repo} …")
    proc = subprocess.run(cmd)
    return proc.returncode
