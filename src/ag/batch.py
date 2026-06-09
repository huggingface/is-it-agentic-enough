"""``ag batch <file.yaml>`` — launch a model × revision matrix of suites.

The YAML declares the profile, the models, and the revisions (with optional
display names); ``ag`` expands the full matrix and, with ``--submit``, launches
each cell — **pi** cells as detached HF Jobs (tracked by id), **claude** cells
locally — recording job ids to ``batches/<name>.json``. With ``--watch`` it then
polls the jobs until they finish and reports any that didn't complete.

Dry-run by default: ``ag batch m.yaml`` prints the matrix and the plan without
launching anything.

Example ``m.yaml``::

    profile: transformers
    runner: pi                       # default for models without a "/" rule below
    tasks: [classify-sentiment, extract-entities, tokenize-count]
    flavor: t4-small
    models:
      - claude                       # → runs locally (claude can't be a Job)
      - nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
      - Qwen/Qwen3-Coder-Next
    revisions:
      - v5.8.0
      - v5.9.0
      - {ref: 4d15b215f3, name: "w/ CLI + Skill"}
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import yaml

from .job import DEFAULT_BUCKET, DEFAULT_FLAVOR, DEFAULT_IMAGE, DEFAULT_TIMEOUT
from .log import log
from .paths import state_root

# Job stages that mean "stopped" (matched case-insensitively; superset for safety).
_TERMINAL = {"COMPLETED", "ERROR", "FAILED", "CANCELED", "CANCELLED", "DELETED"}


@dataclass
class Cell:
    profile: str
    ref: str
    name: str | None
    model: str | None
    runner: str

    def label(self) -> str:
        return f"{self.runner}:{self.model or 'default'} @ {self.name or self.ref}"


def _as_revision(item) -> tuple[str, str | None]:
    if isinstance(item, dict):
        ref = item.get("ref") or item.get("revision")
        if not ref:
            raise SystemExit(f"revision entry missing `ref`: {item!r}")
        return str(ref), item.get("name")
    return str(item), None


def _as_model(item, default_runner: str) -> tuple[str | None, str]:
    """Resolve a models[] entry to ``(model, runner)``.

    - ``claude`` → ``(None, "claude")`` (default Claude model, runs locally)
    - ``"<org>/<id>"`` → ``(id, "pi")`` (HF-served via the pi runner)
    - ``{model, runner}`` → explicit
    - any other bare string → ``(string, default_runner)``
    """
    if isinstance(item, dict):
        return item.get("model"), item.get("runner", default_runner)
    if item == "claude":
        return None, "claude"
    if "/" in item:
        return item, "pi"
    return item, default_runner


def load_batch(path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    for key in ("profile", "models", "revisions"):
        if not cfg.get(key):
            raise SystemExit(f"{path}: missing required `{key}`")
    return cfg


def expand(cfg: dict) -> list[Cell]:
    """The model × revision matrix, in models-then-revisions order."""
    default_runner = cfg.get("runner", "pi")
    cells: list[Cell] = []
    for mitem in cfg["models"]:
        model, runner = _as_model(mitem, default_runner)
        for ritem in cfg["revisions"]:
            ref, name = _as_revision(ritem)
            cells.append(Cell(cfg["profile"], ref, name, model, runner))
    return cells


def _cell_args(cfg: dict, cell: Cell) -> SimpleNamespace:
    """A suite-args namespace for one cell (consumed by job/run_suite)."""
    return SimpleNamespace(
        profile=cell.profile, ref=cell.ref, name=cell.name, model=cell.model, runner=cell.runner,
        tasks=cfg.get("tasks"), tiers=cfg.get("tiers"), runs=cfg.get("runs"),
        max_tool_calls=cfg.get("max_tool_calls", 50), force_rerun=cfg.get("force_rerun", False),
        flavor=cfg.get("flavor", DEFAULT_FLAVOR), timeout=cfg.get("timeout", DEFAULT_TIMEOUT),
        image=cfg.get("image", DEFAULT_IMAGE), bucket=cfg.get("bucket", DEFAULT_BUCKET),
    )


def plan_lines(cfg: dict, cells: list[Cell]) -> str:
    head = (f"batch: profile={cfg['profile']}  {len(cells)} cells "
            f"({len(cfg['models'])} models × {len(cfg['revisions'])} revisions)")
    meta = (f"  tasks: {cfg.get('tasks') or '(all)'}   tiers: {cfg.get('tiers') or '(all)'}   "
            f"runs: {cfg.get('runs') or 'per-task'}")
    rows = []
    for c in cells:
        where = f"HF Job (flavor={cfg.get('flavor', DEFAULT_FLAVOR)})" if c.runner == "pi" else "local"
        rows.append(f"  • {c.label():48} → {where}")
    return "\n".join([head, meta, *rows])


def _watch(submitted: list[tuple[Cell, str]], poll: int) -> list[tuple[str, str]]:
    """Poll until every submitted job is terminal; return [(job_id, stage)] of
    the ones that did not COMPLETE."""
    from huggingface_hub import HfApi

    api = HfApi()
    pending = {jid: cell for cell, jid in submitted}  # job_id -> Cell
    log(f"watching {len(pending)} job(s) (poll every {poll}s)…")
    finished: dict[str, str] = {}
    while pending:
        time.sleep(poll)
        for jid in list(pending):
            try:
                ji = api.inspect_job(job_id=jid)
                st = getattr(ji, "status", None)
                stage = st.get("stage") if isinstance(st, dict) else getattr(st, "stage", st)
            except Exception as e:  # noqa: BLE001
                stage = f"inspect-error: {e}"
            if str(stage).upper() in _TERMINAL or "error" in str(stage).lower():
                cell = pending.pop(jid)
                finished[jid] = str(stage)
                ok = str(stage).upper() == "COMPLETED"
                log(f"  {'✓' if ok else '✗'} {cell.label()}  [{stage}]")
    return [(jid, s) for jid, s in finished.items() if s.upper() != "COMPLETED"]


def run_batch(path, *, submit: bool = False, watch: bool = False, poll: int = 30) -> int:
    cfg = load_batch(path)
    cells = expand(cfg)
    log(plan_lines(cfg, cells))
    if not submit:
        log("DRY RUN — re-run with --submit to launch (pi cells → HF Jobs, claude cells → local).")
        return 0

    from .job import submit_job_api
    from .profile import get_profile
    from .run_suite import run_suite

    submitted: list[tuple[Cell, str]] = []  # (cell, job_id)
    failures: list[tuple[Cell, str]] = []
    for c in cells:
        args = _cell_args(cfg, c)
        try:
            if c.runner == "pi":
                ji = submit_job_api(args)
                submitted.append((c, ji.id))
                log(f"  ▶ submitted {c.label()}  job={ji.id}  {getattr(ji, 'url', '') or ''}")
            else:
                log(f"  ▶ running locally: {c.label()} …")
                run_suite(
                    c.ref, profile=get_profile(c.profile), runner=c.runner, model=c.model,
                    tasks=args.tasks, tiers=args.tiers, runs=args.runs,
                    max_tool_calls=args.max_tool_calls, live=False, name=c.name,
                    skip_existing=not args.force_rerun,
                )
        except Exception as e:  # noqa: BLE001
            failures.append((c, str(e)))
            log(f"  ! {c.label()}  {e}")

    if submitted:
        sdir = state_root() / "batches"
        sdir.mkdir(parents=True, exist_ok=True)
        sfile = sdir / f"{Path(path).stem}.json"
        sfile.write_text(json.dumps(
            {"profile": cfg["profile"],
             "jobs": [{"label": c.label(), "ref": c.ref, "name": c.name,
                       "model": c.model, "job_id": jid} for c, jid in submitted]},
            indent=2,
        ))
        log(f"recorded {len(submitted)} job(s) → {sfile}")

    if watch and submitted:
        failures += [(Cell(cfg["profile"], "?", None, None, "pi"), f"job {jid}: {st} (hf jobs logs {jid})")
                     for jid, st in _watch(submitted, poll)]
    elif submitted:
        log("track with `hf jobs ps` / `hf jobs logs <id>`; "
            f"then `ag report {cfg['profile']} --pull`.")

    if failures:
        log(f"⚠ {len(failures)} issue(s):")
        for c, why in failures:
            log(f"  - {why if c.ref == '?' else c.label() + ': ' + why}")
        return 1
    log("✓ batch complete.")
    return 0
