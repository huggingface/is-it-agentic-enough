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


def _job_stage(api, jid: str) -> str:
    """Current stage of a job, defensively (JobInfo.status may be a dict or obj)."""
    try:
        ji = api.inspect_job(job_id=jid)
        st = getattr(ji, "status", None)
        stage = st.get("stage") if isinstance(st, dict) else getattr(st, "stage", st)
        return str(stage)
    except Exception as e:  # noqa: BLE001
        return f"inspect-error: {e}"


def _is_terminal(stage: str) -> bool:
    return stage.upper() in _TERMINAL or "error" in stage.lower()


def _track(entries: list[tuple[str, str]], *, watch: bool, poll: int) -> list[tuple[str, str, str]]:
    """``entries`` = ``[(label, job_id)]``. Snapshot (``watch=False``) prints each
    job's current stage and returns ``[]`` (running ≠ failed). ``watch=True`` polls
    until every job is terminal and returns ``[(label, job_id, stage)]`` for the
    ones that did not COMPLETE."""
    from huggingface_hub import HfApi

    api = HfApi()
    if not watch:
        for label, jid in entries:
            stage = _job_stage(api, jid)
            mark = "✓" if stage.upper() == "COMPLETED" else ("✗" if _is_terminal(stage) else "…")
            log(f"  {mark} {label}  [{stage}]")
        return []

    pending = {jid: label for label, jid in entries}
    bad: list[tuple[str, str, str]] = []
    log(f"watching {len(pending)} job(s) (poll every {poll}s)…")
    while pending:
        time.sleep(poll)
        for jid in list(pending):
            stage = _job_stage(api, jid)
            if _is_terminal(stage):
                label = pending.pop(jid)
                ok = stage.upper() == "COMPLETED"
                log(f"  {'✓' if ok else '✗'} {label}  [{stage}]")
                if not ok:
                    bad.append((label, jid, stage))
    return bad


def _state_file(path) -> Path:
    return state_root() / "batches" / f"{Path(path).stem}.json"


def _status(path, *, watch: bool, poll: int) -> int:
    sfile = _state_file(path)
    if not sfile.exists():
        log(f"no recorded jobs at {sfile} — run `ag batch {path} --submit` first.")
        return 1
    entries = [(j["label"], j["job_id"]) for j in json.loads(sfile.read_text()).get("jobs", [])]
    if not entries:
        log(f"{sfile}: no jobs recorded.")
        return 0
    log(f"status of {len(entries)} job(s) from {sfile.name}:")
    bad = _track(entries, watch=watch, poll=poll)
    if bad:
        log(f"⚠ {len(bad)} job(s) did not complete:")
        for label, jid, st in bad:
            log(f"  - {label}: {st}  (hf jobs logs {jid})")
        return 1
    return 0


def run_batch(path, *, submit: bool = False, watch: bool = False, status: bool = False,
              poll: int = 30) -> int:
    if status:
        return _status(path, watch=watch, poll=poll)

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
    failures: list[str] = []
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
            failures.append(f"{c.label()}: {e}")
            log(f"  ! {c.label()}  {e}")

    if submitted:
        _state_file(path).parent.mkdir(parents=True, exist_ok=True)
        _state_file(path).write_text(json.dumps(
            {"profile": cfg["profile"],
             "jobs": [{"label": c.label(), "ref": c.ref, "name": c.name,
                       "model": c.model, "job_id": jid} for c, jid in submitted]},
            indent=2,
        ))
        log(f"recorded {len(submitted)} job(s) → {_state_file(path)}")

    if watch and submitted:
        failures += [f"{label}: {st} (hf jobs logs {jid})"
                     for label, jid, st in _track([(c.label(), jid) for c, jid in submitted],
                                                   watch=True, poll=poll)]
    elif submitted:
        log(f"track with `ag batch {path} --status` (snapshot) or `--status --watch`; "
            f"then `ag report {cfg['profile']} --pull`.")

    if failures:
        log(f"⚠ {len(failures)} issue(s):")
        for why in failures:
            log(f"  - {why}")
        return 1
    log("✓ batch complete.")
    return 0
