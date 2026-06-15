import json

import pytest

from ag import batch


def _write(tmp_path, text):
    p = tmp_path / "m.yaml"
    p.write_text(text)
    return p


CONFIG = """
profile: transformers
tasks: [classify-sentiment, tokenize-count]
flavor: t4-small
models:
  - claude
  - Qwen/Qwen3-Coder-Next
  - {model: custom-thing, runner: pi}
revisions:
  - v5.8.0
  - {ref: 4d15b215f3, name: "w/ CLI + Skill"}
"""


def test_load_requires_keys(tmp_path):
    for missing in ("profile: x\nmodels: [a]\n", "profile: x\nrevisions: [v1]\n", "models: [a]\nrevisions: [v1]\n"):
        with pytest.raises(SystemExit):
            batch.load_batch(_write(tmp_path, missing))


def test_expand_matrix_and_runner_resolution(tmp_path):
    cfg = batch.load_batch(_write(tmp_path, CONFIG))
    cells = batch.expand(cfg)
    assert len(cells) == 3 * 2  # 3 models × 2 revisions

    # claude → local (runner claude, default model)
    claude_cells = [c for c in cells if c.runner == "claude"]
    assert len(claude_cells) == 2 and all(c.model is None for c in claude_cells)

    # "<org>/<id>" → pi
    qwen = [c for c in cells if c.model == "Qwen/Qwen3-Coder-Next"]
    assert qwen and all(c.runner == "pi" for c in qwen)

    # explicit {model, runner}
    custom = [c for c in cells if c.model == "custom-thing"]
    assert custom and all(c.runner == "pi" for c in custom)

    # named revision carried through
    named = [c for c in cells if c.ref == "4d15b215f3"]
    assert named and all(c.name == "w/ CLI + Skill" for c in named)


def test_plan_is_dry_run(tmp_path, capsys):
    # skip_complete=False keeps this offline (no bucket lookup)
    rc = batch.run_batch(_write(tmp_path, CONFIG), submit=False, skip_complete=False)
    assert rc == 0
    # run_batch logs via ag.log → stderr; just assert it didn't try to launch
    # (no exception, returns 0). The plan content is asserted directly below.


def test_plan_lines_content(tmp_path):
    cfg = batch.load_batch(_write(tmp_path, CONFIG))
    txt = batch.plan_lines(cfg, batch.expand(cfg))
    assert "6 cells (3 models × 2 revisions)" in txt
    assert "claude:default @ v5.8.0" in txt
    assert "pi:Qwen/Qwen3-Coder-Next @ w/ CLI + Skill" in txt
    assert "→ local" in txt and "→ HF Job (flavor=t4-small)" in txt


def test_cell_args_threads_config(tmp_path):
    cfg = batch.load_batch(_write(tmp_path, CONFIG))
    cell = batch.expand(cfg)[0]
    a = batch._cell_args(cfg, cell)
    assert a.profile == "transformers"
    assert a.tasks == ["classify-sentiment", "tokenize-count"]
    assert a.flavor == "t4-small"
    assert a.runs is None  # not set → per-task default downstream


def test_per_task_expansion(tmp_path):
    cfg = batch.load_batch(_write(tmp_path, CONFIG))   # 3 models × 2 revisions, 2 tasks
    assert len(batch.expand(cfg)) == 6                  # default: one cell per model×revision
    pt = batch.expand(cfg, per_task=True)
    assert len(pt) == 6 * 2                             # × 2 tasks
    assert all(c.task for c in pt)
    # each per-task cell runs exactly its one task
    a = batch._cell_args(cfg, pt[0])
    assert a.tasks == [pt[0].task]
    # label carries the task so jobs/state stay distinguishable
    assert pt[0].task in pt[0].label()


def test_force_flag_overrides_cfg(tmp_path):
    cfg = batch.load_batch(_write(tmp_path, CONFIG))
    cell = batch.expand(cfg)[0]
    assert batch._cell_args(cfg, cell).force_rerun is False           # neither YAML nor flag
    assert batch._cell_args(cfg, cell, force=True).force_rerun is True  # --force-rerun wins


def test_skip_complete_predicate():
    all_tasks = {"a": {}, "b": {"runs": 2}}          # a → default 3 runs, b → 2
    cfg = {"tasks": ["a", "b"], "tiers": ["bare", "clone"]}
    expected = batch._expected_run_keys(cfg, {"bare", "clone"}, all_tasks)
    assert len(expected) == 2 * (3 + 2)               # 2 tiers × (3 + 2) runs

    assert batch._is_complete(expected, {"bare", "clone"}, cfg, all_tasks) is True
    assert batch._is_complete(expected | {("bare", "a", 99)}, {"bare", "clone"}, cfg, all_tasks) is True
    # missing one run → not complete
    short = expected - {("clone", "b", 2)}
    assert batch._is_complete(short, {"bare", "clone"}, cfg, all_tasks) is False
    # empty present / empty tiers → never complete (never-run cell is launched)
    assert batch._is_complete(set(), {"bare"}, cfg, all_tasks) is False
    assert batch._is_complete(expected, set(), cfg, all_tasks) is False


def test_runs_for_explicit_overrides_per_task():
    all_tasks = {"a": {"runs": 7}}
    assert batch._runs_for({"runs": 2}, "a", all_tasks) == 2      # --runs / cfg runs wins
    assert batch._runs_for({}, "a", all_tasks) == 7              # else per-task
    assert batch._runs_for({}, "z", all_tasks) == 3              # else default 3


def test_status_without_state_file_errors(tmp_path, data_root):
    p = tmp_path / "m.yaml"
    p.write_text("ignored")  # status mode doesn't parse the YAML, only its stem
    assert batch.run_batch(p, status=True) == 1


def _seed_state(data_root, stem, jobs):
    sdir = data_root / "batches"
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / f"{stem}.json").write_text(json.dumps({"profile": "transformers", "jobs": jobs}))


class _FakeJI:
    def __init__(self, stage):
        self.status = {"stage": stage}


def test_status_snapshot_all_completed(tmp_path, data_root, monkeypatch):
    import huggingface_hub
    _seed_state(data_root, "m", [{"label": "pi:x @ v1", "job_id": "j1"},
                                 {"label": "pi:y @ v1", "job_id": "j2"}])
    monkeypatch.setattr(huggingface_hub.HfApi, "inspect_job",
                        lambda self, job_id: _FakeJI("COMPLETED"))
    p = tmp_path / "m.yaml"; p.write_text("x")
    assert batch.run_batch(p, status=True, watch=False) == 0


def test_watch_reports_non_completed(tmp_path, data_root, monkeypatch):
    import huggingface_hub
    _seed_state(data_root, "m", [{"label": "pi:x @ v1", "job_id": "j1"}])
    monkeypatch.setattr(huggingface_hub.HfApi, "inspect_job",
                        lambda self, job_id: _FakeJI("ERROR"))
    p = tmp_path / "m.yaml"; p.write_text("x")
    # watch with poll=0 → terminal ERROR on first pass → reported as a failure (rc 1)
    assert batch.run_batch(p, status=True, watch=True, poll=0) == 1
