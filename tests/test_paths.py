from ae import paths


def test_state_root_honors_env(data_root):
    assert paths.state_root() == data_root.resolve()


def test_results_label_pi_default():
    assert paths.results_label("pi", None) == "pi/default"


def test_results_label_pi_sanitizes_slashes():
    # model slashes become `--` so model_id is one path segment
    assert paths.results_label("pi", "Qwen/Qwen3-Coder") == "pi/Qwen--Qwen3-Coder"


def test_model_id_default_when_missing():
    assert paths.model_id(None) == "default"
    assert paths.harness_id(None) == "pi"


def test_results_dir_layout(data_root):
    d = paths.results_dir("0ea540efff", "pi/default")
    assert d == data_root.resolve() / "results" / "0ea540efff" / "pi" / "default"
    assert d.is_dir()  # created on access


def test_traces_dir_mirrors_results(data_root):
    d = paths.traces_dir("abc", "pi/m")
    assert d == data_root.resolve() / "traces" / "abc" / "pi" / "m"


def test_mirror_persists_each_upsert(data_root, tmp_path, monkeypatch):
    """With AE_MIRROR_DIR set, every upsert also lands in the mirror immediately
    (the HF Job points this at /bucket so a mid-suite crash keeps finished runs)."""
    from ae import store

    mirror = tmp_path / "bucket"
    monkeypatch.setenv("AE_MIRROR_DIR", str(mirror))
    store.upsert_run("abc1234567", "pi/m", store.RunRecord("bare", "t1", 1, {"status": "ok"}, []))
    store.upsert_trace("abc1234567", "pi/m", "bare", "t1", 1, "native-session-text")

    # results: one task shard per (model, task); traces: one native session per run
    res = mirror / "results" / "abc1234567" / "pi" / "m" / "t1.jsonl"
    tr = mirror / "traces" / "abc1234567" / "pi" / "m" / "bare__t1__run1.jsonl"
    assert res.exists() and '"task": "t1"' in res.read_text()
    # the trace file is the raw session verbatim (no wrapper), so the Hub detects it
    assert tr.exists() and tr.read_text() == "native-session-text"
