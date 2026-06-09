from ag import paths


def test_state_root_honors_env(data_root):
    assert paths.state_root() == data_root.resolve()


def test_results_label_claude_default():
    assert paths.results_label("claude", None) == "claude/default"
    assert paths.results_label("claude", "opus") == "claude/opus"


def test_results_label_pi_sanitizes_slashes():
    # pi (any non-claude) — model slashes become `--` so model_id is one path segment
    assert paths.results_label("pi", "Qwen/Qwen3-Coder") == "pi/Qwen--Qwen3-Coder"


def test_model_id_default_when_missing():
    assert paths.model_id(None) == "default"
    assert paths.harness_id(None) == "claude"


def test_results_dir_layout(data_root):
    d = paths.results_dir("0ea540efff", "claude/opus")
    assert d == data_root.resolve() / "results" / "0ea540efff" / "claude" / "opus"
    assert d.is_dir()  # created on access


def test_traces_dir_mirrors_results(data_root):
    d = paths.traces_dir("abc", "pi/m")
    assert d == data_root.resolve() / "traces" / "abc" / "pi" / "m"
