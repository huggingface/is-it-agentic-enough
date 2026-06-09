from ag import analyze
from ag.profiles.transformers import MARKERS


def test_parse_fields_and_match(write_run):
    # classify-sentiment has expected "positive" (substring) in the packaged tasks.yaml
    jsonl = write_run(
        "abc1234567", "bare", "classify-sentiment",
        tool_calls=[("Bash", {"command": "transformers classify --text x"}, "label: POSITIVE", False)],
        final="The sentiment is positive.",
    )
    run = analyze.parse(jsonl, "classify-sentiment")
    assert run.tool_calls == [("Bash", {"command": "transformers classify --text x"})]
    assert run.matched_expected is True
    assert run.first_success_turn == 1     # "positive" appears in the step-1 result
    assert run.errored_calls == 0
    assert run.exit_code == 0 and run.status == "ok"


def test_parse_errored_call(write_run):
    jsonl = write_run(
        "abc1234567", "bare", "classify-sentiment",
        tool_calls=[("Bash", {"command": "python x.py"}, "Traceback:\nKeyError", True)],
        final="could not finish",
    )
    run = analyze.parse(jsonl, "classify-sentiment")
    assert run.errored_calls == 1
    assert run.error_details == ["Traceback:"]
    assert run.matched_expected is False


def test_cell_renders_match_marker_and_tokens(write_run):
    jsonl = write_run(
        "abc1234567", "skill", "classify-sentiment",
        tool_calls=[("Bash", {"command": "transformers classify --text x"}, "POSITIVE", False)],
        final="positive",
    )
    run = analyze.parse(jsonl, "classify-sentiment")
    cell = analyze.cell([run], MARKERS)
    assert "✓1/1" in cell
    assert "🏷cli=1/1" in cell
    assert "out:" in cell  # token accounting present
    # no exclusive CLI/Python bucket wording anymore
    assert "CLI-clean" not in cell and "Python" not in cell


def test_cell_without_markers_has_no_tag(write_run):
    jsonl = write_run("abc1234567", "bare", "classify-sentiment", final="positive")
    run = analyze.parse(jsonl, "classify-sentiment")
    assert "🏷" not in analyze.cell([run], [])


def test_discover_task_ids(write_run, data_root):
    write_run("abc1234567", "bare", "classify-sentiment")
    write_run("abc1234567", "clone", "tokenize-count")
    ids = analyze.discover_task_ids("abc1234567", "claude/default")
    assert ids == ["classify-sentiment", "tokenize-count"]
