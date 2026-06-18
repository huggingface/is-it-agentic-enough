from ae import runs

# classify-sentiment scores against expected "positive" (substring); pass the
# task dict directly — runs.parse no longer looks tasks up by id.
SENTIMENT = {"expected": "positive", "match": "substring"}


def test_parse_fields_and_match(write_run):
    rec = write_run(
        "abc1234567", "bare", "classify-sentiment",
        tool_calls=[("Bash", {"command": "transformers classify --text x"}, "label: POSITIVE", False)],
        final="The sentiment is positive.",
    )
    run = runs.parse(rec, SENTIMENT)
    assert run.tool_calls == [("Bash", {"command": "transformers classify --text x"})]
    assert run.matched_expected is True
    assert run.first_success_turn == 1     # "positive" appears in the step-1 result
    assert run.errored_calls == 0
    assert run.exit_code == 0 and run.status == "ok"


def test_parse_errored_call(write_run):
    rec = write_run(
        "abc1234567", "bare", "classify-sentiment",
        tool_calls=[("Bash", {"command": "python x.py"}, "Traceback:\nKeyError", True)],
        final="could not finish",
    )
    run = runs.parse(rec, SENTIMENT)
    assert run.errored_calls == 1
    assert run.error_details == ["Traceback:"]
    assert run.matched_expected is False


def test_no_expected_task_has_no_match(write_run):
    rec = write_run("abc1234567", "bare", "generate-text", final="anything at all")
    run = runs.parse(rec, {})            # behavior-only task: no expected → matched is None
    assert run.matched_expected is None


def test_empty_run_reclassified_from_ok(write_run):
    # no tool calls, no final answer, zero output tokens, but the runner exited "ok"
    # (e.g. an unknown model id that returned an empty completion) → "empty", not "ok"
    rec = write_run(
        "abc1234567", "bare", "classify-sentiment",
        final="", status="ok", exit_code=0,
        tokens={"in": 0, "out": 0, "cache_read": 0, "cache_creation": 0},
    )
    assert runs.parse(rec, SENTIMENT).status == "empty"


def test_zero_tokens_with_output_stays_ok(write_run):
    # a run that produced a final answer is real work even if token accounting is 0
    rec = write_run(
        "abc1234567", "bare", "classify-sentiment",
        final="positive", status="ok",
        tokens={"in": 0, "out": 0, "cache_read": 0, "cache_creation": 0},
    )
    assert runs.parse(rec, SENTIMENT).status == "ok"


def test_empty_does_not_override_existing_failure(write_run):
    # a non-ok status (timeout/error/…) is never downgraded to "empty"
    rec = write_run(
        "abc1234567", "bare", "classify-sentiment",
        final="", status="timeout",
        tokens={"in": 0, "out": 0, "cache_read": 0, "cache_creation": 0},
    )
    assert runs.parse(rec, SENTIMENT).status == "timeout"


def test_step_kind_labels():
    assert runs.step_kind("Bash", {"command": "transformers classify"}) == "bash:transformers"
    assert runs.step_kind("Bash", {"command": "/usr/bin/python3 x.py"}) == "bash:python3"
    assert runs.step_kind("Read", {"file_path": "x"}) == "read"
