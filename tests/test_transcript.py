import json

from ae.transcript import parse_transcript


def _write(tmp_path, lines):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(lines) + "\n" if lines else "")
    return p


def test_pairs_tool_use_with_result(tmp_path):
    p = _write(tmp_path, [
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "a", "name": "Bash", "input": {"command": "ls"}}]}}),
        json.dumps({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "boom", "is_error": True}]}}),
        json.dumps({"type": "result", "result": "final answer"}),
    ])
    tx = parse_transcript(p)
    assert len(tx.steps) == 1
    assert tx.steps[0].name == "Bash"
    assert tx.steps[0].result == "boom"
    assert tx.steps[0].is_error is True
    assert tx.final == "final answer"
    assert tx.broken is False and tx.missing is False


def test_missing_file(tmp_path):
    tx = parse_transcript(tmp_path / "nope.jsonl")
    assert tx.missing is True and tx.steps == [] and tx.final is None


def test_partial_last_line_is_broken_not_crash(tmp_path):
    p = _write(tmp_path, [
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "a", "name": "Bash", "input": {}}]}}),
        '{"type": "result", "resu',  # truncated in-flight write
    ])
    tx = parse_transcript(p)
    assert tx.broken is True
    assert len(tx.steps) == 1          # what parsed before the break is kept
    assert tx.final is None            # no fallback for broken transcripts


def test_pi_style_final_fallback_to_last_text(tmp_path):
    # Pi never emits a `result` event; final is the last assistant text.
    p = _write(tmp_path, [
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "the answer is positive"}]}}),
    ])
    tx = parse_transcript(p)
    assert tx.final == "the answer is positive"
