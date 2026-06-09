from types import SimpleNamespace

from ag import markers
from ag.markers import Marker
from ag.profiles.transformers import MARKERS


def _run(tool_calls=(), tool_results=(), final=""):
    return SimpleNamespace(tool_calls=list(tool_calls), tool_results=list(tool_results), final=final)


def test_run_corpus_scopes():
    run = _run(
        tool_calls=[
            ("Bash", {"command": "transformers classify --text x"}),
            ("Write", {"file_path": "go.py", "content": "from transformers import pipeline"}),
            ("Read", {"file_path": "/repo/src/transformers/cli/agentic/text.py"}),
        ],
        tool_results=["ok", "", ""],
        final="the answer",
    )
    c = markers.run_corpus(run)
    assert "transformers classify" in c["commands"]
    assert "pipeline" in c["wrote"]
    assert "cli/agentic/text.py" in c["reads"]
    assert c["final"] == "the answer"
    assert "transformers classify" in c["any"] and "pipeline" in c["any"]


def test_fired_scopes_are_isolated():
    # a `transformers` mention only in WROTE must not fire the commands-scoped cli marker
    run = _run(tool_calls=[("Write", {"file_path": "x.sh", "content": "transformers classify x"})])
    fired = markers.fired([Marker("cli", r"transformers\s+\S", "commands")], run)
    assert fired["cli"] is False


def test_transformers_cli_marker_matches_after_chain():
    # the bug the markers fixed: `cd ... && transformers ...` (first token is cd)
    run = _run(tool_calls=[("Bash", {"command": "cd /ws && transformers --format json classify --text x"})])
    fired = markers.fired(MARKERS, run)
    assert fired["cli"] is True
    assert fired["pipeline"] is False


def test_transformers_pipeline_and_exemplar_markers():
    run = _run(
        tool_calls=[
            ("Read", {"file_path": "/x/src/transformers/cli/agentic/vision.py"}),
            ("Bash", {"command": "python3 -c 'from transformers import pipeline; pipeline()'"}),
        ],
    )
    fired = markers.fired(MARKERS, run)
    assert fired["pipeline"] is True
    assert fired["agentic-exemplar"] is True
    assert fired["cli"] is False


def test_cli_marker_not_fooled_by_pip_install():
    run = _run(tool_calls=[("Bash", {"command": "pip install transformers && python go.py"})])
    assert markers.fired(MARKERS, run)["cli"] is False
