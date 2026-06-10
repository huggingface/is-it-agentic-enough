"""End-to-end: drive a real suite through the mock profile + mock runner (no
agent, no network, no install) and assert the whole pipeline — run → write →
trace → cleanup → read-side reports — behaves."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ag import analyze, compare
from ag.profile import get_profile
from ag.run_suite import run_suite

PROFILE = get_profile("mock")
MARKERS = PROFILE.markers()
TASKS = ["classify-sentiment", "tokenize-count"]
TIERS = ["bare", "clone", "skill"]


def _run_suite(binding):
    run_suite(
        binding, profile=PROFILE, runner="mock", tasks=TASKS, tiers=TIERS,
        runs=2, live=False, skip_existing=False,
    )


def test_mock_suite_writes_expected_layout(data_root):
    from ag import store

    _run_suite("dev1")
    # All 12 runs (2 tasks × 3 tiers × 2 runs) live in ONE cell file, not 24
    # little ones — that's the point of the bundled format.
    cell_files = list((data_root / "results" / "dev1" / "mock").glob("*.jsonl"))
    assert [p.name for p in cell_files] == ["default.jsonl"]

    runs = store.list_runs("dev1", "mock/default")
    assert len(runs) == 12
    assert ("bare", "classify-sentiment", 1) in {r.key for r in runs}

    rec = store.get_run("dev1", "mock/default", "skill", "tokenize-count", 2)
    assert rec.meta["variant"] == "skill"        # tier recorded
    assert rec.meta["runner"] == "mock"
    assert rec.meta["status"] in ("ok", "error")
    assert isinstance(rec.meta["tool_call_count"], int)
    assert rec.events  # transcript embedded in the line


def test_mock_suite_populates_traces(data_root):
    from ag import store

    _run_suite("dev1")
    # one bundled traces file per cell, one line (native session) per run
    cell_files = list((data_root / "traces" / "dev1" / "mock").glob("*.jsonl"))
    assert [p.name for p in cell_files] == ["default.jsonl"]
    traces = store.list_traces("dev1", "mock/default")
    assert len(traces) == 12 and all(t.get("raw") for t in traces)


def test_mock_suite_cleans_up_workspaces(data_root):
    _run_suite("dev1")
    # per-run session staging dirs under <root>/workspaces are removed
    ws_root = data_root / "workspaces"
    leftover = list(ws_root.glob("*")) if ws_root.exists() else []
    assert leftover == []
    # the mock profile's tempdir workspaces are removed too
    assert not list(Path(tempfile.gettempdir()).glob("dev1__*"))


def test_read_side_over_mock_data(data_root):
    _run_suite("dev1")
    md = analyze.analyze("dev1", "classify-sentiment", ns="mock/default",
                         tiers=TIERS, markers=MARKERS)
    assert "# Agent behavior @ dev1" in md
    assert "### bare" in md and "### skill" in md
    # markers and/or match should surface somewhere in the cells
    assert "🏷" in md or "✓" in md


def test_compare_two_mock_bindings(data_root):
    _run_suite("dev1")
    _run_suite("dev2")
    out = compare.compare(["dev1", "dev2"], ns="mock/default", tiers=TIERS, markers=MARKERS)
    assert "dev1 → dev2" in out
    # one adoption row per marker, with both binding columns
    assert "🏷 `cli` adoption" in out
    assert "## Tier: bare" in out


def test_report_payload_and_render(data_root):
    """`ag report` over mock data: generic payload + data actually embedded in HTML."""
    from ag import report

    run_suite("v1", profile=PROFILE, runner="mock", tasks=["classify-sentiment"],
              tiers=["bare", "skill"], runs=2, live=False, name="my run")

    payload = report.collect_records(markers=MARKERS, profile_name="mock")
    assert payload["profile"] == "mock"
    assert payload["marker_names"]  # generic marker list carried
    rec = payload["runs"][0]
    assert "tier" in rec and "markers" in rec
    assert "approach" not in rec and "variant" not in rec  # transformers-isms gone
    assert payload["revisions"][0]["name"] == "my run"     # --name maps through

    # render offline (pre-seed the plotly stub so _ensure_plotly skips the network)
    out = data_root / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plotly.min.js").write_text("/*stub*/")
    html = report.render(payload, out).read_text()
    assert "__AG_DATA__" not in html and "__ISTH_DATA__" not in html  # placeholder substituted
    assert '"profile":"mock"' in html.replace(" ", "")                # data embedded
