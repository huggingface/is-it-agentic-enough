"""End-to-end: drive a real suite through the mock profile + mock runner (no
agent, no network, no install) and assert the whole pipeline — run → write →
trace → cleanup → web report — behaves."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ae.profile import get_profile
from ae.run_suite import run_suite

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
    from ae import store

    _run_suite("dev1")
    # 12 runs (2 tasks × 3 tiers × 2 runs) live in ONE shard per task (not 24 tiny
    # per-run files), under the model's shard dir — that's the sharded format.
    shards = sorted(p.name for p in (data_root / "results" / "dev1" / "mock" / "default").glob("*.jsonl"))
    assert shards == ["classify-sentiment.jsonl", "tokenize-count.jsonl"]

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
    from ae import store

    _run_suite("dev1")
    # one native-session file per run (Hub-detectable), named <tier>__<task>__run<N>.jsonl
    files = sorted(p.name for p in (data_root / "traces" / "dev1" / "mock" / "default").glob("*.jsonl"))
    assert len(files) == 12  # 2 tasks × 3 tiers × 2 runs
    assert "bare__classify-sentiment__run1.jsonl" in files
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


def test_report_payload_and_render(data_root):
    """`agent-eval report` over mock data: generic payload + data actually embedded in HTML."""
    from ae import report

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
    # index.html stays small: it loads the payload from data.js, not inline (so a
    # large report doesn't get LFS'd by the Hub and served as a download).
    assert '<script src="data.js">' in html                # data loaded externally
    assert '"profile":"mock"' not in html.replace(" ", "")  # payload NOT inlined
    data_js = (out / "data.js").read_text()
    assert data_js.startswith("window.__AG_DATA__=")
    assert '"profile":"mock"' in data_js.replace(" ", "")  # payload written to data.js
