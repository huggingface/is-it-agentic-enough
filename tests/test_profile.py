import pytest

from ag.profile import BuiltEnv, get_profile


def test_registry_has_builtins():
    assert get_profile("transformers").name == "transformers"
    assert get_profile("mock").name == "mock"
    assert get_profile(None).name == "transformers"  # default


def test_unknown_profile_raises():
    with pytest.raises(SystemExit):
        get_profile("does-not-exist")


def test_transformers_pure_bits():
    p = get_profile("transformers")
    assert p.all_tiers() == ["bare", "clone", "skill"]
    # hex SHAs pass through expand_bindings without touching git
    assert p.expand_bindings(["0ea540efff"]) == ["0ea540efff"]
    assert p.expand_bindings(["a" * 12 + "..", "b" * 12]) == ["a" * 10, "b" * 10]
    assert {m.name for m in p.markers()} == {"cli", "pipeline", "ran-help", "agentic-exemplar"}


def test_transformers_agent_assets_only_for_skill(tmp_path):
    p = get_profile("transformers")
    built = BuiltEnv(binding="x", python=tmp_path, available_tiers=["bare", "clone", "skill"], cfg_dir=tmp_path)
    assert p.agent_assets(built, "bare") == {}
    assets = p.agent_assets(built, "skill")
    assert "plugin_dir" in assets and "skill_dir" in assets


def test_mock_profile_is_instant_and_self_contained(tmp_path):
    p = get_profile("mock")
    assert p.expand_bindings(["dev1..dev2..dev1"]) == ["dev1", "dev2"]  # split + dedup, no git
    built = p.build("some/ref")
    assert built.available_tiers == ["bare", "clone", "skill"]
    ws = p.prepare_workspace(built, "bare", "classify-sentiment", 1)
    assert ws.exists()
    assert ws.name.startswith(f"{built.binding}__bare__classify-sentiment__run1__")
    p.remove_workspace(ws)
    assert not ws.exists()
    assert p.markers()  # reuses transformers markers
