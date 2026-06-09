import pytest

from ag import matcher


def test_substring_case_insensitive():
    assert matcher.check("Positive", "the answer is POSITIVE (0.99)") is True
    assert matcher.check("negative", "the answer is positive") is False


def test_default_mode_is_substring():
    assert matcher.DEFAULT_MODE == "substring"
    assert matcher.check("pos", "exposition")  # substring, not word-boundary


def test_exact():
    assert matcher.check("positive", "  Positive  ", "exact") is True
    assert matcher.check("positive", "very positive", "exact") is False


def test_regex_ignorecase():
    assert matcher.check(r"score:\s*\d\.\d+", "Score: 0.99", "regex") is True
    assert matcher.check(r"^\d+$", "not a number", "regex") is False


def test_none_final_never_matches():
    assert matcher.check("x", None) is False


def test_judge_not_implemented():
    with pytest.raises(NotImplementedError):
        matcher.check("x", "y", "judge")


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        matcher.check("x", "y", "nonsense")
