"""Path discovery for the harness.

All runtime state (configs, workspaces, results) lives under ``state_root()``
— by default the current working directory, overridable via the
``AE_DATA_DIR`` env var. Each profile's target repo (the library under test)
is located via :func:`repo_src`, defaulting to ``<state_root>/<name>`` as a
sibling of the repo (where the harness has historically sat next to it),
overridable via ``AE_<NAME>_SRC`` (e.g. ``AE_TRANSFORMERS_SRC``,
``AE_DIFFUSERS_SRC``).
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path


def state_root() -> Path:
    root = Path(os.environ.get("AE_DATA_DIR") or Path.cwd()).resolve()
    return root


def configs_dir() -> Path:
    d = state_root() / "configs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def workspaces_dir() -> Path:
    d = state_root() / "workspaces"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_component(s: str) -> str:
    """Make a model id usable as a single path component (``Qwen/Qwen3`` →
    ``Qwen--Qwen3``) so the layout stays exactly three levels deep."""
    return s.replace("/", "--")


def harness_id(runner: str | None) -> str:
    """The ``{harness}`` path component — the coding agent driving the run."""
    return runner or "pi"


def model_id(model: str | None) -> str:
    """The ``{model_id}`` path component: the model name (``/`` → ``--`` so it
    stays a single path segment), or ``default`` when no ``--model`` was given.
    For the HF-served ``pi`` runner this is e.g.
    ``Qwen--Qwen3-Coder-480B-A35B-Instruct``."""
    return _safe_component(model or "default")


def results_label(runner: str | None, model: str | None) -> str:
    """Per-commit namespace ``<harness>/<model_id>``.

    Results for a single run live at
    ``results/<commit>/<harness>/<model_id>/<variant>__<task>__runN.jsonl``;
    this returns the ``<harness>/<model_id>`` part. Callers join it under a
    commit via :func:`results_dir`.
    """
    return f"{harness_id(runner)}/{model_id(model)}"


def results_dir(commit: str | None = None, ns: str | None = None) -> Path:
    """Results directory, laid out as ``results/<commit>/<harness>/<model_id>/``.

    - ``results_dir()`` — the ``$ROOT/results/`` root.
    - ``results_dir(commit)`` — all runs for one commit.
    - ``results_dir(commit, ns)`` — one ``<harness>/<model_id>`` namespace
      (``ns`` from :func:`results_label`) for one commit.
    """
    d = state_root() / "results"
    if commit:
        d = d / commit
    if ns:
        d = d / ns
    d.mkdir(parents=True, exist_ok=True)
    return d


def traces_dir(commit: str | None = None, ns: str | None = None) -> Path:
    """Native-session collection dir, mirroring :func:`results_dir`'s layout
    under ``traces/`` for Hub upload."""
    d = state_root() / "traces"
    if commit:
        d = d / commit
    if ns:
        d = d / ns
    d.mkdir(parents=True, exist_ok=True)
    return d


def repo_src(name: str) -> Path:
    """Locate the source repo of the library under test (``transformers``,
    ``diffusers``, …). ``AE_<NAME>_SRC`` overrides; the default is a sibling of
    the state root, matching the harness's historical layout."""
    env_var = f"AE_{name.upper()}_SRC"
    override = os.environ.get(env_var)
    if override:
        return Path(override).resolve()
    candidate = (state_root().parent / name).resolve()
    if (candidate / ".git").exists():
        return candidate
    raise SystemExit(
        f"Could not locate the {name} source repo. Set {env_var} "
        f"to the repo path (tried {candidate})."
    )


def transformers_src() -> Path:
    return repo_src("transformers")


def package_data_path(*parts: str) -> Path:
    """Return a filesystem path to a packaged data file (tasks.yaml, inputs/...)."""
    ref = resources.files("ae").joinpath("data", *parts)
    # `resources.files` returns a MultiplexedPath-like; .resolve() for a wheel install
    # isn't guaranteed, but for editable installs (our main use case) it's a real Path.
    return Path(str(ref))
