"""Path discovery for the harness.

All runtime state (configs, workspaces, results) lives under ``state_root()``
— by default the current working directory, overridable via the
``ISTH_DATA_DIR`` env var. The transformers source repo is located via
``transformers_src()``, defaulting to ``<state_root>/../transformers``
(where the harness has historically sat next to the repo), overridable via
``ISTH_TRANSFORMERS_SRC``.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path


def state_root() -> Path:
    root = Path(os.environ.get("ISTH_DATA_DIR") or Path.cwd()).resolve()
    return root


def configs_dir() -> Path:
    d = state_root() / "configs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def workspaces_dir() -> Path:
    d = state_root() / "workspaces"
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(model: str | None = None) -> Path:
    """Results directory, optionally namespaced by model.

    - ``results_dir()`` — the default ``$ROOT/results/`` (backwards compatible
      with runs that didn't specify a model).
    - ``results_dir("sonnet-shared-on-slack-old")`` — ``$ROOT/results/sonnet-shared-on-slack-old/``. Model-specific
      runs land in their own subdirectory so default-model runs aren't
      polluted and cross-model comparison stays cheap.
    """
    d = state_root() / "results"
    if model:
        d = d / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_label(runner: str | None, provider: str | None, model: str | None) -> str | None:
    """Namespace key passed to :func:`results_dir`.

    - ``claude`` (the default runner) keeps the historical scheme: ``model`` as
      the label (so ``results/`` and ``results/<model>/`` are unchanged).
    - other runners namespace under ``<runner>/<provider>/<model>`` so e.g. a
      Pi/HF run never collides with a Claude run of a same-named model. The
      label may contain ``/`` (incl. inside model ids); ``results_dir`` joins it
      as a sub-path, and run filenames carry no model, so nesting is safe.
    """
    if (runner or "claude") == "claude":
        return model
    parts = [runner or "claude"]
    if provider:
        parts.append(provider)
    if model:
        parts.append(model)
    return "/".join(parts)


def traces_dir(label: str | None = None) -> Path:
    """Directory where native agent session files are collected for Hub upload,
    namespaced the same way as :func:`results_dir`."""
    d = state_root() / "traces"
    if label:
        d = d / label
    d.mkdir(parents=True, exist_ok=True)
    return d


def transformers_src() -> Path:
    override = os.environ.get("ISTH_TRANSFORMERS_SRC")
    if override:
        return Path(override).resolve()
    # Historical default: sibling directory.
    candidate = (state_root().parent / "transformers").resolve()
    if (candidate / ".git").exists():
        return candidate
    raise SystemExit(
        "Could not locate the transformers source repo. Set ISTH_TRANSFORMERS_SRC "
        f"to the repo path (tried {candidate})."
    )


def package_data_path(*parts: str) -> Path:
    """Return a filesystem path to a packaged data file (tasks.yaml, inputs/...)."""
    ref = resources.files("isth").joinpath("data", *parts)
    # `resources.files` returns a MultiplexedPath-like; .resolve() for a wheel install
    # isn't guaranteed, but for editable installs (our main use case) it's a real Path.
    return Path(str(ref))
