"""Built-in environment profiles. Importing this package registers them."""

from __future__ import annotations

from . import transformers  # noqa: F401  (registration side-effect)
from . import diffusers  # noqa: F401
from . import mock  # noqa: F401
