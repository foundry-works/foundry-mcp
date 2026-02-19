"""Backward-compatible shim — canonical location: foundry_mcp.core.metrics.registry

.. deprecated::
    Import from ``foundry_mcp.core.metrics.registry`` instead.
"""

import warnings as _warnings

_warnings.warn(
    "foundry_mcp.core.metrics_registry is deprecated, use foundry_mcp.core.metrics.registry instead",
    DeprecationWarning,
    stacklevel=2,
)

from foundry_mcp.core.metrics.registry import *  # noqa: F401,F403,E402
from foundry_mcp.core.metrics.registry import __all__  # noqa: F401,E402
