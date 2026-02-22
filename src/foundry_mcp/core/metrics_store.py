"""Backward-compatible shim — canonical location: foundry_mcp.core.metrics.store

.. deprecated::
    Import from ``foundry_mcp.core.metrics.store`` instead.
"""

import warnings as _warnings

_warnings.warn(
    "foundry_mcp.core.metrics_store is deprecated, use foundry_mcp.core.metrics.store instead",
    DeprecationWarning,
    stacklevel=2,
)

from foundry_mcp.core.metrics.store import *  # noqa: F401,F403,E402
