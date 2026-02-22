"""Backward-compatible shim — canonical location: foundry_mcp.core.metrics.persistence

.. deprecated::
    Import from ``foundry_mcp.core.metrics.persistence`` instead.
"""

import warnings as _warnings

_warnings.warn(
    "foundry_mcp.core.metrics_persistence is deprecated, use foundry_mcp.core.metrics.persistence instead",
    DeprecationWarning,
    stacklevel=2,
)

from foundry_mcp.core.metrics.persistence import *  # noqa: F401,F403,E402
from foundry_mcp.core.metrics.persistence import __all__  # noqa: F401,E402
