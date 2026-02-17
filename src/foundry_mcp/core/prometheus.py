"""Backward-compatible shim — canonical location: foundry_mcp.core.observability.prometheus

.. deprecated::
    Import from ``foundry_mcp.core.observability.prometheus`` instead.
"""

import warnings as _warnings

_warnings.warn(
    "foundry_mcp.core.prometheus is deprecated, "
    "use foundry_mcp.core.observability.prometheus instead",
    DeprecationWarning,
    stacklevel=2,
)

from foundry_mcp.core.observability.prometheus import *  # noqa: F401,F403,E402
from foundry_mcp.core.observability.prometheus import __all__  # noqa: F401,E402
