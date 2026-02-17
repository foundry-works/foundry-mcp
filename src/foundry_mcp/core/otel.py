"""Backward-compatible shim — canonical location: foundry_mcp.core.observability.otel

.. deprecated::
    Import from ``foundry_mcp.core.observability.otel`` instead.
"""

import warnings as _warnings

_warnings.warn(
    "foundry_mcp.core.otel is deprecated, "
    "use foundry_mcp.core.observability.otel instead",
    DeprecationWarning,
    stacklevel=2,
)

from foundry_mcp.core.observability.otel import *  # noqa: F401,F403,E402
from foundry_mcp.core.observability.otel import __all__  # noqa: F401,E402
