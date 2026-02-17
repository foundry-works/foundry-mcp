"""Backward-compatible shim — canonical location: foundry_mcp.core.observability.stubs

.. deprecated::
    Import from ``foundry_mcp.core.observability.stubs`` instead.
"""

import warnings as _warnings

_warnings.warn(
    "foundry_mcp.core.otel_stubs is deprecated, "
    "use foundry_mcp.core.observability.stubs instead",
    DeprecationWarning,
    stacklevel=2,
)

from foundry_mcp.core.observability.stubs import *  # noqa: F401,F403,E402
from foundry_mcp.core.observability.stubs import __all__  # noqa: F401,E402
