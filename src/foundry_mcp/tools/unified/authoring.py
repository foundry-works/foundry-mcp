"""Unified authoring router — delegates to authoring_handlers/ package.

This module is a backward-compatible shim. All handler logic now lives in
``foundry_mcp.tools.unified.authoring_handlers``.

The dispatch function is defined here (not imported) so that tests can patch
``authoring._AUTHORING_ROUTER`` and have it take effect.
"""

from __future__ import annotations

from typing import Any, Dict

from foundry_mcp.config.server import ServerConfig
from foundry_mcp.tools.unified.authoring_handlers import (  # noqa: F401
    _AUTHORING_ROUTER,
    register_unified_authoring_tool,
)
from foundry_mcp.tools.unified.authoring_handlers._helpers import (  # noqa: F401
    _metric_name as _metric_name,
)
from foundry_mcp.tools.unified.common import (
    build_request_id,  # noqa: F401
    dispatch_with_standard_errors,
)


def _request_id() -> str:
    return build_request_id("authoring")


def _dispatch_authoring_action(*, action: str, payload: Dict[str, Any], config: ServerConfig) -> dict:
    return dispatch_with_standard_errors(_AUTHORING_ROUTER, "authoring", action, config=config, **payload)


__all__ = [
    "register_unified_authoring_tool",
]
