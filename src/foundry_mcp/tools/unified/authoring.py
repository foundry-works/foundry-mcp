"""Unified authoring router — delegates to authoring_handlers/ package.

This module is a backward-compatible shim. All handler logic now lives in
``foundry_mcp.tools.unified.authoring_handlers``.
"""

from foundry_mcp.tools.unified.authoring_handlers import (  # noqa: F401
    _AUTHORING_ROUTER,
    _dispatch_authoring_action,
    register_unified_authoring_tool,
)

__all__ = [
    "register_unified_authoring_tool",
]
