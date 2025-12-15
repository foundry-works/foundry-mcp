"""MCP tool registration surface.

Only the unified 17-router tool surface is exported.
"""

from foundry_mcp.tools.unified import register_unified_tools

__all__ = [
    "register_unified_tools",
]
