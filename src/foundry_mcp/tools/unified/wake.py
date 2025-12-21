"""Wake tool for minimal mode."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from foundry_mcp.config import ServerConfig


def register_wake_tool(mcp: "FastMCP", config: "ServerConfig") -> None:
    """Register the minimal wake tool."""

    @mcp.tool()
    def wake() -> dict:
        """
        SDD tools are in minimal mode to save context tokens.

        To enable the full SDD toolkit (17 tools):
        1. Use /sdd-on command, OR
        2. Call mcp__foundry-ctl__set_sdd_mode with mode="full"

        The server will restart automatically (~1-2 seconds).
        """
        return {
            "status": "minimal_mode",
            "message": "Run /sdd-on to enable full SDD tools",
            "hint": "Use mcp__foundry-ctl__set_sdd_mode mode='full' to enable",
        }
