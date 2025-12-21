"""Helper MCP server with mode control tool."""

from __future__ import annotations

from typing import Literal

from mcp.server.fastmcp import FastMCP

from .config import get_mode, set_mode, signal_restart

Mode = Literal["minimal", "full"]

# Default server name to signal
DEFAULT_SERVER_NAME = "foundry-mcp"


def create_helper_server() -> FastMCP:
    """Create the helper MCP server with set_mode tool."""
    mcp = FastMCP(name="foundry-ctl")

    @mcp.tool()
    def set_sdd_mode(
        mode: Mode,
        server_name: str = DEFAULT_SERVER_NAME,
    ) -> dict:
        """
        Set the SDD tools mode and trigger restart.

        Args:
            mode: "minimal" for 1 wake tool, "full" for all 17 routers
            server_name: Name of the foundry-mcp server to restart

        Returns:
            Status dict with mode and message
        """
        previous_mode = get_mode()

        if mode == previous_mode:
            return {
                "status": "unchanged",
                "mode": mode,
                "message": f"Already in {mode} mode",
            }

        set_mode(mode)
        signal_restart(server_name)

        if mode == "full":
            return {
                "status": "restarting",
                "mode": mode,
                "message": "Enabling full SDD tools. Server restarting (~1-2s)...",
                "tools_available": 17,
            }
        else:
            return {
                "status": "restarting",
                "mode": mode,
                "message": "Disabling SDD tools. Server restarting (~1-2s)...",
                "tokens_saved": "~3,300",
            }

    @mcp.tool()
    def get_sdd_mode() -> dict:
        """
        Get the current SDD tools mode.

        Returns:
            Status dict with current mode
        """
        mode = get_mode()
        return {
            "mode": mode,
            "tools_available": 17 if mode == "full" else 1,
        }

    return mcp


def run_helper() -> None:
    """Run the helper MCP server."""
    server = create_helper_server()
    server.run()
