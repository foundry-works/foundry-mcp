"""Foundry MCP - MCP server for SDD toolkit spec management."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("foundry-mcp")
except PackageNotFoundError:
    # Package not installed (development mode without editable install)
    __version__ = "0.6.0"

from foundry_mcp.server import create_server, main

__all__ = ["__version__", "create_server", "main"]
