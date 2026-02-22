"""Smoke tests for MCP server tool registration.

Verifies that the server registers the unified 14-router tool surface.
"""

from __future__ import annotations

from foundry_mcp.config.server import ServerConfig
from foundry_mcp.server import create_server

_UNIFIED_TOOL_NAMES = {
    "health",
    "plan",
    "error",
    "journal",
    "authoring",
    "provider",
    "environment",
    "lifecycle",
    "verification",
    "task",
    "spec",
    "review",
    "server",
    "research",
}


def test_server_creates_successfully(test_config: ServerConfig):
    server = create_server(test_config)
    assert server is not None


def test_server_name_matches_config(mcp_server, test_config: ServerConfig):
    assert mcp_server.name == test_config.server_name


def test_unified_tools_registered(mcp_server):
    tools = mcp_server._tool_manager._tools
    assert set(tools.keys()) == _UNIFIED_TOOL_NAMES


def test_all_tools_callable(mcp_server):
    tools = mcp_server._tool_manager._tools
    for tool_name, tool in tools.items():
        assert callable(tool.fn), f"Tool {tool_name} should be callable"
