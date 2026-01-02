"""Tests for disabled_tools configuration."""

import os
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.config import ServerConfig


class TestDisabledToolsConfig:
    """Test disabled_tools configuration field."""

    def test_default_empty(self):
        """Default disabled_tools is empty list."""
        config = ServerConfig()
        assert config.disabled_tools == []

    def test_env_var_single_tool(self):
        """FOUNDRY_MCP_DISABLED_TOOLS with single tool."""
        with patch.dict(os.environ, {"FOUNDRY_MCP_DISABLED_TOOLS": "health"}):
            config = ServerConfig.from_env()
            assert config.disabled_tools == ["health"]

    def test_env_var_multiple_tools(self):
        """FOUNDRY_MCP_DISABLED_TOOLS with multiple tools."""
        with patch.dict(
            os.environ, {"FOUNDRY_MCP_DISABLED_TOOLS": "error,health,metrics"}
        ):
            config = ServerConfig.from_env()
            assert set(config.disabled_tools) == {"error", "health", "metrics"}

    def test_env_var_whitespace_handling(self):
        """Whitespace in FOUNDRY_MCP_DISABLED_TOOLS is trimmed."""
        with patch.dict(
            os.environ, {"FOUNDRY_MCP_DISABLED_TOOLS": " error , health , metrics "}
        ):
            config = ServerConfig.from_env()
            assert set(config.disabled_tools) == {"error", "health", "metrics"}

    def test_env_var_empty_entries_filtered(self):
        """Empty entries from trailing commas are filtered."""
        with patch.dict(
            os.environ, {"FOUNDRY_MCP_DISABLED_TOOLS": "health,,error,"}
        ):
            config = ServerConfig.from_env()
            assert set(config.disabled_tools) == {"health", "error"}


class TestToolRegistrationWithDisabled:
    """Test that disabled tools are not registered."""

    def test_disabled_tools_not_registered(self):
        """Tools in disabled_tools are not registered."""
        from foundry_mcp.tools.unified import register_unified_tools

        config = ServerConfig()
        config.disabled_tools = ["health", "error", "metrics"]

        mcp = MagicMock()

        register_unified_tools(mcp, config)

        # Get all tool names registered via mcp.tool decorator
        registered_tools = set()
        for call in mcp.tool.call_args_list:
            if call.kwargs.get("name"):
                registered_tools.add(call.kwargs["name"])

        # Verify disabled tools were not registered
        assert "health" not in registered_tools
        assert "error" not in registered_tools
        assert "metrics" not in registered_tools

    def test_enabled_tools_still_registered(self):
        """Tools not in disabled_tools are still registered."""
        from foundry_mcp.tools.unified import register_unified_tools

        config = ServerConfig()
        config.disabled_tools = ["health", "error", "metrics"]

        mcp = MagicMock()

        register_unified_tools(mcp, config)

        # Get all tool names registered
        registered_tools = set()
        for call in mcp.tool.call_args_list:
            if call.kwargs.get("name"):
                registered_tools.add(call.kwargs["name"])

        # Verify some enabled tools are registered
        # (note: actual tool names depend on canonical_tool decorator)
        assert len(registered_tools) > 0  # At least some tools registered
