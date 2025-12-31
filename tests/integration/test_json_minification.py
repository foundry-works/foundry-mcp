"""
Integration tests for JSON minification in tool responses.

Verifies that canonical_tool decorator produces TextContent with minified JSON.
"""

import pytest
from mcp.types import TextContent

from foundry_mcp.config import ServerConfig
from foundry_mcp.server import create_server


@pytest.fixture
def test_specs_dir(tmp_path):
    """Create minimal test specs directory."""
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    for d in ["active", "pending", "completed", "archived"]:
        (specs_dir / d).mkdir()
    return specs_dir


@pytest.fixture
def test_config(test_specs_dir):
    """Create test server configuration."""
    return ServerConfig(
        server_name="foundry-mcp-minify-test",
        server_version="0.1.0",
        specs_dir=test_specs_dir,
        log_level="WARNING",
    )


@pytest.fixture
def mcp_server(test_config):
    """Create test MCP server instance."""
    return create_server(test_config)


class TestJsonMinification:
    """Verify tool responses are minified TextContent."""

    def test_tool_returns_textcontent(self, mcp_server):
        """Tool response should be TextContent, not dict."""
        tools = mcp_server._tool_manager._tools
        result = tools["spec"].fn(action="list", status="all")
        assert isinstance(result, TextContent), f"Expected TextContent, got {type(result).__name__}"

    def test_response_has_no_newlines(self, mcp_server):
        """Minified JSON should have no newlines."""
        tools = mcp_server._tool_manager._tools
        result = tools["spec"].fn(action="list", status="all")
        assert "\n" not in result.text, "Minified JSON should not contain newlines"

    def test_response_has_no_indentation(self, mcp_server):
        """Minified JSON should have no indentation (double spaces)."""
        tools = mcp_server._tool_manager._tools
        result = tools["spec"].fn(action="list", status="all")
        assert "  " not in result.text, "Minified JSON should not contain indentation"

    def test_response_is_valid_json(self, mcp_server):
        """Response text should be valid parseable JSON."""
        import json
        tools = mcp_server._tool_manager._tools
        result = tools["spec"].fn(action="list", status="all")
        parsed = json.loads(result.text)
        assert isinstance(parsed, dict), "Parsed JSON should be a dict"
        assert "success" in parsed, "Response should have 'success' key"

    def test_minified_vs_pretty_size_difference(self, mcp_server):
        """Minified JSON should be smaller than pretty-printed."""
        import json
        tools = mcp_server._tool_manager._tools
        result = tools["spec"].fn(action="list", status="all")

        parsed = json.loads(result.text)
        pretty = json.dumps(parsed, indent=2)
        minified = result.text

        assert len(minified) < len(pretty), (
            f"Minified ({len(minified)} chars) should be smaller than "
            f"pretty-printed ({len(pretty)} chars)"
        )
