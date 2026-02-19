"""
Integration tests for JSON minification in tool responses.

Verifies that canonical_tool decorator produces TextContent with minified JSON.
"""

from mcp.types import TextContent


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
            f"Minified ({len(minified)} chars) should be smaller than pretty-printed ({len(pretty)} chars)"
        )
