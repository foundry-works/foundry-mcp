"""Schema regression test for the unified authoring tool.

Ensures critical parameters (e.g. plan_path, plan_review_path) appear in the
MCP-generated JSON Schema.  This catches regressions where required params
accidentally disappear from the tool signature — which would silently drop
them at the MCP protocol layer even though the underlying code is correct.
"""

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config.server import ServerConfig
from foundry_mcp.tools.unified.authoring_handlers import register_unified_authoring_tool


class TestAuthoringToolSchema:
    """Verify the authoring tool exposes expected parameters in its MCP schema."""

    def test_schema_includes_spec_create_params(self):
        """plan_path and plan_review_path must appear in the MCP schema."""
        mcp = FastMCP("test")
        config = ServerConfig()
        register_unified_authoring_tool(mcp, config)

        tool = mcp._tool_manager._tools["authoring"]
        props = tool.parameters.get("properties", {})

        assert "plan_path" in props, "plan_path missing from authoring tool schema"
        assert "plan_review_path" in props, "plan_review_path missing from authoring tool schema"

    def test_schema_includes_core_params(self):
        """Core parameters required by most authoring actions must be present."""
        mcp = FastMCP("test")
        config = ServerConfig()
        register_unified_authoring_tool(mcp, config)

        tool = mcp._tool_manager._tools["authoring"]
        props = tool.parameters.get("properties", {})

        for param in ("action", "spec_id", "name", "template", "dry_run", "path"):
            assert param in props, f"{param} missing from authoring tool schema"
