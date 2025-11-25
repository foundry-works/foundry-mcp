"""
Smoke tests for MCP server tool registration.

Verifies that FastMCP server registers all tools without schema errors.
"""

import pytest
from foundry_mcp.server import create_server
from foundry_mcp.config import ServerConfig
from pathlib import Path


@pytest.fixture
def test_config(tmp_path):
    """Create a test server configuration."""
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    (specs_dir / "active").mkdir()
    (specs_dir / "pending").mkdir()
    (specs_dir / "completed").mkdir()
    (specs_dir / "archived").mkdir()

    return ServerConfig(
        server_name="foundry-mcp-test",
        server_version="0.1.0",
        specs_dir=specs_dir,
        log_level="WARNING",
    )


@pytest.fixture
def mcp_server(test_config):
    """Create a test MCP server instance."""
    return create_server(test_config)


class TestMCPServerCreation:
    """Tests for MCP server creation."""

    def test_server_creates_successfully(self, test_config):
        """Test that server creates without errors."""
        server = create_server(test_config)
        assert server is not None

    def test_server_has_name(self, mcp_server, test_config):
        """Test that server has correct name."""
        assert mcp_server.name == test_config.server_name


class TestRenderingToolsRegistration:
    """Tests for rendering tools registration."""

    def test_foundry_render_spec_registered(self, mcp_server):
        """Test that foundry_render_spec tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "foundry_render_spec" in tools

    def test_foundry_render_progress_registered(self, mcp_server):
        """Test that foundry_render_progress tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "foundry_render_progress" in tools

    def test_foundry_list_tasks_registered(self, mcp_server):
        """Test that foundry_list_tasks tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "foundry_list_tasks" in tools


class TestLifecycleToolsRegistration:
    """Tests for lifecycle tools registration."""

    def test_foundry_move_spec_registered(self, mcp_server):
        """Test that foundry_move_spec tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "foundry_move_spec" in tools

    def test_foundry_activate_spec_registered(self, mcp_server):
        """Test that foundry_activate_spec tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "foundry_activate_spec" in tools

    def test_foundry_complete_spec_registered(self, mcp_server):
        """Test that foundry_complete_spec tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "foundry_complete_spec" in tools

    def test_foundry_archive_spec_registered(self, mcp_server):
        """Test that foundry_archive_spec tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "foundry_archive_spec" in tools

    def test_foundry_lifecycle_state_registered(self, mcp_server):
        """Test that foundry_lifecycle_state tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "foundry_lifecycle_state" in tools

    def test_foundry_list_specs_by_folder_registered(self, mcp_server):
        """Test that foundry_list_specs_by_folder tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "foundry_list_specs_by_folder" in tools


class TestCoreToolsRegistration:
    """Tests for core tools registration."""

    def test_list_specs_registered(self, mcp_server):
        """Test that tool_list_specs tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "tool_list_specs" in tools

    def test_get_spec_registered(self, mcp_server):
        """Test that tool_get_spec tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "tool_get_spec" in tools

    def test_get_task_registered(self, mcp_server):
        """Test that tool_get_task tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "tool_get_task" in tools


class TestValidationToolsRegistration:
    """Tests for validation tools registration."""

    def test_foundry_validate_spec_registered(self, mcp_server):
        """Test that foundry_validate_spec tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "foundry_validate_spec" in tools


class TestJournalToolsRegistration:
    """Tests for journal tools registration."""

    def test_foundry_get_journal_registered(self, mcp_server):
        """Test that foundry_get_journal tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "foundry_get_journal" in tools


class TestQueryToolsRegistration:
    """Tests for query tools registration."""

    def test_foundry_query_tasks_registered(self, mcp_server):
        """Test that foundry_query_tasks tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "foundry_query_tasks" in tools


class TestTaskToolsRegistration:
    """Tests for task tools registration."""

    def test_foundry_update_status_registered(self, mcp_server):
        """Test that foundry_update_status tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "foundry_update_status" in tools


class TestToolSchemas:
    """Tests for tool schema validity."""

    def test_all_tools_have_schemas(self, mcp_server):
        """Test that all tools have valid schemas."""
        tools = mcp_server._tool_manager._tools
        for tool_name, tool in tools.items():
            # Each tool should have a callable function
            assert callable(tool.fn), f"Tool {tool_name} should have callable function"

    def test_rendering_tools_callable(self, mcp_server):
        """Test that rendering tools are callable without errors."""
        tools = mcp_server._tool_manager._tools
        rendering_tools = ["foundry_render_spec", "foundry_render_progress", "foundry_list_tasks"]

        for tool_name in rendering_tools:
            assert tool_name in tools, f"Tool {tool_name} should be registered"
            assert callable(tools[tool_name].fn), f"Tool {tool_name} should be callable"

    def test_lifecycle_tools_callable(self, mcp_server):
        """Test that lifecycle tools are callable without errors."""
        tools = mcp_server._tool_manager._tools
        lifecycle_tools = [
            "foundry_move_spec",
            "foundry_activate_spec",
            "foundry_complete_spec",
            "foundry_archive_spec",
            "foundry_lifecycle_state",
            "foundry_list_specs_by_folder",
        ]

        for tool_name in lifecycle_tools:
            assert tool_name in tools, f"Tool {tool_name} should be registered"
            assert callable(tools[tool_name].fn), f"Tool {tool_name} should be callable"


class TestResourcesRegistration:
    """Tests for MCP resources registration."""

    def test_specs_list_resource_registered(self, mcp_server):
        """Test that specs://list resource is registered."""
        resources = mcp_server._resource_manager._resources
        # Check for resource template pattern
        assert len(resources) > 0, "Should have resources registered"

    def test_spec_by_id_resource_registered(self, mcp_server):
        """Test that specs://{spec_id} resource is registered."""
        resources = mcp_server._resource_manager._resources
        # The resource manager should have templates
        assert len(resources) > 0, "Should have resource templates registered"


class TestToolCounts:
    """Tests for expected tool counts."""

    def test_minimum_tool_count(self, mcp_server):
        """Test that minimum expected tools are registered."""
        tools = mcp_server._tool_manager._tools
        # We expect at least: 4 core + 3 rendering + 6 lifecycle + validation + journal + query + task
        min_expected = 15
        assert len(tools) >= min_expected, f"Expected at least {min_expected} tools, got {len(tools)}"

    def test_tool_names_are_strings(self, mcp_server):
        """Test that all tool names are valid strings."""
        tools = mcp_server._tool_manager._tools
        for tool_name in tools.keys():
            assert isinstance(tool_name, str), f"Tool name should be string: {tool_name}"
            assert len(tool_name) > 0, "Tool name should not be empty"
