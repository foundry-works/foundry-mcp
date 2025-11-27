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

    def test_spec_render_registered(self, mcp_server):
        """Test that spec_render tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "spec-render" in tools

    def test_spec_render_progress_registered(self, mcp_server):
        """Test that spec_render_progress tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "spec-render-progress" in tools

    def test_task_list_registered(self, mcp_server):
        """Test that task_list tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "task-list" in tools


class TestLifecycleToolsRegistration:
    """Tests for lifecycle tools registration."""

    def test_spec_lifecycle_move_registered(self, mcp_server):
        """Test that spec_lifecycle_move tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "spec-lifecycle-move" in tools

    def test_spec_lifecycle_activate_registered(self, mcp_server):
        """Test that spec_lifecycle_activate tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "spec-lifecycle-activate" in tools

    def test_spec_lifecycle_complete_registered(self, mcp_server):
        """Test that spec_lifecycle_complete tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "spec-lifecycle-complete" in tools

    def test_spec_lifecycle_archive_registered(self, mcp_server):
        """Test that spec_lifecycle_archive tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "spec-lifecycle-archive" in tools

    def test_spec_lifecycle_state_registered(self, mcp_server):
        """Test that spec_lifecycle_state tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "spec-lifecycle-state" in tools

    def test_spec_list_by_folder_registered(self, mcp_server):
        """Test that spec_list_by_folder tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "spec-list-by-folder" in tools


class TestCoreToolsRegistration:
    """Tests for core tools registration."""

    def test_spec_list_basic_registered(self, mcp_server):
        """Test that spec-list-basic tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "spec-list-basic" in tools

    def test_spec_get_registered(self, mcp_server):
        """Test that spec-get tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "spec-get" in tools

    def test_task_get_registered(self, mcp_server):
        """Test that task-get tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "task-get" in tools


class TestValidationToolsRegistration:
    """Tests for validation tools registration."""

    def test_spec_validate_registered(self, mcp_server):
        """Test that spec_validate tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "spec-validate" in tools


class TestJournalToolsRegistration:
    """Tests for journal tools registration."""

    def test_journal_list_registered(self, mcp_server):
        """Test that journal_list tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "journal-list" in tools


class TestQueryToolsRegistration:
    """Tests for query tools registration."""

    def test_task_query_registered(self, mcp_server):
        """Test that task_query tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "task-query" in tools


class TestTaskToolsRegistration:
    """Tests for task tools registration."""

    def test_task_update_status_registered(self, mcp_server):
        """Test that task_update_status tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "task-update-status" in tools


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
        rendering_tools = [
            "spec-render",
            "spec-render-progress",
            "task-list",
        ]

        for tool_name in rendering_tools:
            assert tool_name in tools, f"Tool {tool_name} should be registered"
            assert callable(tools[tool_name].fn), f"Tool {tool_name} should be callable"

    def test_lifecycle_tools_callable(self, mcp_server):
        """Test that lifecycle tools are callable without errors."""
        tools = mcp_server._tool_manager._tools
        lifecycle_tools = [
            "spec-lifecycle-move",
            "spec-lifecycle-activate",
            "spec-lifecycle-complete",
            "spec-lifecycle-archive",
            "spec-lifecycle-state",
            "spec-list-by-folder",
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
        assert any("specs://" in str(resource) for resource in resources)


class TestCanonicalToolNames:
    """Ensure canonical tool names are registered alongside legacy aliases."""

    def test_canonical_tools_registered(self, mcp_server):
        tools = mcp_server._tool_manager._tools
        expected = {
            "sdd-server-capabilities",
            "spec-list-basic",
            "spec-get",
            "spec-get-hierarchy",
            "task-get",
            "spec-render",
            "spec-render-progress",
            "task-list",
            "tool-list",
            "tool-get-schema",
            "capability-get",
            "capability-negotiate",
            "tool-list-categories",
            "spec-find",
            "spec-list",
            "task-query",
            "spec-validate",
            "spec-fix",
            "spec-stats",
            "spec-validate-fix",
            "journal-add",
            "journal-list",
            "task-block",
            "task-unblock",
            "task-list-blocked",
            "journal-list-unjournaled",
            "spec-lifecycle-move",
            "spec-lifecycle-activate",
            "spec-lifecycle-complete",
            "spec-lifecycle-archive",
            "spec-lifecycle-state",
            "spec-list-by-folder",
            "task-prepare",
            "task-next",
            "task-info",
            "task-check-deps",
            "task-update-status",
            "task-complete",
            "task-start",
            "task-progress",
            "code-find-class",
            "code-find-function",
            "code-trace-calls",
            "code-impact-analysis",
            "code-get-callers",
            "code-get-callees",
            "doc-stats",
            "test-run",
            "test-discover",
            "test-presets",
            "test-run-quick",
            "test-run-unit",
        }
        missing = sorted(name for name in expected if name not in tools)
        assert not missing, f"Missing canonical tools: {missing}"

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
        assert len(tools) >= min_expected, (
            f"Expected at least {min_expected} tools, got {len(tools)}"
        )

    def test_tool_names_are_strings(self, mcp_server):
        """Test that all tool names are valid strings."""
        tools = mcp_server._tool_manager._tools
        for tool_name in tools.keys():
            assert isinstance(tool_name, str), (
                f"Tool name should be string: {tool_name}"
            )
            assert len(tool_name) > 0, "Tool name should not be empty"
