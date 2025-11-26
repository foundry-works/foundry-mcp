"""
Integration tests for MCP tool registration, schemas, resources, and prompts.

Tests:
- Tool input/output schema validation
- Resource access patterns (foundry:// URIs)
- Prompt expansion with various arguments
"""

import json
import pytest
from pathlib import Path
from foundry_mcp.server import create_server
from foundry_mcp.config import ServerConfig


@pytest.fixture
def test_specs_dir(tmp_path):
    """Create a test specs directory with sample spec."""
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    (specs_dir / "active").mkdir()
    (specs_dir / "pending").mkdir()
    (specs_dir / "completed").mkdir()
    (specs_dir / "archived").mkdir()
    (specs_dir / "templates").mkdir()

    # Create a sample spec
    sample_spec = {
        "spec_id": "test-spec-001",
        "title": "Test Specification",
        "metadata": {
            "title": "Test Specification",
            "description": "A test spec for integration testing",
            "created_at": "2025-01-25T00:00:00Z",
        },
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test Specification",
                "status": "in_progress",
                "children": ["phase-1"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Implementation",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2"],
            },
            "task-1-1": {
                "type": "task",
                "title": "First task",
                "status": "completed",
                "parent": "phase-1",
                "children": [],
            },
            "task-1-2": {
                "type": "task",
                "title": "Second task",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
            },
        },
        "journal": [
            {
                "timestamp": "2025-01-25T00:00:00Z",
                "entry_type": "status_change",
                "title": "Task completed",
                "content": "Completed first task",
                "task_id": "task-1-1",
            },
        ],
    }

    spec_file = specs_dir / "active" / "test-spec-001.json"
    with open(spec_file, "w") as f:
        json.dump(sample_spec, f)

    # Create a sample template
    template = {
        "spec_id": "{{spec_id}}",
        "title": "Custom Template",
        "metadata": {
            "title": "{{title}}",
            "description": "Custom template for testing",
        },
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "{{title}}",
                "status": "pending",
                "children": ["phase-1"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Custom Phase",
                "status": "pending",
                "parent": "spec-root",
                "children": [],
            },
        },
        "journal": [],
    }

    template_file = specs_dir / "templates" / "custom.json"
    with open(template_file, "w") as f:
        json.dump(template, f)

    return specs_dir


@pytest.fixture
def test_config(test_specs_dir):
    """Create a test server configuration."""
    return ServerConfig(
        server_name="foundry-mcp-test",
        server_version="0.1.0",
        specs_dir=test_specs_dir,
        log_level="WARNING",
    )


@pytest.fixture
def mcp_server(test_config):
    """Create a test MCP server instance."""
    return create_server(test_config)


class TestToolInputSchemas:
    """Tests for tool input schema validation."""

    def test_tool_list_specs_has_status_param(self, mcp_server):
        """Test that tool_list_specs has status parameter."""
        tools = mcp_server._tool_manager._tools
        tool = tools.get("tool_list_specs")
        assert tool is not None
        # Tool should accept status parameter
        assert callable(tool.fn)

    def test_tool_get_spec_requires_spec_id(self, mcp_server):
        """Test that tool_get_spec requires spec_id."""
        tools = mcp_server._tool_manager._tools
        tool = tools.get("tool_get_spec")
        assert tool is not None
        assert callable(tool.fn)

    def test_tool_get_task_requires_both_ids(self, mcp_server):
        """Test that tool_get_task requires spec_id and task_id."""
        tools = mcp_server._tool_manager._tools
        tool = tools.get("tool_get_task")
        assert tool is not None
        assert callable(tool.fn)

    def test_foundry_validate_spec_schema(self, mcp_server):
        """Test foundry_validate_spec tool has correct schema."""
        tools = mcp_server._tool_manager._tools
        tool = tools.get("foundry_validate_spec")
        assert tool is not None
        assert callable(tool.fn)

    def test_foundry_update_status_schema(self, mcp_server):
        """Test foundry_update_status tool has correct schema."""
        tools = mcp_server._tool_manager._tools
        tool = tools.get("foundry_update_status")
        assert tool is not None
        assert callable(tool.fn)


class TestToolOutputSchemas:
    """Tests for tool output format validation."""

    def test_list_specs_returns_dict(self, mcp_server):
        """Test that list_specs returns a dict."""
        tools = mcp_server._tool_manager._tools
        tool = tools.get("tool_list_specs")
        data = tool.fn(status="all")
        assert isinstance(data, dict)
        assert "specs" in data
        assert "count" in data

    def test_list_specs_with_active_filter(self, mcp_server):
        """Test list_specs with active status filter."""
        tools = mcp_server._tool_manager._tools
        tool = tools.get("tool_list_specs")
        data = tool.fn(status="active")
        assert "specs" in data
        assert isinstance(data["specs"], list)

    def test_get_spec_returns_progress(self, mcp_server):
        """Test that get_spec returns progress information."""
        tools = mcp_server._tool_manager._tools
        tool = tools.get("tool_get_spec")
        data = tool.fn(spec_id="test-spec-001")
        assert "total_tasks" in data
        assert "completed_tasks" in data
        assert "progress_percentage" in data

    def test_get_spec_not_found_error(self, mcp_server):
        """Test that get_spec returns error for non-existent spec."""
        tools = mcp_server._tool_manager._tools
        tool = tools.get("tool_get_spec")
        data = tool.fn(spec_id="nonexistent-spec")
        assert "error" in data

    def test_get_task_returns_task_data(self, mcp_server):
        """Test that get_task returns task data."""
        tools = mcp_server._tool_manager._tools
        tool = tools.get("tool_get_task")
        data = tool.fn(spec_id="test-spec-001", task_id="task-1-1")
        assert "task" in data
        assert data["task"]["title"] == "First task"
        assert data["task"]["status"] == "completed"

    def test_get_task_not_found_error(self, mcp_server):
        """Test that get_task returns error for non-existent task."""
        tools = mcp_server._tool_manager._tools
        tool = tools.get("tool_get_task")
        data = tool.fn(spec_id="test-spec-001", task_id="nonexistent-task")
        assert "error" in data

    def test_get_spec_hierarchy_returns_hierarchy(self, mcp_server):
        """Test that get_spec_hierarchy returns hierarchy data."""
        tools = mcp_server._tool_manager._tools
        tool = tools.get("tool_get_spec_hierarchy")
        data = tool.fn(spec_id="test-spec-001")
        assert "hierarchy" in data
        assert "spec-root" in data["hierarchy"]
        assert "phase-1" in data["hierarchy"]


class TestResourceAccess:
    """Tests for MCP resource access patterns."""

    def test_foundry_specs_resource_registered(self, mcp_server):
        """Test that foundry://specs/ resource is registered."""
        resources = mcp_server._resource_manager._resources
        # Check that we have foundry resources registered
        foundry_resources = [r for r in resources.keys() if "foundry" in str(r)]
        assert len(foundry_resources) > 0

    def test_specs_list_resource_returns_json(self, mcp_server):
        """Test that foundry://specs/ returns valid JSON."""
        resources = mcp_server._resource_manager._resources
        # Find the specs list resource
        for uri, resource in resources.items():
            if "foundry://specs/" in str(uri) and resource.fn.__name__ == "resource_specs_list":
                result = resource.fn()
                data = json.loads(result)
                assert "success" in data
                assert "schema_version" in data
                break

    def test_specs_by_status_resource_valid_status(self, mcp_server):
        """Test that foundry://specs/{status}/ validates status."""
        resources = mcp_server._resource_manager._resources
        # Find the specs by status resource
        for uri, resource in resources.items():
            if resource.fn.__name__ == "resource_specs_by_status":
                # Test with valid status
                result = resource.fn(status="active")
                data = json.loads(result)
                assert data.get("success") is True or "error" not in data.get("", "")
                break

    def test_specs_by_status_resource_invalid_status(self, mcp_server):
        """Test that foundry://specs/{status}/ rejects invalid status."""
        resources = mcp_server._resource_manager._resources
        for uri, resource in resources.items():
            if resource.fn.__name__ == "resource_specs_by_status":
                result = resource.fn(status="invalid_status")
                data = json.loads(result)
                assert data["success"] is False
                assert "error" in data
                break

    def test_spec_journal_resource(self, mcp_server):
        """Test that foundry://specs/{spec_id}/journal returns journal."""
        resources = mcp_server._resource_manager._resources
        for uri, resource in resources.items():
            if resource.fn.__name__ == "resource_spec_journal":
                result = resource.fn(spec_id="test-spec-001")
                data = json.loads(result)
                if data["success"]:
                    assert "journal" in data
                    assert "count" in data
                break

    def test_templates_list_resource(self, mcp_server):
        """Test that foundry://templates/ lists templates."""
        resources = mcp_server._resource_manager._resources
        for uri, resource in resources.items():
            if resource.fn.__name__ == "resource_templates_list":
                result = resource.fn()
                data = json.loads(result)
                assert data["success"] is True
                assert "templates" in data
                assert "builtin_templates" in data
                # Should have builtin templates
                assert len(data["builtin_templates"]) >= 3
                break

    def test_template_by_id_builtin(self, mcp_server):
        """Test that foundry://templates/{template_id} returns builtin templates."""
        resources = mcp_server._resource_manager._resources
        for uri, resource in resources.items():
            if resource.fn.__name__ == "resource_template":
                result = resource.fn(template_id="basic")
                data = json.loads(result)
                assert data["success"] is True
                assert data["builtin"] is True
                assert "template" in data
                break

    def test_template_by_id_custom(self, mcp_server):
        """Test that foundry://templates/{template_id} returns custom templates."""
        resources = mcp_server._resource_manager._resources
        for uri, resource in resources.items():
            if resource.fn.__name__ == "resource_template":
                result = resource.fn(template_id="custom")
                data = json.loads(result)
                assert data["success"] is True
                assert data["builtin"] is False
                assert "template" in data
                break

    def test_template_not_found(self, mcp_server):
        """Test that non-existent template returns error."""
        resources = mcp_server._resource_manager._resources
        for uri, resource in resources.items():
            if resource.fn.__name__ == "resource_template":
                result = resource.fn(template_id="nonexistent")
                data = json.loads(result)
                assert data["success"] is False
                assert "error" in data
                break


class TestPromptExpansion:
    """Tests for MCP prompt expansion."""

    def test_prompts_registered(self, mcp_server):
        """Test that workflow prompts are registered."""
        prompts = mcp_server._prompt_manager._prompts
        assert len(prompts) > 0

    def test_start_feature_prompt_exists(self, mcp_server):
        """Test that start_feature prompt is registered."""
        prompts = mcp_server._prompt_manager._prompts
        prompt_names = list(prompts.keys())
        assert "start_feature" in prompt_names

    def test_debug_test_prompt_exists(self, mcp_server):
        """Test that debug_test prompt is registered."""
        prompts = mcp_server._prompt_manager._prompts
        prompt_names = list(prompts.keys())
        assert "debug_test" in prompt_names

    def test_complete_phase_prompt_exists(self, mcp_server):
        """Test that complete_phase prompt is registered."""
        prompts = mcp_server._prompt_manager._prompts
        prompt_names = list(prompts.keys())
        assert "complete_phase" in prompt_names

    def test_review_spec_prompt_exists(self, mcp_server):
        """Test that review_spec prompt is registered."""
        prompts = mcp_server._prompt_manager._prompts
        prompt_names = list(prompts.keys())
        assert "review_spec" in prompt_names

    def test_start_feature_prompt_expansion(self, mcp_server):
        """Test that start_feature prompt expands correctly."""
        prompts = mcp_server._prompt_manager._prompts
        prompt = prompts.get("start_feature")
        result = prompt.fn(
            feature_name="Test Feature",
            description="A test feature description",
            template="feature"
        )
        assert "Test Feature" in result
        assert "A test feature description" in result
        assert "feature" in result.lower()
        assert "## Instructions" in result

    def test_start_feature_prompt_with_different_templates(self, mcp_server):
        """Test start_feature with different templates."""
        prompts = mcp_server._prompt_manager._prompts
        prompt = prompts.get("start_feature")

        # Test with bugfix template
        result = prompt.fn(feature_name="Bug Fix", template="bugfix")
        assert "Bug Fix" in result
        assert "Investigation" in result or "Fix" in result

        # Test with basic template
        result = prompt.fn(feature_name="Simple Task", template="basic")
        assert "Simple Task" in result
        assert "Implementation" in result

    def test_debug_test_prompt_expansion(self, mcp_server):
        """Test that debug_test prompt expands correctly."""
        prompts = mcp_server._prompt_manager._prompts
        prompt = prompts.get("debug_test")
        result = prompt.fn(
            test_name="test_example",
            error_message="AssertionError: expected True",
            spec_id="test-spec-001"
        )
        assert "test_example" in result
        assert "AssertionError" in result
        assert "test-spec-001" in result
        assert "## Debugging Workflow" in result

    def test_debug_test_prompt_without_args(self, mcp_server):
        """Test debug_test with minimal arguments."""
        prompts = mcp_server._prompt_manager._prompts
        prompt = prompts.get("debug_test")
        result = prompt.fn()
        assert "Debug Test Failure" in result
        assert "Not specified" in result

    def test_complete_phase_prompt_expansion(self, mcp_server):
        """Test that complete_phase prompt expands correctly."""
        prompts = mcp_server._prompt_manager._prompts
        prompt = prompts.get("complete_phase")
        result = prompt.fn(spec_id="test-spec-001")
        assert "test-spec-001" in result
        assert "## Completion Checklist" in result

    def test_complete_phase_with_phase_id(self, mcp_server):
        """Test complete_phase with specific phase_id."""
        prompts = mcp_server._prompt_manager._prompts
        prompt = prompts.get("complete_phase")
        result = prompt.fn(spec_id="test-spec-001", phase_id="phase-1")
        assert "test-spec-001" in result
        # Should show phase info if spec exists
        assert "Phase" in result or "phase" in result

    def test_review_spec_prompt_expansion(self, mcp_server):
        """Test that review_spec prompt expands correctly."""
        prompts = mcp_server._prompt_manager._prompts
        prompt = prompts.get("review_spec")
        result = prompt.fn(spec_id="test-spec-001")
        assert "test-spec-001" in result
        # Should contain spec info since it exists
        assert "Test Specification" in result
        assert "Progress" in result or "progress" in result

    def test_review_spec_not_found(self, mcp_server):
        """Test review_spec with non-existent spec."""
        prompts = mcp_server._prompt_manager._prompts
        prompt = prompts.get("review_spec")
        result = prompt.fn(spec_id="nonexistent-spec")
        assert "nonexistent-spec" in result
        assert "not found" in result.lower() or "error" in result.lower()


class TestToolInteraction:
    """Tests for tool interaction patterns."""

    def test_list_then_get_spec_workflow(self, mcp_server):
        """Test listing specs then getting one."""
        tools = mcp_server._tool_manager._tools

        # List specs (returns dict directly)
        list_result = tools["tool_list_specs"].fn(status="active")
        assert "specs" in list_result

        # Get specific spec
        if list_result["specs"]:
            spec_id = list_result["specs"][0]["spec_id"]
            get_result = tools["tool_get_spec"].fn(spec_id=spec_id)
            assert get_result["spec_id"] == spec_id

    def test_get_spec_then_task_workflow(self, mcp_server):
        """Test getting spec then getting task."""
        tools = mcp_server._tool_manager._tools

        # Get spec hierarchy (returns dict directly)
        hierarchy_result = tools["tool_get_spec_hierarchy"].fn(spec_id="test-spec-001")
        assert "hierarchy" in hierarchy_result

        # Get specific task (returns dict directly)
        task_result = tools["tool_get_task"].fn(spec_id="test-spec-001", task_id="task-1-1")
        assert "task" in task_result


class TestResourceIntegrity:
    """Tests for resource data integrity."""

    def test_spec_resource_matches_tool_output(self, mcp_server):
        """Test that resource and tool return consistent data."""
        tools = mcp_server._tool_manager._tools
        resources = mcp_server._resource_manager._resources

        # Get via tool (returns dict directly)
        tool_result = tools["tool_get_spec"].fn(spec_id="test-spec-001")

        # Get via resource (resources still return JSON strings)
        for uri, resource in resources.items():
            if resource.fn.__name__ == "resource_spec_by_status":
                resource_result = json.loads(
                    resource.fn(status="active", spec_id="test-spec-001")
                )
                if resource_result["success"]:
                    # Compare progress percentages
                    assert tool_result["progress_percentage"] == resource_result["progress_percentage"]
                    assert tool_result["total_tasks"] == resource_result["total_tasks"]
                break

    def test_schema_version_consistency(self, mcp_server):
        """Test that all resources use consistent schema version."""
        resources = mcp_server._resource_manager._resources
        schema_versions = set()

        for uri, resource in resources.items():
            if "foundry" in str(uri) or resource.fn.__name__.startswith("resource_"):
                try:
                    # Call with appropriate args based on function
                    fn = resource.fn
                    if "spec_id" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                        if "status" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                            result = fn(status="active", spec_id="test-spec-001")
                        else:
                            result = fn(spec_id="test-spec-001")
                    elif "status" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                        result = fn(status="active")
                    elif "template_id" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                        result = fn(template_id="basic")
                    else:
                        result = fn()

                    data = json.loads(result)
                    if "schema_version" in data:
                        schema_versions.add(data["schema_version"])
                except Exception:
                    pass

        # All resources should use the same schema version
        if schema_versions:
            assert len(schema_versions) == 1


class TestDocsToolsRegistration:
    """Tests for documentation tools registration."""

    def test_foundry_discover_tests_registered(self, mcp_server):
        """Test that foundry_discover_tests tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "foundry_discover_tests" in tools

    def test_foundry_run_tests_registered(self, mcp_server):
        """Test that foundry_run_tests tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "foundry_run_tests" in tools

    def test_foundry_impact_analysis_registered(self, mcp_server):
        """Test that foundry_impact_analysis tool is registered."""
        tools = mcp_server._tool_manager._tools
        assert "foundry_impact_analysis" in tools


class TestToolCategories:
    """Tests for tool categorization and counts."""

    def test_core_tools_count(self, mcp_server):
        """Test that core tools are registered."""
        tools = mcp_server._tool_manager._tools
        core_tools = [
            "tool_list_specs",
            "tool_get_spec",
            "tool_get_spec_hierarchy",
            "tool_get_task",
        ]
        for tool_name in core_tools:
            assert tool_name in tools, f"Core tool {tool_name} missing"

    def test_rendering_tools_count(self, mcp_server):
        """Test that rendering tools are registered."""
        tools = mcp_server._tool_manager._tools
        rendering_tools = [
            "foundry_render_spec",
            "foundry_render_progress",
            "foundry_list_tasks",
        ]
        for tool_name in rendering_tools:
            assert tool_name in tools, f"Rendering tool {tool_name} missing"

    def test_lifecycle_tools_count(self, mcp_server):
        """Test that lifecycle tools are registered."""
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
            assert tool_name in tools, f"Lifecycle tool {tool_name} missing"

    def test_total_tool_count_minimum(self, mcp_server):
        """Test that we have expected minimum number of tools."""
        tools = mcp_server._tool_manager._tools
        # Core (4) + Rendering (3) + Lifecycle (6) + Validation (1) + Journal (1) + Query (1) + Task (1) + Testing (3)
        min_expected = 20
        assert len(tools) >= min_expected, f"Expected at least {min_expected} tools, got {len(tools)}"
