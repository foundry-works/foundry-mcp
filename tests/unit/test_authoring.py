"""
Unit tests for foundry_mcp.tools.authoring module.

Tests the authoring tools for creating and modifying SDD specifications.
These tools use direct core API calls instead of CLI subprocess calls.
"""

import json
from dataclasses import asdict
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from foundry_mcp.core.responses import success_response, error_response, ToolResponse


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_mcp():
    """Create a mock FastMCP server instance."""
    mcp = MagicMock()
    # Store registered tools
    mcp._tools = {}

    def mock_tool(*args, **kwargs):
        def decorator(func):
            # Use the 'name' kwarg if provided, otherwise use func.__name__
            tool_name = kwargs.get("name", func.__name__)
            mcp._tools[tool_name] = func
            return func

        return decorator

    mcp.tool = mock_tool
    return mcp


@pytest.fixture
def mock_config():
    """Create a mock server config."""
    config = MagicMock()
    config.project_root = "/test/project"
    config.specs_dir = None  # Will use find_specs_directory
    return config


@pytest.fixture
def temp_spec_file(tmp_path):
    """Create a temporary spec file for testing."""
    spec_data = {
        "spec_id": "test-spec-001",
        "title": "Test Specification",
        "version": "1.0.0",
        "status": "draft",
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test Specification",
                "status": "pending",
                "children": ["phase-1"],
                "parent": None,
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "pending",
                "children": ["task-1-1"],
                "parent": "spec-root",
            },
            "task-1-1": {
                "type": "task",
                "title": "Task 1",
                "status": "pending",
                "children": [],
                "parent": "phase-1",
            },
        },
        "assumptions": [],
        "revision_history": [],
    }

    specs_dir = tmp_path / "specs" / "active"
    specs_dir.mkdir(parents=True)
    spec_file = specs_dir / "test-spec-001.json"

    with open(spec_file, "w") as f:
        json.dump(spec_data, f)

    return spec_file, spec_data


# =============================================================================
# spec-create Tool Tests
# =============================================================================


class TestSpecCreate:
    """Test the spec-create tool."""

    def test_validation_error_on_empty_name(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """spec_create should return validation error on empty name."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        spec_create = mock_mcp._tools["spec-create"]
        result = spec_create(name="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"
        assert result["data"].get("error_type") == "validation"

    def test_validation_error_on_invalid_template(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """spec_create should return validation error on invalid template."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        spec_create = mock_mcp._tools["spec-create"]
        result = spec_create(name="test-spec", template="invalid")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"
        assert "remediation" in result["data"]

    def test_validation_error_on_invalid_category(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """spec_create should return validation error on invalid category."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        spec_create = mock_mcp._tools["spec-create"]
        result = spec_create(name="test-spec", category="invalid")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"


# =============================================================================
# spec-template Tool Tests
# =============================================================================


class TestSpecTemplate:
    """Test the spec-template tool."""

    def test_list_action_returns_templates(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """spec_template list action should return available templates."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        spec_template = mock_mcp._tools["spec-template"]
        result = spec_template(action="list")

        assert_response_contract(result)
        assert result["success"] is True
        assert "templates" in result["data"]
        assert result["data"]["action"] == "list"
        assert isinstance(result["data"]["templates"], list)
        assert len(result["data"]["templates"]) > 0

    def test_validation_error_on_invalid_action(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """spec_template should return validation error on invalid action."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        spec_template = mock_mcp._tools["spec-template"]
        result = spec_template(action="invalid")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"

    def test_show_action_requires_template_name(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """spec_template show action should require template_name."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        spec_template = mock_mcp._tools["spec-template"]
        result = spec_template(action="show")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_show_action_with_valid_template(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """spec_template show action should return template content."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        spec_template = mock_mcp._tools["spec-template"]
        result = spec_template(action="show", template_name="medium")

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["action"] == "show"
        assert result["data"]["template_name"] == "medium"
        assert "content" in result["data"]


# =============================================================================
# task-add Tool Tests
# =============================================================================


class TestTaskAdd:
    """Test the task-add tool."""

    def test_validation_error_on_missing_spec_id(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """task_add should return validation error on missing spec_id."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        task_add = mock_mcp._tools["task-add"]
        result = task_add(spec_id="", parent="phase-1", title="Test task")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_validation_error_on_missing_parent(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """task_add should return validation error on missing parent."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        task_add = mock_mcp._tools["task-add"]
        result = task_add(spec_id="test-spec", parent="", title="Test task")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_validation_error_on_missing_title(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """task_add should return validation error on missing title."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        task_add = mock_mcp._tools["task-add"]
        result = task_add(spec_id="test-spec", parent="phase-1", title="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_validation_error_on_invalid_task_type(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """task_add should return validation error on invalid task_type."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        task_add = mock_mcp._tools["task-add"]
        result = task_add(
            spec_id="test-spec", parent="phase-1", title="Test", task_type="invalid"
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"

    def test_dry_run_returns_preview(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """task_add with dry_run should return preview without changes."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        task_add = mock_mcp._tools["task-add"]
        result = task_add(
            spec_id="test-spec", parent="phase-1", title="Test task", dry_run=True
        )

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["dry_run"] is True
        assert result["data"]["task_id"] == "(preview)"


class TestPhaseAdd:
    """Test the phase-add tool."""

    def test_validation_error_on_missing_spec_id(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """phase_add should return validation error on missing spec_id."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        phase_add = mock_mcp._tools["phase-add"]
        result = phase_add(spec_id="", title="New Phase")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_validation_error_on_missing_title(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """phase_add should return validation error on missing title."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        phase_add = mock_mcp._tools["phase-add"]
        result = phase_add(spec_id="test-spec", title="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_validation_error_on_negative_hours(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """phase_add should validate estimated_hours is non-negative."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        phase_add = mock_mcp._tools["phase-add"]
        result = phase_add(spec_id="test-spec", title="Invalid", estimated_hours=-1)

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"

    def test_dry_run_returns_preview(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """phase_add with dry_run should return preview without changes."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        phase_add = mock_mcp._tools["phase-add"]
        result = phase_add(spec_id="test-spec", title="Phase", dry_run=True)

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["dry_run"] is True
        assert result["data"].get("phase_id") == "(preview)"


# =============================================================================
# task-remove Tool Tests
# =============================================================================


class TestTaskRemove:
    """Test the task-remove tool."""

    def test_validation_error_on_missing_spec_id(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """task_remove should return validation error on missing spec_id."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        task_remove = mock_mcp._tools["task-remove"]
        result = task_remove(spec_id="", task_id="task-1-2")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_validation_error_on_missing_task_id(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """task_remove should return validation error on missing task_id."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        task_remove = mock_mcp._tools["task-remove"]
        result = task_remove(spec_id="test-spec", task_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_dry_run_returns_preview(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """task_remove with dry_run should return preview without changes."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        task_remove = mock_mcp._tools["task-remove"]
        result = task_remove(spec_id="test-spec", task_id="task-1-2", dry_run=True)

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["dry_run"] is True


# =============================================================================
# assumption-add Tool Tests
# =============================================================================


class TestAssumptionAdd:
    """Test the assumption-add tool."""

    def test_validation_error_on_missing_spec_id(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """assumption_add should return validation error on missing spec_id."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        assumption_add = mock_mcp._tools["assumption-add"]
        result = assumption_add(spec_id="", text="API rate limits apply")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_validation_error_on_missing_text(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """assumption_add should return validation error on missing text."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        assumption_add = mock_mcp._tools["assumption-add"]
        result = assumption_add(spec_id="test-spec", text="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_validation_error_on_invalid_type(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """assumption_add should return validation error on invalid assumption_type."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        assumption_add = mock_mcp._tools["assumption-add"]
        result = assumption_add(
            spec_id="test-spec", text="Test assumption", assumption_type="invalid"
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"

    def test_dry_run_returns_preview(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """assumption_add with dry_run should return preview without changes."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        assumption_add = mock_mcp._tools["assumption-add"]
        result = assumption_add(
            spec_id="test-spec", text="Test assumption", dry_run=True
        )

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["dry_run"] is True


# =============================================================================
# assumption-list Tool Tests
# =============================================================================


class TestAssumptionList:
    """Test the assumption-list tool."""

    def test_validation_error_on_missing_spec_id(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """assumption_list should return validation error on missing spec_id."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        assumption_list = mock_mcp._tools["assumption-list"]
        result = assumption_list(spec_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_validation_error_on_invalid_type(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """assumption_list should return validation error on invalid assumption_type."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        assumption_list = mock_mcp._tools["assumption-list"]
        result = assumption_list(spec_id="test-spec", assumption_type="invalid")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"


# =============================================================================
# revision-add Tool Tests
# =============================================================================


class TestRevisionAdd:
    """Test the revision-add tool."""

    def test_validation_error_on_missing_spec_id(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """revision_add should return validation error on missing spec_id."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        revision_add = mock_mcp._tools["revision-add"]
        result = revision_add(spec_id="", version="1.1", changes="Added new tasks")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_validation_error_on_missing_version(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """revision_add should return validation error on missing version."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        revision_add = mock_mcp._tools["revision-add"]
        result = revision_add(
            spec_id="test-spec", version="", changes="Added new tasks"
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_validation_error_on_missing_changes(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """revision_add should return validation error on missing changes."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        revision_add = mock_mcp._tools["revision-add"]
        result = revision_add(spec_id="test-spec", version="1.1", changes="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_dry_run_returns_preview(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """revision_add with dry_run should return preview without changes."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        revision_add = mock_mcp._tools["revision-add"]
        result = revision_add(
            spec_id="test-spec", version="1.1", changes="Added new tasks", dry_run=True
        )

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["dry_run"] is True


# =============================================================================
# spec-update-frontmatter Tool Tests
# =============================================================================


class TestSpecUpdateFrontmatter:
    """Test the spec-update-frontmatter tool."""

    def test_validation_error_on_missing_spec_id(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """spec_update_frontmatter should return validation error on missing spec_id."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        update = mock_mcp._tools["spec-update-frontmatter"]
        result = update(spec_id="", key="title", value="New Title")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_validation_error_on_missing_key(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """spec_update_frontmatter should return validation error on missing key."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        update = mock_mcp._tools["spec-update-frontmatter"]
        result = update(spec_id="test-spec", key="", value="New Title")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_dry_run_returns_preview(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """spec_update_frontmatter with dry_run should return preview without changes."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        update = mock_mcp._tools["spec-update-frontmatter"]
        result = update(
            spec_id="test-spec", key="title", value="New Title", dry_run=True
        )

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["dry_run"] is True


# =============================================================================
# Integration Tests for Tool Registration
# =============================================================================


class TestToolRegistration:
    """Test that all tools are properly registered."""

    def test_all_tools_registered(self, mock_mcp, mock_config):
        """All authoring tools should be registered with the MCP server."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        expected_tools = [
            "spec-create",
            "spec-template",
            "task-add",
            "phase-add",
            "task-remove",
            "assumption-add",
            "assumption-list",
            "revision-add",
            "spec-update-frontmatter",
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_mcp._tools, f"Tool {tool_name} not registered"

    def test_tools_are_callable(self, mock_mcp, mock_config):
        """All registered tools should be callable functions."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        for tool_name, tool_func in mock_mcp._tools.items():
            assert callable(tool_func), f"Tool {tool_name} is not callable"


# =============================================================================
# Response Contract Compliance Tests
# =============================================================================


class TestResponseContractCompliance:
    """Test that all responses comply with the response-v2 contract."""

    def test_validation_error_response_structure(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """Validation error responses should have correct structure."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        spec_create = mock_mcp._tools["spec-create"]
        result = spec_create(name="")

        # Validate structure
        assert "success" in result
        assert "data" in result
        assert "error" in result
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"

    def test_success_response_has_required_fields(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """Success responses should have all required fields."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        spec_template = mock_mcp._tools["spec-template"]
        result = spec_template(action="list")

        # Validate structure
        assert result["success"] is True
        assert result["error"] is None
        assert "templates" in result["data"]
        assert result["meta"]["version"] == "response-v2"

    def test_error_response_has_required_fields(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """Error responses should have all required fields."""
        from foundry_mcp.tools.authoring import register_authoring_tools

        register_authoring_tools(mock_mcp, mock_config)

        task_add = mock_mcp._tools["task-add"]
        result = task_add(spec_id="", parent="phase-1", title="Test")

        # Validate structure
        assert result["success"] is False
        assert result["error"] is not None
        assert "error_code" in result["data"]
        assert "error_type" in result["data"]
        assert "remediation" in result["data"]
        assert result["meta"]["version"] == "response-v2"
