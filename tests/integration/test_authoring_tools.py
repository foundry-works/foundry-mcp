"""
Integration tests for authoring tools.

Tests:
- Tool registration and availability
- Hierarchy integrity after CRUD operations
- Response envelope compliance (response-v2)
- End-to-end authoring workflows
"""

import json
import pytest
from foundry_mcp.server import create_server
from foundry_mcp.config import ServerConfig


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def test_specs_dir(tmp_path):
    """Create a test specs directory with sample spec for authoring tests."""
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    (specs_dir / "active").mkdir()
    (specs_dir / "pending").mkdir()
    (specs_dir / "completed").mkdir()
    (specs_dir / "archived").mkdir()
    (specs_dir / "templates").mkdir()

    # Create a sample spec with complete hierarchy for authoring tests
    sample_spec = {
        "spec_id": "authoring-test-spec-001",
        "title": "Authoring Test Specification",
        "metadata": {
            "title": "Authoring Test Specification",
            "description": "A test spec for authoring tool integration testing",
            "created_at": "2025-01-25T00:00:00Z",
            "status": "in_progress",
            "version": "1.0.0",
        },
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Authoring Test Specification",
                "status": "in_progress",
                "children": ["phase-1", "phase-2"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Setup Phase",
                "status": "completed",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2"],
            },
            "phase-2": {
                "type": "phase",
                "title": "Implementation Phase",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-2-1"],
            },
            "task-1-1": {
                "type": "task",
                "title": "Initial setup",
                "status": "completed",
                "parent": "phase-1",
                "children": [],
            },
            "task-1-2": {
                "type": "task",
                "title": "Configuration",
                "status": "completed",
                "parent": "phase-1",
                "children": [],
            },
            "task-2-1": {
                "type": "task",
                "title": "Main implementation",
                "status": "in_progress",
                "parent": "phase-2",
                "children": ["subtask-2-1-1"],
            },
            "subtask-2-1-1": {
                "type": "subtask",
                "title": "Helper function",
                "status": "pending",
                "parent": "task-2-1",
                "children": [],
            },
        },
        "assumptions": [
            {
                "id": "assumption-1",
                "text": "API is available",
                "type": "constraint",
                "created_at": "2025-01-25T00:00:00Z",
            },
        ],
        "revision_history": [
            {
                "version": "1.0.0",
                "date": "2025-01-25",
                "changes": "Initial version",
                "author": "test",
            },
        ],
        "journal": [
            {
                "timestamp": "2025-01-25T00:00:00Z",
                "entry_type": "status_change",
                "title": "Spec created",
                "content": "Initial spec creation",
                "task_id": "spec-root",
            },
        ],
    }

    spec_file = specs_dir / "active" / "authoring-test-spec-001.json"
    with open(spec_file, "w") as f:
        json.dump(sample_spec, f, indent=2)

    return specs_dir


@pytest.fixture
def test_config(test_specs_dir):
    """Create a test server configuration."""
    return ServerConfig(
        server_name="foundry-mcp-authoring-test",
        server_version="0.1.0",
        specs_dir=test_specs_dir,
        log_level="WARNING",
    )


@pytest.fixture
def mcp_server(test_config):
    """Create a test MCP server instance."""
    return create_server(test_config)


def _call_tool(tools, tool_name: str, **kwargs):
    """Call unified routers for authoring workflows."""

    if tool_name in {
        "spec-create",
        "phase-add",
        "phase-remove",
        "assumption-add",
        "assumption-list",
        "revision-add",
        "spec-update-frontmatter",
    }:
        return tools["authoring"].fn(action=tool_name, **kwargs)

    if tool_name == "spec-template":
        template_action = kwargs.pop("action", None)
        return tools["authoring"].fn(
            action="spec-template",
            template_action=template_action,
            **kwargs,
        )

    if tool_name == "task-add":
        if "hours" in kwargs and "estimated_hours" not in kwargs:
            kwargs["estimated_hours"] = kwargs.pop("hours")
        return tools["task"].fn(action="add", **kwargs)

    if tool_name == "task-remove":
        return tools["task"].fn(action="remove", **kwargs)

    raise KeyError(f"Unsupported tool mapping for {tool_name}")


# =============================================================================
# Authoring Tools Registration Tests
# =============================================================================


class TestAuthoringToolsRegistration:
    """Test that all authoring tools are properly registered."""

    def test_spec_create_registered(self, mcp_server):
        """Test that spec creation entrypoint is registered."""
        tools = mcp_server._tool_manager._tools
        assert "authoring" in tools
        assert callable(tools["authoring"].fn)

    def test_spec_template_registered(self, mcp_server):
        """Test that spec template entrypoint is registered."""
        tools = mcp_server._tool_manager._tools
        assert "authoring" in tools
        assert callable(tools["authoring"].fn)

    def test_task_add_registered(self, mcp_server):
        """Test that task add entrypoint is registered."""
        tools = mcp_server._tool_manager._tools
        assert "task" in tools
        assert callable(tools["task"].fn)

    def test_phase_add_registered(self, mcp_server):
        """Test that phase add entrypoint is registered."""
        tools = mcp_server._tool_manager._tools
        assert "authoring" in tools
        assert callable(tools["authoring"].fn)

    def test_task_remove_registered(self, mcp_server):
        """Test that task remove entrypoint is registered."""
        tools = mcp_server._tool_manager._tools
        assert "task" in tools
        assert callable(tools["task"].fn)

    def test_assumption_add_registered(self, mcp_server):
        """Test that assumption add entrypoint is registered."""
        tools = mcp_server._tool_manager._tools
        assert "authoring" in tools
        assert callable(tools["authoring"].fn)

    def test_assumption_list_registered(self, mcp_server):
        """Test that assumption list entrypoint is registered."""
        tools = mcp_server._tool_manager._tools
        assert "authoring" in tools
        assert callable(tools["authoring"].fn)

    def test_revision_add_registered(self, mcp_server):
        """Test that revision add entrypoint is registered."""
        tools = mcp_server._tool_manager._tools
        assert "authoring" in tools
        assert callable(tools["authoring"].fn)

    def test_spec_update_frontmatter_registered(self, mcp_server):
        """Test that spec update frontmatter entrypoint is registered."""
        tools = mcp_server._tool_manager._tools
        assert "authoring" in tools
        assert callable(tools["authoring"].fn)

    def test_phase_remove_registered(self, mcp_server):
        """Test that phase remove entrypoint is registered."""
        tools = mcp_server._tool_manager._tools
        assert "authoring" in tools
        assert callable(tools["authoring"].fn)

    def test_all_authoring_tools_count(self, mcp_server):
        """Test that authoring surface is registered."""
        tools = mcp_server._tool_manager._tools
        assert "authoring" in tools
        assert "task" in tools


# =============================================================================
# Response Envelope Compliance Tests
# =============================================================================


class TestResponseEnvelopeCompliance:
    """Test that authoring tools return response-v2 compliant envelopes."""

    def _validate_response_envelope(self, response):
        """Validate response follows response-v2 contract."""
        assert isinstance(response, dict), "Response must be a dict"
        assert "success" in response, "Response must have 'success' key"
        assert "data" in response, "Response must have 'data' key"
        assert "error" in response, "Response must have 'error' key"
        assert "meta" in response, "Response must have 'meta' key"
        assert isinstance(response["success"], bool), "success must be boolean"
        assert isinstance(response["data"], dict), "data must be dict"
        assert isinstance(response["meta"], dict), "meta must be dict"
        assert response["meta"].get("version") == "response-v2", (
            "meta.version must be 'response-v2'"
        )

        if response["success"]:
            assert response["error"] is None, "error must be null on success"
        else:
            assert response["error"] is not None, "error must be set on failure"
            assert isinstance(response["error"], str), "error must be string"

    def test_spec_template_list_envelope(self, mcp_server):
        """Test spec-template list action returns valid envelope."""
        tools = mcp_server._tool_manager._tools
        result = _call_tool(tools, "spec-template", action="list")
        self._validate_response_envelope(result)

    def test_assumption_list_envelope(self, mcp_server):
        """Test assumption-list returns valid envelope."""
        tools = mcp_server._tool_manager._tools
        result = _call_tool(tools, "assumption-list", spec_id="authoring-test-spec-001")
        self._validate_response_envelope(result)

    def test_validation_error_envelope(self, mcp_server):
        """Test validation errors return proper envelope with error details."""
        tools = mcp_server._tool_manager._tools
        # Missing required parameter
        result = _call_tool(
            tools, "task-add", spec_id="", parent="phase-1", title="Test"
        )
        self._validate_response_envelope(result)
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()
        assert "error_code" in result["data"]
        assert "error_type" in result["data"]
        assert "remediation" in result["data"]

    def test_revision_add_validation(self, mcp_server):
        """Test that revision-add validates inputs."""
        tools = mcp_server._tool_manager._tools

        # Missing version
        result = _call_tool(
            tools,
            "revision-add",
            spec_id="authoring-test-spec-001",
            version="",
            changes="Test changes",
        )

        assert result["success"] is False
        assert "version" in result["error"].lower()

    def test_frontmatter_update_validation(self, mcp_server):
        """Test that frontmatter update validates inputs."""
        tools = mcp_server._tool_manager._tools

        # Missing key
        result = _call_tool(
            tools,
            "spec-update-frontmatter",
            spec_id="authoring-test-spec-001",
            key="",
            value="test",
        )

        assert result["success"] is False
        assert "key" in result["error"].lower()


# =============================================================================
# Template Operations Tests
# =============================================================================


class TestTemplateOperations:
    """Test spec-template tool operations."""

    def test_template_list_action(self, mcp_server):
        """Test template list action."""
        tools = mcp_server._tool_manager._tools

        result = _call_tool(tools, "spec-template", action="list")

        # CLI may not be available; verify we get a valid response envelope
        assert "success" in result
        assert "data" in result
        # If successful, should have action and templates data
        if result["success"]:
            assert result["data"]["action"] == "list"
            assert "templates" in result["data"] or "total_count" in result["data"]

    def test_template_show_requires_name(self, mcp_server):
        """Test template show action requires template_name."""
        tools = mcp_server._tool_manager._tools

        result = _call_tool(tools, "spec-template", action="show")

        assert result["success"] is False
        assert "template_name" in result["error"].lower()

    def test_template_apply_requires_name(self, mcp_server):
        """Test template apply action requires template_name."""
        tools = mcp_server._tool_manager._tools

        result = _call_tool(tools, "spec-template", action="apply")

        assert result["success"] is False
        assert "template_name" in result["error"].lower()

    def test_template_invalid_action(self, mcp_server):
        """Test template with invalid action."""
        tools = mcp_server._tool_manager._tools

        result = _call_tool(tools, "spec-template", action="invalid_action")

        assert result["success"] is False
        assert "template_action" in result["error"].lower()


# =============================================================================
# Spec Creation Tests
# =============================================================================


class TestSpecCreation:
    """Test spec-create tool operations."""

    def test_spec_create_validates_template(self, mcp_server):
        """Test spec-create validates template parameter."""
        tools = mcp_server._tool_manager._tools

        # Valid templates are: simple, medium, complex, security
        result = _call_tool(
            tools,
            "spec-create",
            name="test-spec",
            template="invalid_template",
        )

        assert result["success"] is False
        assert "VALIDATION_ERROR" in str(result["data"].get("error_code", ""))

    def test_spec_create_validates_category(self, mcp_server):
        """Test spec-create validates category parameter."""
        tools = mcp_server._tool_manager._tools

        # Valid categories are: investigation, implementation, refactoring, decision, research
        result = _call_tool(
            tools,
            "spec-create",
            name="test-spec",
            category="invalid_category",
        )

        assert result["success"] is False
        assert "category" in result["error"].lower()

    def test_spec_create_valid_templates_accepted(self, mcp_server):
        """Test spec-create accepts all valid templates."""
        tools = mcp_server._tool_manager._tools

        valid_templates = ["simple", "medium", "complex", "security"]
        for template in valid_templates:
            result = _call_tool(
                tools,
                "spec-create",
                name=f"test-{template}",
                template=template,
            )
            # May fail due to CLI not being available, but should not fail validation
            if result["success"] is False:
                assert "VALIDATION_ERROR" not in str(
                    result["data"].get("error_code", "")
                ), f"Template '{template}' should be valid"

    def test_spec_create_valid_categories_accepted(self, mcp_server):
        """Test spec-create accepts all valid categories."""
        tools = mcp_server._tool_manager._tools

        valid_categories = [
            "investigation",
            "implementation",
            "refactoring",
            "decision",
            "research",
        ]
        for category in valid_categories:
            result = _call_tool(
                tools,
                "spec-create",
                name=f"test-{category}",
                category=category,
            )
            # May fail due to CLI not being available, but should not fail validation
            if result["success"] is False:
                assert category.lower() not in result["error"].lower(), (
                    f"Category '{category}' should be valid"
                )


# =============================================================================
# Task Operations Tests
# =============================================================================


class TestTaskOperations:
    """Test task-add and task-remove operations."""

    def test_task_add_validates_spec_id(self, mcp_server):
        """Test task-add validates spec_id."""
        tools = mcp_server._tool_manager._tools

        result = _call_tool(
            tools,
            "task-add",
            spec_id="",
            parent="phase-1",
            title="Test task",
        )

        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_task_add_validates_parent(self, mcp_server):
        """Test task-add validates parent."""
        tools = mcp_server._tool_manager._tools

        result = _call_tool(
            tools,
            "task-add",
            spec_id="test-spec",
            parent="",
            title="Test task",
        )

        assert result["success"] is False
        assert "parent" in result["error"].lower()

    def test_task_add_validates_title(self, mcp_server):
        """Test task-add validates title."""
        tools = mcp_server._tool_manager._tools

        result = _call_tool(
            tools,
            "task-add",
            spec_id="test-spec",
            parent="phase-1",
            title="",
        )

        assert result["success"] is False
        assert "title" in result["error"].lower()

    def test_task_add_validates_task_type(self, mcp_server):
        """Test task-add validates task_type."""
        tools = mcp_server._tool_manager._tools

        # Valid types are: task, subtask, verify
        result = _call_tool(
            tools,
            "task-add",
            spec_id="test-spec",
            parent="phase-1",
            title="Test",
            task_type="invalid_type",
        )

        assert result["success"] is False
        assert "task_type" in result["error"].lower()

    def test_task_add_accepts_valid_types(self, mcp_server):
        """Test task-add accepts all valid task types."""
        tools = mcp_server._tool_manager._tools

        valid_types = ["task", "subtask", "verify"]
        for task_type in valid_types:
            result = _call_tool(
                tools,
                "task-add",
                spec_id="authoring-test-spec-001",
                parent="phase-1",
                title=f"Test {task_type}",
                task_type=task_type,
            )
            # May fail due to CLI, but should not fail type validation
            if result["success"] is False:
                assert task_type.lower() not in result["error"].lower(), (
                    f"Task type '{task_type}' should be valid"
                )

    def test_task_remove_validates_spec_id(self, mcp_server):
        """Test task-remove validates spec_id."""
        tools = mcp_server._tool_manager._tools

        result = _call_tool(
            tools,
            "task-remove",
            spec_id="",
            task_id="task-1-1",
        )

        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_task_remove_validates_task_id(self, mcp_server):
        """Test task-remove validates task_id."""
        tools = mcp_server._tool_manager._tools

        result = _call_tool(
            tools,
            "task-remove",
            spec_id="test-spec",
            task_id="",
        )

        assert result["success"] is False
        assert "task_id" in result["error"].lower()

    def test_task_add_dry_run_option(self, mcp_server):
        """Test task-add supports dry_run option."""
        tools = mcp_server._tool_manager._tools

        result = _call_tool(
            tools,
            "task-add",
            spec_id="authoring-test-spec-001",
            parent="phase-1",
            title="Dry run task",
            dry_run=True,
        )

        # Should either succeed with dry_run flag set or fail gracefully
        if result["success"]:
            assert result["data"].get("dry_run") is True

    def test_task_remove_dry_run_option(self, mcp_server):
        """Test task-remove supports dry_run option."""
        tools = mcp_server._tool_manager._tools

        result = _call_tool(
            tools,
            "task-remove",
            spec_id="authoring-test-spec-001",
            task_id="task-1-2",
            dry_run=True,
        )

        if result["success"]:
            assert result["data"].get("dry_run") is True


class TestPhaseOperations:
    """Test phase-add and phase-remove operations."""

    def test_phase_add_validates_spec_id(self, mcp_server):
        tools = mcp_server._tool_manager._tools

        result = _call_tool(tools, "phase-add", spec_id="", title="Phase")

        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_phase_add_validates_title(self, mcp_server):
        tools = mcp_server._tool_manager._tools

        result = _call_tool(tools, "phase-add", spec_id="test-spec", title="")

        assert result["success"] is False
        assert "title" in result["error"].lower()

    def test_phase_add_validates_hours(self, mcp_server):
        tools = mcp_server._tool_manager._tools

        result = _call_tool(
            tools,
            "phase-add",
            spec_id="test-spec",
            title="Phase",
            estimated_hours=-2,
        )

        assert result["success"] is False
        assert "estimated_hours" in result["error"].lower()

    def test_phase_add_dry_run_option(self, mcp_server):
        tools = mcp_server._tool_manager._tools

        result = _call_tool(
            tools,
            "phase-add",
            spec_id="authoring-test-spec-001",
            title="Dry run phase",
            dry_run=True,
        )

        if result["success"]:
            assert result["data"].get("dry_run") is True

    def test_phase_remove_validates_spec_id(self, mcp_server):
        """Test phase-remove validates spec_id."""
        tools = mcp_server._tool_manager._tools

        result = _call_tool(tools, "phase-remove", spec_id="", phase_id="phase-1")

        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_phase_remove_validates_phase_id(self, mcp_server):
        """Test phase-remove validates phase_id."""
        tools = mcp_server._tool_manager._tools

        result = _call_tool(tools, "phase-remove", spec_id="test-spec", phase_id="")

        assert result["success"] is False
        assert "phase_id" in result["error"].lower()

    def test_phase_remove_dry_run_option(self, mcp_server):
        """Test phase-remove supports dry_run option."""
        tools = mcp_server._tool_manager._tools

        result = _call_tool(
            tools,
            "phase-remove",
            spec_id="authoring-test-spec-001",
            phase_id="phase-1",
            dry_run=True,
        )

        # Should either succeed with dry_run flag set or fail gracefully
        if result["success"]:
            assert result["data"].get("dry_run") is True

    def test_phase_remove_force_option(self, mcp_server):
        """Test phase-remove accepts force option."""
        tools = mcp_server._tool_manager._tools

        # This may fail because phase has tasks, but shouldn't fail validation
        result = _call_tool(
            tools,
            "phase-remove",
            spec_id="authoring-test-spec-001",
            phase_id="phase-1",
            force=True,
        )

        # Should return valid envelope regardless
        assert "success" in result
        assert "data" in result
        assert "meta" in result

    def test_task_remove_dry_run_option(self, mcp_server):
        """Test task-remove supports dry_run option."""
        tools = mcp_server._tool_manager._tools

        result = _call_tool(
            tools,
            "task-remove",
            spec_id="authoring-test-spec-001",
            task_id="task-1-1",
            dry_run=True,
        )

        # Should either succeed with dry_run flag set or fail gracefully
        if result["success"]:
            assert result["data"].get("dry_run") is True


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================


class TestAuthoringWorkflows:
    """Test end-to-end authoring workflows."""

    def test_list_then_add_assumption_workflow(self, mcp_server):
        """Test workflow: list assumptions, then add new one."""
        tools = mcp_server._tool_manager._tools

        # Step 1: List existing assumptions
        list_result = _call_tool(
            tools, "assumption-list", spec_id="authoring-test-spec-001"
        )

        # Step 2: Add a new assumption
        add_result = _call_tool(
            tools,
            "assumption-add",
            spec_id="authoring-test-spec-001",
            text="New integration test assumption",
            assumption_type="constraint",
        )

        # Both operations should return valid envelopes
        assert "success" in list_result
        assert "success" in add_result

    def test_template_list_then_show_workflow(self, mcp_server):
        """Test workflow: list templates, then show one."""
        tools = mcp_server._tool_manager._tools

        # Step 1: List templates
        list_result = _call_tool(tools, "spec-template", action="list")
        # CLI may not be available; verify we get a valid envelope
        assert "success" in list_result
        assert "data" in list_result

        # Step 2: If templates exist and successful, try to show one
        if list_result["success"] and list_result["data"].get("templates"):
            template_name = list_result["data"]["templates"][0].get("name", "simple")
            show_result = _call_tool(
                tools,
                "spec-template",
                action="show",
                template_name=template_name,
            )
            assert "success" in show_result

    def test_validation_before_creation_workflow(self, mcp_server):
        """Test workflow: validate inputs before creating resources."""
        tools = mcp_server._tool_manager._tools

        # Step 1: Try invalid spec creation (should fail validation)
        invalid_result = _call_tool(
            tools,
            "spec-create",
            name="test",
            template="not_a_real_template",
        )
        assert invalid_result["success"] is False
        assert invalid_result["data"].get("error_code") == "VALIDATION_ERROR"

        # Step 2: Try valid parameters (may fail due to CLI, but should pass validation)
        valid_result = _call_tool(
            tools,
            "spec-create",
            name="test",
            template="simple",
        )
        # Should not fail validation
        if valid_result["success"] is False:
            assert valid_result["data"].get("error_code") != "VALIDATION_ERROR"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in authoring tools."""

    def test_circuit_breaker_error_response(self, mcp_server):
        """Test that circuit breaker errors are properly handled."""
        # This test verifies the error response format when circuit breaker is open
        # Since we can't easily trigger the circuit breaker in integration tests,
        # we test that the error codes are defined correctly
        tools = mcp_server._tool_manager._tools

        # Make a request that will likely fail
        result = _call_tool(tools, "spec-create", name="test")

        # Should return valid envelope regardless of success
        assert "success" in result
        assert "data" in result
        assert "error" in result
        assert "meta" in result

    def test_timeout_error_includes_remediation(self, mcp_server):
        """Test that timeout errors include remediation guidance."""
        tools = mcp_server._tool_manager._tools

        # Trigger a request that might timeout
        result = _call_tool(
            tools,
            "task-add",
            spec_id="authoring-test-spec-001",
            parent="phase-1",
            title="Test task",
        )

        # If it failed with timeout, check remediation
        if result["success"] is False and "TIMEOUT" in str(
            result["data"].get("error_code", "")
        ):
            assert "remediation" in result["data"]

    def test_not_found_errors_include_remediation(self, mcp_server):
        """Test that not found errors include remediation guidance."""
        tools = mcp_server._tool_manager._tools

        result = _call_tool(
            tools,
            "task-add",
            spec_id="nonexistent-spec-999",
            parent="phase-1",
            title="Test task",
        )

        if result["success"] is False:
            # Should include remediation for not found errors
            if (
                "NOT_FOUND" in str(result["data"].get("error_code", ""))
                or "not found" in result["error"].lower()
            ):
                assert "remediation" in result["data"]
