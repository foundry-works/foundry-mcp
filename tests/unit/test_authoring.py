"""
Unit tests for foundry_mcp.tools.authoring module.

Tests the authoring tools for creating and modifying SDD specifications,
including circuit breaker protection, validation, and response contracts.
"""

import json
import subprocess
import time
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.responses import success_response, error_response, ToolResponse
from foundry_mcp.core.resilience import CircuitBreaker, CircuitBreakerError, CircuitState


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
            mcp._tools[func.__name__] = func
            return func
        return decorator

    mcp.tool = mock_tool
    return mcp


@pytest.fixture
def mock_config():
    """Create a mock server config."""
    config = MagicMock()
    config.project_root = "/test/project"
    return config


@pytest.fixture
def fresh_circuit_breaker():
    """Create a fresh circuit breaker for each test."""
    return CircuitBreaker(
        name="test_sdd_cli",
        failure_threshold=5,
        recovery_timeout=30.0,
        half_open_max_calls=3,
    )


# =============================================================================
# _run_sdd_command Tests
# =============================================================================


class TestRunSddCommand:
    """Test the _run_sdd_command helper function."""

    def test_successful_command_execution(self):
        """Successful command should return result and record success."""
        from foundry_mcp.tools.authoring import _run_sdd_command, _sdd_cli_breaker

        # Reset breaker state
        _sdd_cli_breaker.reset()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "test"],
                returncode=0,
                stdout='{"result": "success"}',
                stderr="",
            )

            result = _run_sdd_command(["foundry-cli", "test"], "test_tool")

            assert result.returncode == 0
            assert '{"result": "success"}' in result.stdout
            mock_run.assert_called_once()

    def test_failed_command_records_failure(self):
        """Failed command should record failure with circuit breaker."""
        from foundry_mcp.tools.authoring import _run_sdd_command, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        initial_failures = _sdd_cli_breaker.failure_count

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "test"],
                returncode=1,
                stdout="",
                stderr="Error occurred",
            )

            result = _run_sdd_command(["foundry-cli", "test"], "test_tool")

            assert result.returncode == 1
            assert _sdd_cli_breaker.failure_count == initial_failures + 1

    def test_circuit_breaker_open_raises_error(self):
        """When circuit breaker is open, should raise CircuitBreakerError."""
        from foundry_mcp.tools.authoring import _run_sdd_command, _sdd_cli_breaker

        _sdd_cli_breaker.reset()

        # Open the circuit breaker
        for _ in range(5):
            _sdd_cli_breaker.record_failure()

        assert _sdd_cli_breaker.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerError) as exc_info:
            _run_sdd_command(["foundry-cli", "test"], "test_tool")

        assert exc_info.value.breaker_name == "sdd_cli_authoring"

        # Reset for other tests
        _sdd_cli_breaker.reset()

    def test_timeout_records_failure(self):
        """Timeout should record failure with circuit breaker."""
        from foundry_mcp.tools.authoring import _run_sdd_command, _sdd_cli_breaker

        _sdd_cli_breaker.reset()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["foundry-cli"], timeout=30)

            with pytest.raises(subprocess.TimeoutExpired):
                _run_sdd_command(["foundry-cli", "test"], "test_tool")

            assert _sdd_cli_breaker.failure_count == 1

        _sdd_cli_breaker.reset()

    def test_file_not_found_records_failure(self):
        """FileNotFoundError should record failure with circuit breaker."""
        from foundry_mcp.tools.authoring import _run_sdd_command, _sdd_cli_breaker

        _sdd_cli_breaker.reset()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("foundry-cli not found")

            with pytest.raises(FileNotFoundError):
                _run_sdd_command(["foundry-cli", "test"], "test_tool")

            assert _sdd_cli_breaker.failure_count == 1

        _sdd_cli_breaker.reset()


# =============================================================================
# spec_create Tool Tests
# =============================================================================


class TestSpecCreate:
    """Test the spec-create tool."""

    def test_basic_spec_creation(self, mock_mcp, mock_config, assert_response_contract):
        """Should create a spec with minimal parameters."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "create"],
                returncode=0,
                stdout=json.dumps({
                    "spec_id": "test-spec-2025-01-01-001",
                    "spec_path": "/test/specs/pending/test-spec-2025-01-01-001.json",
                }),
                stderr="",
            )

            spec_create = mock_mcp._tools["spec_create"]
            result = spec_create(name="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["spec_id"] == "test-spec-2025-01-01-001"
            assert result["data"]["name"] == "test-spec"
            assert result["data"]["template"] == "medium"  # default

    def test_spec_creation_with_all_options(self, mock_mcp, mock_config, assert_response_contract):
        """Should create a spec with all options specified."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "create"],
                returncode=0,
                stdout=json.dumps({
                    "spec_id": "complex-spec-001",
                    "spec_path": "/test/specs/pending/complex-spec-001.json",
                    "structure": {"phases": 5, "tasks": 25},
                }),
                stderr="",
            )

            spec_create = mock_mcp._tools["spec_create"]
            result = spec_create(
                name="complex-spec",
                template="complex",
                category="implementation",
                path="/custom/path",
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["template"] == "complex"
            assert result["data"]["category"] == "implementation"

    def test_invalid_template_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return validation error for invalid template."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        spec_create = mock_mcp._tools["spec_create"]
        result = spec_create(name="test", template="invalid_template")

        assert_response_contract(result)
        assert result["success"] is False
        assert "VALIDATION_ERROR" in str(result["data"].get("error_code", ""))
        assert "invalid_template" in result["error"].lower()

    def test_invalid_category_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return validation error for invalid category."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        spec_create = mock_mcp._tools["spec_create"]
        result = spec_create(name="test", category="invalid_category")

        assert_response_contract(result)
        assert result["success"] is False
        assert "invalid_category" in result["error"].lower()

    def test_duplicate_spec_error(self, mock_mcp, mock_config, assert_response_contract):
        """Should return conflict error for duplicate spec name."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "create"],
                returncode=1,
                stdout="",
                stderr="Spec 'existing-spec' already exists",
            )

            spec_create = mock_mcp._tools["spec_create"]
            result = spec_create(name="existing-spec")

            assert_response_contract(result)
            assert result["success"] is False
            assert "DUPLICATE_ENTRY" in str(result["data"].get("error_code", ""))

    def test_circuit_breaker_error_handling(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle circuit breaker errors gracefully."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.side_effect = CircuitBreakerError(
                "Circuit open",
                breaker_name="sdd_cli_authoring",
                state=CircuitState.OPEN,
                retry_after=30.0,
            )

            spec_create = mock_mcp._tools["spec_create"]
            result = spec_create(name="test")

            assert_response_contract(result)
            assert result["success"] is False
            assert "CIRCUIT_OPEN" in str(result["data"].get("error_code", ""))

    def test_timeout_error_handling(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle timeout errors gracefully."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.side_effect = subprocess.TimeoutExpired(cmd=["foundry-cli"], timeout=30)

            spec_create = mock_mcp._tools["spec_create"]
            result = spec_create(name="test")

            assert_response_contract(result)
            assert result["success"] is False
            assert "TIMEOUT" in str(result["data"].get("error_code", ""))

    def test_cli_not_found_error_handling(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle CLI not found errors gracefully."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.side_effect = FileNotFoundError("foundry-cli not found")

            spec_create = mock_mcp._tools["spec_create"]
            result = spec_create(name="test")

            assert_response_contract(result)
            assert result["success"] is False
            assert "CLI_NOT_FOUND" in str(result["data"].get("error_code", ""))


# =============================================================================
# spec_template Tool Tests
# =============================================================================


class TestSpecTemplate:
    """Test the spec-template tool."""

    def test_list_templates(self, mock_mcp, mock_config, assert_response_contract):
        """Should list available templates."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "template", "list"],
                returncode=0,
                stdout=json.dumps({
                    "templates": [
                        {"name": "simple", "description": "Basic template"},
                        {"name": "medium", "description": "Standard template"},
                    ]
                }),
                stderr="",
            )

            spec_template = mock_mcp._tools["spec_template"]
            result = spec_template(action="list")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["action"] == "list"
            assert result["data"]["total_count"] == 2
            assert len(result["data"]["templates"]) == 2

    def test_show_template(self, mock_mcp, mock_config, assert_response_contract):
        """Should show template details."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "template", "show"],
                returncode=0,
                stdout=json.dumps({
                    "content": {"phases": [], "metadata": {}},
                    "description": "A standard template",
                }),
                stderr="",
            )

            spec_template = mock_mcp._tools["spec_template"]
            result = spec_template(action="show", template_name="medium")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["action"] == "show"
            assert result["data"]["template_name"] == "medium"

    def test_apply_template(self, mock_mcp, mock_config, assert_response_contract):
        """Should apply a template."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "template", "apply"],
                returncode=0,
                stdout=json.dumps({
                    "generated": {"phases": [{"id": "phase-1"}]},
                    "instructions": "Add to your spec",
                }),
                stderr="",
            )

            spec_template = mock_mcp._tools["spec_template"]
            result = spec_template(action="apply", template_name="medium")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["action"] == "apply"

    def test_invalid_action_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return validation error for invalid action."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        spec_template = mock_mcp._tools["spec_template"]
        result = spec_template(action="invalid_action")

        assert_response_contract(result)
        assert result["success"] is False
        assert "invalid_action" in result["error"].lower()

    def test_missing_template_name_for_show(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error when template_name missing for show action."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        spec_template = mock_mcp._tools["spec_template"]
        result = spec_template(action="show")

        assert_response_contract(result)
        assert result["success"] is False
        assert "template_name" in result["error"].lower()

    def test_template_not_found(self, mock_mcp, mock_config, assert_response_contract):
        """Should return not found error for unknown template."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "template", "show"],
                returncode=1,
                stdout="",
                stderr="Template 'unknown' not found",
            )

            spec_template = mock_mcp._tools["spec_template"]
            result = spec_template(action="show", template_name="unknown")

            assert_response_contract(result)
            assert result["success"] is False
            assert "NOT_FOUND" in str(result["data"].get("error_code", ""))


# =============================================================================
# task_add Tool Tests
# =============================================================================


class TestTaskAdd:
    """Test the task-add tool."""

    def test_basic_task_addition(self, mock_mcp, mock_config, assert_response_contract):
        """Should add a task with required parameters."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "add-task"],
                returncode=0,
                stdout=json.dumps({"task_id": "task-1-3"}),
                stderr="",
            )

            task_add = mock_mcp._tools["task_add"]
            result = task_add(
                spec_id="test-spec-001",
                parent="phase-1",
                title="New task",
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["task_id"] == "task-1-3"
            assert result["data"]["parent"] == "phase-1"
            assert result["data"]["title"] == "New task"
            assert result["data"]["type"] == "task"  # default

    def test_task_addition_with_all_options(self, mock_mcp, mock_config, assert_response_contract):
        """Should add a task with all optional parameters."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "add-task"],
                returncode=0,
                stdout=json.dumps({"task_id": "task-1-2-1"}),
                stderr="",
            )

            task_add = mock_mcp._tools["task_add"]
            result = task_add(
                spec_id="test-spec-001",
                parent="task-1-2",
                title="Subtask with details",
                description="Detailed description",
                task_type="subtask",
                hours=4.5,
                position=2,
                dry_run=True,
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["type"] == "subtask"
            assert result["data"]["hours"] == 4.5
            assert result["data"]["position"] == 2
            assert result["data"]["dry_run"] is True

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        task_add = mock_mcp._tools["task_add"]
        result = task_add(spec_id="", parent="phase-1", title="Test")

        assert_response_contract(result)
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_missing_parent_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing parent."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        task_add = mock_mcp._tools["task_add"]
        result = task_add(spec_id="test-spec", parent="", title="Test")

        assert_response_contract(result)
        assert result["success"] is False
        assert "parent" in result["error"].lower()

    def test_missing_title_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing title."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        task_add = mock_mcp._tools["task_add"]
        result = task_add(spec_id="test-spec", parent="phase-1", title="")

        assert_response_contract(result)
        assert result["success"] is False
        assert "title" in result["error"].lower()

    def test_invalid_task_type_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for invalid task type."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        task_add = mock_mcp._tools["task_add"]
        result = task_add(
            spec_id="test-spec",
            parent="phase-1",
            title="Test",
            task_type="invalid_type",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert "invalid_type" in result["error"].lower()

    def test_spec_not_found_error(self, mock_mcp, mock_config, assert_response_contract):
        """Should return spec not found error."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "add-task"],
                returncode=1,
                stdout="",
                stderr="Spec 'unknown-spec' not found",
            )

            task_add = mock_mcp._tools["task_add"]
            result = task_add(
                spec_id="unknown-spec",
                parent="phase-1",
                title="Test",
            )

            assert_response_contract(result)
            assert result["success"] is False
            assert "SPEC_NOT_FOUND" in str(result["data"].get("error_code", ""))

    def test_parent_not_found_error(self, mock_mcp, mock_config, assert_response_contract):
        """Should return parent not found error."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "add-task"],
                returncode=1,
                stdout="",
                stderr="Parent 'unknown-phase' not found in spec",
            )

            task_add = mock_mcp._tools["task_add"]
            result = task_add(
                spec_id="test-spec",
                parent="unknown-phase",
                title="Test",
            )

            assert_response_contract(result)
            assert result["success"] is False
            assert "NOT_FOUND" in str(result["data"].get("error_code", ""))


# =============================================================================
# task_remove Tool Tests
# =============================================================================


class TestTaskRemove:
    """Test the task-remove tool."""

    def test_basic_task_removal(self, mock_mcp, mock_config, assert_response_contract):
        """Should remove a task successfully."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "remove-task"],
                returncode=0,
                stdout=json.dumps({"removed": True}),
                stderr="",
            )

            task_remove = mock_mcp._tools["task_remove"]
            result = task_remove(spec_id="test-spec", task_id="task-1-2")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["task_id"] == "task-1-2"
            assert result["data"]["spec_id"] == "test-spec"
            assert result["data"]["cascade"] is False

    def test_cascade_task_removal(self, mock_mcp, mock_config, assert_response_contract):
        """Should remove task with cascade option."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "remove-task"],
                returncode=0,
                stdout=json.dumps({"removed": True, "children_removed": 3}),
                stderr="",
            )

            task_remove = mock_mcp._tools["task_remove"]
            result = task_remove(
                spec_id="test-spec",
                task_id="task-1-2",
                cascade=True,
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["cascade"] is True
            assert result["data"]["children_removed"] == 3

    def test_dry_run_removal(self, mock_mcp, mock_config, assert_response_contract):
        """Should support dry run for task removal."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "remove-task"],
                returncode=0,
                stdout=json.dumps({"would_remove": True}),
                stderr="",
            )

            task_remove = mock_mcp._tools["task_remove"]
            result = task_remove(
                spec_id="test-spec",
                task_id="task-1-2",
                dry_run=True,
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["dry_run"] is True

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        task_remove = mock_mcp._tools["task_remove"]
        result = task_remove(spec_id="", task_id="task-1-2")

        assert_response_contract(result)
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_missing_task_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing task_id."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        task_remove = mock_mcp._tools["task_remove"]
        result = task_remove(spec_id="test-spec", task_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert "task_id" in result["error"].lower()

    def test_task_has_children_error(self, mock_mcp, mock_config, assert_response_contract):
        """Should return conflict error when task has children."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "remove-task"],
                returncode=1,
                stdout="",
                stderr="Task has children. Use cascade to remove.",
            )

            task_remove = mock_mcp._tools["task_remove"]
            result = task_remove(spec_id="test-spec", task_id="task-1-2")

            assert_response_contract(result)
            assert result["success"] is False
            assert "CONFLICT" in str(result["data"].get("error_code", ""))


# =============================================================================
# assumption_add Tool Tests
# =============================================================================


class TestAssumptionAdd:
    """Test the assumption-add tool."""

    def test_basic_assumption_addition(self, mock_mcp, mock_config, assert_response_contract):
        """Should add an assumption with required parameters."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "add-assumption"],
                returncode=0,
                stdout=json.dumps({"assumption_id": "assumption-1"}),
                stderr="",
            )

            assumption_add = mock_mcp._tools["assumption_add"]
            result = assumption_add(
                spec_id="test-spec",
                text="API rate limits apply",
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["spec_id"] == "test-spec"
            assert result["data"]["text"] == "API rate limits apply"
            assert result["data"]["type"] == "constraint"  # default

    def test_assumption_with_all_options(self, mock_mcp, mock_config, assert_response_contract):
        """Should add assumption with all optional parameters."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "add-assumption"],
                returncode=0,
                stdout=json.dumps({"assumption_id": "assumption-2"}),
                stderr="",
            )

            assumption_add = mock_mcp._tools["assumption_add"]
            result = assumption_add(
                spec_id="test-spec",
                text="Must support Python 3.9+",
                assumption_type="requirement",
                author="test-author",
                dry_run=True,
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["type"] == "requirement"
            assert result["data"]["author"] == "test-author"
            assert result["data"]["dry_run"] is True

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        assumption_add = mock_mcp._tools["assumption_add"]
        result = assumption_add(spec_id="", text="Test assumption")

        assert_response_contract(result)
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_missing_text_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing text."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        assumption_add = mock_mcp._tools["assumption_add"]
        result = assumption_add(spec_id="test-spec", text="")

        assert_response_contract(result)
        assert result["success"] is False
        assert "text" in result["error"].lower()

    def test_invalid_assumption_type_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for invalid assumption type."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        assumption_add = mock_mcp._tools["assumption_add"]
        result = assumption_add(
            spec_id="test-spec",
            text="Test",
            assumption_type="invalid_type",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert "invalid_type" in result["error"].lower()


# =============================================================================
# assumption_list Tool Tests
# =============================================================================


class TestAssumptionList:
    """Test the assumption-list tool."""

    def test_list_all_assumptions(self, mock_mcp, mock_config, assert_response_contract):
        """Should list all assumptions."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "list-assumptions"],
                returncode=0,
                stdout=json.dumps({
                    "assumptions": [
                        {"id": "a1", "text": "Assumption 1", "type": "constraint"},
                        {"id": "a2", "text": "Assumption 2", "type": "requirement"},
                    ]
                }),
                stderr="",
            )

            assumption_list = mock_mcp._tools["assumption_list"]
            result = assumption_list(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["total_count"] == 2
            assert len(result["data"]["assumptions"]) == 2

    def test_filter_by_type(self, mock_mcp, mock_config, assert_response_contract):
        """Should filter assumptions by type."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "list-assumptions"],
                returncode=0,
                stdout=json.dumps({
                    "assumptions": [
                        {"id": "a1", "text": "Constraint 1", "type": "constraint"},
                    ]
                }),
                stderr="",
            )

            assumption_list = mock_mcp._tools["assumption_list"]
            result = assumption_list(
                spec_id="test-spec",
                assumption_type="constraint",
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["filter_type"] == "constraint"

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        assumption_list = mock_mcp._tools["assumption_list"]
        result = assumption_list(spec_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_invalid_type_filter_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for invalid type filter."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        assumption_list = mock_mcp._tools["assumption_list"]
        result = assumption_list(spec_id="test-spec", assumption_type="invalid")

        assert_response_contract(result)
        assert result["success"] is False


# =============================================================================
# revision_add Tool Tests
# =============================================================================


class TestRevisionAdd:
    """Test the revision-add tool."""

    def test_basic_revision_addition(self, mock_mcp, mock_config, assert_response_contract):
        """Should add a revision entry."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "add-revision"],
                returncode=0,
                stdout=json.dumps({"added": True}),
                stderr="",
            )

            revision_add = mock_mcp._tools["revision_add"]
            result = revision_add(
                spec_id="test-spec",
                version="1.1",
                changes="Added new tasks",
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["version"] == "1.1"
            assert result["data"]["changes"] == "Added new tasks"

    def test_revision_with_author(self, mock_mcp, mock_config, assert_response_contract):
        """Should add revision with author."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "add-revision"],
                returncode=0,
                stdout=json.dumps({"added": True}),
                stderr="",
            )

            revision_add = mock_mcp._tools["revision_add"]
            result = revision_add(
                spec_id="test-spec",
                version="2.0",
                changes="Major refactoring",
                author="developer",
                dry_run=True,
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["author"] == "developer"
            assert result["data"]["dry_run"] is True

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        revision_add = mock_mcp._tools["revision_add"]
        result = revision_add(spec_id="", version="1.0", changes="Test")

        assert_response_contract(result)
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_missing_version_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing version."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        revision_add = mock_mcp._tools["revision_add"]
        result = revision_add(spec_id="test-spec", version="", changes="Test")

        assert_response_contract(result)
        assert result["success"] is False
        assert "version" in result["error"].lower()

    def test_missing_changes_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing changes."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        revision_add = mock_mcp._tools["revision_add"]
        result = revision_add(spec_id="test-spec", version="1.0", changes="")

        assert_response_contract(result)
        assert result["success"] is False
        assert "changes" in result["error"].lower()


# =============================================================================
# spec_update_frontmatter Tool Tests
# =============================================================================


class TestSpecUpdateFrontmatter:
    """Test the spec-update-frontmatter tool."""

    def test_update_title(self, mock_mcp, mock_config, assert_response_contract):
        """Should update spec title."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "update-frontmatter"],
                returncode=0,
                stdout=json.dumps({
                    "updated": True,
                    "previous_value": "Old Title",
                }),
                stderr="",
            )

            update = mock_mcp._tools["spec_update_frontmatter"]
            result = update(
                spec_id="test-spec",
                key="title",
                value="New Title",
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["key"] == "title"
            assert result["data"]["value"] == "New Title"
            assert result["data"]["previous_value"] == "Old Title"

    def test_update_status(self, mock_mcp, mock_config, assert_response_contract):
        """Should update spec status."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "update-frontmatter"],
                returncode=0,
                stdout=json.dumps({"updated": True, "old_value": "draft"}),
                stderr="",
            )

            update = mock_mcp._tools["spec_update_frontmatter"]
            result = update(
                spec_id="test-spec",
                key="status",
                value="active",
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["previous_value"] == "draft"

    def test_dry_run_update(self, mock_mcp, mock_config, assert_response_contract):
        """Should support dry run for updates."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "update-frontmatter"],
                returncode=0,
                stdout=json.dumps({"would_update": True}),
                stderr="",
            )

            update = mock_mcp._tools["spec_update_frontmatter"]
            result = update(
                spec_id="test-spec",
                key="version",
                value="2.0.0",
                dry_run=True,
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["dry_run"] is True

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        update = mock_mcp._tools["spec_update_frontmatter"]
        result = update(spec_id="", key="title", value="Test")

        assert_response_contract(result)
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_missing_key_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing key."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        update = mock_mcp._tools["spec_update_frontmatter"]
        result = update(spec_id="test-spec", key="", value="Test")

        assert_response_contract(result)
        assert result["success"] is False
        assert "key" in result["error"].lower()

    def test_invalid_key_error(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for invalid frontmatter key."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "update-frontmatter"],
                returncode=1,
                stdout="",
                stderr="Invalid key 'invalid_key' not found",
            )

            update = mock_mcp._tools["spec_update_frontmatter"]
            result = update(
                spec_id="test-spec",
                key="invalid_key",
                value="test",
            )

            assert_response_contract(result)
            assert result["success"] is False
            assert "INVALID_KEY" in str(result["data"].get("error_code", ""))

    def test_invalid_value_error(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for invalid value."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "update-frontmatter"],
                returncode=1,
                stdout="",
                stderr="Invalid value 'bad' for key 'status'",
            )

            update = mock_mcp._tools["spec_update_frontmatter"]
            result = update(
                spec_id="test-spec",
                key="status",
                value="bad",
            )

            assert_response_contract(result)
            assert result["success"] is False
            assert "VALIDATION_ERROR" in str(result["data"].get("error_code", ""))


# =============================================================================
# Integration Tests for Tool Registration
# =============================================================================


class TestToolRegistration:
    """Test that all tools are properly registered."""

    def test_all_tools_registered(self, mock_mcp, mock_config):
        """All authoring tools should be registered with the MCP server."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        expected_tools = [
            "spec_create",
            "spec_template",
            "task_add",
            "task_remove",
            "assumption_add",
            "assumption_list",
            "revision_add",
            "spec_update_frontmatter",
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_mcp._tools, f"Tool {tool_name} not registered"

    def test_tools_are_callable(self, mock_mcp, mock_config):
        """All registered tools should be callable functions."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        for tool_name, tool_func in mock_mcp._tools.items():
            assert callable(tool_func), f"Tool {tool_name} is not callable"


# =============================================================================
# Response Contract Compliance Tests
# =============================================================================


class TestResponseContractCompliance:
    """Test that all responses comply with the response-v2 contract."""

    def test_success_response_structure(self, mock_mcp, mock_config, assert_response_contract):
        """Success responses should have correct structure."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.authoring._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "create"],
                returncode=0,
                stdout=json.dumps({"spec_id": "test-001"}),
                stderr="",
            )

            spec_create = mock_mcp._tools["spec_create"]
            result = spec_create(name="test")

            # Validate structure
            assert "success" in result
            assert "data" in result
            assert "error" in result
            assert "meta" in result
            assert result["meta"]["version"] == "response-v2"

    def test_error_response_structure(self, mock_mcp, mock_config, assert_response_contract):
        """Error responses should have correct structure."""
        from foundry_mcp.tools.authoring import register_authoring_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_authoring_tools(mock_mcp, mock_config)

        # Trigger validation error
        task_add = mock_mcp._tools["task_add"]
        result = task_add(spec_id="", parent="phase-1", title="Test")

        # Validate structure
        assert result["success"] is False
        assert result["error"] is not None
        assert "error_code" in result["data"]
        assert "error_type" in result["data"]
        assert "remediation" in result["data"]
        assert result["meta"]["version"] == "response-v2"
