"""
Unit tests for foundry_mcp.tools.mutations and git_integration modules.

Tests the mutation and git integration tools for SDD specifications,
including circuit breaker protection, validation, and response contracts.
"""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.resilience import CircuitBreakerError, CircuitState


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_mcp():
    """Create a mock FastMCP server instance."""
    mcp = MagicMock()
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


# =============================================================================
# Mutations Module Tests - _run_sdd_command
# =============================================================================


class TestMutationsRunSddCommand:
    """Test the _run_sdd_command helper function in mutations module."""

    def test_successful_command_execution(self):
        """Successful command should return result and record success."""
        from foundry_mcp.tools.mutations import _run_sdd_command, _sdd_cli_breaker

        _sdd_cli_breaker.reset()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["sdd", "test"],
                returncode=0,
                stdout='{"result": "success"}',
                stderr="",
            )

            result = _run_sdd_command(["sdd", "test"], "test_tool")

            assert result.returncode == 0
            assert '{"result": "success"}' in result.stdout
            mock_run.assert_called_once()

    def test_circuit_breaker_open_raises_error(self):
        """When circuit breaker is open, should raise CircuitBreakerError."""
        from foundry_mcp.tools.mutations import _run_sdd_command, _sdd_cli_breaker

        _sdd_cli_breaker.reset()

        for _ in range(5):
            _sdd_cli_breaker.record_failure()

        assert _sdd_cli_breaker.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerError) as exc_info:
            _run_sdd_command(["sdd", "test"], "test_tool")

        assert exc_info.value.breaker_name == "sdd_cli_mutations"
        _sdd_cli_breaker.reset()


# =============================================================================
# spec_apply_plan Tool Tests
# =============================================================================


class TestSpecApplyPlan:
    """Test the spec-apply-plan tool."""

    def test_basic_plan_application(self, mock_mcp, mock_config, assert_response_contract):
        """Should apply modifications successfully."""
        from foundry_mcp.tools.mutations import register_mutation_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_mutation_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.mutations._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["sdd", "apply-modifications"],
                returncode=0,
                stdout=json.dumps({
                    "modifications_applied": 5,
                    "modifications_skipped": 0,
                }),
                stderr="",
            )

            spec_apply_plan = mock_mcp._tools["spec_apply_plan"]
            result = spec_apply_plan(
                spec_id="test-spec",
                modifications_file="/path/to/mods.json",
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["modifications_applied"] == 5

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.mutations import register_mutation_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_mutation_tools(mock_mcp, mock_config)

        spec_apply_plan = mock_mcp._tools["spec_apply_plan"]
        result = spec_apply_plan(spec_id="", modifications_file="/path/to/file.json")

        assert_response_contract(result)
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_missing_modifications_file_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing modifications_file."""
        from foundry_mcp.tools.mutations import register_mutation_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_mutation_tools(mock_mcp, mock_config)

        spec_apply_plan = mock_mcp._tools["spec_apply_plan"]
        result = spec_apply_plan(spec_id="test-spec", modifications_file="")

        assert_response_contract(result)
        assert result["success"] is False
        assert "modifications_file" in result["error"].lower()


# =============================================================================
# verification_add Tool Tests
# =============================================================================


class TestVerificationAdd:
    """Test the verification-add tool."""

    def test_basic_verification_addition(self, mock_mcp, mock_config, assert_response_contract):
        """Should add verification successfully."""
        from foundry_mcp.tools.mutations import register_mutation_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_mutation_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.mutations._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["sdd", "add-verification"],
                returncode=0,
                stdout=json.dumps({"added": True}),
                stderr="",
            )

            verification_add = mock_mcp._tools["verification_add"]
            result = verification_add(
                spec_id="test-spec",
                verify_id="verify-1-1",
                result="PASSED",
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["result"] == "PASSED"

    def test_invalid_result_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for invalid result."""
        from foundry_mcp.tools.mutations import register_mutation_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_mutation_tools(mock_mcp, mock_config)

        verification_add = mock_mcp._tools["verification_add"]
        result = verification_add(
            spec_id="test-spec",
            verify_id="verify-1-1",
            result="INVALID",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert "INVALID" in result["error"]


# =============================================================================
# verification_execute Tool Tests
# =============================================================================


class TestVerificationExecute:
    """Test the verification-execute tool."""

    def test_basic_execution(self, mock_mcp, mock_config, assert_response_contract):
        """Should execute verification successfully."""
        from foundry_mcp.tools.mutations import register_mutation_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_mutation_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.mutations._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["sdd", "execute-verify"],
                returncode=0,
                stdout=json.dumps({
                    "result": "PASSED",
                    "command": "pytest tests/",
                    "exit_code": 0,
                }),
                stderr="",
            )

            verification_execute = mock_mcp._tools["verification_execute"]
            result = verification_execute(
                spec_id="test-spec",
                verify_id="verify-1-1",
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["result"] == "PASSED"


# =============================================================================
# task_update_estimate Tool Tests
# =============================================================================


class TestTaskUpdateEstimate:
    """Test the task-update-estimate tool."""

    def test_basic_estimate_update(self, mock_mcp, mock_config, assert_response_contract):
        """Should update estimate successfully."""
        from foundry_mcp.tools.mutations import register_mutation_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_mutation_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.mutations._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["sdd", "update-estimate"],
                returncode=0,
                stdout=json.dumps({"updated": True}),
                stderr="",
            )

            task_update_estimate = mock_mcp._tools["task_update_estimate"]
            result = task_update_estimate(
                spec_id="test-spec",
                task_id="task-1-1",
                hours=4.5,
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["hours"] == 4.5

    def test_no_update_fields_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error when no update fields provided."""
        from foundry_mcp.tools.mutations import register_mutation_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_mutation_tools(mock_mcp, mock_config)

        task_update_estimate = mock_mcp._tools["task_update_estimate"]
        result = task_update_estimate(
            spec_id="test-spec",
            task_id="task-1-1",
        )

        assert_response_contract(result)
        assert result["success"] is False


# =============================================================================
# task_update_metadata Tool Tests
# =============================================================================


class TestTaskUpdateMetadata:
    """Test the task-update-metadata tool."""

    def test_basic_metadata_update(self, mock_mcp, mock_config, assert_response_contract):
        """Should update metadata successfully."""
        from foundry_mcp.tools.mutations import register_mutation_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_mutation_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.mutations._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["sdd", "update-task-metadata"],
                returncode=0,
                stdout=json.dumps({"updated": True}),
                stderr="",
            )

            task_update_metadata = mock_mcp._tools["task_update_metadata"]
            result = task_update_metadata(
                spec_id="test-spec",
                task_id="task-1-1",
                file_path="src/module.py",
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert "file_path" in result["data"]["fields_updated"]


# =============================================================================
# spec_sync_metadata Tool Tests
# =============================================================================


class TestSpecSyncMetadata:
    """Test the spec-sync-metadata tool."""

    def test_basic_sync(self, mock_mcp, mock_config, assert_response_contract):
        """Should sync metadata successfully."""
        from foundry_mcp.tools.mutations import register_mutation_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_mutation_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.mutations._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["sdd", "sync-metadata"],
                returncode=0,
                stdout=json.dumps({"synced": True, "changes": ["title", "status"]}),
                stderr="",
            )

            spec_sync_metadata = mock_mcp._tools["spec_sync_metadata"]
            result = spec_sync_metadata(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["synced"] is True


# =============================================================================
# Git Integration Module Tests - _run_sdd_command
# =============================================================================


class TestGitIntegrationRunSddCommand:
    """Test the _run_sdd_command helper function in git_integration module."""

    def test_successful_command_execution(self):
        """Successful command should return result and record success."""
        from foundry_mcp.tools.git_integration import _run_sdd_command, _sdd_cli_breaker

        _sdd_cli_breaker.reset()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["sdd", "test"],
                returncode=0,
                stdout='{"result": "success"}',
                stderr="",
            )

            result = _run_sdd_command(["sdd", "test"], "test_tool")

            assert result.returncode == 0
            mock_run.assert_called_once()

    def test_circuit_breaker_open_raises_error(self):
        """When circuit breaker is open, should raise CircuitBreakerError."""
        from foundry_mcp.tools.git_integration import _run_sdd_command, _sdd_cli_breaker

        _sdd_cli_breaker.reset()

        for _ in range(5):
            _sdd_cli_breaker.record_failure()

        assert _sdd_cli_breaker.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerError) as exc_info:
            _run_sdd_command(["sdd", "test"], "test_tool")

        assert exc_info.value.breaker_name == "sdd_cli_git_integration"
        _sdd_cli_breaker.reset()


# =============================================================================
# task_create_commit Tool Tests
# =============================================================================


class TestTaskCreateCommit:
    """Test the task-create-commit tool."""

    def test_basic_commit_creation(self, mock_mcp, mock_config, assert_response_contract):
        """Should create commit successfully."""
        from foundry_mcp.tools.git_integration import register_git_integration_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_git_integration_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.git_integration._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["sdd", "create-task-commit"],
                returncode=0,
                stdout=json.dumps({
                    "commit_hash": "abc123",
                    "commit_message": "task-1-1: Implement feature",
                    "files_committed": ["src/module.py"],
                }),
                stderr="",
            )

            task_create_commit = mock_mcp._tools["task_create_commit"]
            result = task_create_commit(
                spec_id="test-spec",
                task_id="task-1-1",
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["commit_hash"] == "abc123"

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.git_integration import register_git_integration_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_git_integration_tools(mock_mcp, mock_config)

        task_create_commit = mock_mcp._tools["task_create_commit"]
        result = task_create_commit(spec_id="", task_id="task-1-1")

        assert_response_contract(result)
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_task_not_completed_error(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error when task is not completed."""
        from foundry_mcp.tools.git_integration import register_git_integration_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_git_integration_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.git_integration._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["sdd", "create-task-commit"],
                returncode=1,
                stdout="",
                stderr="Task 'task-1-1' is not completed",
            )

            task_create_commit = mock_mcp._tools["task_create_commit"]
            result = task_create_commit(
                spec_id="test-spec",
                task_id="task-1-1",
            )

            assert_response_contract(result)
            assert result["success"] is False
            assert "TASK_NOT_COMPLETED" in str(result["data"].get("error_code", ""))


# =============================================================================
# journal_bulk_add Tool Tests
# =============================================================================


class TestJournalBulkAdd:
    """Test the journal-bulk-add tool."""

    def test_basic_bulk_journal(self, mock_mcp, mock_config, assert_response_contract):
        """Should bulk add journal entries successfully."""
        from foundry_mcp.tools.git_integration import register_git_integration_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_git_integration_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.git_integration._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["sdd", "bulk-journal"],
                returncode=0,
                stdout=json.dumps({
                    "tasks_journaled": 3,
                    "task_ids": ["task-1-1", "task-1-2", "task-1-3"],
                }),
                stderr="",
            )

            journal_bulk_add = mock_mcp._tools["journal_bulk_add"]
            result = journal_bulk_add(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["tasks_journaled"] == 3

    def test_with_template(self, mock_mcp, mock_config, assert_response_contract):
        """Should apply template successfully."""
        from foundry_mcp.tools.git_integration import register_git_integration_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_git_integration_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.git_integration._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["sdd", "bulk-journal"],
                returncode=0,
                stdout=json.dumps({"tasks_journaled": 2}),
                stderr="",
            )

            journal_bulk_add = mock_mcp._tools["journal_bulk_add"]
            result = journal_bulk_add(
                spec_id="test-spec",
                template="completion",
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["template_used"] == "completion"

    def test_invalid_template_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for invalid template."""
        from foundry_mcp.tools.git_integration import register_git_integration_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_git_integration_tools(mock_mcp, mock_config)

        journal_bulk_add = mock_mcp._tools["journal_bulk_add"]
        result = journal_bulk_add(
            spec_id="test-spec",
            template="invalid_template",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert "invalid_template" in result["error"].lower()


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestMutationToolRegistration:
    """Test that all mutation tools are properly registered."""

    def test_all_mutation_tools_registered(self, mock_mcp, mock_config):
        """All mutation tools should be registered with the MCP server."""
        from foundry_mcp.tools.mutations import register_mutation_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_mutation_tools(mock_mcp, mock_config)

        expected_tools = [
            "spec_apply_plan",
            "verification_add",
            "verification_execute",
            "verification_format_summary",
            "task_update_estimate",
            "task_update_metadata",
            "spec_sync_metadata",
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_mcp._tools, f"Tool {tool_name} not registered"


class TestGitIntegrationToolRegistration:
    """Test that all git integration tools are properly registered."""

    def test_all_git_tools_registered(self, mock_mcp, mock_config):
        """All git integration tools should be registered."""
        from foundry_mcp.tools.git_integration import register_git_integration_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_git_integration_tools(mock_mcp, mock_config)

        expected_tools = [
            "task_create_commit",
            "journal_bulk_add",
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_mcp._tools, f"Tool {tool_name} not registered"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling across mutation tools."""

    def test_circuit_breaker_error_handling(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle circuit breaker errors gracefully."""
        from foundry_mcp.tools.mutations import register_mutation_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_mutation_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.mutations._run_sdd_command") as mock_cmd:
            mock_cmd.side_effect = CircuitBreakerError(
                "Circuit open",
                breaker_name="sdd_cli_mutations",
                state=CircuitState.OPEN,
                retry_after=30.0,
            )

            spec_sync_metadata = mock_mcp._tools["spec_sync_metadata"]
            result = spec_sync_metadata(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is False
            assert "CIRCUIT_OPEN" in str(result["data"].get("error_code", ""))

    def test_timeout_error_handling(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle timeout errors gracefully."""
        from foundry_mcp.tools.mutations import register_mutation_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_mutation_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.mutations._run_sdd_command") as mock_cmd:
            mock_cmd.side_effect = subprocess.TimeoutExpired(cmd=["sdd"], timeout=30)

            spec_sync_metadata = mock_mcp._tools["spec_sync_metadata"]
            result = spec_sync_metadata(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is False
            assert "TIMEOUT" in str(result["data"].get("error_code", ""))

    def test_cli_not_found_error_handling(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle CLI not found errors gracefully."""
        from foundry_mcp.tools.mutations import register_mutation_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_mutation_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.mutations._run_sdd_command") as mock_cmd:
            mock_cmd.side_effect = FileNotFoundError("sdd not found")

            spec_sync_metadata = mock_mcp._tools["spec_sync_metadata"]
            result = spec_sync_metadata(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is False
            assert "CLI_NOT_FOUND" in str(result["data"].get("error_code", ""))
