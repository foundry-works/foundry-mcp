"""
Unit tests for foundry_mcp.tools.planning module.

Tests the planning tools for SDD specifications, including:
- plan_format - Format task plans for sharing
- phase_list - Enumerate phases in a spec
- phase_check_complete - Verify completion readiness
- phase_report_time - Time tracking per phase
- spec_reconcile_state - Filesystem vs spec drift detection
- plan_report_time - Aggregate time tracking reports
- spec_audit - Quality audits on specifications

Includes circuit breaker protection, validation, and response contract tests.
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
# _run_sdd_command Tests
# =============================================================================


class TestPlanningRunSddCommand:
    """Test the _run_sdd_command helper function in planning module."""

    def test_successful_command_execution(self):
        """Successful command should return result and record success."""
        from foundry_mcp.tools.planning import _run_sdd_command, _sdd_cli_breaker

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
        """Failed command should record failure for circuit breaker."""
        from foundry_mcp.tools.planning import _run_sdd_command, _sdd_cli_breaker

        _sdd_cli_breaker.reset()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "test"],
                returncode=1,
                stdout="",
                stderr="Error message",
            )

            result = _run_sdd_command(["foundry-cli", "test"], "test_tool")

            assert result.returncode == 1
            # Failure was recorded (public attribute, no underscore prefix)
            assert _sdd_cli_breaker.failure_count > 0

        _sdd_cli_breaker.reset()

    def test_circuit_breaker_open_raises_error(self):
        """When circuit breaker is open, should raise CircuitBreakerError."""
        from foundry_mcp.tools.planning import _run_sdd_command, _sdd_cli_breaker

        _sdd_cli_breaker.reset()

        # Trip the circuit breaker
        for _ in range(5):
            _sdd_cli_breaker.record_failure()

        assert _sdd_cli_breaker.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerError) as exc_info:
            _run_sdd_command(["foundry-cli", "test"], "test_tool")

        assert exc_info.value.breaker_name == "sdd_cli_planning"
        _sdd_cli_breaker.reset()


# =============================================================================
# plan_format Tool Tests
# =============================================================================


class TestPlanFormat:
    """Test the plan-format tool."""

    def test_basic_plan_formatting(self, mock_mcp, mock_config, assert_response_contract):
        """Should format plan successfully."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "format-plan"],
                returncode=0,
                stdout=json.dumps({
                    "formatted": "# Task 1-1: Implement feature\n\n## Description\n...",
                    "title": "Implement feature",
                    "status": "pending",
                }),
                stderr="",
            )

            plan_format = mock_mcp._tools["plan_format"]
            result = plan_format(
                spec_id="test-spec",
                task_id="task-1-1",
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert "formatted" in result["data"]
            assert result["data"]["spec_id"] == "test-spec"
            assert result["data"]["task_id"] == "task-1-1"

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        plan_format = mock_mcp._tools["plan_format"]
        result = plan_format(spec_id="", task_id="task-1-1")

        assert_response_contract(result)
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_missing_task_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing task_id."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        plan_format = mock_mcp._tools["plan_format"]
        result = plan_format(spec_id="test-spec", task_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert "task_id" in result["error"].lower()

    def test_not_found_error(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle not found error."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "format-plan"],
                returncode=1,
                stdout="",
                stderr="Spec 'test-spec' not found",
            )

            plan_format = mock_mcp._tools["plan_format"]
            result = plan_format(spec_id="test-spec", task_id="task-1-1")

            assert_response_contract(result)
            assert result["success"] is False
            assert "NOT_FOUND" in str(result["data"].get("error_code", ""))


# =============================================================================
# phase_list Tool Tests
# =============================================================================


class TestPhaseList:
    """Test the phase-list tool."""

    def test_basic_phase_listing(self, mock_mcp, mock_config, assert_response_contract):
        """Should list phases successfully."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "list-phases"],
                returncode=0,
                stdout=json.dumps({
                    "phases": [
                        {"id": "phase-1", "title": "Setup", "status": "completed"},
                        {"id": "phase-2", "title": "Implementation", "status": "in_progress"},
                        {"id": "phase-3", "title": "Testing", "status": "pending"},
                    ]
                }),
                stderr="",
            )

            phase_list = mock_mcp._tools["phase_list"]
            result = phase_list(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["total_phases"] == 3
            assert result["data"]["completed_phases"] == 1
            assert len(result["data"]["phases"]) == 3

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        phase_list = mock_mcp._tools["phase_list"]
        result = phase_list(spec_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_empty_phases(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle specs with no phases."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "list-phases"],
                returncode=0,
                stdout=json.dumps({"phases": []}),
                stderr="",
            )

            phase_list = mock_mcp._tools["phase_list"]
            result = phase_list(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["total_phases"] == 0
            assert result["data"]["completed_phases"] == 0


# =============================================================================
# phase_check_complete Tool Tests
# =============================================================================


class TestPhaseCheckComplete:
    """Test the phase-check-complete tool."""

    def test_spec_complete(self, mock_mcp, mock_config, assert_response_contract):
        """Should verify spec completion."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "check-complete"],
                returncode=0,
                stdout=json.dumps({
                    "is_complete": True,
                    "total_tasks": 10,
                    "completed_tasks": 10,
                    "pending_tasks": [],
                    "blocked_tasks": [],
                }),
                stderr="",
            )

            phase_check_complete = mock_mcp._tools["phase_check_complete"]
            result = phase_check_complete(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["is_complete"] is True
            assert result["data"]["scope"] == "spec"
            assert result["data"]["total_tasks"] == 10

    def test_phase_incomplete(self, mock_mcp, mock_config, assert_response_contract):
        """Should detect incomplete phase."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "check-complete"],
                returncode=0,
                stdout=json.dumps({
                    "is_complete": False,
                    "total_tasks": 5,
                    "completed_tasks": 3,
                    "pending_tasks": ["task-2-4", "task-2-5"],
                    "blocked_tasks": [],
                }),
                stderr="",
            )

            phase_check_complete = mock_mcp._tools["phase_check_complete"]
            result = phase_check_complete(spec_id="test-spec", phase_id="phase-2")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["is_complete"] is False
            assert result["data"]["scope"] == "phase"
            assert result["data"]["phase_id"] == "phase-2"
            assert len(result["data"]["pending_tasks"]) == 2

    def test_mutual_exclusivity_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error when both phase_id and task_id provided."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        phase_check_complete = mock_mcp._tools["phase_check_complete"]
        result = phase_check_complete(
            spec_id="test-spec",
            phase_id="phase-1",
            task_id="task-1-1",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert "mutually exclusive" in result["error"].lower()


# =============================================================================
# phase_report_time Tool Tests
# =============================================================================


class TestPhaseReportTime:
    """Test the phase-report-time tool."""

    def test_basic_time_report(self, mock_mcp, mock_config, assert_response_contract):
        """Should report phase time successfully."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "phase-time"],
                returncode=0,
                stdout=json.dumps({
                    "estimated_hours": 20,
                    "actual_hours": 25,
                    "task_count": 10,
                    "completed_count": 8,
                    "phase_title": "Implementation",
                }),
                stderr="",
            )

            phase_report_time = mock_mcp._tools["phase_report_time"]
            result = phase_report_time(spec_id="test-spec", phase_id="phase-2")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["estimated_hours"] == 20
            assert result["data"]["actual_hours"] == 25
            assert result["data"]["variance_hours"] == 5
            assert result["data"]["variance_percent"] == 25.0

    def test_missing_phase_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing phase_id."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        phase_report_time = mock_mcp._tools["phase_report_time"]
        result = phase_report_time(spec_id="test-spec", phase_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert "phase_id" in result["error"].lower()

    def test_zero_estimated_variance(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle zero estimated hours without division error."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "phase-time"],
                returncode=0,
                stdout=json.dumps({
                    "estimated_hours": 0,
                    "actual_hours": 5,
                    "task_count": 2,
                    "completed_count": 2,
                }),
                stderr="",
            )

            phase_report_time = mock_mcp._tools["phase_report_time"]
            result = phase_report_time(spec_id="test-spec", phase_id="phase-1")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["variance_percent"] == 0  # No division by zero


# =============================================================================
# spec_reconcile_state Tool Tests
# =============================================================================


class TestSpecReconcileState:
    """Test the spec-reconcile-state tool."""

    def test_no_drift_detected(self, mock_mcp, mock_config, assert_response_contract):
        """Should report no drift when filesystem matches spec."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "reconcile-state"],
                returncode=0,
                stdout=json.dumps({
                    "modified_files": [],
                    "new_files": [],
                    "missing_files": [],
                }),
                stderr="",
            )

            spec_reconcile_state = mock_mcp._tools["spec_reconcile_state"]
            result = spec_reconcile_state(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["has_drift"] is False
            assert result["data"]["drift_count"] == 0

    def test_drift_detected(self, mock_mcp, mock_config, assert_response_contract):
        """Should report drift when differences found."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "reconcile-state"],
                returncode=0,
                stdout=json.dumps({
                    "modified_files": ["src/module.py"],
                    "new_files": ["src/new_file.py"],
                    "missing_files": ["src/deleted.py"],
                    "recommendations": ["Update spec to reflect current state"],
                }),
                stderr="",
            )

            spec_reconcile_state = mock_mcp._tools["spec_reconcile_state"]
            result = spec_reconcile_state(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["has_drift"] is True
            assert result["data"]["drift_count"] == 3
            assert len(result["data"]["modified_files"]) == 1
            assert len(result["data"]["new_files"]) == 1
            assert len(result["data"]["missing_files"]) == 1
            assert "recommendations" in result["data"]

    def test_dry_run_mode(self, mock_mcp, mock_config, assert_response_contract):
        """Should support dry run mode."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "reconcile-state"],
                returncode=0,
                stdout=json.dumps({"modified_files": [], "new_files": [], "missing_files": []}),
                stderr="",
            )

            spec_reconcile_state = mock_mcp._tools["spec_reconcile_state"]
            result = spec_reconcile_state(spec_id="test-spec", dry_run=True)

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["dry_run"] is True

            # Verify --dry-run was passed
            call_args = mock_cmd.call_args[0][0]
            assert "--dry-run" in call_args


# =============================================================================
# plan_report_time Tool Tests
# =============================================================================


class TestPlanReportTime:
    """Test the plan-report-time tool."""

    def test_basic_aggregate_report(self, mock_mcp, mock_config, assert_response_contract):
        """Should generate aggregate time report."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "time-report"],
                returncode=0,
                stdout=json.dumps({
                    "total_estimated_hours": 100,
                    "total_actual_hours": 110,
                    "total_tasks": 50,
                    "completed_tasks": 40,
                    "spec_title": "Feature Implementation",
                    "phases": [
                        {"phase_id": "phase-1", "estimated": 30, "actual": 25},
                        {"phase_id": "phase-2", "estimated": 40, "actual": 50},
                        {"phase_id": "phase-3", "estimated": 30, "actual": 35},
                    ],
                }),
                stderr="",
            )

            plan_report_time = mock_mcp._tools["plan_report_time"]
            result = plan_report_time(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["total_estimated_hours"] == 100
            assert result["data"]["total_actual_hours"] == 110
            assert result["data"]["total_variance_hours"] == 10
            assert result["data"]["total_variance_percent"] == 10.0
            assert result["data"]["completion_rate"] == 80.0
            assert len(result["data"]["phases"]) == 3

    def test_no_tasks_completion_rate(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle zero tasks without division error."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "time-report"],
                returncode=0,
                stdout=json.dumps({
                    "total_estimated_hours": 0,
                    "total_actual_hours": 0,
                    "total_tasks": 0,
                    "completed_tasks": 0,
                }),
                stderr="",
            )

            plan_report_time = mock_mcp._tools["plan_report_time"]
            result = plan_report_time(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["completion_rate"] == 0


# =============================================================================
# spec_audit Tool Tests
# =============================================================================


class TestSpecAudit:
    """Test the spec-audit tool."""

    def test_audit_passed(self, mock_mcp, mock_config, assert_response_contract):
        """Should report passed audit."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "audit-spec"],
                returncode=0,
                stdout=json.dumps({
                    "passed": True,
                    "score": 95,
                    "findings": [
                        {"severity": "warning", "message": "Consider adding more tests"},
                    ],
                    "recommendations": ["Add integration tests"],
                    "categories": {"completeness": 100, "quality": 90},
                }),
                stderr="",
            )

            spec_audit = mock_mcp._tools["spec_audit"]
            result = spec_audit(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["passed"] is True
            assert result["data"]["score"] == 95
            assert result["data"]["error_count"] == 0
            assert result["data"]["warning_count"] == 1

    def test_audit_failed(self, mock_mcp, mock_config, assert_response_contract):
        """Should report failed audit with errors."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "audit-spec"],
                returncode=0,
                stdout=json.dumps({
                    "passed": False,
                    "score": 45,
                    "findings": [
                        {"severity": "error", "message": "Missing required field"},
                        {"severity": "error", "message": "Invalid task dependency"},
                        {"severity": "warning", "message": "No verification defined"},
                    ],
                    "recommendations": ["Fix dependency cycle", "Add verifications"],
                }),
                stderr="",
            )

            spec_audit = mock_mcp._tools["spec_audit"]
            result = spec_audit(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["passed"] is False
            assert result["data"]["score"] == 45
            assert result["data"]["error_count"] == 2
            assert result["data"]["warning_count"] == 1

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        spec_audit = mock_mcp._tools["spec_audit"]
        result = spec_audit(spec_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestPlanningToolRegistration:
    """Test that all planning tools are properly registered."""

    def test_all_planning_tools_registered(self, mock_mcp, mock_config):
        """All planning tools should be registered with the MCP server."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        expected_tools = [
            "plan_format",
            "phase_list",
            "phase_check_complete",
            "phase_report_time",
            "spec_reconcile_state",
            "plan_report_time",
            "spec_audit",
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_mcp._tools, f"Tool {tool_name} not registered"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling across planning tools."""

    def test_circuit_breaker_error_handling(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle circuit breaker errors gracefully."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.side_effect = CircuitBreakerError(
                "Circuit open",
                breaker_name="sdd_cli_planning",
                state=CircuitState.OPEN,
                retry_after=30.0,
            )

            phase_list = mock_mcp._tools["phase_list"]
            result = phase_list(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is False
            assert "CIRCUIT_OPEN" in str(result["data"].get("error_code", ""))

    def test_timeout_error_handling(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle timeout errors gracefully."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.side_effect = subprocess.TimeoutExpired(cmd=["foundry-cli"], timeout=30)

            plan_report_time = mock_mcp._tools["plan_report_time"]
            result = plan_report_time(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is False
            assert "TIMEOUT" in str(result["data"].get("error_code", ""))

    def test_cli_not_found_error_handling(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle CLI not found errors gracefully."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.side_effect = FileNotFoundError("foundry-cli not found")

            spec_audit = mock_mcp._tools["spec_audit"]
            result = spec_audit(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is False
            assert "CLI_NOT_FOUND" in str(result["data"].get("error_code", ""))

    def test_generic_exception_handling(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle unexpected exceptions gracefully."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.side_effect = RuntimeError("Unexpected error")

            spec_reconcile_state = mock_mcp._tools["spec_reconcile_state"]
            result = spec_reconcile_state(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is False
            assert "INTERNAL_ERROR" in str(result["data"].get("error_code", ""))


# =============================================================================
# Path Parameter Tests
# =============================================================================


class TestPathParameter:
    """Test that path parameter is correctly passed to all tools."""

    def test_plan_format_with_path(self, mock_mcp, mock_config, assert_response_contract):
        """plan_format should pass path to CLI."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "format-plan"],
                returncode=0,
                stdout=json.dumps({"formatted": "test"}),
                stderr="",
            )

            plan_format = mock_mcp._tools["plan_format"]
            plan_format(spec_id="test-spec", task_id="task-1", path="/custom/path")

            call_args = mock_cmd.call_args[0][0]
            assert "--path" in call_args
            assert "/custom/path" in call_args

    def test_phase_list_with_path(self, mock_mcp, mock_config, assert_response_contract):
        """phase_list should pass path to CLI."""
        from foundry_mcp.tools.planning import register_planning_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_planning_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.planning._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "list-phases"],
                returncode=0,
                stdout=json.dumps({"phases": []}),
                stderr="",
            )

            phase_list = mock_mcp._tools["phase_list"]
            phase_list(spec_id="test-spec", path="/custom/path")

            call_args = mock_cmd.call_args[0][0]
            assert "--path" in call_args
            assert "/custom/path" in call_args
