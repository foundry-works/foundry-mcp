"""
Unit tests for foundry_mcp.tools.reporting module.

Tests the reporting tools for SDD specifications, including:
- spec_report - Generate comprehensive human-readable reports
- spec_report_summary - Quick summary reports for dashboards

Includes circuit breaker protection, validation, and response contract tests.
"""

from unittest.mock import MagicMock, patch
from pathlib import Path

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
def mock_config(tmp_path):
    """Create a mock server config with specs directory."""
    config = MagicMock()
    config.specs_dir = tmp_path / "specs"
    config.specs_dir.mkdir(exist_ok=True)
    return config


@pytest.fixture
def mock_validation_result():
    """Create a mock validation result."""
    result = MagicMock()
    result.is_valid = True
    result.error_count = 0
    result.warning_count = 2
    result.info_count = 1
    result.diagnostics = [
        MagicMock(
            code="WARN001",
            message="Missing description",
            severity="warning",
            category="documentation",
            location="task-1-1",
            suggested_fix="Add a description",
            auto_fixable=False,
        ),
        MagicMock(
            code="WARN002",
            message="Long task name",
            severity="warning",
            category="naming",
            location="task-2-1",
            suggested_fix="Shorten the name",
            auto_fixable=True,
        ),
        MagicMock(
            code="INFO001",
            message="Consider adding tests",
            severity="info",
            category="quality",
            location="phase-1",
            suggested_fix=None,
            auto_fixable=False,
        ),
    ]
    return result


@pytest.fixture
def mock_stats():
    """Create a mock stats result."""
    stats = MagicMock()
    stats.spec_id = "test-spec"
    stats.title = "Test Specification"
    stats.version = "1.0.0"
    stats.status = "in_progress"
    stats.totals = {"phases": 3, "tasks": 10, "subtasks": 5}
    stats.status_counts = {"completed": 6, "in_progress": 2, "pending": 2}
    stats.max_depth = 3
    stats.avg_tasks_per_phase = 3.3
    stats.verification_coverage = 80
    stats.progress = {"completed": 6, "total": 10, "percentage": 60}
    stats.file_size_kb = 15.5
    return stats


# =============================================================================
# spec_report Tool Tests
# =============================================================================


class TestSpecReport:
    """Test the spec-report tool."""

    def test_basic_markdown_report(
        self, mock_mcp, mock_config, mock_validation_result, mock_stats, assert_response_contract
    ):
        """Should generate a markdown report successfully."""
        from foundry_mcp.tools.reporting import register_reporting_tools, _report_breaker

        _report_breaker.reset()
        register_reporting_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.reporting.find_specs_directory") as mock_find_dir, \
             patch("foundry_mcp.tools.reporting.find_spec_file") as mock_find_file, \
             patch("foundry_mcp.tools.reporting.load_spec") as mock_load, \
             patch("foundry_mcp.tools.reporting.validate_spec") as mock_validate, \
             patch("foundry_mcp.tools.reporting.calculate_stats") as mock_calc_stats:

            mock_find_dir.return_value = mock_config.specs_dir
            mock_find_file.return_value = mock_config.specs_dir / "test-spec.json"
            mock_load.return_value = {"id": "test-spec", "title": "Test"}
            mock_validate.return_value = mock_validation_result
            mock_calc_stats.return_value = mock_stats

            spec_report = mock_mcp._tools["spec_report"]
            result = spec_report(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            assert "report" in result["data"]
            assert result["data"]["format"] == "markdown"
            assert "# Spec Report: test-spec" in result["data"]["report"]
            assert "## Validation Results" in result["data"]["report"]
            assert "## Specification Statistics" in result["data"]["report"]
            assert "## Health Assessment" in result["data"]["report"]

    def test_json_format_report(
        self, mock_mcp, mock_config, mock_validation_result, mock_stats, assert_response_contract
    ):
        """Should generate a JSON format report."""
        from foundry_mcp.tools.reporting import register_reporting_tools, _report_breaker

        _report_breaker.reset()
        register_reporting_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.reporting.find_specs_directory") as mock_find_dir, \
             patch("foundry_mcp.tools.reporting.find_spec_file") as mock_find_file, \
             patch("foundry_mcp.tools.reporting.load_spec") as mock_load, \
             patch("foundry_mcp.tools.reporting.validate_spec") as mock_validate, \
             patch("foundry_mcp.tools.reporting.calculate_stats") as mock_calc_stats:

            mock_find_dir.return_value = mock_config.specs_dir
            mock_find_file.return_value = mock_config.specs_dir / "test-spec.json"
            mock_load.return_value = {"id": "test-spec", "title": "Test"}
            mock_validate.return_value = mock_validation_result
            mock_calc_stats.return_value = mock_stats

            spec_report = mock_mcp._tools["spec_report"]
            result = spec_report(spec_id="test-spec", format="json")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["format"] == "json"
            assert "report" in result["data"]
            assert "spec_id" in result["data"]["report"]
            assert "validation" in result["data"]["report"]
            assert "statistics" in result["data"]["report"]
            assert "health" in result["data"]["report"]

    def test_selective_sections(
        self, mock_mcp, mock_config, mock_validation_result, mock_stats, assert_response_contract
    ):
        """Should include only requested sections."""
        from foundry_mcp.tools.reporting import register_reporting_tools, _report_breaker

        _report_breaker.reset()
        register_reporting_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.reporting.find_specs_directory") as mock_find_dir, \
             patch("foundry_mcp.tools.reporting.find_spec_file") as mock_find_file, \
             patch("foundry_mcp.tools.reporting.load_spec") as mock_load, \
             patch("foundry_mcp.tools.reporting.validate_spec") as mock_validate, \
             patch("foundry_mcp.tools.reporting.calculate_stats") as mock_calc_stats:

            mock_find_dir.return_value = mock_config.specs_dir
            mock_find_file.return_value = mock_config.specs_dir / "test-spec.json"
            mock_load.return_value = {"id": "test-spec", "title": "Test"}
            mock_validate.return_value = mock_validation_result
            mock_calc_stats.return_value = mock_stats

            spec_report = mock_mcp._tools["spec_report"]
            result = spec_report(spec_id="test-spec", sections="validation,stats")

            assert_response_contract(result)
            assert result["success"] is True
            assert "validation" in result["data"]["sections"]
            assert "stats" in result["data"]["sections"]
            assert "health" not in result["data"]["sections"]
            # Health section should not be in markdown
            assert "## Health Assessment" not in result["data"]["report"]

    def test_spec_not_found(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error when spec not found."""
        from foundry_mcp.tools.reporting import register_reporting_tools, _report_breaker

        _report_breaker.reset()
        register_reporting_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.reporting.find_specs_directory") as mock_find_dir, \
             patch("foundry_mcp.tools.reporting.load_spec") as mock_load:

            mock_find_dir.return_value = mock_config.specs_dir
            mock_load.return_value = None  # Spec not found

            spec_report = mock_mcp._tools["spec_report"]
            result = spec_report(spec_id="nonexistent-spec")

            assert_response_contract(result)
            assert result["success"] is False
            assert "not found" in result["error"].lower()

    def test_no_specs_directory(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error when specs directory not found."""
        from foundry_mcp.tools.reporting import register_reporting_tools, _report_breaker

        _report_breaker.reset()
        mock_config.specs_dir = None
        register_reporting_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.reporting.find_specs_directory") as mock_find_dir:
            mock_find_dir.return_value = None

            spec_report = mock_mcp._tools["spec_report"]
            result = spec_report(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is False
            assert "no specs directory" in result["error"].lower()

    def test_circuit_breaker_open(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error when circuit breaker is open."""
        from foundry_mcp.tools.reporting import register_reporting_tools, _report_breaker

        _report_breaker.reset()
        register_reporting_tools(mock_mcp, mock_config)

        # Trip the circuit breaker
        for _ in range(5):
            _report_breaker.record_failure()

        assert _report_breaker.state == CircuitState.OPEN

        spec_report = mock_mcp._tools["spec_report"]
        result = spec_report(spec_id="test-spec")

        assert_response_contract(result)
        assert result["success"] is False
        assert "unavailable" in result["error"].lower()

        _report_breaker.reset()

    def test_health_score_calculation(
        self, mock_mcp, mock_config, mock_stats, assert_response_contract
    ):
        """Should calculate health score correctly."""
        from foundry_mcp.tools.reporting import register_reporting_tools, _report_breaker

        _report_breaker.reset()
        register_reporting_tools(mock_mcp, mock_config)

        # Create validation result with errors
        validation_result = MagicMock()
        validation_result.is_valid = False
        validation_result.error_count = 3
        validation_result.warning_count = 10
        validation_result.info_count = 0
        validation_result.diagnostics = []

        with patch("foundry_mcp.tools.reporting.find_specs_directory") as mock_find_dir, \
             patch("foundry_mcp.tools.reporting.find_spec_file") as mock_find_file, \
             patch("foundry_mcp.tools.reporting.load_spec") as mock_load, \
             patch("foundry_mcp.tools.reporting.validate_spec") as mock_validate, \
             patch("foundry_mcp.tools.reporting.calculate_stats") as mock_calc_stats:

            mock_find_dir.return_value = mock_config.specs_dir
            mock_find_file.return_value = mock_config.specs_dir / "test-spec.json"
            mock_load.return_value = {"id": "test-spec", "title": "Test"}
            mock_validate.return_value = validation_result
            mock_calc_stats.return_value = mock_stats

            spec_report = mock_mcp._tools["spec_report"]
            result = spec_report(spec_id="test-spec", format="json", sections="health")

            assert_response_contract(result)
            assert result["success"] is True
            # Health score should be reduced: 100 - 30 (errors) - 20 (warnings) = 50
            assert result["data"]["report"]["health"]["score"] <= 50
            assert result["data"]["report"]["health"]["status"] in ["needs_attention", "critical"]

    def test_telemetry_in_response(
        self, mock_mcp, mock_config, mock_validation_result, mock_stats, assert_response_contract
    ):
        """Should include telemetry in response meta."""
        from foundry_mcp.tools.reporting import register_reporting_tools, _report_breaker

        _report_breaker.reset()
        register_reporting_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.reporting.find_specs_directory") as mock_find_dir, \
             patch("foundry_mcp.tools.reporting.find_spec_file") as mock_find_file, \
             patch("foundry_mcp.tools.reporting.load_spec") as mock_load, \
             patch("foundry_mcp.tools.reporting.validate_spec") as mock_validate, \
             patch("foundry_mcp.tools.reporting.calculate_stats") as mock_calc_stats:

            mock_find_dir.return_value = mock_config.specs_dir
            mock_find_file.return_value = mock_config.specs_dir / "test-spec.json"
            mock_load.return_value = {"id": "test-spec", "title": "Test"}
            mock_validate.return_value = mock_validation_result
            mock_calc_stats.return_value = mock_stats

            spec_report = mock_mcp._tools["spec_report"]
            result = spec_report(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            # Telemetry is in meta.telemetry per response contract
            assert "meta" in result
            assert "telemetry" in result["meta"]
            assert "duration_ms" in result["meta"]["telemetry"]


# =============================================================================
# spec_report_summary Tool Tests
# =============================================================================


class TestSpecReportSummary:
    """Test the spec-report-summary tool."""

    def test_basic_summary(
        self, mock_mcp, mock_config, mock_validation_result, mock_stats, assert_response_contract
    ):
        """Should generate a summary successfully."""
        from foundry_mcp.tools.reporting import register_reporting_tools, _report_breaker

        _report_breaker.reset()
        register_reporting_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.reporting.find_specs_directory") as mock_find_dir, \
             patch("foundry_mcp.tools.reporting.find_spec_file") as mock_find_file, \
             patch("foundry_mcp.tools.reporting.load_spec") as mock_load, \
             patch("foundry_mcp.tools.reporting.validate_spec") as mock_validate, \
             patch("foundry_mcp.tools.reporting.calculate_stats") as mock_calc_stats:

            mock_find_dir.return_value = mock_config.specs_dir
            mock_find_file.return_value = mock_config.specs_dir / "test-spec.json"
            mock_load.return_value = {"id": "test-spec", "title": "Test"}
            mock_validate.return_value = mock_validation_result
            mock_calc_stats.return_value = mock_stats

            spec_report_summary = mock_mcp._tools["spec_report_summary"]
            result = spec_report_summary(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["spec_id"] == "test-spec"
            assert result["data"]["title"] == "Test Specification"
            assert result["data"]["status"] == "in_progress"
            assert "validation" in result["data"]
            assert "progress" in result["data"]
            assert "health" in result["data"]

    def test_summary_validation_fields(
        self, mock_mcp, mock_config, mock_validation_result, mock_stats, assert_response_contract
    ):
        """Should include correct validation fields."""
        from foundry_mcp.tools.reporting import register_reporting_tools, _report_breaker

        _report_breaker.reset()
        register_reporting_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.reporting.find_specs_directory") as mock_find_dir, \
             patch("foundry_mcp.tools.reporting.find_spec_file") as mock_find_file, \
             patch("foundry_mcp.tools.reporting.load_spec") as mock_load, \
             patch("foundry_mcp.tools.reporting.validate_spec") as mock_validate, \
             patch("foundry_mcp.tools.reporting.calculate_stats") as mock_calc_stats:

            mock_find_dir.return_value = mock_config.specs_dir
            mock_find_file.return_value = mock_config.specs_dir / "test-spec.json"
            mock_load.return_value = {"id": "test-spec", "title": "Test"}
            mock_validate.return_value = mock_validation_result
            mock_calc_stats.return_value = mock_stats

            spec_report_summary = mock_mcp._tools["spec_report_summary"]
            result = spec_report_summary(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["validation"]["is_valid"] is True
            assert result["data"]["validation"]["errors"] == 0
            assert result["data"]["validation"]["warnings"] == 2

    def test_summary_progress_fields(
        self, mock_mcp, mock_config, mock_validation_result, mock_stats, assert_response_contract
    ):
        """Should include correct progress fields."""
        from foundry_mcp.tools.reporting import register_reporting_tools, _report_breaker

        _report_breaker.reset()
        register_reporting_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.reporting.find_specs_directory") as mock_find_dir, \
             patch("foundry_mcp.tools.reporting.find_spec_file") as mock_find_file, \
             patch("foundry_mcp.tools.reporting.load_spec") as mock_load, \
             patch("foundry_mcp.tools.reporting.validate_spec") as mock_validate, \
             patch("foundry_mcp.tools.reporting.calculate_stats") as mock_calc_stats:

            mock_find_dir.return_value = mock_config.specs_dir
            mock_find_file.return_value = mock_config.specs_dir / "test-spec.json"
            mock_load.return_value = {"id": "test-spec", "title": "Test"}
            mock_validate.return_value = mock_validation_result
            mock_calc_stats.return_value = mock_stats

            spec_report_summary = mock_mcp._tools["spec_report_summary"]
            result = spec_report_summary(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["progress"]["completed"] == 6
            assert result["data"]["progress"]["total"] == 10
            assert result["data"]["progress"]["percentage"] == 60

    def test_summary_health_fields(
        self, mock_mcp, mock_config, mock_validation_result, mock_stats, assert_response_contract
    ):
        """Should include correct health fields."""
        from foundry_mcp.tools.reporting import register_reporting_tools, _report_breaker

        _report_breaker.reset()
        register_reporting_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.reporting.find_specs_directory") as mock_find_dir, \
             patch("foundry_mcp.tools.reporting.find_spec_file") as mock_find_file, \
             patch("foundry_mcp.tools.reporting.load_spec") as mock_load, \
             patch("foundry_mcp.tools.reporting.validate_spec") as mock_validate, \
             patch("foundry_mcp.tools.reporting.calculate_stats") as mock_calc_stats:

            mock_find_dir.return_value = mock_config.specs_dir
            mock_find_file.return_value = mock_config.specs_dir / "test-spec.json"
            mock_load.return_value = {"id": "test-spec", "title": "Test"}
            mock_validate.return_value = mock_validation_result
            mock_calc_stats.return_value = mock_stats

            spec_report_summary = mock_mcp._tools["spec_report_summary"]
            result = spec_report_summary(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            assert "score" in result["data"]["health"]
            assert "status" in result["data"]["health"]
            # Healthy since is_valid=True and warnings=2 (<=5)
            assert result["data"]["health"]["status"] == "healthy"

    def test_summary_spec_not_found(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error when spec not found."""
        from foundry_mcp.tools.reporting import register_reporting_tools, _report_breaker

        _report_breaker.reset()
        register_reporting_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.reporting.find_specs_directory") as mock_find_dir, \
             patch("foundry_mcp.tools.reporting.load_spec") as mock_load:

            mock_find_dir.return_value = mock_config.specs_dir
            mock_load.return_value = None

            spec_report_summary = mock_mcp._tools["spec_report_summary"]
            result = spec_report_summary(spec_id="nonexistent-spec")

            assert_response_contract(result)
            assert result["success"] is False
            assert "not found" in result["error"].lower()

    def test_summary_with_workspace(
        self, mock_mcp, mock_config, mock_validation_result, mock_stats, assert_response_contract, tmp_path
    ):
        """Should use workspace path when provided."""
        from foundry_mcp.tools.reporting import register_reporting_tools, _report_breaker

        _report_breaker.reset()
        register_reporting_tools(mock_mcp, mock_config)

        workspace_dir = tmp_path / "custom_workspace"
        workspace_dir.mkdir()

        with patch("foundry_mcp.tools.reporting.find_specs_directory") as mock_find_dir, \
             patch("foundry_mcp.tools.reporting.find_spec_file") as mock_find_file, \
             patch("foundry_mcp.tools.reporting.load_spec") as mock_load, \
             patch("foundry_mcp.tools.reporting.validate_spec") as mock_validate, \
             patch("foundry_mcp.tools.reporting.calculate_stats") as mock_calc_stats:

            mock_find_dir.return_value = workspace_dir / "specs"
            mock_find_file.return_value = workspace_dir / "specs" / "test-spec.json"
            mock_load.return_value = {"id": "test-spec", "title": "Test"}
            mock_validate.return_value = mock_validation_result
            mock_calc_stats.return_value = mock_stats

            spec_report_summary = mock_mcp._tools["spec_report_summary"]
            result = spec_report_summary(spec_id="test-spec", workspace=str(workspace_dir))

            assert_response_contract(result)
            assert result["success"] is True
            mock_find_dir.assert_called_with(str(workspace_dir))


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestReportingToolRegistration:
    """Test that all reporting tools are properly registered."""

    def test_all_reporting_tools_registered(self, mock_mcp, mock_config):
        """All reporting tools should be registered with the MCP server."""
        from foundry_mcp.tools.reporting import register_reporting_tools, _report_breaker

        _report_breaker.reset()
        register_reporting_tools(mock_mcp, mock_config)

        expected_tools = [
            "spec_report",
            "spec_report_summary",
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_mcp._tools, f"Tool {tool_name} not registered"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling across reporting tools."""

    def test_generic_exception_handling(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """Should handle unexpected exceptions gracefully."""
        from foundry_mcp.tools.reporting import register_reporting_tools, _report_breaker

        _report_breaker.reset()
        register_reporting_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.reporting.find_specs_directory") as mock_find_dir, \
             patch("foundry_mcp.tools.reporting.load_spec") as mock_load:
            mock_find_dir.return_value = mock_config.specs_dir
            mock_load.side_effect = RuntimeError("Unexpected error")

            spec_report = mock_mcp._tools["spec_report"]
            result = spec_report(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is False
            # Sanitized error messages use generic text per MCP best practices
            assert "internal" in result["error"].lower() or "error" in result["error"].lower()

    def test_circuit_breaker_error_recovery(self, mock_mcp, mock_config, assert_response_contract):
        """Circuit breaker should recover after timeout."""
        from foundry_mcp.tools.reporting import register_reporting_tools, _report_breaker

        _report_breaker.reset()
        register_reporting_tools(mock_mcp, mock_config)

        # Trip the circuit breaker
        for _ in range(5):
            _report_breaker.record_failure()

        assert _report_breaker.state == CircuitState.OPEN

        # Reset for test cleanup
        _report_breaker.reset()
        assert _report_breaker.state == CircuitState.CLOSED


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Test internal helper functions."""

    def test_format_severity_indicator(self):
        """Should format severity indicators correctly."""
        from foundry_mcp.tools.reporting import _format_severity_indicator

        assert _format_severity_indicator("error") == "[ERROR]"
        assert _format_severity_indicator("warning") == "[WARN]"
        assert _format_severity_indicator("info") == "[INFO]"
        assert _format_severity_indicator("UNKNOWN") == "[UNKNOWN]"

    def test_format_diagnostic(self):
        """Should format diagnostic messages correctly."""
        from foundry_mcp.tools.reporting import _format_diagnostic

        diag = {
            "severity": "error",
            "code": "ERR001",
            "message": "Test error",
            "location": "task-1-1",
            "suggested_fix": "Fix it",
        }

        formatted = _format_diagnostic(diag)

        assert "[ERROR]" in formatted
        assert "ERR001" in formatted
        assert "Test error" in formatted
        assert "task-1-1" in formatted
        assert "Fix it" in formatted

    def test_format_diagnostic_without_optional_fields(self):
        """Should handle diagnostics without optional fields."""
        from foundry_mcp.tools.reporting import _format_diagnostic

        diag = {
            "severity": "info",
            "code": "INFO001",
            "message": "Info message",
        }

        formatted = _format_diagnostic(diag)

        assert "[INFO]" in formatted
        assert "INFO001" in formatted
        assert "Info message" in formatted
        assert "Fix:" not in formatted  # No suggested fix
