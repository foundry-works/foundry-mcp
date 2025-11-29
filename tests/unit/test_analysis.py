"""
Unit tests for foundry_mcp.tools.analysis module.

Tests the analysis tools for SDD specifications,
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
# Analysis Module Tests - _run_sdd_command
# =============================================================================


class TestAnalysisRunSddCommand:
    """Test the _run_sdd_command helper function in analysis module."""

    def test_successful_command_execution(self):
        """Successful command should return result and record success."""
        from foundry_mcp.tools.analysis import _run_sdd_command, _sdd_cli_breaker

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

    def test_circuit_breaker_open_raises_error(self):
        """When circuit breaker is open, should raise CircuitBreakerError."""
        from foundry_mcp.tools.analysis import _run_sdd_command, _sdd_cli_breaker

        _sdd_cli_breaker.reset()

        for _ in range(5):
            _sdd_cli_breaker.record_failure()

        assert _sdd_cli_breaker.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerError) as exc_info:
            _run_sdd_command(["foundry-cli", "test"], "test_tool")

        assert exc_info.value.breaker_name == "sdd_cli_analysis"
        _sdd_cli_breaker.reset()


# =============================================================================
# spec_analyze Tool Tests
# =============================================================================


class TestSpecAnalyze:
    """Test the spec-analyze tool."""

    def test_basic_analysis(self, mock_mcp, mock_config, assert_response_contract):
        """Should analyze specs successfully."""
        from foundry_mcp.tools.analysis import register_analysis_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_analysis_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.analysis._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "analyze"],
                returncode=0,
                stdout=json.dumps({
                    "has_specs": True,
                    "documentation_available": True,
                    "spec_count": 5,
                }),
                stderr="",
            )

            spec_analyze = mock_mcp._tools["spec_analyze"]
            result = spec_analyze()

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["has_specs"] is True

    def test_analysis_with_directory(self, mock_mcp, mock_config, assert_response_contract):
        """Should pass directory parameter to CLI."""
        from foundry_mcp.tools.analysis import register_analysis_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_analysis_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.analysis._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "analyze"],
                returncode=0,
                stdout=json.dumps({"has_specs": True}),
                stderr="",
            )

            spec_analyze = mock_mcp._tools["spec_analyze"]
            result = spec_analyze(directory="/custom/dir")

            assert_response_contract(result)
            assert result["success"] is True
            # Verify directory was passed to command
            call_args = mock_cmd.call_args[0][0]
            assert "/custom/dir" in call_args

    def test_analysis_not_found_error(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle not found errors."""
        from foundry_mcp.tools.analysis import register_analysis_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_analysis_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.analysis._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "analyze"],
                returncode=1,
                stdout="",
                stderr="Directory not found",
            )

            spec_analyze = mock_mcp._tools["spec_analyze"]
            result = spec_analyze(directory="/nonexistent")

            assert_response_contract(result)
            assert result["success"] is False
            assert result["data"]["error_code"] == "NOT_FOUND"


# =============================================================================
# review_parse_feedback Tool Tests
# =============================================================================


class TestReviewParseFeedback:
    """Test the review-parse-feedback tool."""

    def test_basic_parse(self, mock_mcp, mock_config, assert_response_contract):
        """Should parse review feedback successfully."""
        from foundry_mcp.tools.analysis import register_analysis_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_analysis_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.analysis._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "parse-review"],
                returncode=0,
                stdout=json.dumps({
                    "suggestions_count": 3,
                    "output_file": "/tmp/suggestions.json",
                    "categories": {"improvement": 2, "fix": 1},
                }),
                stderr="",
            )

            review_parse_feedback = mock_mcp._tools["review_parse_feedback"]
            result = review_parse_feedback(
                spec_id="test-spec",
                review_path="/path/to/review.md",
            )

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["suggestions_count"] == 3

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.analysis import register_analysis_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_analysis_tools(mock_mcp, mock_config)

        review_parse_feedback = mock_mcp._tools["review_parse_feedback"]
        result = review_parse_feedback(spec_id="", review_path="/path/to/review.md")

        assert_response_contract(result)
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_missing_review_path_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing review_path."""
        from foundry_mcp.tools.analysis import register_analysis_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_analysis_tools(mock_mcp, mock_config)

        review_parse_feedback = mock_mcp._tools["review_parse_feedback"]
        result = review_parse_feedback(spec_id="test-spec", review_path="")

        assert_response_contract(result)
        assert result["success"] is False
        assert "review_path" in result["error"].lower()

    def test_parse_error_handling(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle parse errors correctly."""
        from foundry_mcp.tools.analysis import register_analysis_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_analysis_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.analysis._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "parse-review"],
                returncode=1,
                stdout="",
                stderr="Parse error: invalid format",
            )

            review_parse_feedback = mock_mcp._tools["review_parse_feedback"]
            result = review_parse_feedback(
                spec_id="test-spec",
                review_path="/path/to/invalid.txt",
            )

            assert_response_contract(result)
            assert result["success"] is False
            assert result["data"]["error_code"] == "PARSE_ERROR"


# =============================================================================
# spec_analyze_deps Tool Tests
# =============================================================================


class TestSpecAnalyzeDeps:
    """Test the spec-analyze-deps tool."""

    def test_basic_dependency_analysis(self, mock_mcp, mock_config, assert_response_contract):
        """Should analyze dependencies successfully."""
        from foundry_mcp.tools.analysis import register_analysis_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_analysis_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.analysis._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "analyze-deps"],
                returncode=0,
                stdout=json.dumps({
                    "dependency_count": 15,
                    "bottlenecks": ["task-1-2"],
                    "circular_deps": [],
                    "critical_path": ["task-1-1", "task-1-2", "task-2-1"],
                }),
                stderr="",
            )

            spec_analyze_deps = mock_mcp._tools["spec_analyze_deps"]
            result = spec_analyze_deps(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is True
            assert result["data"]["dependency_count"] == 15

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.analysis import register_analysis_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_analysis_tools(mock_mcp, mock_config)

        spec_analyze_deps = mock_mcp._tools["spec_analyze_deps"]
        result = spec_analyze_deps(spec_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_bottleneck_threshold_parameter(self, mock_mcp, mock_config, assert_response_contract):
        """Should pass bottleneck_threshold to CLI."""
        from foundry_mcp.tools.analysis import register_analysis_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_analysis_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.analysis._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "analyze-deps"],
                returncode=0,
                stdout=json.dumps({"dependency_count": 10}),
                stderr="",
            )

            spec_analyze_deps = mock_mcp._tools["spec_analyze_deps"]
            result = spec_analyze_deps(spec_id="test-spec", bottleneck_threshold=5)

            assert_response_contract(result)
            assert result["success"] is True
            # Verify threshold was passed to command
            call_args = mock_cmd.call_args[0][0]
            assert "--bottleneck-threshold" in call_args
            assert "5" in call_args

    def test_circular_dependency_error(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle circular dependency errors."""
        from foundry_mcp.tools.analysis import register_analysis_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_analysis_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.analysis._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "analyze-deps"],
                returncode=1,
                stdout="",
                stderr="Circular dependency detected: task-1 -> task-2 -> task-1",
            )

            spec_analyze_deps = mock_mcp._tools["spec_analyze_deps"]
            result = spec_analyze_deps(spec_id="test-spec")

            assert_response_contract(result)
            assert result["success"] is False
            assert result["data"]["error_code"] == "CIRCULAR_DEPENDENCY"

    def test_spec_not_found_error(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle spec not found errors."""
        from foundry_mcp.tools.analysis import register_analysis_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_analysis_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.analysis._run_sdd_command") as mock_cmd:
            mock_cmd.return_value = subprocess.CompletedProcess(
                args=["foundry-cli", "analyze-deps"],
                returncode=1,
                stdout="",
                stderr="Spec not found: nonexistent-spec",
            )

            spec_analyze_deps = mock_mcp._tools["spec_analyze_deps"]
            result = spec_analyze_deps(spec_id="nonexistent-spec")

            assert_response_contract(result)
            assert result["success"] is False
            assert result["data"]["error_code"] == "NOT_FOUND"


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestAnalysisToolRegistration:
    """Test that all analysis tools are registered correctly."""

    def test_all_analysis_tools_registered(self, mock_mcp, mock_config):
        """All analysis tools should be registered."""
        from foundry_mcp.tools.analysis import register_analysis_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_analysis_tools(mock_mcp, mock_config)

        expected_tools = [
            "spec_analyze",
            "review_parse_feedback",
            "spec_analyze_deps",
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_mcp._tools, f"Tool {tool_name} not registered"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestAnalysisErrorHandling:
    """Test error handling scenarios for analysis tools."""

    def test_circuit_breaker_error_handling(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error response when circuit breaker is open."""
        from foundry_mcp.tools.analysis import register_analysis_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_analysis_tools(mock_mcp, mock_config)

        # Open the circuit breaker
        for _ in range(5):
            _sdd_cli_breaker.record_failure()

        spec_analyze = mock_mcp._tools["spec_analyze"]
        result = spec_analyze()

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"]["error_code"] == "CIRCUIT_OPEN"

        _sdd_cli_breaker.reset()

    def test_timeout_error_handling(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle timeout errors gracefully."""
        from foundry_mcp.tools.analysis import register_analysis_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_analysis_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.analysis._run_sdd_command") as mock_cmd:
            mock_cmd.side_effect = subprocess.TimeoutExpired(cmd=["foundry-cli"], timeout=30)

            spec_analyze = mock_mcp._tools["spec_analyze"]
            result = spec_analyze()

            assert_response_contract(result)
            assert result["success"] is False
            assert result["data"]["error_code"] == "TIMEOUT"

    def test_cli_not_found_error_handling(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle CLI not found errors."""
        from foundry_mcp.tools.analysis import register_analysis_tools, _sdd_cli_breaker

        _sdd_cli_breaker.reset()
        register_analysis_tools(mock_mcp, mock_config)

        with patch("foundry_mcp.tools.analysis._run_sdd_command") as mock_cmd:
            mock_cmd.side_effect = FileNotFoundError("foundry-cli not found")

            spec_analyze = mock_mcp._tools["spec_analyze"]
            result = spec_analyze()

            assert_response_contract(result)
            assert result["success"] is False
            assert result["data"]["error_code"] == "CLI_NOT_FOUND"
