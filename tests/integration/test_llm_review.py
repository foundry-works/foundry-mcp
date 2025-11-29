"""
Integration tests for LLM-powered review tools.

Focused tests for spec_review and pr_create_with_spec tools with:
- Mocked provider matrix support
- Data-only fallback assertions
- Circuit breaker integration
- Response envelope validation
"""

import json
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from foundry_mcp.config import ServerConfig


@pytest.fixture
def mock_mcp():
    """Create a mock FastMCP server instance.

    Note: Tools are registered with function names (snake_case) by the
    canonical_tool decorator, not the kebab-case canonical names.
    """
    mcp = MagicMock()
    mcp._tools = {}

    def mock_tool(*args, **kwargs):
        def decorator(func):
            # Use function name as key (this is what actually gets registered)
            name = func.__name__
            mcp._tools[name] = MagicMock(fn=func)
            return func
        return decorator

    mcp.tool = mock_tool
    return mcp


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock server config with temp specs dir."""
    return ServerConfig(specs_dir=tmp_path / "specs")


@pytest.fixture
def sample_review_output():
    """Sample JSON output from sdd review command."""
    return json.dumps({
        "spec_id": "test-spec-001",
        "review_type": "quick",
        "findings": [
            {"type": "suggestion", "message": "Consider adding error handling", "severity": "low"},
        ],
        "suggestions": ["Add input validation", "Include unit tests"],
        "summary": "Review completed successfully with minor suggestions",
    })


@pytest.fixture
def sample_pr_output():
    """Sample JSON output from sdd create-pr command."""
    return json.dumps({
        "spec_id": "test-spec-001",
        "pr_url": "https://github.com/org/repo/pull/123",
        "title": "feat: Implement authentication flow",
        "description_preview": "## Summary\n- Implemented JWT auth...",
    })


class TestSpecReviewProviderMatrix:
    """Test spec_review with mocked provider matrix support."""

    def test_spec_review_registers_successfully(self, mock_mcp, mock_config):
        """Test spec_review tool registers without error."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)
        assert "spec_review" in mock_mcp._tools

    def test_spec_review_with_cursor_agent(self, mock_mcp, mock_config, sample_review_output):
        """Test spec_review with cursor-agent tool."""
        from foundry_mcp.tools.review import register_review_tools, _review_breaker

        _review_breaker.reset()
        register_review_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.review.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=sample_review_output,
                stderr="",
            )

            spec_review = mock_mcp._tools["spec_review"]
            result = spec_review.fn(
                spec_id="test-spec-001",
                review_type="full",
                tools="cursor-agent",
            )

            assert result["success"] is True
            assert result["data"]["spec_id"] == "test-spec-001"
            assert "cursor-agent" in mock_run.call_args[0][0]

        _review_breaker.reset()

    def test_spec_review_with_gemini(self, mock_mcp, mock_config, sample_review_output):
        """Test spec_review with gemini tool."""
        from foundry_mcp.tools.review import register_review_tools, _review_breaker

        _review_breaker.reset()
        register_review_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.review.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=sample_review_output,
                stderr="",
            )

            spec_review = mock_mcp._tools["spec_review"]
            result = spec_review.fn(
                spec_id="test-spec-001",
                review_type="security",
                tools="gemini",
            )

            assert result["success"] is True
            assert "gemini" in mock_run.call_args[0][0]

        _review_breaker.reset()

    def test_spec_review_with_multiple_tools(self, mock_mcp, mock_config, sample_review_output):
        """Test spec_review with multiple tools (cursor-agent,gemini,codex)."""
        from foundry_mcp.tools.review import register_review_tools, _review_breaker

        _review_breaker.reset()
        register_review_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.review.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=sample_review_output,
                stderr="",
            )

            spec_review = mock_mcp._tools["spec_review"]
            result = spec_review.fn(
                spec_id="test-spec-001",
                review_type="full",
                tools="cursor-agent,gemini,codex",
            )

            assert result["success"] is True
            cmd = mock_run.call_args[0][0]
            assert "--tools" in cmd
            assert "cursor-agent,gemini,codex" in cmd

        _review_breaker.reset()

    def test_spec_review_model_override(self, mock_mcp, mock_config, sample_review_output):
        """Test spec_review with custom model override."""
        from foundry_mcp.tools.review import register_review_tools, _review_breaker

        _review_breaker.reset()
        register_review_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.review.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=sample_review_output,
                stderr="",
            )

            spec_review = mock_mcp._tools["spec_review"]
            result = spec_review.fn(
                spec_id="test-spec-001",
                review_type="quick",
                model="gpt-4-turbo",
            )

            assert result["success"] is True
            cmd = mock_run.call_args[0][0]
            assert "--model" in cmd
            assert "gpt-4-turbo" in cmd

        _review_breaker.reset()


class TestSpecReviewDataOnlyFallback:
    """Test spec_review data-only fallback when LLM unavailable."""

    def test_spec_review_includes_llm_status(self, mock_mcp, mock_config, sample_review_output):
        """Test spec_review always includes llm_status in response."""
        from foundry_mcp.tools.review import register_review_tools, _review_breaker

        _review_breaker.reset()
        register_review_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.review.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=sample_review_output,
                stderr="",
            )

            spec_review = mock_mcp._tools["spec_review"]
            result = spec_review.fn(spec_id="test-spec-001")

            assert "llm_status" in result["data"]
            assert "configured" in result["data"]["llm_status"]

        _review_breaker.reset()

    def test_spec_review_fallback_on_llm_error(self, mock_mcp, mock_config):
        """Test spec_review falls back to data-only on LLM error."""
        from foundry_mcp.tools.review import register_review_tools, _review_breaker

        _review_breaker.reset()
        register_review_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.review.subprocess.run') as mock_run:
            # Simulate LLM failure but structural review succeeds
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Structural review completed",  # Non-JSON fallback
                stderr="",
            )

            spec_review = mock_mcp._tools["spec_review"]
            result = spec_review.fn(spec_id="test-spec-001", review_type="quick")

            assert result["success"] is True
            # Should have raw_output when JSON parsing fails
            assert "raw_output" in result["data"]

        _review_breaker.reset()

    def test_spec_review_dry_run_without_llm(self, mock_mcp, mock_config):
        """Test spec_review dry run works without LLM execution."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        spec_review = mock_mcp._tools["spec_review"]
        result = spec_review.fn(spec_id="test-spec-001", dry_run=True)

        assert result["success"] is True
        assert result["data"]["dry_run"] is True
        assert "command" in result["data"]
        assert "foundry-cli" in result["data"]["command"]

    def test_review_list_tools_shows_llm_unconfigured(self, mock_mcp, mock_config):
        """Test review_list_tools shows unconfigured LLM status."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.review._get_llm_status') as mock_status:
            mock_status.return_value = {"configured": False, "error": "No API key"}

            list_tools = mock_mcp._tools["review_list_tools"]
            result = list_tools.fn()

            assert result["success"] is True
            assert result["data"]["llm_status"]["configured"] is False


class TestSpecReviewValidation:
    """Test spec_review input validation and error handling."""

    def test_spec_review_invalid_type_returns_error(self, mock_mcp, mock_config):
        """Test invalid review type returns error with valid types."""
        from foundry_mcp.tools.review import register_review_tools, REVIEW_TYPES

        register_review_tools(mock_mcp, mock_config)

        spec_review = mock_mcp._tools["spec_review"]
        result = spec_review.fn(spec_id="test-spec-001", review_type="invalid_type")

        assert result["success"] is False
        assert "valid_types" in result["data"]
        assert set(result["data"]["valid_types"]) == set(REVIEW_TYPES)

    def test_spec_review_all_review_types(self, mock_mcp, mock_config, sample_review_output):
        """Test all review types are accepted."""
        from foundry_mcp.tools.review import register_review_tools, REVIEW_TYPES, _review_breaker

        register_review_tools(mock_mcp, mock_config)

        for review_type in REVIEW_TYPES:
            _review_breaker.reset()

            with patch('foundry_mcp.tools.review.subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout=sample_review_output,
                    stderr="",
                )

                spec_review = mock_mcp._tools["spec_review"]
                result = spec_review.fn(spec_id="test-spec-001", review_type=review_type)

                assert result["success"] is True, f"Failed for review_type={review_type}"
                assert result["data"]["review_type"] == review_type

            _review_breaker.reset()


class TestPRCreateWithSpec:
    """Test pr_create_with_spec tool functionality."""

    def test_pr_create_registers_successfully(self, mock_mcp, mock_config):
        """Test pr_create_with_spec tool registers without error."""
        from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools

        register_pr_workflow_tools(mock_mcp, mock_config)
        assert "pr_create_with_spec" in mock_mcp._tools

    def test_pr_create_dry_run(self, mock_mcp, mock_config, sample_pr_output):
        """Test pr_create_with_spec dry run mode."""
        from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools, _pr_breaker

        _pr_breaker.reset()
        register_pr_workflow_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.pr_workflow.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=sample_pr_output,
                stderr="",
            )

            pr_create = mock_mcp._tools["pr_create_with_spec"]
            result = pr_create.fn(spec_id="test-spec-001", dry_run=True)

            assert result["success"] is True
            cmd = mock_run.call_args[0][0]
            assert "--dry-run" in cmd

        _pr_breaker.reset()

    def test_pr_create_with_custom_title(self, mock_mcp, mock_config, sample_pr_output):
        """Test pr_create_with_spec with custom title."""
        from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools, _pr_breaker

        _pr_breaker.reset()
        register_pr_workflow_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.pr_workflow.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=sample_pr_output,
                stderr="",
            )

            pr_create = mock_mcp._tools["pr_create_with_spec"]
            result = pr_create.fn(
                spec_id="test-spec-001",
                title="feat: Custom PR title",
                dry_run=True,
            )

            assert result["success"] is True
            cmd = mock_run.call_args[0][0]
            assert "--title" in cmd
            assert "feat: Custom PR title" in cmd

        _pr_breaker.reset()

    def test_pr_create_includes_journals_by_default(self, mock_mcp, mock_config, sample_pr_output):
        """Test pr_create_with_spec includes journals by default."""
        from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools, _pr_breaker

        _pr_breaker.reset()
        register_pr_workflow_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.pr_workflow.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=sample_pr_output,
                stderr="",
            )

            pr_create = mock_mcp._tools["pr_create_with_spec"]
            result = pr_create.fn(spec_id="test-spec-001", dry_run=True)

            assert result["success"] is True
            cmd = mock_run.call_args[0][0]
            assert "--include-journals" in cmd

        _pr_breaker.reset()

    def test_pr_create_includes_diffs_by_default(self, mock_mcp, mock_config, sample_pr_output):
        """Test pr_create_with_spec includes diffs by default."""
        from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools, _pr_breaker

        _pr_breaker.reset()
        register_pr_workflow_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.pr_workflow.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=sample_pr_output,
                stderr="",
            )

            pr_create = mock_mcp._tools["pr_create_with_spec"]
            result = pr_create.fn(spec_id="test-spec-001", dry_run=True)

            assert result["success"] is True
            cmd = mock_run.call_args[0][0]
            assert "--include-diffs" in cmd

        _pr_breaker.reset()

    def test_pr_create_includes_llm_status(self, mock_mcp, mock_config, sample_pr_output):
        """Test pr_create_with_spec includes LLM status in response."""
        from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools, _pr_breaker

        _pr_breaker.reset()
        register_pr_workflow_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.pr_workflow.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=sample_pr_output,
                stderr="",
            )

            pr_create = mock_mcp._tools["pr_create_with_spec"]
            result = pr_create.fn(spec_id="test-spec-001", dry_run=True)

            assert result["success"] is True
            assert "llm_status" in result["data"]

        _pr_breaker.reset()


class TestPRCreateDataOnlyFallback:
    """Test pr_create_with_spec data-only fallback."""

    def test_pr_create_non_json_output_fallback(self, mock_mcp, mock_config):
        """Test pr_create_with_spec handles non-JSON output gracefully."""
        from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools, _pr_breaker

        _pr_breaker.reset()
        register_pr_workflow_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.pr_workflow.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="PR created: https://github.com/org/repo/pull/123",
                stderr="",
            )

            pr_create = mock_mcp._tools["pr_create_with_spec"]
            result = pr_create.fn(spec_id="test-spec-001", dry_run=True)

            assert result["success"] is True
            assert "raw_output" in result["data"]

        _pr_breaker.reset()

    def test_pr_create_cli_not_found(self, mock_mcp, mock_config):
        """Test pr_create_with_spec handles missing CLI gracefully."""
        from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools, _pr_breaker

        _pr_breaker.reset()
        register_pr_workflow_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.pr_workflow.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("foundry-cli not found")

            pr_create = mock_mcp._tools["pr_create_with_spec"]
            result = pr_create.fn(spec_id="test-spec-001", dry_run=True)

            assert result["success"] is False
            assert "not found" in result["error"].lower()

        _pr_breaker.reset()


class TestCircuitBreakerIntegration:
    """Test circuit breaker behavior for review tools."""

    def test_spec_review_circuit_breaker_trips_after_failures(self, mock_mcp, mock_config):
        """Test spec_review circuit breaker trips after consecutive failures."""
        from foundry_mcp.tools.review import register_review_tools, _review_breaker
        from foundry_mcp.core.resilience import CircuitBreakerError

        _review_breaker.reset()
        register_review_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.review.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("foundry-cli not found")

            spec_review = mock_mcp._tools["spec_review"]

            # Trip the circuit breaker
            for _ in range(_review_breaker.failure_threshold):
                result = spec_review.fn(spec_id="test-spec-001")
                assert result["success"] is False

            # Next call should get circuit breaker error
            result = spec_review.fn(spec_id="test-spec-001")
            assert result["success"] is False
            assert "circuit breaker" in result["error"].lower()

        _review_breaker.reset()

    def test_pr_create_circuit_breaker_trips_after_failures(self, mock_mcp, mock_config):
        """Test pr_create_with_spec circuit breaker trips after consecutive failures."""
        from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools, _pr_breaker

        _pr_breaker.reset()
        register_pr_workflow_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.pr_workflow.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("foundry-cli not found")

            pr_create = mock_mcp._tools["pr_create_with_spec"]

            # Trip the circuit breaker
            for _ in range(_pr_breaker.failure_threshold):
                result = pr_create.fn(spec_id="test-spec-001", dry_run=True)
                assert result["success"] is False

            # Next call should get circuit breaker error
            result = pr_create.fn(spec_id="test-spec-001", dry_run=True)
            assert result["success"] is False
            assert "circuit breaker" in result["error"].lower()

        _pr_breaker.reset()

    def test_spec_review_circuit_breaker_recovers(self, mock_mcp, mock_config, sample_review_output):
        """Test spec_review circuit breaker recovers after reset."""
        from foundry_mcp.tools.review import register_review_tools, _review_breaker

        _review_breaker.reset()
        register_review_tools(mock_mcp, mock_config)

        # First, trip the breaker
        with patch('foundry_mcp.tools.review.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("foundry-cli not found")

            spec_review = mock_mcp._tools["spec_review"]
            for _ in range(_review_breaker.failure_threshold + 1):
                spec_review.fn(spec_id="test-spec-001")

        # Reset and verify recovery
        _review_breaker.reset()

        with patch('foundry_mcp.tools.review.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=sample_review_output,
                stderr="",
            )

            result = spec_review.fn(spec_id="test-spec-001")
            assert result["success"] is True

        _review_breaker.reset()


class TestResponseEnvelope:
    """Test response envelope compliance for review tools."""

    def test_spec_review_success_envelope(self, mock_mcp, mock_config, sample_review_output):
        """Test spec_review success response has required envelope fields."""
        from foundry_mcp.tools.review import register_review_tools, _review_breaker

        _review_breaker.reset()
        register_review_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.review.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=sample_review_output,
                stderr="",
            )

            spec_review = mock_mcp._tools["spec_review"]
            result = spec_review.fn(spec_id="test-spec-001")

            # Required envelope fields
            assert "success" in result
            assert result["success"] is True
            assert "data" in result
            assert "meta" in result

        _review_breaker.reset()

    def test_spec_review_error_envelope(self, mock_mcp, mock_config):
        """Test spec_review error response has required envelope fields."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        spec_review = mock_mcp._tools["spec_review"]
        result = spec_review.fn(spec_id="test-spec-001", review_type="invalid")

        # Required envelope fields for errors
        assert "success" in result
        assert result["success"] is False
        assert "error" in result
        assert "data" in result

    def test_pr_create_success_envelope(self, mock_mcp, mock_config, sample_pr_output):
        """Test pr_create_with_spec success response has required envelope fields."""
        from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools, _pr_breaker

        _pr_breaker.reset()
        register_pr_workflow_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.pr_workflow.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=sample_pr_output,
                stderr="",
            )

            pr_create = mock_mcp._tools["pr_create_with_spec"]
            result = pr_create.fn(spec_id="test-spec-001", dry_run=True)

            assert "success" in result
            assert result["success"] is True
            assert "data" in result
            assert "meta" in result

        _pr_breaker.reset()


class TestTimeoutHandling:
    """Test timeout handling for review tools."""

    def test_spec_review_timeout_returns_error(self, mock_mcp, mock_config):
        """Test spec_review timeout returns appropriate error."""
        from foundry_mcp.tools.review import register_review_tools, _review_breaker, REVIEW_TIMEOUT

        _review_breaker.reset()
        register_review_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.review.subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="foundry-cli", timeout=REVIEW_TIMEOUT)

            spec_review = mock_mcp._tools["spec_review"]
            result = spec_review.fn(spec_id="test-spec-001")

            assert result["success"] is False
            assert "timed out" in result["error"].lower()
            assert "timeout_seconds" in result["data"]

        _review_breaker.reset()

    def test_pr_create_timeout_returns_error(self, mock_mcp, mock_config):
        """Test pr_create_with_spec timeout returns appropriate error."""
        from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools, _pr_breaker, PR_TIMEOUT

        _pr_breaker.reset()
        register_pr_workflow_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.pr_workflow.subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="foundry-cli", timeout=PR_TIMEOUT)

            pr_create = mock_mcp._tools["pr_create_with_spec"]
            result = pr_create.fn(spec_id="test-spec-001", dry_run=True)

            assert result["success"] is False
            assert "timed out" in result["error"].lower()

        _pr_breaker.reset()


class TestMetricsEmission:
    """Test metrics emission for review tools."""

    def test_spec_review_emits_duration_metric(self, mock_mcp, mock_config, sample_review_output):
        """Test spec_review emits timer metric on success."""
        from foundry_mcp.tools.review import register_review_tools, _review_breaker, _metrics

        _review_breaker.reset()
        register_review_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.review.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=sample_review_output,
                stderr="",
            )

            with patch.object(_metrics, 'timer') as mock_timer:
                spec_review = mock_mcp._tools["spec_review"]
                result = spec_review.fn(spec_id="test-spec-001")

                assert result["success"] is True
                # Timer should have been called with duration
                mock_timer.assert_called()

        _review_breaker.reset()

    def test_spec_review_response_includes_duration(self, mock_mcp, mock_config, sample_review_output):
        """Test spec_review response includes duration_ms."""
        from foundry_mcp.tools.review import register_review_tools, _review_breaker

        _review_breaker.reset()
        register_review_tools(mock_mcp, mock_config)

        with patch('foundry_mcp.tools.review.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=sample_review_output,
                stderr="",
            )

            spec_review = mock_mcp._tools["spec_review"]
            result = spec_review.fn(spec_id="test-spec-001")

            assert result["success"] is True
            assert "duration_ms" in result["data"]
            assert isinstance(result["data"]["duration_ms"], (int, float))

        _review_breaker.reset()
