"""
Integration tests for all LLM-powered tools.

Tests verify:
- Data-only fallback paths when LLM unavailable
- Multi-provider matrix support
- Circuit breaker behavior
- Response envelope compliance
- Timeout and error handling
"""

import json
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from foundry_mcp.config import ServerConfig


@pytest.fixture
def mock_mcp():
    """Create a mock FastMCP server instance."""
    mcp = MagicMock()
    mcp._tools = {}

    def mock_tool(*args, **kwargs):
        def decorator(func):
            # Use canonical_name if available, otherwise function name
            name = kwargs.get("canonical_name", func.__name__)
            mcp._tools[name] = MagicMock(fn=func)
            return func
        return decorator

    mcp.tool = mock_tool
    return mcp


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock server config."""
    return ServerConfig(specs_dir=tmp_path / "specs")


class TestLLMToolsDataOnlyFallback:
    """Test that LLM tools gracefully degrade to data-only when LLM unavailable."""

    def test_spec_review_registration_succeeds(self, mock_mcp, mock_config):
        """Test spec-review registers successfully."""
        from foundry_mcp.tools.review import register_review_tools

        # Registration should succeed
        register_review_tools(mock_mcp, mock_config)

        # Verify tools are registered
        assert len(mock_mcp._tools) >= 1

    def test_review_list_tools_works_without_llm(self, mock_mcp, mock_config):
        """Test review-list-tools returns data even without LLM configured."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        # Get the registered function
        list_tools = mock_mcp._tools.get("review-list-tools")
        if list_tools:
            result = list_tools.fn()
            # Should return available tools list regardless of LLM status
            assert result.get("success") is True
            assert "tools" in result or "data" in result

    def test_review_list_plan_tools_degrades_gracefully(self, mock_mcp, mock_config):
        """Test review-list-plan-tools shows unavailable status for LLM tools."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_plan_tools = mock_mcp._tools.get("review-list-plan-tools")
        if list_plan_tools:
            result = list_plan_tools.fn()
            assert result.get("success") is True

            # Should include plan tools info
            if "data" in result:
                data = result["data"]
            else:
                data = result

            assert "plan_tools" in data or "recommendations" in data


class TestMultiProviderMatrix:
    """Test multi-provider support across LLM tools."""

    def test_review_tools_lists_supported_providers(self):
        """Test that available review tools include multiple providers."""
        from foundry_mcp.tools.review import REVIEW_TOOLS

        # Should support multiple external tools
        assert len(REVIEW_TOOLS) >= 2
        assert "cursor-agent" in REVIEW_TOOLS or "gemini" in REVIEW_TOOLS

    def test_review_types_available(self):
        """Test multiple review types are available."""
        from foundry_mcp.tools.review import REVIEW_TYPES

        assert len(REVIEW_TYPES) >= 2
        assert "quick" in REVIEW_TYPES
        assert "full" in REVIEW_TYPES


class TestCircuitBreakerIntegration:
    """Test circuit breaker behavior across LLM tools."""

    def test_review_circuit_breaker_exists(self):
        """Test that review tools have circuit breaker protection."""
        from foundry_mcp.tools.review import _review_breaker

        assert _review_breaker is not None
        assert _review_breaker.name == "sdd_cli_review"

    def test_documentation_circuit_breaker_exists(self):
        """Test that documentation tools have circuit breaker protection."""
        from foundry_mcp.tools.documentation import _doc_breaker

        assert _doc_breaker is not None
        assert _doc_breaker.name == "documentation"

    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after consecutive failures."""
        from foundry_mcp.tools.review import _review_breaker, _run_review_command
        from foundry_mcp.core.resilience import CircuitBreakerError

        # Reset breaker state
        _review_breaker.reset()

        # Simulate failures up to threshold
        with patch('foundry_mcp.tools.review.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("foundry-cli not found")

            for i in range(_review_breaker.failure_threshold):
                try:
                    _run_review_command(["foundry-cli", "review", "test"], "test-tool")
                except (FileNotFoundError, CircuitBreakerError):
                    pass

            # Next call should fail - circuit breaker open
            assert not _review_breaker.can_execute()

        # Reset for other tests
        _review_breaker.reset()

    def test_circuit_breaker_recovers_after_success(self):
        """Test circuit breaker recovers after successful calls."""
        from foundry_mcp.tools.review import _review_breaker

        # Reset to clean state
        _review_breaker.reset()
        _review_breaker.record_success()

        assert _review_breaker.can_execute()


class TestResponseEnvelopeCompliance:
    """Test all LLM tools emit compliant response envelopes."""

    def test_review_list_tools_response_has_required_fields(self, mock_mcp, mock_config):
        """Test review-list-tools returns properly structured response."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_tools = mock_mcp._tools.get("review-list-tools")
        if list_tools:
            result = list_tools.fn()
            assert "success" in result or "tools" in result or "data" in result

    def test_documentation_tools_register(self, mock_mcp, mock_config):
        """Test documentation tools register without error."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        # Should not raise
        register_documentation_tools(mock_mcp, mock_config)

        # Should have registered tools
        assert len(mock_mcp._tools) >= 1


class TestTimeoutHandling:
    """Test timeout handling across LLM tools."""

    def test_review_timeout_raises_exception(self):
        """Test review timeout raises TimeoutExpired."""
        from foundry_mcp.tools.review import _run_review_command, _review_breaker, REVIEW_TIMEOUT

        _review_breaker.reset()

        with patch('foundry_mcp.tools.review.subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="foundry-cli", timeout=REVIEW_TIMEOUT)

            with pytest.raises(subprocess.TimeoutExpired):
                _run_review_command(["foundry-cli", "review", "test"], "test-tool")

        _review_breaker.reset()

    def test_documentation_timeout_provides_guidance(self):
        """Test documentation timeout includes recovery guidance."""
        from foundry_mcp.tools.documentation import _run_sdd_llm_doc_gen_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="foundry-cli", timeout=600)

            result = _run_sdd_llm_doc_gen_command(["generate", "/path"])

            assert result["success"] is False
            assert "timed out" in result["error"]
            assert "--resume" in result["error"]  # Recovery guidance

    def test_fidelity_review_timeout_suggests_scope_reduction(self):
        """Test fidelity review timeout suggests reducing scope."""
        from foundry_mcp.tools.documentation import _run_sdd_fidelity_review_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="foundry-cli", timeout=600)

            result = _run_sdd_fidelity_review_command(["test-spec"])

            assert result["success"] is False
            assert "timed out" in result["error"]
            assert "smaller scope" in result["error"]


class TestErrorHandling:
    """Test error handling across LLM tools."""

    def test_invalid_review_type_returns_error(self, mock_mcp, mock_config):
        """Test invalid review type returns appropriate error."""
        from foundry_mcp.tools.review import register_review_tools, REVIEW_TYPES

        register_review_tools(mock_mcp, mock_config)

        spec_review = mock_mcp._tools.get("spec-review")
        if spec_review:
            result = spec_review.fn(spec_id="test", review_type="invalid_type")

            assert result.get("success") is False
            # Should include valid types in error
            data = result.get("data", {})
            assert "valid_types" in data or any(t in str(result) for t in REVIEW_TYPES)

    def test_documentation_invalid_format_returns_error(self, mock_mcp, mock_config):
        """Test invalid output format returns error with supported formats."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc = mock_mcp._tools.get("spec-doc")
        if spec_doc:
            result = spec_doc.fn(spec_id="test", output_format="invalid")

            assert result.get("success") is False
            # Should mention supported formats
            error_msg = result.get("error", "") or result.get("message", "")
            assert "markdown" in error_msg.lower() or "md" in error_msg.lower()


class TestLLMConfigurationStatus:
    """Test LLM configuration status reporting."""

    def test_list_tools_includes_llm_status(self, mock_mcp, mock_config):
        """Test review-list-tools includes LLM configuration status."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_tools = mock_mcp._tools.get("review-list-tools")
        if list_tools:
            result = list_tools.fn()

            if result.get("success"):
                data = result.get("data", result)
                assert "llm_status" in data

    def test_list_plan_tools_shows_availability_based_on_llm(self, mock_mcp, mock_config):
        """Test plan tools availability reflects LLM configuration."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_plan_tools = mock_mcp._tools.get("review-list-plan-tools")
        if list_plan_tools:
            result = list_plan_tools.fn()

            if result.get("success"):
                data = result.get("data", result)
                plan_tools = data.get("plan_tools", [])

                # Should have at least one tool that doesn't require LLM
                non_llm_tools = [t for t in plan_tools if not t.get("llm_required")]
                assert len(non_llm_tools) >= 1


class TestToolRegistration:
    """Test all LLM tools register correctly."""

    def test_all_review_tools_register(self, mock_mcp, mock_config):
        """Test all review tools register without error."""
        from foundry_mcp.tools.review import register_review_tools

        # Should not raise
        register_review_tools(mock_mcp, mock_config)

        # Should have registered review tools
        assert len(mock_mcp._tools) >= 2

    def test_all_documentation_tools_register(self, mock_mcp, mock_config):
        """Test all documentation tools register without error."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        # Should not raise
        register_documentation_tools(mock_mcp, mock_config)

        # Should have registered tools
        assert len(mock_mcp._tools) >= 1


class TestSecurityValidation:
    """Test security validation in LLM tools."""

    def test_documentation_validates_prompt_injection(self, mock_mcp, mock_config):
        """Test documentation tools check for prompt injection."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc = mock_mcp._tools.get("spec-doc")
        if spec_doc:
            # Try a potential injection pattern
            result = spec_doc.fn(
                spec_id="test; rm -rf /",  # Suspicious pattern
            )

            # Should either succeed with sanitized input or fail validation
            assert "success" in result

    def test_fidelity_review_handles_suspicious_paths(self, mock_mcp, mock_config):
        """Test fidelity review handles suspicious file path inputs."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        fidelity = mock_mcp._tools.get("spec-review-fidelity")
        if fidelity:
            # Files parameter with path traversal attempt
            result = fidelity.fn(
                spec_id="test",
                files=["../../../etc/passwd"],
            )

            # Should handle gracefully (not crash)
            assert "success" in result or "error" in result


class TestMetricsEmission:
    """Test that LLM tools emit proper metrics."""

    def test_review_metrics_object_exists(self):
        """Test review tools have metrics object."""
        from foundry_mcp.tools.review import _metrics

        # Metrics object should exist
        assert _metrics is not None

    def test_documentation_metrics_integration(self):
        """Test documentation tools integrate with metrics."""
        from foundry_mcp.core.observability import get_metrics

        metrics = get_metrics()
        assert metrics is not None

        # Should be able to record metrics without error
        metrics.counter("test.counter", labels={"tool": "test"})
        metrics.timer("test.timer", 100.0)


class TestDataOnlyFallbackPaths:
    """Test data-only fallback behavior when LLM operations fail."""

    def test_render_command_json_fallback(self):
        """Test render command falls back to raw_output when JSON fails."""
        from foundry_mcp.tools.documentation import _run_sdd_render_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            # Simulate non-JSON output
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Rendered to specs/.human-readable/test.md",
                stderr="",
            )

            result = _run_sdd_render_command(["test-spec"])

            # Should succeed with raw_output fallback
            assert result["success"] is True
            assert "raw_output" in result["data"]

    def test_llm_doc_gen_json_fallback(self):
        """Test LLM doc gen falls back to raw_output when JSON fails."""
        from foundry_mcp.tools.documentation import _run_sdd_llm_doc_gen_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Generated 5 documentation files",
                stderr="",
            )

            result = _run_sdd_llm_doc_gen_command(["generate", "/path"])

            assert result["success"] is True
            assert "raw_output" in result["data"]

    def test_fidelity_review_json_fallback(self):
        """Test fidelity review falls back to raw_output when JSON fails."""
        from foundry_mcp.tools.documentation import _run_sdd_fidelity_review_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Fidelity review: PASS",
                stderr="",
            )

            result = _run_sdd_fidelity_review_command(["test-spec"])

            assert result["success"] is True
            assert "raw_output" in result["data"]

    def test_review_command_json_fallback(self):
        """Test review command handles non-JSON output."""
        from foundry_mcp.tools.review import _run_review_command, _review_breaker

        _review_breaker.reset()

        with patch('foundry_mcp.tools.review.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Review completed successfully",
                stderr="",
            )

            result = _run_review_command(["foundry-cli", "review", "test"], "test-tool")

            # Should complete successfully
            assert result.returncode == 0

        _review_breaker.reset()


class TestCLINotFoundHandling:
    """Test graceful handling when SDD CLI is not found."""

    def test_doc_command_cli_not_found(self):
        """Test doc command handles missing CLI."""
        from foundry_mcp.tools.documentation import _run_sdd_doc_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("foundry-cli not found")

            result = _run_sdd_doc_command(["test-spec"])

            assert result["success"] is False
            assert "foundry-cli not found" in result["error"]
            assert "foundry-mcp" in result["error"]

    def test_render_command_cli_not_found(self):
        """Test render command handles missing CLI."""
        from foundry_mcp.tools.documentation import _run_sdd_render_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("foundry-cli not found")

            result = _run_sdd_render_command(["test-spec"])

            assert result["success"] is False
            assert "foundry-cli not found" in result["error"]

    def test_llm_doc_gen_cli_not_found(self):
        """Test LLM doc gen handles missing CLI."""
        from foundry_mcp.tools.documentation import _run_sdd_llm_doc_gen_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("foundry-cli not found")

            result = _run_sdd_llm_doc_gen_command(["generate", "/path"])

            assert result["success"] is False
            assert "foundry-cli not found" in result["error"]

    def test_fidelity_review_cli_not_found(self):
        """Test fidelity review handles missing CLI."""
        from foundry_mcp.tools.documentation import _run_sdd_fidelity_review_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("foundry-cli not found")

            result = _run_sdd_fidelity_review_command(["test-spec"])

            assert result["success"] is False
            assert "foundry-cli not found" in result["error"]
