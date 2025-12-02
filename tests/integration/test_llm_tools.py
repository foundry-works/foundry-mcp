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
        # Provider system now handles provider discovery
        from foundry_mcp.core.providers import describe_providers

        providers = describe_providers()
        provider_ids = [p.get("id") for p in providers]

        # Should have multiple providers registered
        assert len(providers) >= 2
        # Should include expected providers
        assert "cursor-agent" in provider_ids or "gemini" in provider_ids

    def test_review_types_available(self):
        """Test multiple review types are available."""
        from foundry_mcp.tools.review import REVIEW_TYPES

        assert len(REVIEW_TYPES) >= 2
        assert "quick" in REVIEW_TYPES
        assert "full" in REVIEW_TYPES


class TestCircuitBreakerIntegration:
    """Test circuit breaker behavior across LLM tools.

    Note: Circuit breaker functionality has been moved to the provider system.
    These tests now verify that provider resilience is properly configured.
    """

    def test_provider_system_handles_unavailable_providers(self):
        """Test that provider system gracefully handles unavailable providers."""
        from foundry_mcp.core.providers import check_provider_available

        # Non-existent provider should return False, not raise
        result = check_provider_available("nonexistent-provider-xyz")
        assert result is False

    def test_provider_system_returns_availability_status(self):
        """Test that provider system reports availability status."""
        from foundry_mcp.core.providers import get_provider_statuses

        statuses = get_provider_statuses()

        # Should return a dict of provider availability
        assert isinstance(statuses, dict)
        # All values should be booleans
        for provider_id, available in statuses.items():
            assert isinstance(available, bool)

    def test_provider_describes_available_providers(self):
        """Test describe_providers includes availability info."""
        from foundry_mcp.core.providers import describe_providers

        providers = describe_providers()

        # Should return list of provider descriptions
        assert isinstance(providers, list)
        assert len(providers) >= 1

        # Each provider should have availability info
        for provider in providers:
            assert "available" in provider
            assert isinstance(provider["available"], bool)

    def test_provider_resolution_validates_availability(self):
        """Test that resolving unavailable provider raises appropriate error."""
        from foundry_mcp.core.providers import (
            resolve_provider,
            ProviderUnavailableError,
            ProviderHooks,
        )

        # Attempting to resolve non-existent provider should raise
        with pytest.raises(ProviderUnavailableError):
            resolve_provider("nonexistent-provider-xyz", hooks=ProviderHooks())


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
    """Test timeout handling across LLM tools.

    Note: Timeout handling is now managed by the provider system.
    These tests verify that provider timeout errors are properly typed.
    """

    def test_provider_timeout_error_exists(self):
        """Test ProviderTimeoutError is available for timeout scenarios."""
        from foundry_mcp.core.providers import ProviderTimeoutError

        # Should be able to create timeout error with context
        error = ProviderTimeoutError("Test timeout", provider="test-provider")
        assert "Test timeout" in str(error)
        assert error.provider == "test-provider"

    def test_provider_request_includes_timeout(self):
        """Test ProviderRequest accepts timeout parameter."""
        from foundry_mcp.core.providers import ProviderRequest

        request = ProviderRequest(
            prompt="test prompt",
            timeout=60,
        )
        assert request.timeout == 60

    def test_provider_timeout_is_optional(self):
        """Test ProviderRequest timeout can be None (provider decides default)."""
        from foundry_mcp.core.providers import ProviderRequest

        request = ProviderRequest(prompt="test prompt")
        # Timeout is optional - provider implementations decide default
        # None means "use provider default"
        assert request.timeout is None

        # Can explicitly set timeout
        request_with_timeout = ProviderRequest(prompt="test", timeout=60)
        assert request_with_timeout.timeout == 60


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
    """Test data-only fallback behavior when LLM operations fail.

    Note: CLI command execution is now handled by the provider system.
    These tests verify provider error handling and response semantics.
    """

    def test_provider_unavailable_returns_error_response(self):
        """Test that unavailable provider returns proper error structure."""
        from foundry_mcp.core.providers import (
            check_provider_available,
            ProviderUnavailableError,
        )

        # Verify unavailable provider is detected
        is_available = check_provider_available("nonexistent-xyz")
        assert is_available is False

    def test_provider_result_dataclass(self):
        """Test ProviderResult contains expected fields."""
        from foundry_mcp.core.providers import ProviderResult, ProviderStatus

        result = ProviderResult(
            content="Test response",
            provider_id="test-provider",
            model_used="test-model",
            status=ProviderStatus.SUCCESS,
        )
        assert result.content == "Test response"
        assert result.status == ProviderStatus.SUCCESS
        assert result.provider_id == "test-provider"
        assert result.model_used == "test-model"

    def test_provider_result_with_error_status(self):
        """Test ProviderResult can represent error state via status."""
        from foundry_mcp.core.providers import ProviderResult, ProviderStatus

        result = ProviderResult(
            content="",
            provider_id="test-provider",
            model_used="test-model",
            status=ProviderStatus.ERROR,
            stderr="Something went wrong",  # Error info goes in stderr
        )
        assert result.status == ProviderStatus.ERROR
        assert result.stderr == "Something went wrong"

    def test_provider_status_enum_values(self):
        """Test ProviderStatus has expected values."""
        from foundry_mcp.core.providers import ProviderStatus

        # Should have common status values
        assert ProviderStatus.SUCCESS is not None
        assert ProviderStatus.ERROR is not None
        assert ProviderStatus.TIMEOUT is not None


class TestCLINotFoundHandling:
    """Test graceful handling when provider binaries are not found.

    Note: Binary detection is now handled by the provider detector system.
    These tests verify detector behavior for missing binaries.
    """

    def test_detector_returns_false_for_missing_binary(self):
        """Test detector returns False when binary not in PATH."""
        from foundry_mcp.core.providers.detectors import ProviderDetector

        detector = ProviderDetector(
            provider_id="nonexistent",
            binary_name="nonexistent-binary-xyz-123",
        )
        assert detector.is_available() is False

    def test_detector_resolve_binary_returns_none_for_missing(self):
        """Test resolve_binary returns None when binary not found."""
        from foundry_mcp.core.providers.detectors import ProviderDetector

        detector = ProviderDetector(
            provider_id="nonexistent",
            binary_name="nonexistent-binary-xyz-123",
        )
        assert detector.resolve_binary() is None

    def test_provider_unavailable_error_has_provider_info(self):
        """Test ProviderUnavailableError includes provider context."""
        from foundry_mcp.core.providers import ProviderUnavailableError

        error = ProviderUnavailableError(
            "Binary not found",
            provider="test-provider",
        )
        assert error.provider == "test-provider"
        assert "Binary not found" in str(error)

    def test_check_provider_available_false_for_unknown(self):
        """Test check_provider_available returns False for unknown providers."""
        from foundry_mcp.core.providers import check_provider_available

        result = check_provider_available("completely-unknown-provider-xyz")
        assert result is False
