"""Tests for deep research timeout resilience enhancement.

Tests the _execute_provider_async method with timeout protection, retry,
and fallback logic.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.config.research import ResearchConfig
from foundry_mcp.core.errors.provider import ProviderTimeoutError
from foundry_mcp.core.providers import ProviderResult, ProviderStatus
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow


class TestConfigFallbackProviders:
    """Tests for phase fallback provider configuration."""

    def test_get_phase_fallback_providers_empty_by_default(self) -> None:
        """Test that fallback providers are empty by default."""
        config = ResearchConfig()
        assert config.get_phase_fallback_providers("planning") == []
        assert config.get_phase_fallback_providers("synthesis") == []

    def test_get_phase_fallback_providers_configured(self) -> None:
        """Test that configured fallback providers are returned."""
        config = ResearchConfig(
            deep_research_planning_providers=["gemini", "claude"],
            deep_research_synthesis_providers=["claude:opus", "gemini:pro"],
        )
        assert config.get_phase_fallback_providers("planning") == ["gemini", "claude"]
        assert config.get_phase_fallback_providers("synthesis") == ["claude:opus", "gemini:pro"]

    def test_get_phase_fallback_providers_unknown_phase(self) -> None:
        """Test that unknown phases return empty list."""
        config = ResearchConfig(
            deep_research_planning_providers=["gemini"],
        )
        assert config.get_phase_fallback_providers("unknown_phase") == []

    def test_retry_settings_default(self) -> None:
        """Test default retry settings."""
        config = ResearchConfig()
        assert config.deep_research_max_retries == 2
        assert config.deep_research_retry_delay == 5.0

    def test_retry_settings_custom(self) -> None:
        """Test custom retry settings."""
        config = ResearchConfig(
            deep_research_max_retries=5,
            deep_research_retry_delay=10.0,
        )
        assert config.deep_research_max_retries == 5
        assert config.deep_research_retry_delay == 10.0


class TestConfigFromTomlDict:
    """Tests for parsing fallback config from TOML."""

    def test_parse_phase_fallback_providers(self) -> None:
        """Test that phase fallback providers are parsed from TOML dict."""
        data = {
            "deep_research_planning_providers": ["gemini:pro", "claude:sonnet"],
            "deep_research_synthesis_providers": ["gemini:pro"],
            "deep_research_max_retries": 3,
            "deep_research_retry_delay": 8.5,
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_planning_providers == ["gemini:pro", "claude:sonnet"]
        assert config.deep_research_synthesis_providers == ["gemini:pro"]
        assert config.deep_research_max_retries == 3
        assert config.deep_research_retry_delay == 8.5

    def test_parse_phase_fallback_providers_string(self) -> None:
        """Test that comma-separated string is parsed correctly."""
        data = {
            "deep_research_planning_providers": "gemini,claude,codex",
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_planning_providers == ["gemini", "claude", "codex"]


class TestExecuteProviderAsyncExists:
    """Tests that _execute_provider_async method exists and has correct signature."""

    def test_method_exists(self) -> None:
        """Test that the async method exists on the workflow class."""
        config = ResearchConfig()
        workflow = DeepResearchWorkflow(config)
        assert hasattr(workflow, "_execute_provider_async")
        assert asyncio.iscoroutinefunction(workflow._execute_provider_async)

    def test_method_signature(self) -> None:
        """Test that the method has the expected parameters."""
        import inspect

        from foundry_mcp.core.research.workflows.base import ResearchWorkflowBase

        method = getattr(ResearchWorkflowBase, "_execute_provider_async", None)
        assert method is not None

        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        expected_params = [
            "self",
            "prompt",
            "provider_id",
            "system_prompt",
            "model",
            "timeout",
            "temperature",
            "max_tokens",
            "hooks",
            "phase",
            "fallback_providers",
            "max_retries",
            "retry_delay",
            "max_timeout_retries",
        ]
        assert params == expected_params


class TestWorkflowResultTimeoutMetadata:
    """Tests for WorkflowResult with timeout metadata."""

    def test_timeout_metadata_in_result(self) -> None:
        """Test that WorkflowResult can carry timeout metadata."""
        result = WorkflowResult(
            success=False,
            content="",
            error="Timed out after 60s",
            metadata={
                "phase": "planning",
                "timeout": True,
                "retries": 2,
                "providers_tried": ["gemini", "claude"],
            },
        )
        assert result.success is False
        assert result.metadata["timeout"] is True
        assert result.metadata["phase"] == "planning"
        assert result.metadata["retries"] == 2
        assert result.metadata["providers_tried"] == ["gemini", "claude"]


def _make_workflow() -> DeepResearchWorkflow:
    """Create a workflow with minimal config for testing."""
    return DeepResearchWorkflow(ResearchConfig())


def _make_success_result() -> ProviderResult:
    """Create a successful ProviderResult."""
    return ProviderResult(
        content="ok",
        provider_id="test",
        model_used="test-model",
        status=ProviderStatus.SUCCESS,
    )


class TestTimeoutRetryCap:
    """Tests for max_timeout_retries parameter in _execute_provider_async."""

    @pytest.mark.asyncio
    async def test_timeout_no_retry_by_default(self) -> None:
        """Timeout errors should NOT be retried when max_timeout_retries=0 (default)."""
        workflow = _make_workflow()
        call_count = 0

        def fake_generate(request):
            nonlocal call_count
            call_count += 1
            raise ProviderTimeoutError("timed out", provider="test")

        mock_provider = MagicMock()
        mock_provider.generate = fake_generate

        with patch.object(workflow, "_resolve_provider", return_value=mock_provider):
            result = await workflow._execute_provider_async(
                prompt="test prompt",
                provider_id="test",
                timeout=1.0,
                phase="test",
                max_retries=3,
                retry_delay=0.0,
                max_timeout_retries=0,
            )

        assert result.success is False
        assert result.metadata.get("timeout") is True
        # Should be called exactly once — no retries for timeout
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_non_timeout_errors_still_retry(self) -> None:
        """Non-timeout errors should still retry up to max_retries."""
        workflow = _make_workflow()
        call_count = 0

        def fake_generate(request):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("transient API error")

        mock_provider = MagicMock()
        mock_provider.generate = fake_generate

        with patch.object(workflow, "_resolve_provider", return_value=mock_provider):
            result = await workflow._execute_provider_async(
                prompt="test prompt",
                provider_id="test",
                timeout=1.0,
                phase="test",
                max_retries=2,
                retry_delay=0.0,
                max_timeout_retries=0,
            )

        assert result.success is False
        # Should be called 3 times (initial + 2 retries)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_retry_cap_of_one(self) -> None:
        """max_timeout_retries=1 allows exactly one timeout retry."""
        workflow = _make_workflow()
        call_count = 0

        def fake_generate(request):
            nonlocal call_count
            call_count += 1
            raise ProviderTimeoutError("timed out", provider="test")

        mock_provider = MagicMock()
        mock_provider.generate = fake_generate

        with patch.object(workflow, "_resolve_provider", return_value=mock_provider):
            result = await workflow._execute_provider_async(
                prompt="test prompt",
                provider_id="test",
                timeout=1.0,
                phase="test",
                max_retries=5,
                retry_delay=0.0,
                max_timeout_retries=1,
            )

        assert result.success is False
        assert result.metadata.get("timeout") is True
        # Should be called exactly 2 times (initial + 1 timeout retry)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_asyncio_timeout_also_capped(self) -> None:
        """asyncio.TimeoutError should also be capped by max_timeout_retries."""
        workflow = _make_workflow()
        call_count = 0

        async def fake_wait_for(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise asyncio.TimeoutError()

        mock_provider = MagicMock()
        mock_provider.generate = MagicMock()  # won't be called directly

        with (
            patch.object(workflow, "_resolve_provider", return_value=mock_provider),
            patch("asyncio.wait_for", side_effect=fake_wait_for),
        ):
            result = await workflow._execute_provider_async(
                prompt="test prompt",
                provider_id="test",
                timeout=1.0,
                phase="test",
                max_retries=3,
                retry_delay=0.0,
                max_timeout_retries=0,
            )

        assert result.success is False
        assert result.metadata.get("timeout") is True
        # Should be called exactly once — no retries for asyncio.TimeoutError
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_timeout_then_success_on_retry(self) -> None:
        """When max_timeout_retries=1, a timeout followed by success should work."""
        workflow = _make_workflow()
        call_count = 0

        def fake_generate(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ProviderTimeoutError("timed out", provider="test")
            return _make_success_result()

        mock_provider = MagicMock()
        mock_provider.generate = fake_generate

        with patch.object(workflow, "_resolve_provider", return_value=mock_provider):
            result = await workflow._execute_provider_async(
                prompt="test prompt",
                provider_id="test",
                timeout=1.0,
                phase="test",
                max_retries=3,
                retry_delay=0.0,
                max_timeout_retries=1,
            )

        assert result.success is True
        assert call_count == 2
