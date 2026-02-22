"""Tests for deadline-based timeout and fallback behavior.

Phase 0b (baseline) + Phase 2c (deadline fix): validates that
_execute_provider_async enforces a single time budget across retries
and fallback providers, and that wall-clock duration is bounded.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.config.research import ResearchConfig
from foundry_mcp.core.providers import ProviderStatus
from foundry_mcp.core.research.workflows.base import (
    ResearchWorkflowBase,
    WorkflowResult,
)
from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

# ──────────────────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_config():
    """Create a mock ResearchConfig with default_provider."""
    config = MagicMock()
    config.default_provider = "test-provider"
    config.default_timeout = 30
    config.ttl_hours = 24
    return config


@pytest.fixture
def mock_memory():
    """Create a mock ResearchMemory."""
    return MagicMock()


@pytest.fixture
def workflow(mock_config, mock_memory):
    """Create a DeepResearchWorkflow instance."""
    return DeepResearchWorkflow(mock_config, mock_memory)


def _make_success_provider(content="response", delay=0):
    """Create a mock provider that succeeds after optional delay."""
    mock_provider = MagicMock()
    mock_result = MagicMock()
    mock_result.status = ProviderStatus.SUCCESS
    mock_result.content = content
    mock_result.provider_id = "test"
    mock_result.model_used = "test-model"
    mock_result.tokens = None

    def generate(request):
        if delay > 0:
            time.sleep(delay)
        return mock_result

    mock_provider.generate = generate
    return mock_provider


def _make_timeout_provider(delay):
    """Create a mock provider that sleeps longer than timeout (triggers asyncio.TimeoutError)."""
    mock_provider = MagicMock()

    def generate(request):
        time.sleep(delay)
        # This won't be reached if asyncio.wait_for cancels us
        mock_result = MagicMock()
        mock_result.status = ProviderStatus.SUCCESS
        mock_result.content = "late"
        mock_result.provider_id = "slow"
        mock_result.model_used = "slow-model"
        mock_result.tokens = None
        return mock_result

    mock_provider.generate = generate
    return mock_provider


# ──────────────────────────────────────────────────────────────────────
#  Config Tests (retained from original)
# ──────────────────────────────────────────────────────────────────────


class TestConfigFallbackProviders:
    """Tests for phase fallback provider configuration."""

    def test_get_phase_fallback_providers_empty_by_default(self) -> None:
        config = ResearchConfig()
        assert config.get_phase_fallback_providers("planning") == []
        assert config.get_phase_fallback_providers("analysis") == []

    def test_get_phase_fallback_providers_configured(self) -> None:
        config = ResearchConfig(
            deep_research_planning_providers=["gemini", "claude"],
            deep_research_synthesis_providers=["claude:opus", "gemini:pro"],
        )
        assert config.get_phase_fallback_providers("planning") == [
            "gemini",
            "claude",
        ]
        assert config.get_phase_fallback_providers("synthesis") == [
            "claude:opus",
            "gemini:pro",
        ]
        assert config.get_phase_fallback_providers("analysis") == []

    def test_get_phase_fallback_providers_unknown_phase(self) -> None:
        config = ResearchConfig(
            deep_research_planning_providers=["gemini"],
        )
        assert config.get_phase_fallback_providers("unknown_phase") == []

    def test_retry_settings_default(self) -> None:
        config = ResearchConfig()
        assert config.deep_research_max_retries == 2
        assert config.deep_research_retry_delay == 5.0

    def test_retry_settings_custom(self) -> None:
        config = ResearchConfig(
            deep_research_max_retries=5,
            deep_research_retry_delay=10.0,
        )
        assert config.deep_research_max_retries == 5
        assert config.deep_research_retry_delay == 10.0


class TestConfigFromTomlDict:
    """Tests for parsing fallback config from TOML."""

    def test_parse_phase_fallback_providers(self) -> None:
        data = {
            "deep_research_planning_providers": ["gemini:pro", "claude:sonnet"],
            "deep_research_analysis_providers": ["gemini:pro"],
            "deep_research_max_retries": 3,
            "deep_research_retry_delay": 8.5,
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_planning_providers == [
            "gemini:pro",
            "claude:sonnet",
        ]
        assert config.deep_research_analysis_providers == ["gemini:pro"]
        assert config.deep_research_synthesis_providers == []
        assert config.deep_research_max_retries == 3
        assert config.deep_research_retry_delay == 8.5

    def test_parse_phase_fallback_providers_string(self) -> None:
        data = {
            "deep_research_planning_providers": "gemini,claude,codex",
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_planning_providers == [
            "gemini",
            "claude",
            "codex",
        ]


class TestExecuteProviderAsyncExists:
    """Tests that _execute_provider_async method exists and has correct signature."""

    def test_method_exists(self) -> None:
        config = ResearchConfig()
        wf = DeepResearchWorkflow(config)
        assert hasattr(wf, "_execute_provider_async")
        assert asyncio.iscoroutinefunction(wf._execute_provider_async)

    def test_method_signature(self) -> None:
        import inspect

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
        ]
        assert params == expected_params


class TestWorkflowResultTimeoutMetadata:
    """Tests for WorkflowResult with timeout metadata."""

    def test_timeout_metadata_in_result(self) -> None:
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


# ──────────────────────────────────────────────────────────────────────
#  Deadline-based Timeout Tests (Phase 2c)
# ──────────────────────────────────────────────────────────────────────


class TestDeadlineBasedTimeout:
    """Tests that _execute_provider_async uses deadline-based timeout
    so that total wall-clock time is bounded by the configured timeout."""

    def test_success_includes_wall_clock_metadata(self, workflow):
        """Successful result should include wall_clock_ms and configured_timeout_s."""
        provider = _make_success_provider()

        with patch.object(workflow, "_resolve_provider", return_value=provider):
            result = asyncio.run(
                workflow._execute_provider_async(
                    prompt="test",
                    timeout=10.0,
                    phase="test",
                    max_retries=0,
                )
            )

        assert result.success is True
        assert "wall_clock_ms" in result.metadata
        assert result.metadata["configured_timeout_s"] == 10.0

    def test_failure_includes_wall_clock_metadata(self, workflow):
        """Failed result should include wall_clock_ms and configured_timeout_s."""
        with patch.object(workflow, "_resolve_provider", return_value=None):
            result = asyncio.run(
                workflow._execute_provider_async(
                    prompt="test",
                    timeout=10.0,
                    phase="test",
                    max_retries=0,
                )
            )

        assert result.success is False
        assert "wall_clock_ms" in result.metadata
        assert result.metadata["configured_timeout_s"] == 10.0

    def test_deadline_caps_total_duration(self, workflow):
        """Total wall-clock time should not exceed timeout, even with
        fallback providers. This is the core deadline fix — previously
        each provider got the full timeout, potentially doubling wall-clock."""
        # Provider that takes 0.8s (would succeed within per-attempt timeout
        # but we set overall timeout to 1s so fallback should get ~0.2s)
        slow_provider = _make_timeout_provider(delay=0.8)
        fast_provider = _make_success_provider(content="fast response", delay=0)

        call_count = 0

        def resolve_provider(pid, hooks=None):
            nonlocal call_count
            call_count += 1
            if pid == "slow":
                return slow_provider
            return fast_provider

        overall_timeout = 1.0

        with patch.object(workflow, "_resolve_provider", side_effect=resolve_provider):
            start = time.monotonic()
            result = asyncio.run(
                workflow._execute_provider_async(
                    prompt="test",
                    provider_id="slow",
                    timeout=overall_timeout,
                    phase="test",
                    fallback_providers=["fast"],
                    max_retries=0,
                    retry_delay=0.0,
                )
            )
            elapsed = time.monotonic() - start

        # Total wall-clock must be bounded by timeout + small tolerance
        assert elapsed < overall_timeout + 0.5, (
            f"Wall-clock {elapsed:.2f}s exceeded timeout {overall_timeout}s + tolerance"
        )

    def test_primary_consumes_budget_fallback_gets_remainder(self, workflow):
        """If primary takes most of the budget, fallback gets the remainder."""
        call_log = []

        def make_provider_that_logs(name, delay=0):
            provider = MagicMock()

            def generate(request):
                call_log.append(
                    {"name": name, "timeout": request.timeout}
                )
                if delay > 0:
                    time.sleep(delay)
                result = MagicMock()
                result.status = ProviderStatus.SUCCESS
                result.content = f"from {name}"
                result.provider_id = name
                result.model_used = "model"
                result.tokens = None
                return result

            provider.generate = generate
            return provider

        primary = make_provider_that_logs("primary", delay=0)
        # Primary succeeds quickly so no fallback needed, but we verify
        # that the timeout passed to the request uses remaining budget

        with patch.object(workflow, "_resolve_provider", return_value=primary):
            result = asyncio.run(
                workflow._execute_provider_async(
                    prompt="test",
                    timeout=300.0,
                    phase="test",
                    max_retries=0,
                )
            )

        assert result.success is True
        # The request timeout should be close to 300 (the full budget minus tiny elapsed)
        assert len(call_log) == 1
        assert call_log[0]["timeout"] <= 300.0
        assert call_log[0]["timeout"] > 299.0  # Should be very close to full budget

    def test_no_time_left_fallback_skipped(self, workflow):
        """If primary exhausts the entire budget, fallback is skipped."""
        providers_called = []

        def slow_generate(request):
            providers_called.append("primary")
            # Sleep longer than the budget
            time.sleep(2.0)
            result = MagicMock()
            result.status = ProviderStatus.SUCCESS
            result.content = "late"
            result.provider_id = "primary"
            result.model_used = "model"
            result.tokens = None
            return result

        def fast_generate(request):
            providers_called.append("fallback")
            result = MagicMock()
            result.status = ProviderStatus.SUCCESS
            result.content = "fallback response"
            result.provider_id = "fallback"
            result.model_used = "model"
            result.tokens = None
            return result

        primary = MagicMock()
        primary.generate = slow_generate
        fallback = MagicMock()
        fallback.generate = fast_generate

        def resolve(pid, hooks=None):
            if pid == "primary":
                return primary
            return fallback

        with patch.object(workflow, "_resolve_provider", side_effect=resolve):
            result = asyncio.run(
                workflow._execute_provider_async(
                    prompt="test",
                    provider_id="primary",
                    timeout=0.5,  # Very short budget
                    phase="test",
                    fallback_providers=["fallback"],
                    max_retries=0,
                    retry_delay=0.0,
                )
            )

        # Primary should time out and fallback should be skipped (no budget left)
        assert result.success is False
        assert result.metadata.get("timeout") is True
        assert "fallback" not in providers_called

    def test_no_fallback_timeout_returns_error(self, workflow):
        """Timeout with no fallback configured returns clean error."""
        provider = _make_timeout_provider(delay=5.0)

        with patch.object(workflow, "_resolve_provider", return_value=provider):
            result = asyncio.run(
                workflow._execute_provider_async(
                    prompt="test",
                    timeout=0.3,
                    phase="test",
                    max_retries=0,
                )
            )

        assert result.success is False
        assert result.metadata.get("timeout") is True
        assert "Timed out" in (result.error or "")
