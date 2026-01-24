"""Tests for deep research timeout resilience enhancement.

Tests the _execute_provider_async method with timeout protection, retry,
and fallback logic.
"""

import asyncio

from foundry_mcp.config import ResearchConfig
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow


class TestConfigFallbackProviders:
    """Tests for phase fallback provider configuration."""

    def test_get_phase_fallback_providers_empty_by_default(self) -> None:
        """Test that fallback providers are empty by default."""
        config = ResearchConfig()
        assert config.get_phase_fallback_providers("planning") == []
        assert config.get_phase_fallback_providers("analysis") == []
        assert config.get_phase_fallback_providers("synthesis") == []
        assert config.get_phase_fallback_providers("refinement") == []

    def test_get_phase_fallback_providers_configured(self) -> None:
        """Test that configured fallback providers are returned."""
        config = ResearchConfig(
            deep_research_planning_providers=["gemini", "claude"],
            deep_research_synthesis_providers=["claude:opus", "gemini:pro"],
        )
        assert config.get_phase_fallback_providers("planning") == ["gemini", "claude"]
        assert config.get_phase_fallback_providers("synthesis") == ["claude:opus", "gemini:pro"]
        # Unconfigured phases return empty
        assert config.get_phase_fallback_providers("analysis") == []
        assert config.get_phase_fallback_providers("refinement") == []

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
            "deep_research_analysis_providers": ["gemini:pro"],
            "deep_research_max_retries": 3,
            "deep_research_retry_delay": 8.5,
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_planning_providers == ["gemini:pro", "claude:sonnet"]
        assert config.deep_research_analysis_providers == ["gemini:pro"]
        assert config.deep_research_synthesis_providers == []
        assert config.deep_research_refinement_providers == []
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
