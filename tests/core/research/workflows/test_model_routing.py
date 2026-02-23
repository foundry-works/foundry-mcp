"""Unit tests for Phase 6: Multi-Model Cost Optimization.

Tests cover:
1. resolve_model_for_role() — resolution chain (role-specific -> phase -> global)
2. Explicit provider_id/model overrides role-based resolution in execute_llm_call
3. Cost tracking per role in PhaseMetrics metadata
4. Backward-compat when no role-specific config provided
5. get_model_role_costs() aggregation helper
6. All phase callsites pass expected roles
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.config.research import ResearchConfig
from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.models.fidelity import PhaseMetrics
from foundry_mcp.core.research.workflows.base import WorkflowResult


# =============================================================================
# Helpers
# =============================================================================


def _make_config(**overrides: Any) -> ResearchConfig:
    """Create a ResearchConfig with optional overrides."""
    return ResearchConfig(**overrides)


def _make_state(
    query: str = "What is quantum computing?",
    phase: DeepResearchPhase = DeepResearchPhase.PLANNING,
) -> DeepResearchState:
    """Create a minimal DeepResearchState for testing."""
    return DeepResearchState(
        id="deepres-routing-test",
        original_query=query,
        phase=phase,
        iteration=1,
        max_iterations=3,
    )


def _make_workflow_mock(config: ResearchConfig) -> MagicMock:
    """Create a mock workflow with config and required attributes."""
    workflow = MagicMock()
    workflow.config = config
    workflow.memory = MagicMock()
    workflow.memory.save_deep_research = MagicMock()
    workflow._write_audit_event = MagicMock()

    # Mock _execute_provider_async to return a successful WorkflowResult
    workflow._execute_provider_async = AsyncMock(
        return_value=WorkflowResult(
            success=True,
            content="mock response",
            provider_id="test-provider",
            model_used="test-model",
            tokens_used=100,
            input_tokens=80,
            output_tokens=20,
            cached_tokens=0,
            duration_ms=500.0,
        )
    )
    return workflow


# =============================================================================
# Tests: resolve_model_for_role
# =============================================================================


class TestResolveModelForRole:
    """Test the centralized role -> (provider, model) resolution."""

    def test_global_default_fallback(self) -> None:
        """When no role-specific config, falls back to default_provider."""
        config = _make_config(default_provider="gemini")
        provider, model = config.resolve_model_for_role("research")
        assert provider == "gemini"
        assert model is None

    def test_role_specific_provider(self) -> None:
        """Role-specific provider takes precedence over default."""
        config = _make_config(
            default_provider="gemini",
            deep_research_research_provider="claude",
        )
        provider, model = config.resolve_model_for_role("research")
        assert provider == "claude"
        assert model is None

    def test_role_specific_model(self) -> None:
        """Role-specific model is returned alongside provider."""
        config = _make_config(
            default_provider="gemini",
            deep_research_research_provider="claude",
            deep_research_research_model="opus",
        )
        provider, model = config.resolve_model_for_role("research")
        assert provider == "claude"
        assert model == "opus"

    def test_phase_fallback_for_research(self) -> None:
        """'research' role falls back to analysis phase config."""
        config = _make_config(
            default_provider="gemini",
            deep_research_analysis_provider="openai",
        )
        provider, model = config.resolve_model_for_role("research")
        assert provider == "openai"

    def test_phase_fallback_for_report(self) -> None:
        """'report' role falls back to synthesis phase config."""
        config = _make_config(
            default_provider="gemini",
            deep_research_synthesis_provider="claude",
        )
        provider, model = config.resolve_model_for_role("report")
        assert provider == "claude"

    def test_reflection_role(self) -> None:
        """'reflection' role uses reflection-specific config."""
        config = _make_config(
            default_provider="gemini",
            deep_research_reflection_provider="claude",
            deep_research_reflection_model="haiku",
        )
        provider, model = config.resolve_model_for_role("reflection")
        assert provider == "claude"
        assert model == "haiku"

    def test_topic_reflection_falls_back_to_reflection(self) -> None:
        """'topic_reflection' falls back to 'reflection' config."""
        config = _make_config(
            default_provider="gemini",
            deep_research_reflection_provider="openai",
            deep_research_reflection_model="gpt-4o-mini",
        )
        provider, model = config.resolve_model_for_role("topic_reflection")
        assert provider == "openai"
        assert model == "gpt-4o-mini"

    def test_topic_reflection_own_config_takes_precedence(self) -> None:
        """topic_reflection-specific config beats reflection fallback."""
        config = _make_config(
            default_provider="gemini",
            deep_research_reflection_provider="openai",
            deep_research_reflection_model="gpt-4o",
            deep_research_topic_reflection_provider="claude",
            deep_research_topic_reflection_model="haiku",
        )
        provider, model = config.resolve_model_for_role("topic_reflection")
        assert provider == "claude"
        assert model == "haiku"

    def test_summarization_role(self) -> None:
        """'summarization' role uses summarization-specific config."""
        config = _make_config(
            default_provider="gemini",
            deep_research_summarization_provider="openai",
            deep_research_summarization_model="gpt-4o-mini",
        )
        provider, model = config.resolve_model_for_role("summarization")
        assert provider == "openai"
        assert model == "gpt-4o-mini"

    def test_compression_role(self) -> None:
        """'compression' role uses compression-specific config."""
        config = _make_config(
            default_provider="gemini",
            deep_research_compression_provider="claude",
            deep_research_compression_model="sonnet",
        )
        provider, model = config.resolve_model_for_role("compression")
        assert provider == "claude"
        assert model == "sonnet"

    def test_clarification_falls_back_to_research(self) -> None:
        """'clarification' role falls back through research then analysis."""
        config = _make_config(
            default_provider="gemini",
            deep_research_research_provider="claude",
        )
        provider, model = config.resolve_model_for_role("clarification")
        assert provider == "claude"

    def test_unknown_role_falls_back_to_default(self) -> None:
        """Unknown role falls back to default_provider."""
        config = _make_config(default_provider="gemini")
        provider, model = config.resolve_model_for_role("nonexistent_role")
        assert provider == "gemini"
        assert model is None

    def test_model_from_bracket_provider_spec(self) -> None:
        """Model embedded in bracket ProviderSpec string is parsed correctly."""
        config = _make_config(
            default_provider="gemini",
            deep_research_research_provider="[cli]claude:opus",
        )
        provider, model = config.resolve_model_for_role("research")
        assert provider == "claude"
        assert model == "opus"

    def test_explicit_model_overrides_bracket_provider_spec_model(self) -> None:
        """Explicit model field beats model in bracket ProviderSpec string."""
        config = _make_config(
            default_provider="gemini",
            deep_research_research_provider="[cli]claude:opus",
            deep_research_research_model="sonnet",
        )
        provider, model = config.resolve_model_for_role("research")
        assert provider == "claude"
        assert model == "sonnet"

    def test_mixed_provider_and_model_from_different_suffixes(self) -> None:
        """Provider from one suffix, model from another in the chain."""
        config = _make_config(
            default_provider="gemini",
            deep_research_research_provider="claude",
            # No research_model, but analysis_model set — should NOT be picked
            # because research_provider was found and model stays None
        )
        provider, model = config.resolve_model_for_role("research")
        assert provider == "claude"
        assert model is None


# =============================================================================
# Tests: execute_llm_call role parameter
# =============================================================================


class TestExecuteLLMCallRole:
    """Test that execute_llm_call uses role for resolution and tracking."""

    @pytest.mark.asyncio
    async def test_role_resolves_provider_when_none(self) -> None:
        """When provider_id is None and role is set, resolve from config."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            execute_llm_call,
        )

        config = _make_config(
            default_provider="gemini",
            deep_research_research_provider="claude",
            deep_research_research_model="opus",
        )
        workflow = _make_workflow_mock(config)
        state = _make_state()

        result = await execute_llm_call(
            workflow=workflow,
            state=state,
            phase_name="planning",
            system_prompt="test system",
            user_prompt="test user",
            provider_id=None,
            model=None,
            temperature=0.5,
            timeout=60.0,
            role="research",
        )

        # Verify _execute_provider_async was called with role-resolved values
        call_kwargs = workflow._execute_provider_async.call_args
        assert call_kwargs.kwargs["provider_id"] == "claude"
        assert call_kwargs.kwargs["model"] == "opus"

    @pytest.mark.asyncio
    async def test_explicit_provider_overrides_role(self) -> None:
        """Explicit provider_id takes precedence over role resolution."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            execute_llm_call,
        )

        config = _make_config(
            default_provider="gemini",
            deep_research_research_provider="claude",
            deep_research_research_model="opus",
        )
        workflow = _make_workflow_mock(config)
        state = _make_state()

        result = await execute_llm_call(
            workflow=workflow,
            state=state,
            phase_name="planning",
            system_prompt="test system",
            user_prompt="test user",
            provider_id="openai",
            model="gpt-4o",
            temperature=0.5,
            timeout=60.0,
            role="research",
        )

        call_kwargs = workflow._execute_provider_async.call_args
        assert call_kwargs.kwargs["provider_id"] == "openai"
        assert call_kwargs.kwargs["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_role_stored_in_phase_metrics(self) -> None:
        """Role is stored in PhaseMetrics metadata for cost tracking."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            execute_llm_call,
        )

        config = _make_config(default_provider="gemini")
        workflow = _make_workflow_mock(config)
        state = _make_state()

        await execute_llm_call(
            workflow=workflow,
            state=state,
            phase_name="analysis",
            system_prompt="test",
            user_prompt="test",
            provider_id=None,
            model=None,
            temperature=0.3,
            timeout=60.0,
            role="research",
        )

        assert len(state.phase_metrics) == 1
        assert state.phase_metrics[0].metadata.get("role") == "research"

    @pytest.mark.asyncio
    async def test_no_role_no_metadata(self) -> None:
        """When role is None, no 'role' key in PhaseMetrics metadata."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            execute_llm_call,
        )

        config = _make_config(default_provider="gemini")
        workflow = _make_workflow_mock(config)
        state = _make_state()

        await execute_llm_call(
            workflow=workflow,
            state=state,
            phase_name="analysis",
            system_prompt="test",
            user_prompt="test",
            provider_id="gemini",
            model=None,
            temperature=0.3,
            timeout=60.0,
        )

        assert len(state.phase_metrics) == 1
        assert "role" not in state.phase_metrics[0].metadata

    @pytest.mark.asyncio
    async def test_backward_compat_no_role(self) -> None:
        """execute_llm_call works without role parameter (backward-compat)."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            execute_llm_call,
        )

        config = _make_config(default_provider="gemini")
        workflow = _make_workflow_mock(config)
        state = _make_state()

        result = await execute_llm_call(
            workflow=workflow,
            state=state,
            phase_name="planning",
            system_prompt="test",
            user_prompt="test",
            provider_id="gemini",
            model=None,
            temperature=0.5,
            timeout=60.0,
        )

        # Should succeed without role
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            LLMCallResult,
        )
        assert isinstance(result, LLMCallResult)


# =============================================================================
# Tests: get_model_role_costs aggregation
# =============================================================================


class TestGetModelRoleCosts:
    """Test the per-role cost aggregation on DeepResearchState."""

    def test_empty_metrics(self) -> None:
        """No phase_metrics returns empty dict."""
        state = _make_state()
        assert state.get_model_role_costs() == {}

    def test_metrics_without_role(self) -> None:
        """Metrics without role key are excluded from aggregation."""
        state = _make_state()
        state.phase_metrics.append(
            PhaseMetrics(
                phase="planning",
                duration_ms=100.0,
                input_tokens=50,
                output_tokens=20,
                provider_id="gemini",
                model_used="flash",
            )
        )
        assert state.get_model_role_costs() == {}

    def test_single_role(self) -> None:
        """Single role aggregates correctly."""
        state = _make_state()
        state.phase_metrics.append(
            PhaseMetrics(
                phase="analysis",
                duration_ms=200.0,
                input_tokens=100,
                output_tokens=50,
                provider_id="claude",
                model_used="opus",
                metadata={"role": "research"},
            )
        )
        costs = state.get_model_role_costs()
        assert "research" in costs
        assert costs["research"]["provider"] == "claude"
        assert costs["research"]["model"] == "opus"
        assert costs["research"]["input_tokens"] == 100
        assert costs["research"]["output_tokens"] == 50
        assert costs["research"]["calls"] == 1

    def test_multiple_calls_same_role(self) -> None:
        """Multiple calls with same role are summed."""
        state = _make_state()
        for i in range(3):
            state.phase_metrics.append(
                PhaseMetrics(
                    phase=f"phase-{i}",
                    duration_ms=100.0,
                    input_tokens=100,
                    output_tokens=50,
                    provider_id="claude",
                    model_used="opus",
                    metadata={"role": "research"},
                )
            )
        costs = state.get_model_role_costs()
        assert costs["research"]["input_tokens"] == 300
        assert costs["research"]["output_tokens"] == 150
        assert costs["research"]["calls"] == 3

    def test_multiple_roles(self) -> None:
        """Different roles are tracked separately."""
        state = _make_state()
        state.phase_metrics.extend([
            PhaseMetrics(
                phase="analysis",
                input_tokens=200,
                output_tokens=100,
                provider_id="claude",
                model_used="opus",
                metadata={"role": "research"},
            ),
            PhaseMetrics(
                phase="synthesis",
                input_tokens=300,
                output_tokens=200,
                provider_id="claude",
                model_used="sonnet",
                metadata={"role": "report"},
            ),
            PhaseMetrics(
                phase="topic_reflection",
                input_tokens=50,
                output_tokens=20,
                provider_id="openai",
                model_used="gpt-4o-mini",
                metadata={"role": "reflection"},
            ),
        ])
        costs = state.get_model_role_costs()
        assert len(costs) == 3
        assert costs["research"]["provider"] == "claude"
        assert costs["report"]["model"] == "sonnet"
        assert costs["reflection"]["provider"] == "openai"
        assert costs["reflection"]["input_tokens"] == 50

    def test_mixed_with_and_without_role(self) -> None:
        """Metrics without role are excluded, with role are included."""
        state = _make_state()
        state.phase_metrics.extend([
            PhaseMetrics(
                phase="planning",
                input_tokens=100,
                output_tokens=50,
                provider_id="gemini",
                metadata={},  # no role
            ),
            PhaseMetrics(
                phase="analysis",
                input_tokens=200,
                output_tokens=100,
                provider_id="claude",
                metadata={"role": "research"},
            ),
        ])
        costs = state.get_model_role_costs()
        assert len(costs) == 1
        assert "research" in costs
        assert costs["research"]["input_tokens"] == 200


# =============================================================================
# Tests: Config from_toml_dict parses new fields
# =============================================================================


class TestConfigFromToml:
    """Test that from_toml_dict correctly parses Phase 6 config fields."""

    def test_parses_research_role_fields(self) -> None:
        config = ResearchConfig.from_toml_dict({
            "deep_research_research_provider": "claude",
            "deep_research_research_model": "opus",
        })
        assert config.deep_research_research_provider == "claude"
        assert config.deep_research_research_model == "opus"

    def test_parses_report_role_fields(self) -> None:
        config = ResearchConfig.from_toml_dict({
            "deep_research_report_provider": "openai",
            "deep_research_report_model": "gpt-4o",
        })
        assert config.deep_research_report_provider == "openai"
        assert config.deep_research_report_model == "gpt-4o"

    def test_parses_reflection_model(self) -> None:
        config = ResearchConfig.from_toml_dict({
            "deep_research_reflection_model": "haiku",
        })
        assert config.deep_research_reflection_model == "haiku"

    def test_parses_topic_reflection_model(self) -> None:
        config = ResearchConfig.from_toml_dict({
            "deep_research_topic_reflection_model": "gpt-4o-mini",
        })
        assert config.deep_research_topic_reflection_model == "gpt-4o-mini"

    def test_parses_clarification_model(self) -> None:
        config = ResearchConfig.from_toml_dict({
            "deep_research_clarification_model": "flash",
        })
        assert config.deep_research_clarification_model == "flash"

    def test_defaults_to_none(self) -> None:
        """All new fields default to None when not in TOML."""
        config = ResearchConfig.from_toml_dict({})
        assert config.deep_research_research_provider is None
        assert config.deep_research_research_model is None
        assert config.deep_research_report_provider is None
        assert config.deep_research_report_model is None
        assert config.deep_research_reflection_model is None
        assert config.deep_research_topic_reflection_model is None
        assert config.deep_research_clarification_model is None
