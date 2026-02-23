"""Unit and integration tests for Phase 2: LLM-Driven Supervisor Reflection.

Tests cover:
1. ReflectionDecision dataclass and serialization
2. _parse_reflection_response() — valid JSON, malformed, missing fields, edge cases
3. _build_reflection_llm_prompt() — per-phase context inclusion
4. async_think_pause() — LLM call success, failure, fallback behavior
5. _maybe_reflect() — enabled/disabled paths, audit event recording
6. Integration: reflection doesn't break existing workflow (disabled by default)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.models.enums import ConfidenceLevel
from foundry_mcp.core.research.models.sources import (
    ResearchSource,
    SourceQuality,
    SubQuery,
)
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research.orchestration import (
    AgentRole,
    ReflectionDecision,
    SupervisorOrchestrator,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_state(
    query: str = "How does deep learning work?",
    phase: DeepResearchPhase = DeepResearchPhase.PLANNING,
) -> DeepResearchState:
    """Create a minimal DeepResearchState for testing."""
    return DeepResearchState(
        id="deepres-test-reflect",
        original_query=query,
        phase=phase,
        iteration=1,
        max_iterations=3,
    )


def _make_source(
    source_id: str = "src-1",
    quality: SourceQuality = SourceQuality.HIGH,
) -> ResearchSource:
    return ResearchSource(
        id=source_id,
        title=f"Source {source_id}",
        content="Test content",
        quality=quality,
        url=f"https://example.com/{source_id}",
    )


# =============================================================================
# Unit tests: ReflectionDecision
# =============================================================================


class TestReflectionDecision:
    """Tests for ReflectionDecision dataclass."""

    def test_to_dict(self) -> None:
        """to_dict serializes all fields correctly."""
        decision = ReflectionDecision(
            quality_assessment="Good quality output",
            proceed=True,
            adjustments=["Consider more sources"],
            rationale="Output is sufficient",
            phase="planning",
            provider_id="test-provider",
            model_used="test-model",
            tokens_used=150,
            duration_ms=500.0,
        )
        d = decision.to_dict()

        assert d["quality_assessment"] == "Good quality output"
        assert d["proceed"] is True
        assert d["adjustments"] == ["Consider more sources"]
        assert d["rationale"] == "Output is sufficient"
        assert d["phase"] == "planning"
        assert d["provider_id"] == "test-provider"
        assert d["model_used"] == "test-model"
        assert d["tokens_used"] == 150
        assert d["duration_ms"] == 500.0

    def test_default_values(self) -> None:
        """Defaults are sensible."""
        decision = ReflectionDecision(
            quality_assessment="OK",
            proceed=True,
        )
        assert decision.adjustments == []
        assert decision.rationale == ""
        assert decision.phase == ""
        assert decision.provider_id is None
        assert decision.tokens_used == 0


# =============================================================================
# Unit tests: _parse_reflection_response
# =============================================================================


class TestParseReflectionResponse:
    """Tests for SupervisorOrchestrator._parse_reflection_response()."""

    def setup_method(self) -> None:
        self.orchestrator = SupervisorOrchestrator()

    def test_valid_json_proceed_true(self) -> None:
        """Valid JSON with proceed=true returns correct decision."""
        content = json.dumps(
            {
                "quality_assessment": "Planning produced comprehensive sub-queries",
                "proceed": True,
                "adjustments": [],
                "rationale": "Sub-queries cover all key aspects",
            }
        )
        decision = self.orchestrator._parse_reflection_response(
            content,
            phase=DeepResearchPhase.PLANNING,
            provider_id="test",
            tokens_used=100,
            duration_ms=300.0,
        )

        assert decision.proceed is True
        assert "comprehensive" in decision.quality_assessment
        assert decision.phase == "planning"
        assert decision.tokens_used == 100

    def test_valid_json_proceed_false(self) -> None:
        """Valid JSON with proceed=false and adjustments."""
        content = json.dumps(
            {
                "quality_assessment": "Only 1 sub-query generated",
                "proceed": False,
                "adjustments": ["Add more sub-queries", "Cover alternative angles"],
                "rationale": "Insufficient query decomposition",
            }
        )
        decision = self.orchestrator._parse_reflection_response(
            content,
            phase=DeepResearchPhase.PLANNING,
        )

        assert decision.proceed is False
        assert len(decision.adjustments) == 2
        assert "Insufficient" in decision.rationale

    def test_empty_content_defaults_proceed(self) -> None:
        """Empty content falls back to proceed=True."""
        decision = self.orchestrator._parse_reflection_response(
            "",
            phase=DeepResearchPhase.ANALYSIS,
        )

        assert decision.proceed is True
        assert "parse failure" in decision.rationale.lower()

    def test_malformed_json_defaults_proceed(self) -> None:
        """Malformed JSON falls back to proceed=True."""
        decision = self.orchestrator._parse_reflection_response(
            "{broken!!}",
            phase=DeepResearchPhase.GATHERING,
        )

        assert decision.proceed is True
        assert decision.phase == "gathering"

    def test_no_json_in_text_defaults_proceed(self) -> None:
        """Plain text without JSON falls back to proceed=True."""
        decision = self.orchestrator._parse_reflection_response(
            "The quality looks good, we should continue.",
            phase=DeepResearchPhase.SYNTHESIS,
        )

        assert decision.proceed is True

    def test_json_in_code_block(self) -> None:
        """JSON wrapped in markdown code block is extracted."""
        content = """```json
{"quality_assessment": "Good", "proceed": true, "adjustments": [], "rationale": "OK"}
```"""
        decision = self.orchestrator._parse_reflection_response(
            content,
            phase=DeepResearchPhase.ANALYSIS,
        )

        assert decision.proceed is True
        assert decision.quality_assessment == "Good"

    def test_missing_proceed_defaults_true(self) -> None:
        """Missing proceed key defaults to True."""
        content = json.dumps(
            {
                "quality_assessment": "Decent output",
                "rationale": "Looks fine",
            }
        )
        decision = self.orchestrator._parse_reflection_response(
            content,
            phase=DeepResearchPhase.PLANNING,
        )

        assert decision.proceed is True

    def test_adjustments_truncated_to_three(self) -> None:
        """More than 3 adjustments are truncated."""
        content = json.dumps(
            {
                "quality_assessment": "Needs work",
                "proceed": False,
                "adjustments": ["A1", "A2", "A3", "A4", "A5"],
                "rationale": "Many issues",
            }
        )
        decision = self.orchestrator._parse_reflection_response(
            content,
            phase=DeepResearchPhase.ANALYSIS,
        )

        assert len(decision.adjustments) == 3

    def test_non_list_adjustments_returns_empty(self) -> None:
        """Non-list adjustments value returns empty list."""
        content = json.dumps(
            {
                "quality_assessment": "OK",
                "proceed": True,
                "adjustments": "add more sources",
                "rationale": "Fine",
            }
        )
        decision = self.orchestrator._parse_reflection_response(
            content,
            phase=DeepResearchPhase.GATHERING,
        )

        assert decision.adjustments == []

    def test_provider_metadata_preserved(self) -> None:
        """Provider metadata is passed through to decision."""
        content = json.dumps(
            {
                "quality_assessment": "OK",
                "proceed": True,
                "adjustments": [],
                "rationale": "Fine",
            }
        )
        decision = self.orchestrator._parse_reflection_response(
            content,
            phase=DeepResearchPhase.PLANNING,
            provider_id="gemini",
            model_used="gemini-2.0-flash",
            tokens_used=200,
            duration_ms=450.0,
        )

        assert decision.provider_id == "gemini"
        assert decision.model_used == "gemini-2.0-flash"
        assert decision.tokens_used == 200
        assert decision.duration_ms == 450.0


# =============================================================================
# Unit tests: _build_reflection_llm_prompt
# =============================================================================


class TestBuildReflectionLLMPrompt:
    """Tests for per-phase reflection prompt building."""

    def setup_method(self) -> None:
        self.orchestrator = SupervisorOrchestrator()

    def test_planning_prompt_includes_sub_queries(self) -> None:
        """Planning reflection prompt includes sub-query count."""
        state = _make_state(phase=DeepResearchPhase.PLANNING)
        state.sub_queries = [
            SubQuery(query="What is deep learning?", source_types=[]),
            SubQuery(query="How do neural networks work?", source_types=[]),
        ]
        state.research_brief = "Test brief"

        prompt = self.orchestrator._build_reflection_llm_prompt(state, DeepResearchPhase.PLANNING)

        assert "Sub-queries generated: 2" in prompt
        assert "Research brief available: True" in prompt
        assert "deep learning" in state.original_query

    def test_gathering_prompt_includes_source_stats(self) -> None:
        """Gathering reflection prompt includes source quality distribution."""
        state = _make_state(phase=DeepResearchPhase.GATHERING)
        state.sources = [
            _make_source("s1", SourceQuality.HIGH),
            _make_source("s2", SourceQuality.HIGH),
            _make_source("s3", SourceQuality.MEDIUM),
        ]

        prompt = self.orchestrator._build_reflection_llm_prompt(state, DeepResearchPhase.GATHERING)

        assert "Sources collected: 3" in prompt
        assert "HIGH=2" in prompt
        assert "MEDIUM=1" in prompt

    def test_analysis_prompt_includes_findings(self) -> None:
        """Analysis reflection prompt includes finding counts."""
        state = _make_state(phase=DeepResearchPhase.ANALYSIS)
        # Mock findings
        finding = MagicMock()
        finding.confidence = ConfidenceLevel.HIGH
        state.findings = [finding]
        gap = MagicMock()
        gap.resolved = False
        state.gaps = [gap]

        prompt = self.orchestrator._build_reflection_llm_prompt(state, DeepResearchPhase.ANALYSIS)

        assert "Findings extracted: 1" in prompt
        assert "High confidence findings: 1" in prompt
        assert "Gaps identified: 1" in prompt

    def test_synthesis_prompt_includes_report_stats(self) -> None:
        """Synthesis reflection prompt includes report length."""
        state = _make_state(phase=DeepResearchPhase.SYNTHESIS)
        state.report = "A" * 500

        prompt = self.orchestrator._build_reflection_llm_prompt(state, DeepResearchPhase.SYNTHESIS)

        assert "Report generated: True" in prompt
        assert "Report length: 500 chars" in prompt

    def test_prompt_always_includes_base_context(self) -> None:
        """All prompts include research query, phase, and iteration."""
        state = _make_state(query="Test query", phase=DeepResearchPhase.CLARIFICATION)

        prompt = self.orchestrator._build_reflection_llm_prompt(state, DeepResearchPhase.CLARIFICATION)

        assert "Test query" in prompt
        assert "clarification" in prompt
        assert "1/3" in prompt


# =============================================================================
# Unit tests: async_think_pause
# =============================================================================


class TestAsyncThinkPause:
    """Tests for SupervisorOrchestrator.async_think_pause()."""

    def setup_method(self) -> None:
        self.orchestrator = SupervisorOrchestrator()

    @pytest.mark.asyncio
    async def test_no_workflow_returns_proceed(self) -> None:
        """Without workflow instance, returns proceed=True."""
        state = _make_state()
        decision = await self.orchestrator.async_think_pause(
            state=state,
            phase=DeepResearchPhase.PLANNING,
            workflow=None,
        )

        assert decision.proceed is True
        assert "no workflow" in decision.rationale.lower()

    @pytest.mark.asyncio
    async def test_successful_reflection(self) -> None:
        """Successful LLM call returns parsed reflection decision."""
        state = _make_state()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.content = json.dumps(
            {
                "quality_assessment": "Strong sub-query coverage",
                "proceed": True,
                "adjustments": [],
                "rationale": "All key angles covered",
            }
        )
        mock_result.provider_id = "test-provider"
        mock_result.model_used = "test-model"
        mock_result.tokens_used = 120
        mock_result.duration_ms = 400.0

        mock_workflow = MagicMock()
        mock_workflow.config.get_reflection_provider.return_value = "test-provider"
        mock_workflow.config.deep_research_reflection_timeout = 60.0
        mock_workflow._execute_provider_async = AsyncMock(return_value=mock_result)

        decision = await self.orchestrator.async_think_pause(
            state=state,
            phase=DeepResearchPhase.PLANNING,
            workflow=mock_workflow,
        )

        assert decision.proceed is True
        assert "Strong sub-query coverage" in decision.quality_assessment
        assert decision.provider_id == "test-provider"

    @pytest.mark.asyncio
    async def test_llm_call_exception_returns_proceed(self) -> None:
        """Exception during LLM call falls back to proceed=True."""
        state = _make_state()

        mock_workflow = MagicMock()
        mock_workflow.config.get_reflection_provider.return_value = "test-provider"
        mock_workflow.config.deep_research_reflection_timeout = 60.0
        mock_workflow._execute_provider_async = AsyncMock(side_effect=RuntimeError("Provider unavailable"))

        decision = await self.orchestrator.async_think_pause(
            state=state,
            phase=DeepResearchPhase.ANALYSIS,
            workflow=mock_workflow,
        )

        assert decision.proceed is True
        assert "error" in decision.rationale.lower() or "failed" in decision.rationale.lower()

    @pytest.mark.asyncio
    async def test_llm_returns_failure_falls_back(self) -> None:
        """LLM result.success=False falls back to proceed=True."""
        state = _make_state()

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Timeout"
        mock_result.provider_id = "test-provider"
        mock_result.model_used = "test-model"

        mock_workflow = MagicMock()
        mock_workflow.config.get_reflection_provider.return_value = "test-provider"
        mock_workflow.config.deep_research_reflection_timeout = 60.0
        mock_workflow._execute_provider_async = AsyncMock(return_value=mock_result)

        decision = await self.orchestrator.async_think_pause(
            state=state,
            phase=DeepResearchPhase.GATHERING,
            workflow=mock_workflow,
        )

        assert decision.proceed is True
        assert "Timeout" in decision.rationale

    @pytest.mark.asyncio
    async def test_records_agent_decision(self) -> None:
        """Successful reflection records an AgentDecision."""
        state = _make_state()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.content = json.dumps(
            {
                "quality_assessment": "OK",
                "proceed": True,
                "adjustments": [],
                "rationale": "Fine",
            }
        )
        mock_result.provider_id = "test"
        mock_result.model_used = "test"
        mock_result.tokens_used = 50
        mock_result.duration_ms = 200.0

        mock_workflow = MagicMock()
        mock_workflow.config.get_reflection_provider.return_value = "test"
        mock_workflow.config.deep_research_reflection_timeout = 60.0
        mock_workflow._execute_provider_async = AsyncMock(return_value=mock_result)

        await self.orchestrator.async_think_pause(
            state=state,
            phase=DeepResearchPhase.PLANNING,
            workflow=mock_workflow,
        )

        assert len(self.orchestrator._decisions) == 1
        d = self.orchestrator._decisions[0]
        assert d.agent == AgentRole.SUPERVISOR
        assert d.action == "reflect_planning"


# =============================================================================
# Unit tests: _maybe_reflect (workflow integration)
# =============================================================================


class TestMaybeReflect:
    """Tests for WorkflowExecutionMixin._maybe_reflect()."""

    @pytest.fixture
    def mock_workflow(self) -> MagicMock:
        """Create a mock workflow with _maybe_reflect accessible."""
        from foundry_mcp.core.research.workflows.deep_research.workflow_execution import (
            WorkflowExecutionMixin,
        )

        class StubWorkflow(WorkflowExecutionMixin):
            def __init__(self) -> None:
                self.config = MagicMock()
                self.memory = MagicMock()
                self.hooks = MagicMock()
                self.orchestrator = SupervisorOrchestrator()
                self._tasks: dict[str, Any] = {}
                self._audit_events: list[tuple] = []
                import threading

                self._tasks_lock = threading.Lock()
                self._search_providers: dict[str, Any] = {}

            def _write_audit_event(self, state: Any, event: str, **kwargs: Any) -> None:
                self._audit_events.append((event, kwargs))

            def _flush_state(self, state: Any) -> None:
                pass

            def _record_workflow_error(self, *args: Any, **kwargs: Any) -> None:
                pass

            def _safe_orchestrator_transition(self, *args: Any, **kwargs: Any) -> None:
                pass

        return StubWorkflow()

    @pytest.mark.asyncio
    async def test_reflection_disabled_is_noop(self, mock_workflow: Any) -> None:
        """When reflection is disabled, _maybe_reflect does nothing."""
        mock_workflow.config.deep_research_enable_reflection = False
        state = _make_state()

        await mock_workflow._maybe_reflect(state, DeepResearchPhase.PLANNING)

        # No audit events should be recorded
        assert len(mock_workflow._audit_events) == 0

    @pytest.mark.asyncio
    async def test_reflection_enabled_calls_orchestrator(self, mock_workflow: Any) -> None:
        """When reflection is enabled, calls async_think_pause and records audit."""
        mock_workflow.config.deep_research_enable_reflection = True
        mock_workflow.config.get_reflection_provider = MagicMock(return_value="test")
        mock_workflow.config.deep_research_reflection_timeout = 60.0

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.content = json.dumps(
            {
                "quality_assessment": "Looks good",
                "proceed": True,
                "adjustments": [],
                "rationale": "Quality sufficient",
            }
        )
        mock_result.provider_id = "test"
        mock_result.model_used = "test-model"
        mock_result.tokens_used = 80
        mock_result.duration_ms = 300.0

        mock_workflow._execute_provider_async = AsyncMock(return_value=mock_result)

        state = _make_state()
        state.total_tokens_used = 0

        await mock_workflow._maybe_reflect(state, DeepResearchPhase.PLANNING)

        # Audit event should be recorded
        assert len(mock_workflow._audit_events) == 1
        event_name, event_data = mock_workflow._audit_events[0]
        assert event_name == "reflection_complete"
        assert event_data["data"]["proceed"] is True
        assert event_data["data"]["phase"] == "planning"

        # Tokens should be tracked
        assert state.total_tokens_used == 80

    @pytest.mark.asyncio
    async def test_reflection_exception_caught(self, mock_workflow: Any) -> None:
        """Reflection errors are caught and don't crash the workflow."""
        mock_workflow.config.deep_research_enable_reflection = True

        # Make the orchestrator raise an exception
        mock_workflow.orchestrator.async_think_pause = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        state = _make_state()

        # Should not raise
        await mock_workflow._maybe_reflect(state, DeepResearchPhase.PLANNING)

        # No audit event recorded (exception occurred before recording)
        assert len(mock_workflow._audit_events) == 0

    @pytest.mark.asyncio
    async def test_reflection_proceed_false_logs_adjustments(self, mock_workflow: Any) -> None:
        """When reflection says proceed=false, adjustments are logged and token tracked."""
        mock_workflow.config.deep_research_enable_reflection = True
        mock_workflow.config.get_reflection_provider = MagicMock(return_value="test")
        mock_workflow.config.deep_research_reflection_timeout = 60.0

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.content = json.dumps(
            {
                "quality_assessment": "Insufficient coverage",
                "proceed": False,
                "adjustments": ["Add more sources", "Try different queries"],
                "rationale": "Only 1 source found",
            }
        )
        mock_result.provider_id = "test"
        mock_result.model_used = "test-model"
        mock_result.tokens_used = 100
        mock_result.duration_ms = 350.0

        mock_workflow._execute_provider_async = AsyncMock(return_value=mock_result)

        state = _make_state()
        state.total_tokens_used = 500

        await mock_workflow._maybe_reflect(state, DeepResearchPhase.GATHERING)

        # Audit event records proceed=false
        event_name, event_data = mock_workflow._audit_events[0]
        assert event_data["data"]["proceed"] is False
        assert len(event_data["data"]["adjustments"]) == 2

        # Tokens still tracked
        assert state.total_tokens_used == 600


# =============================================================================
# Integration tests: config validation
# =============================================================================


class TestReflectionConfig:
    """Tests for reflection config keys."""

    def test_default_reflection_enabled(self) -> None:
        """Reflection is enabled by default."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig()
        assert config.deep_research_enable_reflection is True
        assert config.deep_research_reflection_provider is None
        assert config.deep_research_reflection_timeout == 60.0

    def test_from_toml_dict_parses_reflection_keys(self) -> None:
        """from_toml_dict correctly parses reflection config."""
        from foundry_mcp.config.research import ResearchConfig

        data = {
            "deep_research_enable_reflection": True,
            "deep_research_reflection_provider": "[cli]gemini:flash",
            "deep_research_reflection_timeout": 30.0,
        }
        config = ResearchConfig.from_toml_dict(data)

        assert config.deep_research_enable_reflection is True
        assert config.deep_research_reflection_provider == "[cli]gemini:flash"
        assert config.deep_research_reflection_timeout == 30.0

    def test_get_reflection_provider_with_explicit(self) -> None:
        """get_reflection_provider returns configured provider."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig(
            deep_research_reflection_provider="[cli]gemini:flash",
        )
        provider = config.get_reflection_provider()
        assert provider is not None

    def test_get_reflection_provider_falls_back(self) -> None:
        """get_reflection_provider falls back to default_provider."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig(default_provider="gemini")
        provider = config.get_reflection_provider()
        assert provider == "gemini"


# =============================================================================
# Integration test: reflection doesn't break existing workflow
# =============================================================================


class TestReflectionWorkflowIntegration:
    """Verify reflection integration with existing workflow patterns."""

    def test_reflection_system_prompt_is_valid(self) -> None:
        """System prompt contains required JSON schema elements."""
        orchestrator = SupervisorOrchestrator()
        prompt = orchestrator._build_reflection_system_prompt()

        assert "quality_assessment" in prompt
        assert "proceed" in prompt
        assert "adjustments" in prompt
        assert "rationale" in prompt
        assert "JSON" in prompt

    def test_existing_evaluate_phase_completion_unchanged(self) -> None:
        """Existing heuristic evaluate_phase_completion still works."""
        orchestrator = SupervisorOrchestrator()
        state = _make_state(phase=DeepResearchPhase.PLANNING)
        state.sub_queries = [
            SubQuery(query="Q1", source_types=[]),
            SubQuery(query="Q2", source_types=[]),
        ]

        decision = orchestrator.evaluate_phase_completion(state, DeepResearchPhase.PLANNING)

        assert decision.outputs["quality_ok"] is True
        assert decision.outputs["sub_query_count"] == 2

    def test_existing_decide_iteration_unchanged(self) -> None:
        """Existing decide_iteration still works."""
        orchestrator = SupervisorOrchestrator()
        state = _make_state()

        decision = orchestrator.decide_iteration(state)

        assert decision.outputs["should_iterate"] is False
        assert decision.outputs["next_phase"] == "COMPLETED"

    def test_existing_get_reflection_prompt_unchanged(self) -> None:
        """Existing get_reflection_prompt still returns text for all phases."""
        orchestrator = SupervisorOrchestrator()
        state = _make_state()

        for phase in DeepResearchPhase:
            prompt = orchestrator.get_reflection_prompt(state, phase)
            assert isinstance(prompt, str)
            assert len(prompt) > 0
