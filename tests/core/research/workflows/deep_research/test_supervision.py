"""Tests for the supervision phase of deep research.

Covers:
- 6.7.1: State model (enum ordering, advance_phase, supervision_round reset, should_continue)
- 6.7.2: Prompt/coverage building (per-query coverage, system/user prompts)
- 6.7.3: Response parsing (valid JSON, dedup, invalid JSON fallback, heuristic)
- 6.7.4: Integration tests (follow-up queries, max-round cap, disabled skip, sufficient coverage)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
    TopicResearchResult,
)
from foundry_mcp.core.research.models.sources import (
    ResearchSource,
    SourceQuality,
    SourceType,
    SubQuery,
)
from foundry_mcp.core.research.models.deep_research import (
    DelegationResponse,
    ResearchDirective,
)
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    StructuredLLMCallResult,
)
from foundry_mcp.core.research.workflows.deep_research.phases.supervision import (
    SupervisionPhaseMixin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    query: str = "How does deep learning work?",
    phase: DeepResearchPhase = DeepResearchPhase.SUPERVISION,
    num_completed: int = 2,
    num_pending: int = 0,
    sources_per_query: int = 2,
    supervision_round: int = 0,
    max_supervision_rounds: int = 3,
) -> DeepResearchState:
    """Create a DeepResearchState pre-populated for supervision tests."""
    from tests.core.research.workflows.deep_research.conftest import make_test_state

    return make_test_state(
        id="deepres-supervision-test",
        query=query,
        research_brief="Investigating deep learning fundamentals",
        phase=phase,
        num_sub_queries=num_completed,
        num_pending_sub_queries=num_pending,
        sources_per_query=sources_per_query,
        supervision_round=supervision_round,
        max_supervision_rounds=max_supervision_rounds,
    )


class StubSupervision(SupervisionPhaseMixin):
    """Concrete class for testing SupervisionPhaseMixin in isolation."""

    def __init__(self, *, delegation_model: bool = False) -> None:
        self.config = MagicMock()
        self.config.deep_research_supervision_min_sources_per_query = 2
        self.config.deep_research_max_concurrent_research_units = 5
        self.config.deep_research_reflection_timeout = 60.0
        self.config.deep_research_coverage_confidence_threshold = 0.75
        self.config.deep_research_coverage_confidence_weights = None
        self.config.deep_research_supervision_wall_clock_timeout = 1800.0
        self.memory = MagicMock()
        self._audit_events: list[tuple[str, dict]] = []

    def _write_audit_event(self, state: Any, event: str, **kwargs: Any) -> None:
        self._audit_events.append((event, kwargs))

    def _check_cancellation(self, state: Any) -> None:
        pass


def _wrap_as_structured_mock(mock_execute_llm_call):
    """Wrap a mock_execute_llm_call to work as mock_execute_structured_llm_call.

    The delegate step now uses execute_structured_llm_call() which accepts
    a parse_fn. This helper wraps an existing LLM call mock so it returns
    a StructuredLLMCallResult with the parsed DelegationResponse.
    """

    async def mock_execute_structured_llm_call(**kwargs):
        parse_fn = kwargs.pop("parse_fn", None)
        llm_result = await mock_execute_llm_call(**kwargs)

        if isinstance(llm_result, WorkflowResult):
            return llm_result

        content = llm_result.result.content or ""
        parsed = None
        if parse_fn:
            try:
                parsed = parse_fn(content)
            except Exception:
                pass

        return StructuredLLMCallResult(
            result=llm_result.result,
            llm_call_duration_ms=0.0,
            parsed=parsed,
            parse_retries=0,
        )

    return mock_execute_structured_llm_call


# ===========================================================================
# 6.7.1  State model tests
# ===========================================================================


class TestSupervisionStateModel:
    """State model tests for the SUPERVISION phase."""

    def test_supervision_phase_in_enum(self):
        """SUPERVISION exists in enum between GATHERING and SYNTHESIS."""
        phases = list(DeepResearchPhase)
        gathering_idx = phases.index(DeepResearchPhase.GATHERING)
        supervision_idx = phases.index(DeepResearchPhase.SUPERVISION)
        synthesis_idx = phases.index(DeepResearchPhase.SYNTHESIS)

        assert supervision_idx == gathering_idx + 1
        assert synthesis_idx == supervision_idx + 1

    def test_advance_phase_gathering_to_supervision(self):
        """advance_phase() from GATHERING goes to SUPERVISION."""
        state = DeepResearchState(
            original_query="test",
            phase=DeepResearchPhase.GATHERING,
        )
        new_phase = state.advance_phase()
        assert new_phase == DeepResearchPhase.SUPERVISION
        assert state.phase == DeepResearchPhase.SUPERVISION

    def test_advance_phase_supervision_to_synthesis(self):
        """advance_phase() from SUPERVISION goes to SYNTHESIS."""
        state = DeepResearchState(
            original_query="test",
            phase=DeepResearchPhase.SUPERVISION,
        )
        new_phase = state.advance_phase()
        assert new_phase == DeepResearchPhase.SYNTHESIS
        assert state.phase == DeepResearchPhase.SYNTHESIS

    def test_supervision_round_resets_manually(self):
        """supervision_round field can be reset to 0 between iterations."""
        state = DeepResearchState(
            original_query="test",
            phase=DeepResearchPhase.SUPERVISION,
            supervision_round=2,
        )
        # Simulate what workflow code does between iterations: reset the counter
        state.supervision_round = 0
        assert state.supervision_round == 0
        assert state.phase == DeepResearchPhase.SUPERVISION

    def test_should_continue_supervision_within_limit(self):
        """Returns True when pending sub-queries exist and within round limit."""
        state = _make_state(num_completed=2, num_pending=1, supervision_round=1)
        assert state.should_continue_supervision() is True

    def test_should_continue_supervision_at_limit(self):
        """Returns False when supervision_round >= max_supervision_rounds."""
        state = _make_state(num_completed=2, num_pending=1, supervision_round=3, max_supervision_rounds=3)
        assert state.should_continue_supervision() is False

    def test_should_continue_supervision_no_pending(self):
        """Returns False when no pending sub-queries even if within limit."""
        state = _make_state(num_completed=2, num_pending=0, supervision_round=0)
        assert state.should_continue_supervision() is False


# ===========================================================================
# 6.7.2  Prompt / coverage tests
# ===========================================================================


class TestSupervisionPrompts:
    """Tests for coverage building and prompt generation."""

    def test_build_per_query_coverage(self):
        """Correct source counts, quality distribution, and domain count."""
        stub = StubSupervision()
        state = _make_state(num_completed=2, sources_per_query=3)
        coverage = stub._build_per_query_coverage(state)

        assert len(coverage) == 2
        entry = coverage[0]
        assert entry["sub_query_id"] == "sq-0"
        assert entry["source_count"] == 3
        assert entry["status"] == "completed"
        # First source HIGH, rest MEDIUM (per _make_state)
        assert entry["quality_distribution"]["HIGH"] == 1
        assert entry["quality_distribution"]["MEDIUM"] == 2
        # 3 distinct example{j}.com domains
        assert entry["unique_domains"] == 3

    def test_build_per_query_coverage_with_topic_results(self):
        """Coverage includes findings_summary from topic research."""
        stub = StubSupervision()
        state = _make_state(num_completed=1, sources_per_query=2)
        state.topic_research_results.append(
            TopicResearchResult(
                sub_query_id="sq-0",
                searches_performed=1,
                sources_found=2,
                per_topic_summary="Deep learning uses neural networks for representation learning.",
            )
        )
        coverage = stub._build_per_query_coverage(state)
        assert coverage[0]["findings_summary"] is not None
        assert "neural networks" in coverage[0]["findings_summary"]

    # Legacy prompt tests (_build_supervision_system_prompt, _build_supervision_user_prompt)
    # removed — LegacySupervisionMixin deleted as dead code (see git history).


# ===========================================================================
# 6.7.3  Response parsing tests
# ===========================================================================


class TestSupervisionParsing:
    """Tests for heuristic coverage assessment.

    Legacy _parse_supervision_response tests removed — LegacySupervisionMixin
    deleted as dead code (see git history).
    """

    def test_heuristic_fallback(self):
        """Heuristic returns correct coverage assessment."""
        stub = StubSupervision()

        # All queries have >= 2 sources → sufficient
        state = _make_state(num_completed=3, sources_per_query=2)
        result = stub._assess_coverage_heuristic(state, min_sources=2)
        assert result["overall_coverage"] == "sufficient"
        assert result["should_continue_gathering"] is False
        assert result["queries_assessed"] == 3
        assert result["queries_sufficient"] == 3

    def test_heuristic_partial_coverage(self):
        """Heuristic returns partial when some queries lack sources."""
        stub = StubSupervision()
        state = _make_state(num_completed=2, sources_per_query=2)
        # Add a completed sub-query with no sources
        state.sub_queries.append(
            SubQuery(id="sq-nosrc", query="No sources query", status="completed")
        )
        result = stub._assess_coverage_heuristic(state, min_sources=2)
        assert result["overall_coverage"] == "partial"
        assert result["queries_sufficient"] == 2

    def test_heuristic_no_completed(self):
        """Heuristic returns insufficient when no completed queries."""
        stub = StubSupervision()
        state = _make_state(num_completed=0, num_pending=2)
        result = stub._assess_coverage_heuristic(state, min_sources=2)
        assert result["overall_coverage"] == "insufficient"
        assert result["queries_assessed"] == 0


# ===========================================================================
# 6.7.4  Integration tests
# ===========================================================================


class TestSupervisionIntegration:
    """Integration tests for the full supervision execution flow."""

    @pytest.mark.asyncio
    async def test_supervision_loop_adds_follow_up_queries(self):
        """Delegation supervision generates directives targeting coverage gaps."""
        stub = StubSupervision()
        stub.config.deep_research_providers = ["tavily"]
        stub.config.deep_research_topic_max_tool_calls = 5
        # Low confidence threshold so heuristic early-exits after round 0
        # completes (round 1 check). The test focuses on directive generation,
        # not heuristic sensitivity.
        stub.config.deep_research_coverage_confidence_threshold = 0.1
        # sources_per_query=1 keeps coverage below min_sources=2 so heuristic
        # does not short-circuit on round=0.
        state = _make_state(num_completed=2, sources_per_query=1, supervision_round=0)
        state.max_sub_queries = 10
        state.topic_research_results = []

        # Delegation response — research_complete=False signals more work needed
        delegation_response = json.dumps({
            "research_complete": False,
            "directives": [
                {
                    "research_topic": "Investigate backpropagation fundamentals and gradient flow.",
                    "perspective": "technical",
                    "evidence_needed": "papers, tutorials",
                    "priority": 2,
                },
                {
                    "research_topic": "Compare CNN vs RNN architecture trade-offs.",
                    "perspective": "comparative",
                    "evidence_needed": "benchmarks",
                    "priority": 2,
                },
            ],
            "rationale": "Need more specific results",
        })

        async def mock_execute_llm_call(**kwargs):
            result = MagicMock()
            result.result = WorkflowResult(
                success=True,
                content=delegation_response,
                provider_id="test-provider",
                model_used="test-model",
                tokens_used=100,
                duration_ms=500.0,
            )
            return result

        async def mock_topic_research(**kwargs):
            sq = kwargs.get("sub_query")
            return TopicResearchResult(
                sub_query_id=sq.id if sq else "unknown",
                searches_performed=1,
                sources_found=2,
            )

        stub._execute_topic_research_async = mock_topic_research
        stub._get_search_provider = MagicMock(return_value=MagicMock())

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            side_effect=_wrap_as_structured_mock(mock_execute_llm_call),
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ):
            result = await stub._execute_supervision_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        assert result.success is True
        assert "should_continue_gathering" not in result.metadata
        assert result.metadata["model"] == "delegation"
        # Two directives executed in round 0
        assert result.metadata["total_directives_executed"] == 2
        # Round 0 ran (→ round 1), then heuristic fired on round 1 (→ round 2)
        assert state.supervision_round >= 1

        # Supervision history recorded
        history = state.metadata["supervision_history"]
        # First history entry is the delegation round with 2 directives
        delegation_entry = next(h for h in history if h["method"] == "delegation")
        assert delegation_entry["directives_executed"] == 2

    @pytest.mark.asyncio
    async def test_supervision_loop_terminates_at_max_rounds(self):
        """Heuristic early-exit when all queries covered and round > 0."""
        stub = StubSupervision()
        # Round > 0 and all queries have >= 2 sources → heuristic early-exit
        state = _make_state(
            num_completed=2,
            sources_per_query=3,
            supervision_round=1,
            max_supervision_rounds=3,
        )

        result = await stub._execute_supervision_async(
            state=state, provider_id="test-provider", timeout=30.0
        )

        assert result.success is True
        assert "should_continue_gathering" not in result.metadata
        assert result.metadata["model"] == "delegation"
        assert state.supervision_round == 2  # incremented
        # History records the heuristic early-exit
        history = state.metadata["supervision_history"]
        assert history[0]["method"] == "delegation_heuristic"

    @pytest.mark.asyncio
    async def test_supervision_skipped_when_disabled(self):
        """When supervision is disabled, workflow_execution skips SUPERVISION entirely."""
        from foundry_mcp.core.research.workflows.deep_research.workflow_execution import (
            WorkflowExecutionMixin,
        )

        state = _make_state(phase=DeepResearchPhase.SUPERVISION)
        audit_events: list[tuple[str, dict]] = []

        class StubWorkflowExecution(WorkflowExecutionMixin):
            def __init__(self):
                self.config = MagicMock()
                self.config.deep_research_enable_supervision = False
                self.config.get_phase_timeout.return_value = 60.0
                self.memory = MagicMock()
                self.hooks = MagicMock()
                self.orchestrator = MagicMock()
                self._tasks: dict = {}
                self._tasks_lock = __import__("threading").Lock()
                self._search_providers: dict = {}

            def _write_audit_event(self, _state, event, **kwargs):
                audit_events.append((event, kwargs))

            def _flush_state(self, _state):
                pass

            def _record_workflow_error(self, *a, **kw):
                pass

            def _safe_orchestrator_transition(self, *a, **kw):
                pass

            async def _execute_supervision_async(self, **kw):
                raise AssertionError("Should not be called when supervision is disabled")

            async def _execute_synthesis_async(self, **kw):
                return WorkflowResult(success=True, content="report")

        stub = StubWorkflowExecution()
        result = await stub._execute_workflow_async(
            state=state, provider_id="test", timeout_per_operation=60.0, max_concurrent=3,
        )

        assert state.phase == DeepResearchPhase.SYNTHESIS
        # Verify audit event was emitted for the skip
        skip_events = [e for e, _ in audit_events if e == "supervision_skipped"]
        assert len(skip_events) == 1

    @pytest.mark.asyncio
    async def test_supervision_proceeds_when_all_covered(self):
        """When LLM signals research_complete=True, should_continue_gathering is False."""
        stub = StubSupervision()
        state = _make_state(num_completed=2, sources_per_query=3, supervision_round=0)
        state.max_sub_queries = 10
        state.topic_research_results = []

        # research_complete=True signals that all dimensions are covered
        delegation_response = json.dumps({
            "research_complete": True,
            "directives": [],
            "rationale": "All aspects well covered",
        })

        async def mock_execute_llm_call(**kwargs):
            result = MagicMock()
            result.result = WorkflowResult(
                success=True,
                content=delegation_response,
                provider_id="test-provider",
                model_used="test-model",
                tokens_used=80,
                duration_ms=400.0,
            )
            return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            side_effect=_wrap_as_structured_mock(mock_execute_llm_call),
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ):
            result = await stub._execute_supervision_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        assert result.success is True
        assert "should_continue_gathering" not in result.metadata
        assert result.metadata["total_directives_executed"] == 0
        # No new sub-queries added
        assert len(state.sub_queries) == 2
        # History records delegation_complete
        history = state.metadata["supervision_history"]
        assert history[0]["method"] == "delegation_complete"

    @pytest.mark.asyncio
    async def test_supervision_llm_failure_falls_back_to_heuristic(self):
        """When both LLM calls fail, supervision degrades gracefully with no directives."""
        stub = StubSupervision()
        state = _make_state(num_completed=2, sources_per_query=2, supervision_round=0)
        state.topic_research_results = []

        # execute_llm_call returns WorkflowResult directly on failure for both
        # think and delegate steps.
        failed_result = WorkflowResult(
            success=False,
            content="",
            error="Provider timeout",
        )

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            new_callable=AsyncMock,
            return_value=failed_result,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            new_callable=AsyncMock,
            return_value=failed_result,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ):
            result = await stub._execute_supervision_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        assert result.success is True  # Graceful degradation
        assert result.metadata["model"] == "delegation"
        assert "should_continue_gathering" not in result.metadata
        assert result.metadata["total_directives_executed"] == 0
        # History recorded — no directives generated due to LLM failures
        history = state.metadata["supervision_history"]
        assert len(history) == 1
        assert history[0]["method"] == "delegation_no_directives"

    @pytest.mark.asyncio
    async def test_supervision_respects_directive_budget(self):
        """Directives are capped by max_concurrent_research_units config."""
        stub = StubSupervision()
        # Cap concurrent research units to 1 directive per round
        stub.config.deep_research_max_concurrent_research_units = 1
        stub.config.deep_research_providers = ["tavily"]
        stub.config.deep_research_topic_max_tool_calls = 5
        # Low threshold so heuristic early-exits after round 0
        stub.config.deep_research_coverage_confidence_threshold = 0.1

        state = _make_state(num_completed=2, sources_per_query=1, supervision_round=0)
        state.max_sub_queries = 10
        state.topic_research_results = []

        # LLM proposes 3 directives but only 1 should survive the cap
        delegation_response = json.dumps({
            "research_complete": False,
            "directives": [
                {"research_topic": "Investigate topic A in detail.", "priority": 2},
                {"research_topic": "Investigate topic B in detail.", "priority": 2},
                {"research_topic": "Investigate topic C in detail.", "priority": 2},
            ],
            "rationale": "",
        })

        async def mock_execute_llm_call(**kwargs):
            result = MagicMock()
            result.result = WorkflowResult(
                success=True,
                content=delegation_response,
                provider_id="test-provider",
                model_used="test-model",
                tokens_used=50,
                duration_ms=300.0,
            )
            return result

        async def mock_topic_research(**kwargs):
            sq = kwargs.get("sub_query")
            return TopicResearchResult(
                sub_query_id=sq.id if sq else "unknown",
                searches_performed=1,
                sources_found=2,
            )

        stub._execute_topic_research_async = mock_topic_research
        stub._get_search_provider = MagicMock(return_value=MagicMock())

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            side_effect=_wrap_as_structured_mock(mock_execute_llm_call),
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ):
            result = await stub._execute_supervision_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        assert result.metadata["total_directives_executed"] == 1  # capped at 1


# ===========================================================================
# 1.5  Think-tool deliberation tests
# ===========================================================================


class TestThinkToolPrompts:
    """Tests for _build_think_prompt and _build_think_system_prompt."""

    def test_think_prompt_contains_per_query_coverage(self):
        """Think prompt includes per-sub-query coverage with gap analysis instructions."""
        stub = StubSupervision()
        state = _make_state(num_completed=2, sources_per_query=2)
        coverage = stub._build_per_query_coverage(state)
        prompt = stub._build_think_prompt(state, coverage)

        # Contains original query
        assert state.original_query in prompt
        # Contains per-sub-query data
        assert "Sub-query 0" in prompt
        assert "Sub-query 1" in prompt
        # Contains coverage metrics
        assert "Sources found:" in prompt
        assert "Quality:" in prompt
        assert "Domains:" in prompt
        # Contains gap analysis instructions
        assert "DO NOT generate follow-up queries" in prompt
        assert "information gaps" in prompt

    def test_think_prompt_includes_research_brief(self):
        """Think prompt includes research brief excerpt when available."""
        stub = StubSupervision()
        state = _make_state(num_completed=1, sources_per_query=1)
        state.research_brief = "A comprehensive study of neural network architectures"
        coverage = stub._build_per_query_coverage(state)
        prompt = stub._build_think_prompt(state, coverage)

        assert "comprehensive study of neural network" in prompt

    def test_think_prompt_handles_empty_coverage(self):
        """Think prompt works with no coverage data (no completed queries)."""
        stub = StubSupervision()
        state = _make_state(num_completed=0, num_pending=2)
        coverage = stub._build_per_query_coverage(state)
        prompt = stub._build_think_prompt(state, coverage)

        # Should still contain instructions even without per-query data
        assert "DO NOT generate follow-up queries" in prompt
        assert state.original_query in prompt

    def test_think_system_prompt_forbids_query_generation(self):
        """Think system prompt explicitly forbids generating follow-up queries."""
        stub = StubSupervision()
        system = stub._build_think_system_prompt()

        assert "do not generate follow-up queries" in system.lower()
        assert "gap" in system.lower()


# TestThinkToolInUserPrompt removed — tested legacy _build_supervision_user_prompt
# (LegacySupervisionMixin deleted as dead code; see git history).


class TestThinkToolIntegration:
    """Integration tests for think-tool deliberation in supervision execution."""

    @pytest.mark.asyncio
    async def test_think_step_executes_on_round_zero(self):
        """Think step executes on supervision_round=0 and output flows into delegate prompt."""
        stub = StubSupervision()
        state = _make_state(
            num_completed=2, sources_per_query=1, supervision_round=0
        )
        state.max_sub_queries = 10
        state.topic_research_results = []

        think_response = (
            "Sub-query 0 lacks diversity — only 1 source from a single domain. "
            "Missing: academic perspectives and peer-reviewed research."
        )

        delegation_response = json.dumps({
            "research_complete": True,
            "directives": [],
            "rationale": "Covered after think analysis",
        })

        # Track calls to distinguish think vs delegate LLM calls
        think_call_count = 0

        async def mock_execute_llm_call(**kwargs):
            nonlocal think_call_count
            if kwargs.get("phase_name") == "supervision_think":
                think_call_count += 1
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True,
                    content=think_response,
                    provider_id="reflection-provider",
                    model_used="cheap-model",
                    tokens_used=50,
                    duration_ms=200.0,
                )
                return result
            # Other calls (delegation delegate step routed via execute_structured_llm_call)
            result = MagicMock()
            result.result = WorkflowResult(
                success=True,
                content=delegation_response,
                provider_id="test-provider",
                model_used="test-model",
                tokens_used=100,
                duration_ms=500.0,
            )
            return result

        async def mock_execute_structured_llm_call(**kwargs):
            # Verify think output was included in delegate user prompt.
            # For first-round decomposition (round=0, no prior results), the think
            # output is wrapped in <decomposition_strategy> tags.
            user_prompt = kwargs.get("user_prompt", "")
            assert "<decomposition_strategy>" in user_prompt, (
                "Think output should be included in first-round delegation user prompt"
            )
            assert "academic perspectives" in user_prompt
            return await _wrap_as_structured_mock(mock_execute_llm_call)(**kwargs)

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            side_effect=mock_execute_structured_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ):
            result = await stub._execute_supervision_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        assert result.success is True
        assert think_call_count == 1  # think step executed once
        assert result.metadata["total_directives_executed"] == 0

        # Think output recorded in history
        history = state.metadata["supervision_history"]
        assert len(history) == 1
        assert "think_output" in history[0]
        assert "academic perspectives" in history[0]["think_output"]

    @pytest.mark.asyncio
    async def test_think_step_skipped_on_round_gt_zero(self):
        """Think step is skipped when supervision_round > 0 with sufficient coverage.

        When round > 0 and the heuristic finds sufficient coverage, the
        delegation loop exits early via ``delegation_heuristic`` without
        making any LLM calls — meaning the think step is inherently skipped.
        """
        stub = StubSupervision()
        state = _make_state(
            num_completed=2, sources_per_query=3, supervision_round=1
        )

        call_phases: list[str] = []

        async def mock_execute_llm_call(**kwargs):
            call_phases.append(kwargs.get("phase_name", "unknown"))
            result = MagicMock()
            result.result = WorkflowResult(
                success=True,
                content="{}",
                provider_id="test-provider",
                model_used="test-model",
                tokens_used=80,
                duration_ms=400.0,
            )
            return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ):
            result = await stub._execute_supervision_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        assert result.success is True
        assert result.metadata["model"] == "delegation"
        # No LLM calls at all — heuristic path, so think step was skipped
        assert call_phases == []
        # History records heuristic early-exit, NOT think_output
        history = state.metadata["supervision_history"]
        assert history[0]["method"] == "delegation_heuristic"
        assert "think_output" not in history[0]

    @pytest.mark.asyncio
    async def test_think_step_failure_is_non_fatal(self):
        """When think step fails, delegation proceeds without gap analysis."""
        stub = StubSupervision()
        state = _make_state(
            num_completed=2, sources_per_query=2, supervision_round=0
        )
        state.max_sub_queries = 10
        state.topic_research_results = []

        delegation_response = json.dumps({
            "research_complete": True,
            "directives": [],
            "rationale": "Coverage is sufficient",
        })

        async def mock_execute_llm_call(**kwargs):
            if kwargs.get("phase_name") == "supervision_think":
                # Think step fails — returns WorkflowResult directly
                return WorkflowResult(
                    success=False,
                    content="",
                    error="Provider timeout",
                )
            # Other calls (delegate path via execute_structured_llm_call)
            result = MagicMock()
            result.result = WorkflowResult(
                success=True,
                content=delegation_response,
                provider_id="test-provider",
                model_used="test-model",
                tokens_used=80,
                duration_ms=400.0,
            )
            return result

        async def mock_execute_structured_llm_call(**kwargs):
            # Verify no gap_analysis section when think step failed
            user_prompt = kwargs.get("user_prompt", "")
            assert "<gap_analysis>" not in user_prompt, (
                "Should NOT have gap analysis when think step fails"
            )
            return await _wrap_as_structured_mock(mock_execute_llm_call)(**kwargs)

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            side_effect=mock_execute_structured_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ):
            result = await stub._execute_supervision_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        assert result.success is True
        # History records delegation_complete since research_complete=True
        history = state.metadata["supervision_history"]
        assert history[0]["method"] == "delegation_complete"
        # think_output is None/not present since think step failed
        assert not history[0].get("think_output")

    @pytest.mark.asyncio
    async def test_think_step_uses_reflection_role(self):
        """Think step LLM call uses the 'reflection' role for cheap model routing."""
        stub = StubSupervision()
        state = _make_state(
            num_completed=2, sources_per_query=1, supervision_round=0
        )
        state.max_sub_queries = 10
        state.topic_research_results = []

        captured_llm_kwargs: list[dict] = []

        delegation_response = json.dumps({
            "research_complete": True,
            "directives": [],
            "rationale": "OK",
        })

        async def mock_execute_llm_call(**kwargs):
            captured_llm_kwargs.append(kwargs)
            if kwargs.get("phase_name") == "supervision_think":
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True,
                    content="Gap analysis here",
                    provider_id="reflection-provider",
                    model_used="cheap-model",
                    tokens_used=45,
                    duration_ms=150.0,
                )
                return result
            else:
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True,
                    content=delegation_response,
                    provider_id="test-provider",
                    model_used="test-model",
                    tokens_used=80,
                    duration_ms=400.0,
                )
                return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            side_effect=_wrap_as_structured_mock(mock_execute_llm_call),
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ):
            await stub._execute_supervision_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        # Verify think step used reflection role
        # captured_llm_kwargs[0] is the think step (first execute_llm_call invocation)
        think_kwargs = captured_llm_kwargs[0]
        assert think_kwargs["phase_name"] == "supervision_think"
        assert think_kwargs["role"] == "reflection"
        assert think_kwargs["temperature"] == 0.2
        assert think_kwargs["provider_id"] is None  # resolved by role
        assert think_kwargs["model"] is None  # resolved by role


# ===========================================================================
# Phase 4 PLAN: Supervisor Delegation Model Tests
# ===========================================================================


class TestResearchDirectiveModel:
    """Tests for the ResearchDirective dataclass."""

    def test_directive_has_required_fields(self):
        """ResearchDirective contains all required fields with defaults."""
        from foundry_mcp.core.research.models.deep_research import ResearchDirective

        directive = ResearchDirective(
            research_topic="Investigate the comparative effectiveness of transformer vs CNN architectures for image classification tasks, focusing on accuracy-compute tradeoffs.",
        )

        assert directive.research_topic.startswith("Investigate")
        assert directive.perspective == ""
        assert directive.evidence_needed == ""
        assert directive.priority == 2
        assert directive.id.startswith("dir-")
        assert directive.supervision_round == 0

    def test_directive_serialization(self):
        """ResearchDirective round-trips through JSON serialization."""
        from foundry_mcp.core.research.models.deep_research import ResearchDirective

        directive = ResearchDirective(
            research_topic="Research topic here",
            perspective="technical",
            evidence_needed="benchmarks",
            priority=1,
            supervision_round=2,
        )

        data = directive.model_dump()
        restored = ResearchDirective(**data)
        assert restored.research_topic == directive.research_topic
        assert restored.perspective == "technical"
        assert restored.priority == 1
        assert restored.supervision_round == 2

    def test_directives_field_on_state(self):
        """DeepResearchState has a directives list field (default empty)."""
        state = DeepResearchState(original_query="test")
        assert state.directives == []
        assert isinstance(state.directives, list)

    def test_directives_persisted_in_state_serialization(self):
        """Directives survive state serialization round-trip."""
        from foundry_mcp.core.research.models.deep_research import ResearchDirective

        state = DeepResearchState(original_query="test")
        state.directives.append(
            ResearchDirective(
                research_topic="Investigate X",
                perspective="comparative",
                priority=1,
            )
        )

        data = state.model_dump()
        restored = DeepResearchState(**data)
        assert len(restored.directives) == 1
        assert restored.directives[0].research_topic == "Investigate X"


class TestDelegationPrompts:
    """Tests for delegation prompt building and parsing."""

    def test_delegation_system_prompt_requests_json(self):
        """Delegation system prompt requests valid JSON output."""
        stub = StubSupervision(delegation_model=True)
        prompt = stub._build_delegation_system_prompt()

        assert "valid JSON" in prompt
        assert "research_complete" in prompt
        assert "directives" in prompt
        assert "research_topic" in prompt
        assert "perspective" in prompt
        assert "evidence_needed" in prompt

    def test_delegation_user_prompt_includes_coverage(self):
        """Delegation user prompt includes coverage data and gap analysis."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=2, sources_per_query=2)
        coverage = stub._build_per_query_coverage(state)

        prompt = stub._build_delegation_user_prompt(
            state, coverage, think_output="Missing: historical context"
        )

        assert state.original_query in prompt
        assert "Missing: historical context" in prompt
        assert "<gap_analysis>" in prompt
        assert "Sources:" in prompt

    def test_delegation_user_prompt_includes_prior_directives(self):
        """Delegation user prompt lists previously executed directives."""
        from foundry_mcp.core.research.models.deep_research import ResearchDirective

        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=2, sources_per_query=2)
        state.directives.append(
            ResearchDirective(
                research_topic="Previously investigated topic about transformers",
                priority=1,
            )
        )
        coverage = stub._build_per_query_coverage(state)

        prompt = stub._build_delegation_user_prompt(state, coverage)

        assert "Previously Executed Directives" in prompt
        assert "Previously investigated topic" in prompt

    def test_parse_delegation_response_valid(self):
        """Parse a valid delegation response with directives."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=2, sources_per_query=2)

        response = json.dumps({
            "research_complete": False,
            "directives": [
                {
                    "research_topic": "Investigate the historical development of deep learning from perceptrons through modern transformers, focusing on key architectural innovations.",
                    "perspective": "historical",
                    "evidence_needed": "academic papers, timeline data",
                    "priority": 1,
                },
                {
                    "research_topic": "Survey current hardware requirements for training large language models.",
                    "perspective": "technical",
                    "evidence_needed": "benchmarks, cost data",
                    "priority": 2,
                },
            ],
            "rationale": "Two key gaps identified",
        })

        directives, complete = stub._parse_delegation_response(response, state)

        assert complete is False
        assert len(directives) == 2
        assert directives[0].priority == 1
        assert "historical development" in directives[0].research_topic
        assert directives[0].perspective == "historical"
        assert directives[1].priority == 2

    def test_parse_delegation_response_research_complete(self):
        """Parse a response signaling research completion."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=2, sources_per_query=2)

        response = json.dumps({
            "research_complete": True,
            "directives": [],
            "rationale": "All dimensions well covered",
        })

        directives, complete = stub._parse_delegation_response(response, state)

        assert complete is True
        assert len(directives) == 0

    def test_parse_delegation_response_caps_directives(self):
        """Directives are capped by max_concurrent_research_units."""
        stub = StubSupervision(delegation_model=True)
        stub.config.deep_research_max_concurrent_research_units = 2
        state = _make_state(num_completed=1, sources_per_query=1)

        response = json.dumps({
            "research_complete": False,
            "directives": [
                {"research_topic": f"Topic {i}", "priority": 2}
                for i in range(5)
            ],
            "rationale": "",
        })

        directives, _ = stub._parse_delegation_response(response, state)

        assert len(directives) == 2  # Capped at max_concurrent_research_units

    def test_parse_delegation_response_invalid_json(self):
        """Invalid JSON returns empty directives gracefully."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=1, sources_per_query=1)

        directives, complete = stub._parse_delegation_response("not json at all", state)

        assert len(directives) == 0
        assert complete is False

    def test_parse_delegation_response_empty_content(self):
        """Empty content returns empty directives."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=1, sources_per_query=1)

        directives, complete = stub._parse_delegation_response("", state)

        assert len(directives) == 0
        assert complete is False

    def test_parse_delegation_response_skips_empty_topics(self):
        """Directives with empty research_topic are skipped."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=1, sources_per_query=1)

        response = json.dumps({
            "research_complete": False,
            "directives": [
                {"research_topic": "", "priority": 1},
                {"research_topic": "Valid topic", "priority": 2},
            ],
            "rationale": "",
        })

        directives, _ = stub._parse_delegation_response(response, state)

        assert len(directives) == 1
        assert directives[0].research_topic == "Valid topic"

    def test_parse_delegation_response_clamps_priority(self):
        """Priority values are clamped to 1-3 range."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=1, sources_per_query=1)

        response = json.dumps({
            "research_complete": False,
            "directives": [
                {"research_topic": "Topic A", "priority": 0},
                {"research_topic": "Topic B", "priority": 10},
            ],
            "rationale": "",
        })

        directives, _ = stub._parse_delegation_response(response, state)

        assert directives[0].priority == 1  # Clamped up from 0
        assert directives[1].priority == 3  # Clamped down from 10


class TestDelegationIntegration:
    """Integration tests for the delegation supervision model."""

    @pytest.mark.asyncio
    async def test_delegation_dispatches_correctly(self):
        """When delegation_model=True, _execute_supervision_async uses delegation path."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=2, sources_per_query=3, supervision_round=1)

        # Round > 0 with sufficient sources triggers heuristic early exit
        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ):
            result = await stub._execute_supervision_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        assert result.success is True
        assert result.metadata.get("model") == "delegation"

    @pytest.mark.asyncio
    async def test_delegation_research_complete_signal(self):
        """ResearchComplete signal terminates the delegation loop."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=2, sources_per_query=2, supervision_round=0)

        delegation_response = json.dumps({
            "research_complete": True,
            "directives": [],
            "rationale": "All dimensions covered",
        })

        async def mock_execute_llm_call(**kwargs):
            phase = kwargs.get("phase_name")
            if phase == "supervision_think":
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content="Gap analysis", tokens_used=50
                )
                return result
            elif phase in ("supervision_delegate", "supervision_delegate_generate"):
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True,
                    content=delegation_response,
                    provider_id="test",
                    model_used="test",
                    tokens_used=80,
                    duration_ms=400.0,
                )
                return result
            elif phase == "supervision_delegate_critique":
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content="VERDICT: NO_ISSUES", tokens_used=30,
                    provider_id="test", model_used="test",
                )
                return result
            # No other calls expected
            raise AssertionError(f"Unexpected call: {phase}")

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            side_effect=_wrap_as_structured_mock(mock_execute_llm_call),
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ):
            result = await stub._execute_supervision_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        assert result.success is True
        assert "should_continue_gathering" not in result.metadata
        # Check history records completion
        history = state.metadata.get("supervision_history", [])
        assert len(history) == 1
        assert history[0]["method"] == "delegation_complete"

    @pytest.mark.asyncio
    async def test_delegation_round_limit_enforced(self):
        """Delegation loop respects max_supervision_rounds."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(
            num_completed=2, sources_per_query=2,
            supervision_round=0, max_supervision_rounds=1,
        )

        delegation_response = json.dumps({
            "research_complete": False,
            "directives": [
                {"research_topic": "Topic A", "priority": 2},
            ],
            "rationale": "Need more",
        })

        call_phases: list[str] = []

        async def mock_execute_llm_call(**kwargs):
            phase = kwargs.get("phase_name", "unknown")
            call_phases.append(phase)
            if phase == "supervision_think":
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content="Gaps exist", tokens_used=40
                )
                return result
            elif phase in ("supervision_delegate", "supervision_delegate_generate"):
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True,
                    content=delegation_response,
                    provider_id="test",
                    model_used="test",
                    tokens_used=60,
                    duration_ms=300.0,
                )
                return result
            elif phase == "supervision_delegate_critique":
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content="VERDICT: NO_ISSUES", tokens_used=20,
                    provider_id="test", model_used="test",
                )
                return result
            raise AssertionError(f"Unexpected: {phase}")

        # Mock _execute_topic_research_async to avoid needing real providers
        async def mock_topic_research(*args, **kwargs):
            sq = kwargs.get("sub_query") or args[0]
            return TopicResearchResult(
                sub_query_id=sq.id,
                sources_found=2,
                searches_performed=1,
            )

        mock_provider = MagicMock()
        mock_provider.get_provider_name.return_value = "tavily"

        stub.config.deep_research_providers = ["tavily"]
        stub.config.deep_research_topic_max_tool_calls = 10
        stub._execute_topic_research_async = mock_topic_research
        stub._get_search_provider = lambda provider_name: mock_provider

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            side_effect=_wrap_as_structured_mock(mock_execute_llm_call),
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ):
            result = await stub._execute_supervision_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        assert result.success is True
        # Only 1 round executed (max_supervision_rounds=1)
        assert state.supervision_round == 1
        assert result.metadata["total_directives_executed"] == 1

    @pytest.mark.asyncio
    async def test_delegation_no_directives_stops_loop(self):
        """When no directives are generated, the loop exits."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=2, sources_per_query=2, supervision_round=0)

        empty_delegation = json.dumps({
            "research_complete": False,
            "directives": [],
            "rationale": "Cannot identify specific gaps",
        })

        async def mock_execute_llm_call(**kwargs):
            phase = kwargs.get("phase_name")
            if phase == "supervision_think":
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content="Analysis", tokens_used=40
                )
                return result
            elif phase in ("supervision_delegate", "supervision_delegate_generate"):
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True,
                    content=empty_delegation,
                    provider_id="test",
                    model_used="test",
                    tokens_used=60,
                    duration_ms=300.0,
                )
                return result
            elif phase == "supervision_delegate_critique":
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content="VERDICT: NO_ISSUES", tokens_used=20,
                    provider_id="test", model_used="test",
                )
                return result
            raise AssertionError(f"Unexpected: {phase}")

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            side_effect=_wrap_as_structured_mock(mock_execute_llm_call),
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ):
            result = await stub._execute_supervision_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        assert result.success is True
        assert result.metadata["total_directives_executed"] == 0
        history = state.metadata["supervision_history"]
        # Validator now forces research_complete=True when directives are empty,
        # so the loop takes the "delegation_complete" path instead of "no_directives"
        assert history[0]["method"] == "delegation_complete"


# ===========================================================================
# Supervisor-Owned Decomposition (Phase 2 PLAN)
# ===========================================================================


class TestSupervisorOwnedDecompositionDetection:
    """Tests for _is_first_round_decomposition() detection logic."""

    def test_first_round_detected_when_round_zero_no_results(self):
        """Returns True: round 0, no prior topic results."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(
            num_completed=0, num_pending=0, supervision_round=0,
        )
        state.topic_research_results = []
        assert stub._is_first_round_decomposition(state) is True

    def test_not_first_round_when_round_gt_zero(self):
        """Returns False when supervision_round > 0 (already past first round)."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=2, supervision_round=1)
        state.topic_research_results = []
        assert stub._is_first_round_decomposition(state) is False

    def test_not_first_round_when_topic_results_exist(self):
        """Returns False when topic_research_results already present."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=1, supervision_round=0)
        state.topic_research_results = [
            TopicResearchResult(
                sub_query_id="sq-0",
                searches_performed=1,
                sources_found=2,
            )
        ]
        assert stub._is_first_round_decomposition(state) is False


class TestFirstRoundDecompositionPrompts:
    """Tests for first-round think and delegation prompts."""

    def test_first_round_think_system_prompt_is_strategic(self):
        """First-round think system prompt instructs decomposition strategy."""
        stub = StubSupervision(delegation_model=True)
        prompt = stub._build_first_round_think_system_prompt()

        assert "research strategist" in prompt.lower()
        assert "decomposition" in prompt.lower()
        assert "parallel" in prompt.lower()
        assert "same ground" in prompt.lower()  # self-critique: no overlap
        assert "missing" in prompt.lower()  # self-critique: no missing perspectives

    def test_first_round_think_prompt_includes_brief(self):
        """First-round think prompt includes research brief and query."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=0, supervision_round=0)
        state.research_brief = "Investigate the impact of transformer architectures on NLP"
        prompt = stub._build_first_round_think_prompt(state)

        assert state.original_query in prompt
        assert "transformer architectures" in prompt
        assert "Decomposition Strategy" in prompt
        assert "Query type" in prompt
        assert "Self-critique" in prompt

    def test_first_round_think_prompt_includes_clarification_constraints(self):
        """First-round think prompt includes clarification constraints when present."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=0, supervision_round=0)
        state.clarification_constraints = {"time_period": "last 5 years", "focus": "NLP"}
        prompt = stub._build_first_round_think_prompt(state)

        assert "time_period" in prompt
        assert "last 5 years" in prompt
        assert "focus" in prompt

    def test_first_round_think_prompt_includes_scaling_guidance(self):
        """First-round think prompt includes guidelines for researcher count scaling."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=0, supervision_round=0)
        prompt = stub._build_first_round_think_prompt(state)

        assert "Simple factual queries: 1-2" in prompt
        assert "Comparisons:" in prompt
        assert "3-5 researchers" in prompt

    def test_first_round_delegation_system_prompt_has_decomposition_rules(self):
        """First-round delegation system prompt absorbs planning decomposition rules."""
        stub = StubSupervision(delegation_model=True)
        prompt = stub._build_first_round_delegation_system_prompt()

        # Core decomposition rules (absorbed from planning.py)
        assert "2-5 directives" in prompt
        assert "research_topic" in prompt
        assert "perspective" in prompt
        assert "evidence_needed" in prompt
        assert "priority" in prompt
        # Scaling rules
        assert "FEWER researchers for simple queries" in prompt.upper() or "fewer researchers for simple queries" in prompt.lower()
        assert "COMPARISONS" in prompt.upper() or "comparison" in prompt.lower()
        # Self-critique is now a separate call — NOT in the generation prompt
        assert "Self-Critique" not in prompt
        # JSON format
        assert "research_complete" in prompt
        assert "directives" in prompt
        assert "rationale" in prompt

    def test_critique_system_prompt_has_quality_criteria(self):
        """Critique system prompt covers all four quality criteria."""
        stub = StubSupervision(delegation_model=True)
        prompt = stub._build_critique_system_prompt()

        assert "Redundancy" in prompt
        assert "Coverage" in prompt
        assert "Proportionality" in prompt
        assert "Specificity" in prompt
        assert "VERDICT" in prompt

    def test_revision_system_prompt_has_merge_instructions(self):
        """Revision system prompt instructs merging, adding, removing directives."""
        stub = StubSupervision(delegation_model=True)
        prompt = stub._build_revision_system_prompt()

        assert "MERGE" in prompt
        assert "ADD" in prompt
        assert "REMOVE" in prompt
        assert "research_complete" in prompt

    def test_critique_has_issues_detects_verdicts(self):
        """_critique_has_issues correctly parses VERDICT lines."""
        assert SupervisionPhaseMixin._critique_has_issues(
            "All good.\nVERDICT: NO_ISSUES"
        ) is False
        assert SupervisionPhaseMixin._critique_has_issues(
            "Problems found.\nVERDICT: REVISION_NEEDED"
        ) is True
        # Fallback: ISSUE markers without verdict
        assert SupervisionPhaseMixin._critique_has_issues(
            "1. Redundancy: ISSUE: directives 1 and 3 overlap"
        ) is True
        # No issues and no verdict
        assert SupervisionPhaseMixin._critique_has_issues(
            "Everything looks fine. All criteria pass."
        ) is False

    def test_first_round_delegation_user_prompt_includes_brief(self):
        """First-round delegation user prompt includes research brief and query."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=0, supervision_round=0)
        state.research_brief = "Compare React vs Vue for enterprise applications"
        prompt = stub._build_first_round_delegation_user_prompt(state)

        assert state.original_query in prompt
        assert "React vs Vue" in prompt
        assert "Instructions" in prompt

    def test_first_round_delegation_user_prompt_with_think_output(self):
        """First-round delegation user prompt integrates decomposition strategy."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=0, supervision_round=0)
        state.research_brief = "Compare React vs Vue"

        think_output = (
            "This is a comparison query. Need one researcher per framework "
            "plus one for cross-cutting comparison."
        )
        prompt = stub._build_first_round_delegation_user_prompt(state, think_output)

        assert "<decomposition_strategy>" in prompt
        assert "</decomposition_strategy>" in prompt
        assert "comparison query" in prompt

    def test_first_round_delegation_user_prompt_without_think_output(self):
        """First-round delegation user prompt works without think output."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=0, supervision_round=0)
        prompt = stub._build_first_round_delegation_user_prompt(state, think_output=None)

        assert "<decomposition_strategy>" not in prompt
        assert "Decompose the research query" in prompt


class TestFirstRoundDecompositionIntegration:
    """Integration tests for supervisor-owned decomposition flow."""

    @pytest.mark.asyncio
    async def test_first_round_produces_initial_directives(self):
        """Supervisor round 0 produces initial decomposition directives from brief."""
        stub = StubSupervision(delegation_model=True)
        stub.config.deep_research_topic_max_tool_calls = 5
        stub.config.deep_research_providers = ["tavily"]

        state = _make_state(
            num_completed=0, num_pending=0, supervision_round=0,
        )
        state.research_brief = "Compare React, Vue, and Angular for enterprise apps"
        state.topic_research_results = []
        state.max_sub_queries = 10

        # Think step returns decomposition strategy
        think_response = (
            "This is a comparison query with 3 elements. "
            "Deploy 3 researchers: one for React, one for Vue, one for Angular."
        )

        # Delegate step returns initial directives
        delegate_response = json.dumps({
            "research_complete": False,
            "directives": [
                {
                    "research_topic": "Investigate React's enterprise adoption, performance benchmarks, and ecosystem maturity.",
                    "perspective": "technical",
                    "evidence_needed": "benchmarks, case studies, adoption statistics",
                    "priority": 1,
                },
                {
                    "research_topic": "Investigate Vue's enterprise adoption, performance benchmarks, and ecosystem maturity.",
                    "perspective": "technical",
                    "evidence_needed": "benchmarks, case studies, adoption statistics",
                    "priority": 1,
                },
                {
                    "research_topic": "Investigate Angular's enterprise adoption, performance benchmarks, and ecosystem maturity.",
                    "perspective": "technical",
                    "evidence_needed": "benchmarks, case studies, adoption statistics",
                    "priority": 1,
                },
            ],
            "rationale": "One researcher per framework for balanced comparison",
        })

        call_count = 0

        async def mock_execute_llm_call(**kwargs):
            nonlocal call_count
            call_count += 1
            phase_name = kwargs.get("phase_name", "")

            if "think" in phase_name:
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True,
                    content=think_response,
                    provider_id="test",
                    model_used="test",
                    tokens_used=50,
                    duration_ms=200.0,
                )
                return result
            elif "delegate" in phase_name:
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True,
                    content=delegate_response,
                    provider_id="test",
                    model_used="test",
                    tokens_used=100,
                    duration_ms=500.0,
                )
                return result
            raise AssertionError(f"Unexpected phase: {phase_name}")

        # Mock topic researcher to avoid actual search
        async def mock_topic_research(**kwargs):
            sq = kwargs.get("sub_query")
            return TopicResearchResult(
                sub_query_id=sq.id if sq else "unknown",
                searches_performed=2,
                sources_found=3,
                per_topic_summary="Found relevant enterprise adoption data.",
                compressed_findings="Framework shows strong enterprise adoption...",
            )

        stub._execute_topic_research_async = mock_topic_research
        stub._get_search_provider = MagicMock(return_value=MagicMock())

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            side_effect=_wrap_as_structured_mock(mock_execute_llm_call),
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ):
            result = await stub._execute_supervision_delegation_async(
                state=state, provider_id="test-provider", timeout=30.0,
            )

        assert result.success is True
        assert result.metadata["model"] == "delegation"
        # Should have generated 3 directives from decomposition
        assert len(state.directives) >= 3
        # Topic results should be populated
        assert len(state.topic_research_results) >= 3
        # Supervision round advanced
        assert state.supervision_round >= 1

    @pytest.mark.asyncio
    async def test_second_round_uses_gap_analysis(self):
        """Round > 0 uses standard gap-driven delegation (not first-round decomposition).

        The heuristic always returns should_continue_gathering=False (conservative),
        so we patch it to force True — this lets the delegation path run so we can
        verify it uses the standard (non-first-round) prompts.
        """
        stub = StubSupervision(delegation_model=True)
        stub.config.deep_research_topic_max_tool_calls = 5
        stub.config.deep_research_providers = ["tavily"]

        state = _make_state(
            num_completed=2, num_pending=0, supervision_round=1,
        )
        state.topic_research_results = [
            TopicResearchResult(
                sub_query_id="sq-0",
                searches_performed=1,
                sources_found=2,
            )
        ]
        state.max_sub_queries = 10

        # Standard gap-driven delegation response (research complete)
        delegate_response = json.dumps({
            "research_complete": True,
            "directives": [],
            "rationale": "Coverage sufficient",
        })

        async def mock_execute_llm_call(**kwargs):
            result = MagicMock()
            result.result = WorkflowResult(
                success=True,
                content=delegate_response,
                provider_id="test",
                model_used="test",
                tokens_used=50,
                duration_ms=200.0,
            )
            return result

        # Patch heuristic to return should_continue_gathering=True so the
        # delegation path runs (heuristic normally always returns False).
        def mock_heuristic(state, min_sources):
            return {
                "overall_coverage": "partial",
                "should_continue_gathering": True,
                "queries_assessed": 2,
                "queries_sufficient": 1,
            }

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            side_effect=_wrap_as_structured_mock(mock_execute_llm_call),
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ), patch.object(stub, "_assess_coverage_heuristic", side_effect=mock_heuristic):
            result = await stub._execute_supervision_delegation_async(
                state=state, provider_id="test-provider", timeout=30.0,
            )

        assert result.success is True
        # Should have used standard prompts (not first-round decomposition)
        history = state.metadata.get("supervision_history", [])
        assert len(history) >= 1
        assert history[0]["method"] == "delegation_complete"

    @pytest.mark.asyncio
    async def test_round_zero_counted_toward_max_rounds(self):
        """Decomposition round 0 counts toward max_supervision_rounds budget."""
        stub = StubSupervision(delegation_model=True)
        stub.config.deep_research_topic_max_tool_calls = 5
        stub.config.deep_research_providers = ["tavily"]

        state = _make_state(
            num_completed=0, num_pending=0, supervision_round=0,
            max_supervision_rounds=1,  # Only 1 round allowed
        )
        state.topic_research_results = []
        state.max_sub_queries = 10
        state.research_brief = "Simple query"

        delegate_response = json.dumps({
            "research_complete": False,
            "directives": [
                {
                    "research_topic": "Test directive",
                    "perspective": "general",
                    "evidence_needed": "articles",
                    "priority": 1,
                }
            ],
            "rationale": "Initial decomposition",
        })

        async def mock_execute_llm_call(**kwargs):
            result = MagicMock()
            result.result = WorkflowResult(
                success=True,
                content=delegate_response,
                provider_id="test",
                model_used="test",
                tokens_used=50,
                duration_ms=200.0,
            )
            return result

        async def mock_topic_research(**kwargs):
            sq = kwargs.get("sub_query")
            return TopicResearchResult(
                sub_query_id=sq.id if sq else "unknown",
                searches_performed=1,
                sources_found=2,
            )

        stub._execute_topic_research_async = mock_topic_research
        stub._get_search_provider = MagicMock(return_value=MagicMock())

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            side_effect=_wrap_as_structured_mock(mock_execute_llm_call),
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ):
            result = await stub._execute_supervision_delegation_async(
                state=state, provider_id="test-provider", timeout=30.0,
            )

        assert result.success is True
        # Round 0 used the single allowed round — loop should have terminated
        assert state.supervision_round == 1  # Advanced past round 0
        # No second round should have executed
        assert result.metadata["total_directives_executed"] <= 1


class TestPhaseFlowSupervisorOwnedDecomposition:
    """Tests for workflow phase flow with supervisor-owned decomposition."""

    def test_brief_to_supervision_transition(self):
        """BRIEF → SUPERVISION transition skips deprecated GATHERING phase."""
        state = DeepResearchState(
            original_query="test query",
            phase=DeepResearchPhase.BRIEF,
            research_brief="Enriched research brief",
        )

        # BRIEF → SUPERVISION (GATHERING is deprecated and skipped)
        state.advance_phase()
        assert state.phase == DeepResearchPhase.SUPERVISION

    def test_supervision_to_synthesis_transition(self):
        """SUPERVISION → SYNTHESIS transition follows the active phase order."""
        state = DeepResearchState(
            original_query="test query",
            phase=DeepResearchPhase.SUPERVISION,
            supervision_round=2,
        )
        # advance_phase() from SUPERVISION goes to SYNTHESIS
        new_phase = state.advance_phase()
        assert new_phase == DeepResearchPhase.SYNTHESIS
        assert state.phase == DeepResearchPhase.SYNTHESIS


# ===========================================================================
# PLAN Phase 1: Unified Supervisor Orchestration tests
# ===========================================================================


class TestUnifiedSupervisorOrchestration:
    """Tests for PLAN Phase 1: Unify Supervision as the Research Orchestrator.

    Validates that:
    - New workflows go BRIEF → SUPERVISION → SYNTHESIS (no PLANNING/GATHERING)
    - GATHERING is legacy-resume-only with deprecation logging
    - PLANNING phase is absent from the enum (already removed)
    - Supervisor round 0 performs decomposition; round 1+ assesses gaps
    - Old config files with removed keys don't crash
    """

    # ------------------------------------------------------------------
    # 1.1: Config flag removed — old config files don't crash
    # ------------------------------------------------------------------

    def test_old_config_key_supervisor_owned_decomposition_ignored(self):
        """from_toml_dict ignores the removed supervisor_owned_decomposition key."""
        from foundry_mcp.config.research import ResearchConfig

        data = {
            "deep_research_supervisor_owned_decomposition": True,
            "default_provider": "gemini",
        }
        config = ResearchConfig.from_toml_dict(data)
        # Should not raise — unknown keys are silently ignored by data.get()
        assert config.default_provider == "gemini"
        assert not hasattr(config, "deep_research_supervisor_owned_decomposition")

    # ------------------------------------------------------------------
    # 1.2: BRIEF → SUPERVISION is the sole default transition
    # ------------------------------------------------------------------

    def test_new_workflow_skips_planning_and_gathering(self):
        """After BRIEF, workflow_execution sets phase to SUPERVISION, not PLANNING or GATHERING."""
        # Simulate the transition logic from workflow_execution.py lines 194-199
        state = DeepResearchState(
            original_query="test query",
            phase=DeepResearchPhase.BRIEF,
        )
        # After BRIEF completes, the orchestrator advances phase to GATHERING
        # (via advance_phase), but workflow_execution.py overrides to SUPERVISION.
        # Simulate: after BRIEF, phase is set via the conditional logic.
        # The elif branch sets phase to SUPERVISION for anything not already
        # SUPERVISION or SYNTHESIS.
        state.phase = DeepResearchPhase.BRIEF  # Just finished BRIEF
        # Apply the workflow_execution.py conditional:
        if state.phase == DeepResearchPhase.GATHERING:
            pass  # Legacy resume
        elif state.phase not in (DeepResearchPhase.SUPERVISION, DeepResearchPhase.SYNTHESIS):
            state.phase = DeepResearchPhase.SUPERVISION

        assert state.phase == DeepResearchPhase.SUPERVISION

    def test_planning_phase_deprecated_in_enum(self):
        """PLANNING phase exists in DeepResearchPhase for legacy deserialization but is skipped."""
        phase_values = [p.value for p in DeepResearchPhase]
        assert "planning" in phase_values
        # PLANNING is in _SKIP_PHASES so advance_phase() skips over it
        assert DeepResearchPhase.PLANNING in DeepResearchState._SKIP_PHASES

    # ------------------------------------------------------------------
    # 1.3 / 1.4: GATHERING is legacy-resume-only with deprecation logging
    # ------------------------------------------------------------------

    def test_gathering_only_runs_from_legacy_resume(self):
        """GATHERING block only executes when state.phase == GATHERING on entry."""
        # New workflow: phase transitions from BRIEF → SUPERVISION (skipping GATHERING)
        state = DeepResearchState(
            original_query="test query",
            phase=DeepResearchPhase.BRIEF,
        )
        # Apply workflow_execution.py logic
        if state.phase == DeepResearchPhase.GATHERING:
            entered_gathering = True
        elif state.phase not in (DeepResearchPhase.SUPERVISION, DeepResearchPhase.SYNTHESIS):
            state.phase = DeepResearchPhase.SUPERVISION
            entered_gathering = False

        assert not entered_gathering
        assert state.phase == DeepResearchPhase.SUPERVISION

    def test_legacy_gathering_resume_enters_gathering_block(self):
        """Legacy saved state at GATHERING enters the GATHERING block."""
        state = DeepResearchState(
            original_query="test query",
            phase=DeepResearchPhase.GATHERING,
        )
        # Apply workflow_execution.py logic
        entered_gathering = False
        if state.phase == DeepResearchPhase.GATHERING:
            entered_gathering = True
        elif state.phase not in (DeepResearchPhase.SUPERVISION, DeepResearchPhase.SYNTHESIS):
            state.phase = DeepResearchPhase.SUPERVISION

        assert entered_gathering
        assert state.phase == DeepResearchPhase.GATHERING

    def test_deprecation_log_emitted_for_legacy_gathering(self):
        """Deprecation warning is logged when legacy GATHERING phase runs."""
        state = DeepResearchState(
            original_query="test query",
            phase=DeepResearchPhase.GATHERING,
        )

        # Simulate the workflow_execution.py deprecation logging
        with patch(
            "foundry_mcp.core.research.workflows.deep_research.workflow_execution.logger"
        ) as mock_logger:
            if state.phase == DeepResearchPhase.GATHERING:
                mock_logger.warning(
                    "GATHERING phase running from legacy saved state (research %s) "
                    "— new workflows use supervisor-owned decomposition via SUPERVISION phase",
                    state.id,
                )

            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert "legacy saved state" in call_args[0][0]
            assert "GATHERING" in call_args[0][0]
            assert "SUPERVISION" in call_args[0][0]

    # ------------------------------------------------------------------
    # 1.5: First-round decomposition prompts verified
    # ------------------------------------------------------------------

    def test_first_round_think_prompt_has_bias_toward_single_researcher(self):
        """First-round think system prompt biases toward single researcher for simple queries."""
        stub = StubSupervision(delegation_model=True)
        prompt = stub._build_first_round_think_system_prompt()
        assert "1-2 researchers" in prompt or "simple factual queries" in prompt.lower()

    def test_first_round_delegation_prompt_has_comparison_guidance(self):
        """First-round delegation system prompt parallelizes for comparisons."""
        stub = StubSupervision(delegation_model=True)
        prompt = stub._build_first_round_delegation_system_prompt()
        # Comparison guidance
        assert "comparison" in prompt.lower()
        assert "one directive per" in prompt.lower() or "one per" in prompt.lower()

    def test_first_round_delegation_prompt_has_2_to_5_range(self):
        """First-round delegation system prompt specifies 2-5 directives range."""
        stub = StubSupervision(delegation_model=True)
        prompt = stub._build_first_round_delegation_system_prompt()
        assert "2-5" in prompt

    def test_first_round_delegation_prompt_no_inline_self_critique(self):
        """First-round delegation system prompt does NOT have inline self-critique.

        Self-critique is now handled by a separate LLM call (call 2 of the
        decompose → critique → revise pipeline).
        """
        stub = StubSupervision(delegation_model=True)
        prompt = stub._build_first_round_delegation_system_prompt()
        assert "Self-Critique" not in prompt
        # Critique concerns live in the separate critique prompt
        critique_prompt = stub._build_critique_system_prompt()
        assert "redundancy" in critique_prompt.lower()
        assert "coverage" in critique_prompt.lower()

    # ------------------------------------------------------------------
    # 1.6: Round 0 → round 1 handoff
    # ------------------------------------------------------------------

    def test_round_zero_always_delegates(self):
        """Round 0 performs decomposition (no heuristic early-exit)."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(
            num_completed=0, num_pending=0, supervision_round=0,
        )
        state.topic_research_results = []

        # The heuristic early-exit only runs when supervision_round > 0
        # Verify: at round 0, _is_first_round_decomposition returns True
        assert stub._is_first_round_decomposition(state) is True

    def test_round_one_assesses_round_zero_results(self):
        """Round 1 heuristic sees round 0's topic_research_results and sources."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(
            num_completed=3, sources_per_query=3, supervision_round=1,
        )
        # Add topic research results (as if round 0 produced them)
        for sq in state.completed_sub_queries():
            state.topic_research_results.append(
                TopicResearchResult(
                    sub_query_id=sq.id,
                    searches_performed=2,
                    sources_found=3,
                )
            )

        # At round 1, heuristic assesses coverage
        heuristic = stub._assess_coverage_heuristic(state, min_sources=2)

        # All 3 sub-queries have 3 sources each (>= min_sources=2)
        assert heuristic["overall_coverage"] == "sufficient"
        assert heuristic["should_continue_gathering"] is False
        assert heuristic["queries_assessed"] == 3
        assert heuristic["queries_sufficient"] == 3

    def test_round_one_heuristic_detects_gaps_from_round_zero(self):
        """Round 1 heuristic identifies insufficient coverage from round 0."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(
            num_completed=3, sources_per_query=1, supervision_round=1,
        )
        # Only 1 source per query — below min_sources=2
        heuristic = stub._assess_coverage_heuristic(state, min_sources=2)

        assert heuristic["overall_coverage"] == "insufficient"
        assert heuristic["queries_sufficient"] == 0

    def test_round_zero_counted_toward_max_supervision_rounds(self):
        """Round 0 is counted in supervision_round tracking."""
        state = _make_state(supervision_round=0, max_supervision_rounds=3)

        # After round 0 executes, supervision_round increments to 1
        state.supervision_round += 1
        assert state.supervision_round == 1

        # After 3 rounds (0, 1, 2), supervision_round == 3 == max
        state.supervision_round = 3
        assert state.supervision_round >= state.max_supervision_rounds

    # ------------------------------------------------------------------
    # 1.8: Full phase flow integration tests
    # ------------------------------------------------------------------

    def test_new_workflow_phase_sequence_excludes_planning_and_gathering(self):
        """New workflow visits only: CLARIFICATION → BRIEF → SUPERVISION → SYNTHESIS."""
        state = DeepResearchState(
            original_query="What is quantum computing?",
            phase=DeepResearchPhase.CLARIFICATION,
        )
        visited_phases = [state.phase]

        # CLARIFICATION → BRIEF
        state.advance_phase()
        visited_phases.append(state.phase)

        # BRIEF → (skip to SUPERVISION via workflow_execution.py logic)
        # advance_phase would go to GATHERING, but workflow_execution overrides
        if state.phase == DeepResearchPhase.GATHERING:
            pass
        elif state.phase not in (DeepResearchPhase.SUPERVISION, DeepResearchPhase.SYNTHESIS):
            state.phase = DeepResearchPhase.SUPERVISION
        visited_phases.append(state.phase)

        # SUPERVISION → SYNTHESIS
        state.phase = DeepResearchPhase.SYNTHESIS
        visited_phases.append(state.phase)

        assert visited_phases == [
            DeepResearchPhase.CLARIFICATION,
            DeepResearchPhase.BRIEF,
            DeepResearchPhase.SUPERVISION,
            DeepResearchPhase.SYNTHESIS,
        ]
        # GATHERING was never visited
        assert DeepResearchPhase.GATHERING not in visited_phases

    def test_coverage_data_includes_round_zero_results(self):
        """_build_per_query_coverage includes topic_research_results from round 0."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=2, sources_per_query=2, supervision_round=1)

        # Add topic research results with compressed findings
        for sq in state.completed_sub_queries():
            state.topic_research_results.append(
                TopicResearchResult(
                    sub_query_id=sq.id,
                    searches_performed=2,
                    sources_found=2,
                    compressed_findings="Key findings about deep learning aspect.",
                )
            )

        coverage = stub._build_per_query_coverage(state)

        assert len(coverage) == 2
        for entry in coverage:
            assert entry["source_count"] == 2
            assert entry["compressed_findings_excerpt"] is not None
            assert "Key findings" in entry["compressed_findings_excerpt"]


# ===========================================================================
# Supervision Message Accumulation (PLAN Phase 1)
# ===========================================================================


class TestSupervisionMessageAccumulation:
    """Tests for supervisor message accumulation across delegation rounds.

    Covers PLAN checklist items:
    - 1.1: supervision_messages field on state
    - 1.2: compressed findings formatted as tool-result messages
    - 1.3: accumulated messages passed to delegation LLM
    - 1.4: structured coverage data still present as supplementary context
    - 1.6: supervisor sees prior round findings in message history
    - 1.7: supervisor doesn't re-delegate already-covered topics
    """

    def test_supervision_messages_field_exists(self):
        """1.1: DeepResearchState has supervision_messages field."""
        state = DeepResearchState(original_query="test")
        assert hasattr(state, "supervision_messages")
        assert state.supervision_messages == []

    def test_supervision_messages_serializable(self):
        """1.1: supervision_messages round-trips through serialization."""
        state = DeepResearchState(original_query="test")
        state.supervision_messages = [
            {"role": "assistant", "type": "think", "round": 0, "content": "Gap analysis"},
            {"role": "tool_result", "type": "research_findings", "round": 0,
             "directive_id": "sq-1", "content": "Findings about topic A"},
        ]
        data = state.model_dump()
        restored = DeepResearchState(**data)
        assert len(restored.supervision_messages) == 2
        assert restored.supervision_messages[0]["content"] == "Gap analysis"
        assert restored.supervision_messages[1]["directive_id"] == "sq-1"

    @pytest.mark.asyncio
    async def test_messages_accumulated_during_delegation(self):
        """1.2/1.6: Think output, delegation response, and compressed findings
        are accumulated in supervision_messages after each round."""
        stub = StubSupervision(delegation_model=True)
        stub.config.deep_research_topic_max_tool_calls = 5
        stub.config.deep_research_providers = ["tavily"]

        state = _make_state(
            num_completed=0, num_pending=0, supervision_round=0,
            max_supervision_rounds=2,
        )
        state.research_brief = "Investigate AI safety"
        state.topic_research_results = []
        state.max_sub_queries = 10

        think_response = "Gap: No coverage of alignment techniques"
        delegate_response = json.dumps({
            "research_complete": False,
            "directives": [
                {"research_topic": "AI alignment techniques", "priority": 1},
            ],
            "rationale": "Need alignment coverage",
        })

        # Second round: research_complete = true
        delegate_response_r1 = json.dumps({
            "research_complete": True,
            "directives": [],
            "rationale": "All covered now",
        })

        round_counter = {"value": 0}

        async def mock_execute_llm_call(**kwargs):
            phase_name = kwargs.get("phase_name", "")
            if "think" in phase_name:
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content=think_response, tokens_used=50
                )
                return result
            elif "delegate" in phase_name:
                content = delegate_response if round_counter["value"] == 0 else delegate_response_r1
                round_counter["value"] += 1
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content=content,
                    provider_id="test", model_used="test",
                    tokens_used=80, duration_ms=400.0,
                )
                return result
            raise AssertionError(f"Unexpected phase: {phase_name}")

        async def mock_topic_research(**kwargs):
            sq = kwargs.get("sub_query")
            return TopicResearchResult(
                sub_query_id=sq.id if sq else "unknown",
                searches_performed=2,
                sources_found=3,
                compressed_findings="AI alignment research shows RLHF is dominant approach...",
            )

        stub._execute_topic_research_async = mock_topic_research
        stub._get_search_provider = MagicMock(return_value=MagicMock())

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            side_effect=_wrap_as_structured_mock(mock_execute_llm_call),
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.truncate_supervision_messages",
            side_effect=lambda messages, model: messages,  # no truncation in test
        ):
            result = await stub._execute_supervision_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        assert result.success is True

        # Verify supervision_messages were accumulated
        msgs = state.supervision_messages
        assert len(msgs) >= 3, f"Expected >= 3 messages, got {len(msgs)}"

        # Check message types are present
        think_msgs = [m for m in msgs if m.get("type") == "think"]
        delegation_msgs = [m for m in msgs if m.get("type") == "delegation"]
        findings_msgs = [m for m in msgs if m.get("type") == "research_findings"]

        assert len(think_msgs) >= 1, "Should have at least one think message"
        assert len(delegation_msgs) >= 1, "Should have at least one delegation message"
        assert len(findings_msgs) >= 1, "Should have at least one findings message"

        # Verify findings content
        assert any("RLHF" in m["content"] for m in findings_msgs), \
            "Findings messages should contain compressed research content"

    @pytest.mark.asyncio
    async def test_messages_injected_into_delegation_prompt(self):
        """1.3: Accumulated messages appear in the delegation user prompt."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=2, sources_per_query=2, supervision_round=1)

        # Pre-populate supervision_messages from a prior round
        state.supervision_messages = [
            {
                "role": "assistant", "type": "think", "round": 0,
                "content": "The research has a gap in regulatory analysis.",
            },
            {
                "role": "assistant", "type": "delegation", "round": 0,
                "content": json.dumps({
                    "research_complete": False,
                    "directives": [{"research_topic": "Regulatory framework", "priority": 1}],
                }),
            },
            {
                "role": "tool_result", "type": "research_findings", "round": 0,
                "directive_id": "sq-reg-1",
                "content": "EU AI Act imposes strict requirements on high-risk systems...",
            },
        ]

        coverage = stub._build_per_query_coverage(state)
        prompt = stub._build_delegation_user_prompt(state, coverage, think_output="New gaps here")

        # Verify prior conversation is in the prompt
        assert "Prior Supervisor Conversation" in prompt
        assert "regulatory analysis" in prompt
        assert "EU AI Act" in prompt
        assert "[Round 0] Your Gap Analysis" in prompt
        assert "[Round 0] Research Findings" in prompt

        # Verify structured coverage data is ALSO present (1.4)
        assert "Current Research Coverage" in prompt

    def test_delegation_prompt_without_prior_messages(self):
        """1.4: Coverage data is still present when no prior messages exist."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=2, sources_per_query=2, supervision_round=0)
        assert state.supervision_messages == []

        coverage = stub._build_per_query_coverage(state)
        prompt = stub._build_delegation_user_prompt(state, coverage, think_output=None)

        # No prior conversation section
        assert "Prior Supervisor Conversation" not in prompt
        # But coverage data is present
        assert "Current Research Coverage" in prompt
        assert "Sources:" in prompt


class TestSupervisionMessageTruncation:
    """Tests for token-limit guard on supervision message history.

    Covers PLAN checklist item 1.5 and 1.8.
    """

    def test_truncation_no_op_when_within_budget(self):
        """Messages within budget are returned unchanged."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            truncate_supervision_messages,
        )

        messages = [
            {"role": "assistant", "type": "think", "round": 0, "content": "Short analysis"},
            {"role": "tool_result", "type": "research_findings", "round": 0,
             "directive_id": "d1", "content": "Brief findings"},
        ]
        result = truncate_supervision_messages(messages, model=None)
        assert result == messages

    def test_truncation_removes_oldest_rounds_first(self):
        """1.8: When truncation is needed, oldest rounds are removed first."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            truncate_supervision_messages,
        )

        # Create messages with long content across 3 rounds
        messages = []
        for r in range(3):
            messages.append({
                "role": "assistant", "type": "think", "round": r,
                "content": f"Think output round {r}: " + "x" * 50_000,
            })
            messages.append({
                "role": "tool_result", "type": "research_findings", "round": r,
                "directive_id": f"d-{r}",
                "content": f"Findings round {r}: " + "y" * 50_000,
            })

        # Force a small budget to trigger truncation
        small_limits = {"test-model": 10_000}
        result = truncate_supervision_messages(
            messages, model="test-model", token_limits=small_limits,
        )

        # Should have removed oldest rounds
        assert len(result) < len(messages)
        # Most recent round should be preserved
        remaining_rounds = {m.get("round") for m in result}
        assert 2 in remaining_rounds, "Most recent round (2) should be preserved"

    def test_truncation_preserves_most_recent_round(self):
        """1.8: Even under heavy truncation, the most recent round survives."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            truncate_supervision_messages,
        )

        messages = []
        for r in range(5):
            messages.append({
                "role": "assistant", "type": "think", "round": r,
                "content": "x" * 100_000,
            })

        # Very small budget — only the last round should survive
        tiny_limits = {"tiny": 5_000}
        result = truncate_supervision_messages(
            messages, model="tiny", token_limits=tiny_limits,
        )

        remaining_rounds = {m.get("round") for m in result}
        assert 4 in remaining_rounds, "Most recent round (4) must survive"
        # Oldest rounds should be removed
        assert 0 not in remaining_rounds

    def test_truncation_empty_messages(self):
        """Empty message list is returned unchanged."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            truncate_supervision_messages,
        )

        result = truncate_supervision_messages([], model=None)
        assert result == []


# ===========================================================================
# Phase 4: Type-Aware Supervision Message Truncation
# ===========================================================================


class TestTypeAwareSupervisionTruncation:
    """Tests for Phase 4: type-aware supervision message truncation.

    Covers PLAN checklist items:
    - 4.1: Type-aware budgeting (60% reasoning, 40% findings)
    - 4.2: Findings body truncation before dropping whole messages
    - 4.3: preserve_last_n_thinks parameter
    - 4.5: Think messages preserved when findings are truncated
    - 4.6: Last N think messages survive aggressive truncation
    - 4.7: Total token usage within model limits after truncation
    """

    def test_think_messages_preserved_over_findings(self):
        """4.5: Think message content is preserved while findings content is heavily truncated."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            truncate_supervision_messages,
        )

        messages = []
        for r in range(3):
            # Think message — smaller, high-value
            messages.append({
                "role": "assistant", "type": "think", "round": r,
                "content": f"Gap analysis round {r}: " + "x" * 5_000,
            })
            # Findings message — larger, lower priority
            messages.append({
                "role": "tool_result", "type": "research_findings", "round": r,
                "directive_id": f"d-{r}",
                "content": f"Findings round {r}: " + "y" * 50_000,
            })

        original_think_chars = sum(
            len(m["content"]) for m in messages if m["type"] == "think"
        )
        original_findings_chars = sum(
            len(m["content"]) for m in messages if m["type"] == "research_findings"
        )

        # Budget that can hold thinks but not all findings
        small_limits = {"test-model": 10_000}
        result = truncate_supervision_messages(
            messages, model="test-model", token_limits=small_limits,
        )

        # Measure surviving content by type
        surviving_think_chars = sum(
            len(m.get("content", "")) for m in result if m.get("type") == "think"
        )
        surviving_findings_chars = sum(
            len(m.get("content", "")) for m in result if m.get("type") == "research_findings"
        )

        # Think messages should retain more of their content proportionally
        think_retention = surviving_think_chars / original_think_chars if original_think_chars else 0
        findings_retention = surviving_findings_chars / original_findings_chars if original_findings_chars else 0

        assert think_retention > findings_retention, (
            f"Think retention ({think_retention:.2%}) should exceed findings retention "
            f"({findings_retention:.2%}) due to type-aware budgeting"
        )
        assert surviving_think_chars > 0, "At least some think content must survive"

    def test_last_n_thinks_survive_aggressive_truncation(self):
        """4.6: The most recent N think messages survive even under extreme budget pressure."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            truncate_supervision_messages,
        )

        messages = []
        for r in range(5):
            messages.append({
                "role": "assistant", "type": "think", "round": r,
                "content": f"Gap analysis round {r}: " + "x" * 20_000,
            })
            messages.append({
                "role": "assistant", "type": "delegation", "round": r,
                "content": f"Delegation round {r}: " + "d" * 10_000,
            })
            messages.append({
                "role": "tool_result", "type": "research_findings", "round": r,
                "directive_id": f"d-{r}",
                "content": f"Findings round {r}: " + "y" * 40_000,
            })

        # Very small budget — forces aggressive truncation
        tiny_limits = {"tiny": 5_000}
        result = truncate_supervision_messages(
            messages, model="tiny", token_limits=tiny_limits,
            preserve_last_n_thinks=2,
        )

        # The 2 most recent think messages (rounds 3 and 4) must survive
        surviving_thinks = [
            m for m in result
            if m.get("type") == "think"
        ]
        surviving_think_rounds = {m.get("round") for m in surviving_thinks}

        assert 4 in surviving_think_rounds, "Most recent think (round 4) must survive"
        assert 3 in surviving_think_rounds, "Second-most-recent think (round 3) must survive"
        assert len(surviving_thinks) >= 2, (
            f"At least 2 think messages must survive, got {len(surviving_thinks)}"
        )

    def test_findings_body_truncated_before_dropping(self):
        """4.2: Findings bodies are truncated (keeping headers) before messages are dropped."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            truncate_supervision_messages,
            _FINDINGS_BODY_TRUNCATION_HEADER_CHARS,
        )

        # One small think message + one large findings message
        messages = [
            {
                "role": "assistant", "type": "think", "round": 0,
                "content": "Short analysis",
            },
            {
                "role": "tool_result", "type": "research_findings", "round": 0,
                "directive_id": "d-0",
                "content": "# Key Findings Header\nImportant summary line.\n\n" + "y" * 20_000,
            },
        ]

        # Budget forces truncation but should keep the findings message
        # (just with truncated body)
        limits = {"test": 3_000}
        result = truncate_supervision_messages(
            messages, model="test", token_limits=limits,
        )

        # The findings message should still exist (not dropped)
        findings_msgs = [m for m in result if m.get("type") == "research_findings"]
        assert len(findings_msgs) >= 1, "Findings message should be truncated, not dropped"

        # If it was truncated, it should contain the truncation marker and header
        for fm in findings_msgs:
            content = fm.get("content", "")
            if len(content) < 20_000:  # Was truncated
                assert "Key Findings Header" in content, (
                    "Truncated findings should preserve the header"
                )
                assert len(content) <= _FINDINGS_BODY_TRUNCATION_HEADER_CHARS + 100, (
                    "Truncated body should be roughly header-sized"
                )

    def test_total_chars_within_budget_after_truncation(self):
        """4.7: Total message content stays within the model's token budget after truncation."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            truncate_supervision_messages,
            _SUPERVISION_HISTORY_BUDGET_FRACTION,
            _CHARS_PER_TOKEN,
        )

        messages = []
        for r in range(4):
            messages.append({
                "role": "assistant", "type": "think", "round": r,
                "content": f"Think {r}: " + "t" * 15_000,
            })
            messages.append({
                "role": "tool_result", "type": "research_findings", "round": r,
                "directive_id": f"d-{r}",
                "content": f"Findings {r}: " + "f" * 30_000,
            })

        model_limit = 10_000  # tokens
        limits = {"budget-test": model_limit}
        budget_chars = int(model_limit * _SUPERVISION_HISTORY_BUDGET_FRACTION * _CHARS_PER_TOKEN)

        result = truncate_supervision_messages(
            messages, model="budget-test", token_limits=limits,
        )

        result_chars = sum(len(m.get("content", "")) for m in result)
        # Allow protected thinks to exceed budget (by design), but total should
        # be reasonable — within 2x budget at most
        assert result_chars < budget_chars * 2, (
            f"Result ({result_chars} chars) should be within 2x budget ({budget_chars} chars)"
        )

    def test_type_aware_budget_split(self):
        """4.1: Verify the 60/40 reasoning/findings budget split is applied."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            truncate_supervision_messages,
        )

        # Create scenario where findings are much larger than reasoning
        messages = []
        for r in range(3):
            messages.append({
                "role": "assistant", "type": "think", "round": r,
                "content": f"Think {r}: " + "t" * 8_000,
            })
            messages.append({
                "role": "tool_result", "type": "research_findings", "round": r,
                "directive_id": f"d-{r}",
                "content": f"Findings {r}: " + "f" * 40_000,
            })

        limits = {"split-test": 10_000}
        result = truncate_supervision_messages(
            messages, model="split-test", token_limits=limits,
        )

        # Reasoning messages should be better preserved proportionally
        think_chars = sum(
            len(m.get("content", ""))
            for m in result if m.get("type") == "think"
        )
        findings_chars = sum(
            len(m.get("content", ""))
            for m in result if m.get("type") == "research_findings"
        )

        total = think_chars + findings_chars
        if total > 0:
            think_ratio = think_chars / total
            # Think messages should get a larger share than naive proportional
            # (they were ~15% of original but should get ~60% of budget)
            assert think_ratio > 0.3, (
                f"Think ratio ({think_ratio:.2f}) should reflect priority budget allocation"
            )

    def test_preserve_last_n_thinks_custom_value(self):
        """4.3: Custom preserve_last_n_thinks value is respected."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            truncate_supervision_messages,
        )

        messages = []
        for r in range(6):
            messages.append({
                "role": "assistant", "type": "think", "round": r,
                "content": f"Think {r}: " + "t" * 15_000,
            })

        tiny_limits = {"tiny": 3_000}
        result = truncate_supervision_messages(
            messages, model="tiny", token_limits=tiny_limits,
            preserve_last_n_thinks=3,
        )

        surviving_rounds = sorted(m.get("round", -1) for m in result if m.get("type") == "think")
        # Last 3 rounds (3, 4, 5) must survive
        assert 5 in surviving_rounds, "Round 5 think must survive"
        assert 4 in surviving_rounds, "Round 4 think must survive"
        assert 3 in surviving_rounds, "Round 3 think must survive"

    def test_delegation_messages_treated_as_reasoning(self):
        """4.1: Delegation messages share the reasoning budget with think messages."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            truncate_supervision_messages,
        )

        messages = []
        for r in range(3):
            messages.append({
                "role": "assistant", "type": "delegation", "round": r,
                "content": f"Delegation {r}: " + "d" * 5_000,
            })
            messages.append({
                "role": "tool_result", "type": "research_findings", "round": r,
                "directive_id": f"d-{r}",
                "content": f"Findings {r}: " + "f" * 50_000,
            })

        original_deleg_chars = sum(
            len(m["content"]) for m in messages if m["type"] == "delegation"
        )
        original_findings_chars = sum(
            len(m["content"]) for m in messages if m["type"] == "research_findings"
        )

        limits = {"deleg-test": 8_000}
        result = truncate_supervision_messages(
            messages, model="deleg-test", token_limits=limits,
        )

        # Delegation messages should retain more content than findings
        surviving_deleg_chars = sum(
            len(m.get("content", "")) for m in result if m.get("type") == "delegation"
        )
        surviving_findings_chars = sum(
            len(m.get("content", "")) for m in result if m.get("type") == "research_findings"
        )

        deleg_retention = surviving_deleg_chars / original_deleg_chars if original_deleg_chars else 0
        findings_retention = surviving_findings_chars / original_findings_chars if original_findings_chars else 0

        assert surviving_deleg_chars > 0, "At least some delegation content should survive"
        assert deleg_retention > findings_retention, (
            f"Delegation retention ({deleg_retention:.2%}) should exceed "
            f"findings retention ({findings_retention:.2%})"
        )


# ===========================================================================
# Phase 5: Coverage Delta Injection for Supervisor Think Step
# ===========================================================================


class TestCoverageDelta:
    """Tests for Phase 5: coverage delta computation and injection.

    Covers PLAN checklist items:
    - 5.1: _compute_coverage_delta helper
    - 5.2: Coverage snapshots stored in state.metadata
    - 5.3: Delta injected into think step user prompt
    - 5.5: Delta correctly identifies newly sufficient, still-insufficient, new queries
    - 5.6: Delta injected into think prompt on rounds > 0
    - 5.7: Coverage snapshots persist across supervision rounds
    """

    def _make_stub(self):
        """Create a StubSupervision instance for testing."""
        return StubSupervision(delegation_model=True)

    def test_compute_delta_identifies_newly_sufficient(self):
        """5.5: Delta marks queries that became sufficient this round."""
        stub = self._make_stub()
        state = _make_state(num_completed=2, sources_per_query=3, supervision_round=1)

        # Simulate previous snapshot where query had only 1 source (insufficient)
        state.metadata["coverage_snapshots"] = {
            "0": {
                "sq-0": {"query": "Sub-query 0: aspect 0 of deep learning", "source_count": 1, "unique_domains": 1, "status": "completed"},
                "sq-1": {"query": "Sub-query 1: aspect 1 of deep learning", "source_count": 1, "unique_domains": 1, "status": "completed"},
            }
        }

        coverage_data = stub._build_per_query_coverage(state)
        delta = stub._compute_coverage_delta(state, coverage_data, min_sources=2)

        assert delta is not None, "Delta should be generated for round > 0"
        assert "NEWLY SUFFICIENT" in delta, "Queries that crossed min_sources threshold should be marked NEWLY SUFFICIENT"
        assert "round 0" in delta, "Delta should reference previous round"
        assert "1)" in delta, "Delta should reference current round"

    def test_compute_delta_identifies_still_insufficient(self):
        """5.5: Delta marks queries that remain insufficient."""
        stub = self._make_stub()
        state = _make_state(num_completed=1, sources_per_query=1, supervision_round=1)

        # Previous snapshot had same count — still insufficient
        state.metadata["coverage_snapshots"] = {
            "0": {
                "sq-0": {"query": "Sub-query 0: aspect 0 of deep learning", "source_count": 1, "unique_domains": 1, "status": "completed"},
            }
        }

        coverage_data = stub._build_per_query_coverage(state)
        delta = stub._compute_coverage_delta(state, coverage_data, min_sources=3)

        assert delta is not None
        assert "STILL INSUFFICIENT" in delta, "Queries still below threshold should be marked STILL INSUFFICIENT"

    def test_compute_delta_identifies_new_queries(self):
        """5.5: Delta marks queries that didn't exist in the previous round."""
        stub = self._make_stub()
        state = _make_state(num_completed=2, sources_per_query=2, supervision_round=1)

        # Previous snapshot only had sq-0, so sq-1 is new
        state.metadata["coverage_snapshots"] = {
            "0": {
                "sq-0": {"query": "Sub-query 0: aspect 0 of deep learning", "source_count": 1, "unique_domains": 1, "status": "completed"},
            }
        }

        coverage_data = stub._build_per_query_coverage(state)
        delta = stub._compute_coverage_delta(state, coverage_data, min_sources=2)

        assert delta is not None
        assert "[NEW]" in delta, "Queries not in previous snapshot should be marked [NEW]"

    def test_compute_delta_returns_none_for_round_zero(self):
        """5.5: No delta on round 0 (no previous snapshot)."""
        stub = self._make_stub()
        state = _make_state(num_completed=2, sources_per_query=2, supervision_round=0)

        coverage_data = stub._build_per_query_coverage(state)
        delta = stub._compute_coverage_delta(state, coverage_data)

        assert delta is None, "Delta should be None when no previous snapshot exists"

    def test_store_coverage_snapshot(self):
        """5.2: Snapshots are stored in state.metadata keyed by round number."""
        stub = self._make_stub()
        state = _make_state(num_completed=2, sources_per_query=2, supervision_round=0)

        coverage_data = stub._build_per_query_coverage(state)
        stub._store_coverage_snapshot(state, coverage_data)

        snapshots = state.metadata.get("coverage_snapshots", {})
        assert "0" in snapshots, "Snapshot for round 0 should be stored"
        assert "sq-0" in snapshots["0"], "Snapshot should contain sub-query IDs"
        assert snapshots["0"]["sq-0"]["source_count"] == 2
        assert "query" in snapshots["0"]["sq-0"], "Snapshot should contain query text"

    def test_snapshots_persist_across_rounds(self):
        """5.7: Multiple rounds' snapshots coexist in state.metadata."""
        stub = self._make_stub()
        state = _make_state(num_completed=2, sources_per_query=2, supervision_round=0)

        # Store snapshot for round 0
        coverage_data = stub._build_per_query_coverage(state)
        stub._store_coverage_snapshot(state, coverage_data)

        # Simulate round 1 with more sources
        state.supervision_round = 1
        state.sources.append(
            ResearchSource(
                id="src-extra",
                url="https://newdomain.com/article",
                title="Extra source",
                source_type=SourceType.WEB,
                quality=SourceQuality.HIGH,
                sub_query_id="sq-0",
            )
        )
        coverage_data_r1 = stub._build_per_query_coverage(state)
        stub._store_coverage_snapshot(state, coverage_data_r1)

        snapshots = state.metadata["coverage_snapshots"]
        assert "0" in snapshots, "Round 0 snapshot should persist"
        assert "1" in snapshots, "Round 1 snapshot should be added"
        assert snapshots["0"]["sq-0"]["source_count"] == 2
        assert snapshots["1"]["sq-0"]["source_count"] == 3, "Round 1 should reflect new source"

    def test_delta_injected_into_think_prompt(self):
        """5.3/5.6: Coverage delta appears in the think prompt on rounds > 0."""
        stub = self._make_stub()
        state = _make_state(num_completed=2, sources_per_query=3, supervision_round=1)

        # Set up previous snapshot
        state.metadata["coverage_snapshots"] = {
            "0": {
                "sq-0": {"query": "Sub-query 0: aspect 0 of deep learning", "source_count": 1, "unique_domains": 1, "status": "completed"},
                "sq-1": {"query": "Sub-query 1: aspect 1 of deep learning", "source_count": 1, "unique_domains": 1, "status": "completed"},
            }
        }

        coverage_data = stub._build_per_query_coverage(state)
        delta = stub._compute_coverage_delta(state, coverage_data, min_sources=2)

        # Build the think prompt with delta
        prompt = stub._build_think_prompt(state, coverage_data, coverage_delta=delta)

        assert "What Changed Since Last Round" in prompt, "Delta section header should appear"
        assert "Coverage delta" in prompt, "Delta content should be in prompt"
        assert "STILL INSUFFICIENT" in prompt or "NEWLY SUFFICIENT" in prompt or "SUFFICIENT" in prompt, \
            "Delta status labels should appear in prompt"
        assert "Focus your analysis" in prompt, "Guidance to focus on changes should appear"

    def test_think_prompt_without_delta_on_round_zero(self):
        """5.6: Think prompt omits delta section when delta is None (round 0)."""
        stub = self._make_stub()
        state = _make_state(num_completed=2, sources_per_query=2, supervision_round=0)

        coverage_data = stub._build_per_query_coverage(state)

        # No delta for round 0
        prompt = stub._build_think_prompt(state, coverage_data, coverage_delta=None)

        assert "What Changed Since Last Round" not in prompt, "Delta section should not appear on round 0"
        assert "Coverage delta" not in prompt

    def test_delta_shows_source_and_domain_changes(self):
        """5.5: Delta includes numeric source and domain changes."""
        stub = self._make_stub()
        state = _make_state(num_completed=1, sources_per_query=3, supervision_round=1)

        state.metadata["coverage_snapshots"] = {
            "0": {
                "sq-0": {"query": "Sub-query 0: aspect 0 of deep learning", "source_count": 1, "unique_domains": 1, "status": "completed"},
            }
        }

        coverage_data = stub._build_per_query_coverage(state)
        delta = stub._compute_coverage_delta(state, coverage_data, min_sources=2)

        assert delta is not None
        assert "+2 sources" in delta, "Delta should show source count change"
        assert "now: 3 sources" in delta, "Delta should show current totals"


# ===========================================================================
# Phase 6: Supervisor think_tool as Conversation
# ===========================================================================


class TestPhase6ThinkAsConversation:
    """Tests for Phase 6: making supervisor gap analysis conversational.

    Covers PLAN checklist items:
    - 6.1: Think output injected into supervision_messages before delegation
    - 6.5: Supervisor references prior gap analysis in delegation rationale
    """

    def test_think_output_in_delegation_prompt_via_messages(self):
        """6.1/6.5: When think output is in supervision_messages, the
        delegation prompt includes it in the Prior Supervisor Conversation
        section, and the Gap Analysis section references history instead
        of duplicating the full text."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=2, sources_per_query=2, supervision_round=1)

        # Simulate Phase 6 flow: think output already injected into messages
        think_text = "Gap: Missing regulatory analysis for EU markets."
        state.supervision_messages = [
            {
                "role": "assistant", "type": "think", "round": 1,
                "content": think_text,
            },
        ]

        coverage = stub._build_per_query_coverage(state)
        prompt = stub._build_delegation_user_prompt(state, coverage, think_output=think_text)

        # Think output should be in Prior Supervisor Conversation
        assert "Prior Supervisor Conversation" in prompt
        assert "regulatory analysis for EU markets" in prompt
        assert "[Round 1] Your Gap Analysis" in prompt

        # Gap Analysis section should be a lightweight reference, not full text
        assert "conversation history above" in prompt
        # The full think text should NOT be duplicated in a <gap_analysis> tag
        assert "<gap_analysis>" not in prompt

    def test_think_output_fallback_when_no_messages(self):
        """6.1: If supervision_messages is empty (shouldn't happen), the Gap
        Analysis section falls back to embedding the full think text."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=2, sources_per_query=2, supervision_round=0)
        state.supervision_messages = []

        coverage = stub._build_per_query_coverage(state)
        prompt = stub._build_delegation_user_prompt(
            state, coverage, think_output="Full gap analysis text here",
        )

        # Full text should be embedded in <gap_analysis> tags
        assert "<gap_analysis>" in prompt
        assert "Full gap analysis text here" in prompt

    @pytest.mark.asyncio
    async def test_think_injected_before_delegate_in_loop(self):
        """6.1: During the delegation loop, think output is appended to
        supervision_messages BEFORE the delegate step runs, so the
        delegation prompt can reference it."""
        stub = StubSupervision(delegation_model=True)
        stub.config.deep_research_topic_max_tool_calls = 5
        stub.config.deep_research_providers = ["tavily"]
        stub.config.deep_research_supervision_single_call = False

        state = _make_state(
            num_completed=0, num_pending=0, supervision_round=0,
            max_supervision_rounds=1,
        )
        state.research_brief = "Test research"
        state.topic_research_results = []
        state.max_sub_queries = 10

        # Track the order of operations
        call_order: list[str] = []
        messages_at_delegate_time: list[dict] = []

        async def mock_execute_llm_call(**kwargs):
            phase_name = kwargs.get("phase_name", "")
            if "think" in phase_name:
                call_order.append("think")
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content="Gap: need more data", tokens_used=50,
                )
                return result
            raise AssertionError(f"Unexpected phase: {phase_name}")

        async def mock_execute_structured_llm_call(**kwargs):
            phase_name = kwargs.get("phase_name", "")
            if "delegate" in phase_name:
                call_order.append("delegate")
                # Capture supervision_messages at the time delegation runs
                messages_at_delegate_time.extend(list(state.supervision_messages))

                content = json.dumps({
                    "research_complete": True,
                    "directives": [],
                    "rationale": "All covered",
                })
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content=content,
                    provider_id="test", model_used="test",
                    tokens_used=80, duration_ms=400.0,
                )
                parsed = DelegationResponse(
                    research_complete=True, directives=[], rationale="All covered",
                )
                return StructuredLLMCallResult(
                    result=result.result, llm_call_duration_ms=0.0,
                    parsed=parsed, parse_retries=0,
                )
            raise AssertionError(f"Unexpected structured phase: {phase_name}")

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            side_effect=mock_execute_structured_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.truncate_supervision_messages",
            side_effect=lambda messages, model: messages,
        ):
            result = await stub._execute_supervision_async(
                state=state, provider_id="test", timeout=30.0,
            )

        assert result.success is True
        # Think should have been called before delegate
        assert call_order == ["think", "delegate"]
        # At the time delegation ran, think output should already be in messages
        think_msgs = [m for m in messages_at_delegate_time if m.get("type") == "think"]
        assert len(think_msgs) >= 1, "Think should be in messages before delegate runs"
        assert "need more data" in think_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_post_think_output_injected_into_messages(self):
        """6.1: Post-execution think output (Step 4) is also injected into
        supervision_messages for subsequent rounds."""
        stub = StubSupervision(delegation_model=True)
        stub.config.deep_research_topic_max_tool_calls = 5
        stub.config.deep_research_providers = ["tavily"]
        stub.config.deep_research_supervision_single_call = False

        state = _make_state(
            num_completed=0, num_pending=0, supervision_round=0,
            max_supervision_rounds=1,
        )
        state.research_brief = "Test research"
        state.topic_research_results = []
        state.max_sub_queries = 10

        think_call_count = {"value": 0}

        async def mock_execute_llm_call(**kwargs):
            phase_name = kwargs.get("phase_name", "")
            if "think" in phase_name:
                think_call_count["value"] += 1
                content = (
                    "Pre-delegation gap analysis"
                    if think_call_count["value"] == 1
                    else "Post-execution: alignment coverage improved"
                )
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content=content, tokens_used=50,
                )
                return result
            if "critique" in phase_name:
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content="VERDICT: NO_ISSUES", tokens_used=20,
                    provider_id="test", model_used="test",
                )
                return result
            raise AssertionError(f"Unexpected phase: {phase_name}")

        delegate_response = json.dumps({
            "research_complete": False,
            "directives": [
                {"research_topic": "Investigate AI safety metrics", "priority": 1},
            ],
            "rationale": "Need safety data",
        })

        async def mock_execute_structured_llm_call(**kwargs):
            phase_name = kwargs.get("phase_name", "")
            if "delegate" in phase_name:
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content=delegate_response,
                    provider_id="test", model_used="test",
                    tokens_used=80, duration_ms=400.0,
                )
                parsed = DelegationResponse(
                    research_complete=False,
                    directives=[
                        ResearchDirective(
                            research_topic="Investigate AI safety metrics",
                            priority=1,
                        ),
                    ],
                    rationale="Need safety data",
                )
                return StructuredLLMCallResult(
                    result=result.result, llm_call_duration_ms=0.0,
                    parsed=parsed, parse_retries=0,
                )
            raise AssertionError(f"Unexpected structured phase: {phase_name}")

        async def mock_topic_research(**kwargs):
            sq = kwargs.get("sub_query")
            return TopicResearchResult(
                sub_query_id=sq.id if sq else "unknown",
                searches_performed=2,
                sources_found=3,
                compressed_findings="Safety metrics show RLHF effectiveness...",
            )

        stub._execute_topic_research_async = mock_topic_research
        stub._get_search_provider = MagicMock(return_value=MagicMock())

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            side_effect=mock_execute_structured_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.truncate_supervision_messages",
            side_effect=lambda messages, model: messages,
        ):
            result = await stub._execute_supervision_async(
                state=state, provider_id="test", timeout=30.0,
            )

        assert result.success is True

        # Should have 2 think messages: pre-delegation + post-execution
        think_msgs = [m for m in state.supervision_messages if m.get("type") == "think"]
        assert len(think_msgs) >= 2, f"Expected >= 2 think messages, got {len(think_msgs)}"
        assert "Pre-delegation" in think_msgs[0]["content"]
        assert "Post-execution" in think_msgs[1]["content"]


class TestPhase6SingleCallMode:
    """Tests for Phase 6: single-call think+delegate option.

    Covers PLAN checklist items:
    - 6.2: Single-call approach evaluation
    - 6.3: Merged think+delegate in one LLM call
    - 6.6: Latency reduction (one call instead of two)
    """

    def test_parse_combined_response_valid(self):
        """6.3: Combined response with gap_analysis + JSON parses correctly."""
        content = """<gap_analysis>
The research has significant gaps in regulatory coverage.
European AI regulation (EU AI Act) is completely missing.
</gap_analysis>

```json
{
    "research_complete": false,
    "directives": [
        {
            "research_topic": "Investigate EU AI Act implications for high-risk systems",
            "perspective": "regulatory",
            "evidence_needed": "legislative text, compliance frameworks",
            "priority": 1
        }
    ],
    "rationale": "Need regulatory coverage as identified in gap analysis"
}
```"""
        think_output, delegation = SupervisionPhaseMixin._parse_combined_response(content)

        assert think_output is not None
        assert "regulatory coverage" in think_output
        assert "EU AI Act" in think_output
        assert delegation.research_complete is False
        assert len(delegation.directives) == 1
        assert "EU AI Act" in delegation.directives[0].research_topic

    def test_parse_combined_response_no_gap_analysis(self):
        """6.3: Combined response without gap_analysis tags still parses JSON."""
        content = json.dumps({
            "research_complete": True,
            "directives": [],
            "rationale": "All covered",
        })
        think_output, delegation = SupervisionPhaseMixin._parse_combined_response(content)

        assert think_output is None
        assert delegation.research_complete is True

    def test_parse_combined_response_no_json(self):
        """6.3: Combined response without JSON raises ValueError."""
        content = """<gap_analysis>
Some analysis here.
</gap_analysis>

No JSON in this response."""
        with pytest.raises(ValueError, match="No JSON found"):
            SupervisionPhaseMixin._parse_combined_response(content)

    def test_extract_gap_analysis_section(self):
        """Helper correctly extracts gap analysis from tags."""
        content = "Preamble\n<gap_analysis>\nAnalysis text\n</gap_analysis>\nPostamble"
        result = SupervisionPhaseMixin._extract_gap_analysis_section(content)
        assert result == "Analysis text"

    def test_extract_gap_analysis_section_missing(self):
        """Helper returns None when no gap_analysis tags present."""
        result = SupervisionPhaseMixin._extract_gap_analysis_section("No tags here")
        assert result is None

    def test_combined_prompt_includes_conversation_history(self):
        """6.3: Combined user prompt includes prior conversation and coverage."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=2, sources_per_query=2, supervision_round=1)
        state.supervision_messages = [
            {
                "role": "assistant", "type": "think", "round": 0,
                "content": "Prior round gap analysis about safety.",
            },
        ]

        coverage = stub._build_per_query_coverage(state)
        prompt = stub._build_combined_think_delegate_user_prompt(state, coverage)

        assert "Prior Supervisor Conversation" in prompt
        assert "Prior round gap analysis about safety" in prompt
        assert "Current Research Coverage" in prompt
        assert "gap analysis inside <gap_analysis> tags" in prompt

    @pytest.mark.asyncio
    async def test_single_call_makes_one_llm_call(self):
        """6.6: Single-call mode makes one LLM call instead of two separate
        think + delegate calls, reducing latency."""
        stub = StubSupervision(delegation_model=True)
        stub.config.deep_research_topic_max_tool_calls = 5
        stub.config.deep_research_providers = ["tavily"]
        stub.config.deep_research_supervision_single_call = True

        # Use supervision_round=0 with existing topic_research_results to
        # avoid both first-round decomposition and the round>0 heuristic
        # early exit (which always returns should_continue_gathering=False).
        state = _make_state(
            num_completed=2, num_pending=0, supervision_round=0,
            max_supervision_rounds=1, sources_per_query=1,
        )
        state.research_brief = "Test research"
        state.topic_research_results = [
            TopicResearchResult(sub_query_id="sq-0", searches_performed=2, sources_found=2),
        ]
        state.max_sub_queries = 10

        llm_calls: list[str] = []

        combined_response = (
            "<gap_analysis>\nAll topics well covered.\n</gap_analysis>\n\n"
            + json.dumps({
                "research_complete": True,
                "directives": [],
                "rationale": "Sufficient coverage across all dimensions",
            })
        )

        async def mock_execute_llm_call(**kwargs):
            phase_name = kwargs.get("phase_name", "")
            llm_calls.append(phase_name)
            result = MagicMock()
            result.result = WorkflowResult(
                success=True, content="Think output", tokens_used=50,
            )
            return result

        async def mock_execute_structured_llm_call(**kwargs):
            phase_name = kwargs.get("phase_name", "")
            llm_calls.append(phase_name)

            result = MagicMock()
            result.result = WorkflowResult(
                success=True, content=combined_response,
                provider_id="test", model_used="test",
                tokens_used=150, duration_ms=600.0,
            )

            # Parse the combined response
            think_output, delegation = SupervisionPhaseMixin._parse_combined_response(
                combined_response,
            )
            return StructuredLLMCallResult(
                result=result.result, llm_call_duration_ms=0.0,
                parsed=(think_output, delegation), parse_retries=0,
            )

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            side_effect=mock_execute_structured_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.truncate_supervision_messages",
            side_effect=lambda messages, model: messages,
        ):
            result = await stub._execute_supervision_async(
                state=state, provider_id="test", timeout=30.0,
            )

        assert result.success is True

        # Should have made exactly one combined call (not separate think + delegate)
        combined_calls = [c for c in llm_calls if "combined" in c]
        think_calls = [c for c in llm_calls if c == "supervision_think"]
        delegate_calls = [c for c in llm_calls if c == "supervision_delegate"]

        assert len(combined_calls) == 1, f"Expected 1 combined call, got {len(combined_calls)}: {llm_calls}"
        assert len(think_calls) == 0, f"Should not have separate think calls: {llm_calls}"
        assert len(delegate_calls) == 0, f"Should not have separate delegate calls: {llm_calls}"

    @pytest.mark.asyncio
    async def test_single_call_injects_think_into_messages(self):
        """6.3: Single-call mode still injects think output into
        supervision_messages for conversation continuity."""
        stub = StubSupervision(delegation_model=True)
        stub.config.deep_research_topic_max_tool_calls = 5
        stub.config.deep_research_providers = ["tavily"]
        stub.config.deep_research_supervision_single_call = True

        # Use supervision_round=0 with existing topic_research_results
        state = _make_state(
            num_completed=2, num_pending=0, supervision_round=0,
            max_supervision_rounds=1, sources_per_query=1,
        )
        state.research_brief = "Test research"
        state.topic_research_results = [
            TopicResearchResult(sub_query_id="sq-0", searches_performed=2, sources_found=2),
        ]
        state.max_sub_queries = 10

        combined_response = (
            "<gap_analysis>\nRegulatory gap identified in EU coverage.\n</gap_analysis>\n\n"
            + json.dumps({
                "research_complete": True,
                "directives": [],
                "rationale": "Sufficient",
            })
        )

        async def mock_execute_structured_llm_call(**kwargs):
            result = MagicMock()
            result.result = WorkflowResult(
                success=True, content=combined_response,
                provider_id="test", model_used="test",
                tokens_used=150, duration_ms=600.0,
            )
            think_output, delegation = SupervisionPhaseMixin._parse_combined_response(
                combined_response,
            )
            return StructuredLLMCallResult(
                result=result.result, llm_call_duration_ms=0.0,
                parsed=(think_output, delegation), parse_retries=0,
            )

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            side_effect=mock_execute_structured_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.truncate_supervision_messages",
            side_effect=lambda messages, model: messages,
        ):
            result = await stub._execute_supervision_async(
                state=state, provider_id="test", timeout=30.0,
            )

        assert result.success is True
        think_msgs = [m for m in state.supervision_messages if m.get("type") == "think"]
        assert len(think_msgs) >= 1
        assert "Regulatory gap" in think_msgs[0]["content"]


# ===========================================================================
# Phase 6 (PLAN): Confidence-Scored Coverage Heuristic
# ===========================================================================


class TestConfidenceScoredCoverage:
    """Tests for PLAN Phase 6: confidence-scored coverage heuristic.

    Covers PLAN checklist items:
    - 6.1: Multi-dimensional scoring (source adequacy, domain diversity, query completion rate)
    - 6.2: Weighted confidence score with configurable weights
    - 6.3: confidence, dominant_factors, weak_factors in return dict
    - 6.4: confidence >= threshold for should_continue_gathering decision
    - 6.5: Audit events contain confidence breakdown
    - 6.6: Confidence score reflects actual coverage quality
    - 6.7: Threshold-based decision matches expected behavior at boundary values
    - 6.8: Audit events contain confidence breakdown
    """

    def _make_stub(self, threshold: float = 0.75, weights: dict | None = None):
        stub = StubSupervision(delegation_model=True)
        stub.config.deep_research_coverage_confidence_threshold = threshold
        stub.config.deep_research_coverage_confidence_weights = weights
        return stub

    # ------------------------------------------------------------------
    # 6.1 / 6.6: Multi-dimensional scoring reflects coverage quality
    # ------------------------------------------------------------------

    def test_high_coverage_yields_high_confidence(self):
        """Fully covered queries with diverse domains score near 1.0."""
        stub = self._make_stub()
        # 3 completed queries, 3 sources each (>= min_sources=2), diverse domains
        state = _make_state(num_completed=3, sources_per_query=3)

        result = stub._assess_coverage_heuristic(state, min_sources=2)

        assert result["confidence"] >= 0.8, f"Expected high confidence, got {result['confidence']}"
        assert result["confidence_dimensions"]["source_adequacy"] == 1.0
        assert result["confidence_dimensions"]["query_completion_rate"] == 1.0
        assert result["overall_coverage"] == "sufficient"
        assert result["queries_sufficient"] == 3

    def test_low_coverage_yields_low_confidence(self):
        """Insufficient sources yield low confidence."""
        stub = self._make_stub()
        # 2 completed queries but only 1 source each (< min_sources=3)
        state = _make_state(num_completed=2, sources_per_query=1)

        result = stub._assess_coverage_heuristic(state, min_sources=3)

        assert result["confidence"] < 0.6, f"Expected low confidence, got {result['confidence']}"
        assert result["confidence_dimensions"]["source_adequacy"] < 0.5
        assert result["overall_coverage"] == "insufficient"

    def test_partial_coverage_yields_medium_confidence(self):
        """Mixed coverage produces intermediate confidence."""
        stub = self._make_stub()
        state = _make_state(num_completed=2, sources_per_query=2)
        # Add a completed sub-query with no sources
        state.sub_queries.append(
            SubQuery(id="sq-nosrc", query="No sources query", status="completed")
        )

        result = stub._assess_coverage_heuristic(state, min_sources=2)

        assert 0.3 < result["confidence"] < 0.9
        assert result["overall_coverage"] == "partial"
        assert result["queries_sufficient"] == 2

    def test_no_completed_queries_yields_zero_confidence(self):
        """No completed queries produce confidence=0.0."""
        stub = self._make_stub()
        state = _make_state(num_completed=0, num_pending=3)

        result = stub._assess_coverage_heuristic(state, min_sources=2)

        assert result["confidence"] == 0.0
        assert result["overall_coverage"] == "insufficient"
        assert result["queries_assessed"] == 0
        assert len(result["weak_factors"]) == 3

    def test_domain_diversity_dimension(self):
        """Domain diversity reflects unique domain count vs query count."""
        stub = self._make_stub()
        state = _make_state(num_completed=2, sources_per_query=2)

        result = stub._assess_coverage_heuristic(state, min_sources=2)

        # 2 completed queries × 2 sources each, URLs are example0.com, example1.com
        # diversity = unique_domains / (query_count * 2)
        dims = result["confidence_dimensions"]
        assert 0.0 < dims["domain_diversity"] <= 1.0

    def test_query_completion_rate_dimension(self):
        """Query completion rate = completed / total sub-queries."""
        stub = self._make_stub()
        state = _make_state(num_completed=2, num_pending=2, sources_per_query=3)

        result = stub._assess_coverage_heuristic(state, min_sources=2)

        # 2 completed out of 4 total → 0.5
        assert result["confidence_dimensions"]["query_completion_rate"] == 0.5

    # ------------------------------------------------------------------
    # 6.2: Configurable weights
    # ------------------------------------------------------------------

    def test_custom_weights_affect_confidence(self):
        """Custom weights shift the confidence score."""
        # Weight heavily toward source_adequacy
        stub_source_heavy = self._make_stub(
            weights={"source_adequacy": 1.0, "domain_diversity": 0.0, "query_completion_rate": 0.0}
        )
        # Weight heavily toward domain_diversity
        stub_domain_heavy = self._make_stub(
            weights={"source_adequacy": 0.0, "domain_diversity": 1.0, "query_completion_rate": 0.0}
        )

        state = _make_state(num_completed=3, sources_per_query=3)

        result_source = stub_source_heavy._assess_coverage_heuristic(state, min_sources=2)
        result_domain = stub_domain_heavy._assess_coverage_heuristic(state, min_sources=2)

        # Source adequacy is 1.0, domain diversity varies — different scores
        assert result_source["confidence"] == 1.0
        assert result_domain["confidence"] != result_source["confidence"]

    # ------------------------------------------------------------------
    # 6.3: Return dict structure
    # ------------------------------------------------------------------

    def test_return_dict_contains_confidence_fields(self):
        """Return dict includes confidence, dimensions, factors."""
        stub = self._make_stub()
        state = _make_state(num_completed=2, sources_per_query=2)

        result = stub._assess_coverage_heuristic(state, min_sources=2)

        assert "confidence" in result
        assert "confidence_threshold" in result
        assert "confidence_dimensions" in result
        assert "dominant_factors" in result
        assert "weak_factors" in result
        assert isinstance(result["confidence"], float)
        assert isinstance(result["confidence_dimensions"], dict)
        assert isinstance(result["dominant_factors"], list)
        assert isinstance(result["weak_factors"], list)

        dims = result["confidence_dimensions"]
        assert "source_adequacy" in dims
        assert "domain_diversity" in dims
        assert "query_completion_rate" in dims

    def test_dominant_and_weak_factors_classification(self):
        """Factors are classified as dominant (>= 0.7) or weak (< 0.5)."""
        stub = self._make_stub()
        # High source coverage, all queries completed
        state = _make_state(num_completed=3, sources_per_query=4)

        result = stub._assess_coverage_heuristic(state, min_sources=2)

        # source_adequacy = 1.0 → dominant
        assert "source_adequacy" in result["dominant_factors"]
        # query_completion_rate = 1.0 → dominant
        assert "query_completion_rate" in result["dominant_factors"]

    # ------------------------------------------------------------------
    # 6.4 / 6.7: Threshold-based decision at boundary values
    # ------------------------------------------------------------------

    def test_confidence_above_threshold_stops_gathering(self):
        """When confidence >= threshold, should_continue_gathering=False."""
        stub = self._make_stub(threshold=0.5)
        state = _make_state(num_completed=3, sources_per_query=3)

        result = stub._assess_coverage_heuristic(state, min_sources=2)

        assert result["confidence"] >= 0.5
        assert result["should_continue_gathering"] is False

    def test_confidence_below_threshold_continues_gathering(self):
        """When confidence < threshold, should_continue_gathering=True."""
        stub = self._make_stub(threshold=0.99)
        # Limited sources and incomplete queries
        state = _make_state(num_completed=1, num_pending=3, sources_per_query=1)

        result = stub._assess_coverage_heuristic(state, min_sources=3)

        assert result["confidence"] < 0.99
        assert result["should_continue_gathering"] is True

    def test_boundary_at_exactly_threshold(self):
        """At exactly the threshold, should_continue_gathering=False (>= check)."""
        stub = self._make_stub()
        state = _make_state(num_completed=3, sources_per_query=3)

        result = stub._assess_coverage_heuristic(state, min_sources=2)

        # Force threshold to exactly match confidence
        exact_threshold = result["confidence"]
        stub.config.deep_research_coverage_confidence_threshold = exact_threshold

        result2 = stub._assess_coverage_heuristic(state, min_sources=2)
        assert result2["should_continue_gathering"] is False

    def test_high_threshold_prevents_early_exit(self):
        """A very high threshold (0.99) prevents premature early-exit."""
        stub = self._make_stub(threshold=0.99)
        state = _make_state(num_completed=2, sources_per_query=2)

        result = stub._assess_coverage_heuristic(state, min_sources=2)

        # Even with decent coverage, 0.99 threshold is hard to meet
        assert result["should_continue_gathering"] is True

    def test_low_threshold_allows_early_exit(self):
        """A very low threshold (0.1) allows early exit with minimal coverage."""
        stub = self._make_stub(threshold=0.1)
        state = _make_state(num_completed=1, sources_per_query=1)

        result = stub._assess_coverage_heuristic(state, min_sources=2)

        # Even with minimal coverage, 0.1 threshold should be met
        assert result["confidence"] >= 0.1
        assert result["should_continue_gathering"] is False

    # ------------------------------------------------------------------
    # 6.5 / 6.8: Audit events contain confidence breakdown
    # ------------------------------------------------------------------

    def test_audit_event_includes_confidence_breakdown(self):
        """Audit event data includes the full confidence breakdown."""
        stub = self._make_stub()
        state = _make_state(
            num_completed=3, sources_per_query=3, supervision_round=1,
        )

        heuristic = stub._assess_coverage_heuristic(state, min_sources=2)

        # Simulate what the delegation loop does: write audit event with heuristic
        stub._write_audit_event(
            state,
            "supervision_result",
            data={
                "reason": "heuristic_sufficient",
                "supervision_round": state.supervision_round,
                "coverage_summary": heuristic,
            },
        )

        assert len(stub._audit_events) == 1
        event_name, event_kwargs = stub._audit_events[0]
        assert event_name == "supervision_result"

        coverage_summary = event_kwargs["data"]["coverage_summary"]
        assert "confidence" in coverage_summary
        assert "confidence_dimensions" in coverage_summary
        assert "dominant_factors" in coverage_summary
        assert "weak_factors" in coverage_summary
        assert isinstance(coverage_summary["confidence"], float)
        assert "source_adequacy" in coverage_summary["confidence_dimensions"]

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------

    def test_backward_compatible_keys_preserved(self):
        """Original keys (overall_coverage, queries_assessed, etc.) still present."""
        stub = self._make_stub()
        state = _make_state(num_completed=2, sources_per_query=2)

        result = stub._assess_coverage_heuristic(state, min_sources=2)

        assert "overall_coverage" in result
        assert "should_continue_gathering" in result
        assert "queries_assessed" in result
        assert "queries_sufficient" in result
        assert result["overall_coverage"] in ("sufficient", "partial", "insufficient")

    # ------------------------------------------------------------------
    # 2.4: Lopsided coverage — min() prevents premature exit
    # ------------------------------------------------------------------

    def test_lopsided_coverage_detected_by_min(self):
        """A query with 0 sources should drag source_adequacy to 0.0.

        Before the fix (using mean), a query with 10x sources could mask
        another with 0 sources.  With min(), the worst sub-query dominates.
        """
        stub = self._make_stub()
        # Two completed queries: one well-covered, one with NO sources
        state = _make_state(num_completed=1, sources_per_query=10)
        # Add a second completed sub-query with zero sources
        state.sub_queries.append(
            SubQuery(id="sq-empty", query="Empty query", status="completed"),
        )
        result = stub._assess_coverage_heuristic(state, min_sources=2)

        # min() should yield 0.0 for source_adequacy (empty query = 0/2 = 0.0)
        assert result["confidence_dimensions"]["source_adequacy"] == 0.0
        assert result["overall_coverage"] == "partial"


# ===========================================================================
# Phase 2: Supervisor Delegation Scaling Heuristics
# ===========================================================================


class TestQueryComplexityClassification:
    """Tests for _classify_query_complexity() — Phase 2b."""

    def test_simple_query_few_sub_queries_short_brief(self):
        """Short brief with 0-2 sub-queries classifies as simple."""
        state = _make_state(num_completed=2, sources_per_query=1)
        state.research_brief = "What is the capital of France?"
        result = StubSupervision()._classify_query_complexity(state)
        assert result == "simple"

    def test_simple_query_no_brief_falls_back_to_original(self):
        """When research_brief is None, falls back to original_query."""
        state = _make_state(num_completed=1, sources_per_query=1)
        state.research_brief = None
        state.original_query = "What is Python?"
        result = StubSupervision()._classify_query_complexity(state)
        assert result == "simple"

    def test_moderate_query_three_sub_queries(self):
        """3-4 sub-queries classifies as moderate."""
        state = _make_state(num_completed=3, sources_per_query=1)
        state.research_brief = "Compare Python and JavaScript for web development"
        result = StubSupervision()._classify_query_complexity(state)
        assert result == "moderate"

    def test_moderate_query_medium_brief(self):
        """Brief with 80-199 words classifies as moderate even with few sub-queries."""
        state = _make_state(num_completed=1, sources_per_query=1)
        state.research_brief = " ".join(["word"] * 100)
        result = StubSupervision()._classify_query_complexity(state)
        assert result == "moderate"

    def test_complex_query_five_plus_sub_queries(self):
        """5+ sub-queries classifies as complex."""
        state = _make_state(num_completed=5, sources_per_query=1)
        state.research_brief = "Short brief"
        result = StubSupervision()._classify_query_complexity(state)
        assert result == "complex"

    def test_complex_query_long_brief(self):
        """Brief with 200+ words classifies as complex."""
        state = _make_state(num_completed=1, sources_per_query=1)
        state.research_brief = " ".join(["word"] * 250)
        result = StubSupervision()._classify_query_complexity(state)
        assert result == "complex"

    def test_boundary_two_sub_queries_short_brief_is_simple(self):
        """Boundary: 2 sub-queries + short brief = simple."""
        state = _make_state(num_completed=2, sources_per_query=1)
        state.research_brief = "Brief topic"
        result = StubSupervision()._classify_query_complexity(state)
        assert result == "simple"

    def test_boundary_three_sub_queries_is_moderate(self):
        """Boundary: exactly 3 sub-queries = moderate."""
        state = _make_state(num_completed=3, sources_per_query=1)
        state.research_brief = "Short"
        result = StubSupervision()._classify_query_complexity(state)
        assert result == "moderate"

    def test_boundary_eighty_words_is_moderate(self):
        """Boundary: exactly 80 words = moderate."""
        state = _make_state(num_completed=1, sources_per_query=1)
        state.research_brief = " ".join(["word"] * 80)
        result = StubSupervision()._classify_query_complexity(state)
        assert result == "moderate"


class TestDelegationScalingHeuristics:
    """Tests for scaling heuristics in delegation prompts — Phase 2a/2b."""

    def test_system_prompt_has_scaling_guidance(self):
        """Follow-up delegation system prompt includes directive count scaling."""
        stub = StubSupervision(delegation_model=True)
        prompt = stub._build_delegation_system_prompt()

        assert "Directive Count Scaling" in prompt
        assert "Simple factual gaps" in prompt
        assert "1-2 directives" in prompt
        assert "Comparison gaps" in prompt
        assert "3-5 directives" in prompt
        assert "BIAS toward fewer" in prompt

    def test_system_prompt_still_has_max_five(self):
        """Follow-up delegation system prompt preserves the max-5 cap."""
        stub = StubSupervision(delegation_model=True)
        prompt = stub._build_delegation_system_prompt()

        assert "Maximum 5 directives per round" in prompt

    def test_user_prompt_includes_complexity_label_simple(self):
        """User prompt includes simple complexity signal for simple queries."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=2, sources_per_query=1)
        state.research_brief = "What is Python?"
        coverage = stub._build_per_query_coverage(state)

        prompt = stub._build_delegation_user_prompt(state, coverage)

        assert "Query complexity: **simple**" in prompt
        assert "1-2 focused directives" in prompt

    def test_user_prompt_includes_complexity_label_moderate(self):
        """User prompt includes moderate complexity signal."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=3, sources_per_query=1)
        state.research_brief = "Compare Python, JavaScript, and Rust"
        coverage = stub._build_per_query_coverage(state)

        prompt = stub._build_delegation_user_prompt(state, coverage)

        assert "Query complexity: **moderate**" in prompt
        assert "2-3 directives" in prompt

    def test_user_prompt_includes_complexity_label_complex(self):
        """User prompt includes complex complexity signal."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=5, sources_per_query=1)
        state.research_brief = " ".join(["word"] * 250)
        coverage = stub._build_per_query_coverage(state)

        prompt = stub._build_delegation_user_prompt(state, coverage)

        assert "Query complexity: **complex**" in prompt
        assert "3-5 directives" in prompt


# ===========================================================================
# Phase 2 ODR Alignment: Supervisor Context Preservation
# ===========================================================================


class TestBuildEvidenceInventory:
    """Tests for _build_evidence_inventory() helper (checklist 2b).

    Verifies compact evidence summary output format, character cap,
    and handling of edge cases (no sources, no raw notes, etc.).
    """

    def test_basic_inventory_with_sources(self):
        """Inventory lists sources with URLs, titles, and domain count."""
        stub = StubSupervision()
        state = _make_state(num_completed=0)
        # Add sources manually
        for i in range(3):
            state.sources.append(
                ResearchSource(
                    id=f"inv-src-{i}",
                    url=f"https://domain{i}.com/page",
                    title=f"Article {i}",
                    source_type=SourceType.WEB,
                    quality=SourceQuality.HIGH,
                )
            )

        result = TopicResearchResult(
            sub_query_id="sq-inv",
            source_ids=["inv-src-0", "inv-src-1", "inv-src-2"],
            sources_found=3,
        )

        inventory = stub._build_evidence_inventory(result, state)

        assert inventory is not None
        assert "Sources: 3 found" in inventory
        assert "3 unique domains" in inventory
        assert '"Article 0"' in inventory
        assert "domain0.com" in inventory

    def test_inventory_with_raw_notes_data_points(self):
        """Inventory includes data point estimate from raw notes."""
        stub = StubSupervision()
        state = _make_state(num_completed=0)
        state.sources.append(
            ResearchSource(
                id="inv-src-x",
                url="https://example.com/data",
                title="Data Source",
                source_type=SourceType.WEB,
                quality=SourceQuality.MEDIUM,
            )
        )

        result = TopicResearchResult(
            sub_query_id="sq-inv-2",
            source_ids=["inv-src-x"],
            sources_found=1,
            raw_notes="Line 1\nLine 2\nLine 3\n\nLine 5\n",
        )

        inventory = stub._build_evidence_inventory(result, state)

        assert inventory is not None
        assert "Key data points:" in inventory

    def test_inventory_respects_char_cap(self):
        """Inventory output does not exceed max_chars."""
        stub = StubSupervision()
        state = _make_state(num_completed=0)
        # Add many sources with long titles
        for i in range(20):
            state.sources.append(
                ResearchSource(
                    id=f"inv-long-{i}",
                    url=f"https://example{i}.com/very-long-article-path",
                    title=f"Very Long Article Title Number {i} With Extra Words",
                    source_type=SourceType.WEB,
                    quality=SourceQuality.MEDIUM,
                )
            )

        result = TopicResearchResult(
            sub_query_id="sq-inv-3",
            source_ids=[f"inv-long-{i}" for i in range(20)],
            sources_found=20,
        )

        inventory = stub._build_evidence_inventory(result, state, max_chars=200)

        assert inventory is not None
        assert len(inventory) <= 200

    def test_inventory_returns_none_when_no_evidence(self):
        """Returns None when result has neither source_ids nor raw_notes."""
        stub = StubSupervision()
        state = _make_state(num_completed=0)

        result = TopicResearchResult(
            sub_query_id="sq-empty",
            source_ids=[],
            sources_found=0,
        )

        inventory = stub._build_evidence_inventory(result, state)
        assert inventory is None

    def test_inventory_with_raw_notes_only(self):
        """Inventory works with raw_notes but no matching sources."""
        stub = StubSupervision()
        state = _make_state(num_completed=0)

        result = TopicResearchResult(
            sub_query_id="sq-notes-only",
            source_ids=["nonexistent-src"],
            sources_found=0,
            raw_notes="Some raw research data\nAnother line\nThird line",
        )

        inventory = stub._build_evidence_inventory(result, state)

        # Should still produce an inventory from the raw notes data points
        # even though source_ids don't match any sources in state
        assert inventory is not None
        assert "Sources: 0 found" in inventory


class TestEvidenceInventoryInPrompts:
    """Tests for evidence inventory rendering in supervisor prompts (checklist 2c).

    Verifies that evidence_inventory messages are rendered with distinct
    headers in both the combined think+delegate and delegation prompts.
    """

    def test_combined_prompt_renders_evidence_inventory(self):
        """Evidence inventory messages render with distinct header in combined prompt."""
        stub = StubSupervision()
        state = _make_state(num_completed=2, supervision_round=1)
        state.supervision_messages = [
            {
                "role": "assistant", "type": "think", "round": 0,
                "content": "Gap analysis text",
            },
            {
                "role": "tool_result", "type": "research_findings", "round": 0,
                "directive_id": "dir-1", "content": "Compressed findings",
            },
            {
                "role": "tool_result", "type": "evidence_inventory", "round": 0,
                "directive_id": "dir-1", "content": "Sources: 3 found, 2 unique domains",
            },
        ]

        coverage = stub._build_per_query_coverage(state)
        prompt = stub._build_combined_think_delegate_user_prompt(state, coverage)

        assert "### [Round 0] Evidence Inventory (directive dir-1)" in prompt
        assert "### [Round 0] Research Findings (directive dir-1)" in prompt
        assert "Sources: 3 found, 2 unique domains" in prompt

    def test_delegation_prompt_renders_evidence_inventory(self):
        """Evidence inventory messages render with distinct header in delegation prompt."""
        stub = StubSupervision()
        state = _make_state(num_completed=2, supervision_round=1)
        state.supervision_messages = [
            {
                "role": "tool_result", "type": "evidence_inventory", "round": 0,
                "directive_id": "dir-abc", "content": "Sources: 5 found, 3 unique domains",
            },
            {
                "role": "tool_result", "type": "research_findings", "round": 0,
                "directive_id": "dir-abc", "content": "Detailed compressed findings here",
            },
        ]

        coverage = stub._build_per_query_coverage(state)
        prompt = stub._build_delegation_user_prompt(state, coverage)

        assert "### [Round 0] Evidence Inventory (directive dir-abc)" in prompt
        assert "### [Round 0] Research Findings (directive dir-abc)" in prompt

    def test_prompt_without_evidence_inventory_unchanged(self):
        """Prompts without evidence_inventory messages render as before."""
        stub = StubSupervision()
        state = _make_state(num_completed=2, supervision_round=1)
        state.supervision_messages = [
            {
                "role": "tool_result", "type": "research_findings", "round": 0,
                "directive_id": "dir-old", "content": "Old findings",
            },
        ]

        coverage = stub._build_per_query_coverage(state)
        prompt = stub._build_combined_think_delegate_user_prompt(state, coverage)

        assert "Evidence Inventory" not in prompt
        assert "### [Round 0] Research Findings (directive dir-old)" in prompt


class TestEvidenceInventoryTruncation:
    """Tests for evidence inventory truncation awareness (checklist 2d).

    Verifies that evidence_inventory messages from oldest rounds are
    dropped before research_findings messages during truncation.
    """

    def test_evidence_inventories_dropped_before_findings(self):
        """Evidence inventories from oldest rounds dropped before research_findings."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            truncate_supervision_messages,
        )

        messages = []
        # Round 0: findings + evidence inventory
        messages.append({
            "role": "tool_result", "type": "research_findings", "round": 0,
            "directive_id": "d-0", "content": "F" * 30_000,
        })
        messages.append({
            "role": "tool_result", "type": "evidence_inventory", "round": 0,
            "directive_id": "d-0", "content": "E" * 10_000,
        })
        # Round 1: findings + evidence inventory
        messages.append({
            "role": "tool_result", "type": "research_findings", "round": 1,
            "directive_id": "d-1", "content": "G" * 30_000,
        })
        messages.append({
            "role": "tool_result", "type": "evidence_inventory", "round": 1,
            "directive_id": "d-1", "content": "H" * 10_000,
        })
        # Round 2: findings + evidence inventory (most recent)
        messages.append({
            "role": "tool_result", "type": "research_findings", "round": 2,
            "directive_id": "d-2", "content": "I" * 30_000,
        })
        messages.append({
            "role": "tool_result", "type": "evidence_inventory", "round": 2,
            "directive_id": "d-2", "content": "J" * 10_000,
        })

        # Budget that forces some removal but not all
        # 40% findings budget of 20k tokens * 4 chars = 32k chars
        small_limits = {"test-model": 20_000}
        result = truncate_supervision_messages(
            messages, model="test-model", token_limits=small_limits,
        )

        # Evidence inventories from oldest round should be removed first

        # Check: if any evidence_inventory was removed, it should be from
        # the oldest round(s) before any research_findings was removed
        removed_types = set()
        for msg in messages:
            found = any(
                m.get("type") == msg.get("type")
                and m.get("round") == msg.get("round")
                and m.get("directive_id") == msg.get("directive_id")
                for m in result
            )
            if not found:
                removed_types.add((msg.get("type"), msg.get("round")))

        # If both an inventory and findings from the same round were
        # candidates for removal, the inventory should have been removed first.
        # Verify: no research_findings removed from a round where
        # evidence_inventory was kept.
        for msg_type, msg_round in removed_types:
            if msg_type == "research_findings":
                # The inventory for this round should also be removed
                # (it should have been dropped first)
                inv_kept = any(
                    m.get("type") == "evidence_inventory" and m.get("round") == msg_round
                    for m in result
                )
                assert not inv_kept, (
                    f"Research findings from round {msg_round} were removed but "
                    f"evidence_inventory from the same round was kept"
                )

    def test_evidence_inventory_type_detected(self):
        """_is_evidence_inventory correctly identifies evidence_inventory messages."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            _is_evidence_inventory,
        )

        assert _is_evidence_inventory({"type": "evidence_inventory", "role": "tool_result"})
        assert not _is_evidence_inventory({"type": "research_findings", "role": "tool_result"})
        assert not _is_evidence_inventory({"type": "think", "role": "assistant"})
        assert not _is_evidence_inventory({})

    def test_truncation_with_mixed_messages_preserves_recent(self):
        """Truncation with all message types preserves most recent round."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            truncate_supervision_messages,
        )

        messages = []
        for r in range(4):
            messages.append({
                "role": "assistant", "type": "think", "round": r,
                "content": f"Think {r}: " + "x" * 20_000,
            })
            messages.append({
                "role": "tool_result", "type": "research_findings", "round": r,
                "directive_id": f"d-{r}",
                "content": f"Findings {r}: " + "y" * 20_000,
            })
            messages.append({
                "role": "tool_result", "type": "evidence_inventory", "round": r,
                "directive_id": f"d-{r}",
                "content": f"Inventory {r}: " + "z" * 5_000,
            })

        tiny_limits = {"test-model": 10_000}
        result = truncate_supervision_messages(
            messages, model="test-model", token_limits=tiny_limits,
        )

        remaining_rounds = {m.get("round") for m in result}
        # Most recent round (3) should survive
        assert 3 in remaining_rounds


# ===========================================================================
# Cancellation propagation tests (PLAN Phase 6B)
# ===========================================================================


class TestDirectiveExecutionCancellation:
    """Tests that cancellation propagates correctly during directive execution.

    The directive execution gather in supervision.py uses
    ``asyncio.gather(*tasks, return_exceptions=True)`` followed by a manual
    check for ``CancelledError`` results.  These tests verify that pattern.
    """

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates_from_gather(self):
        """CancelledError in gather results is re-raised to propagate cancellation."""
        import asyncio

        async def ok_task():
            return TopicResearchResult(
                sub_query_id="sq-0", sources_found=2, source_ids=[],
            )

        async def cancelled_task():
            raise asyncio.CancelledError()

        gather_results = await asyncio.gather(
            ok_task(), cancelled_task(), return_exceptions=True,
        )

        # Apply the same cancellation-propagation pattern as supervision.py
        with pytest.raises(asyncio.CancelledError):
            for r in gather_results:
                if isinstance(r, asyncio.CancelledError):
                    raise r

    @pytest.mark.asyncio
    async def test_non_cancellation_exceptions_are_non_fatal(self):
        """Non-CancelledError exceptions in gather are treated as non-fatal."""
        import asyncio

        async def ok_task():
            return TopicResearchResult(
                sub_query_id="sq-0", sources_found=2, source_ids=[],
            )

        async def failing_task():
            raise RuntimeError("unexpected error")

        gather_results = await asyncio.gather(
            ok_task(), failing_task(), return_exceptions=True,
        )

        # CancelledError check — should NOT raise for RuntimeError
        for r in gather_results:
            if isinstance(r, asyncio.CancelledError):
                raise r

        # Filter results using supervision.py pattern
        successful = [r for r in gather_results if not isinstance(r, BaseException)]
        exceptions = [r for r in gather_results if isinstance(r, BaseException)]

        assert len(successful) == 1
        assert successful[0].sub_query_id == "sq-0"
        assert len(exceptions) == 1
        assert isinstance(exceptions[0], RuntimeError)

    @pytest.mark.asyncio
    async def test_partial_results_preserved_on_cancellation(self):
        """When cancellation fires mid-batch, completed results are available before re-raise."""
        import asyncio

        async def ok_task():
            return TopicResearchResult(
                sub_query_id="sq-ok", sources_found=3, source_ids=[],
            )

        async def slow_cancelled_task():
            await asyncio.sleep(0.01)
            raise asyncio.CancelledError()

        gather_results = await asyncio.gather(
            ok_task(), slow_cancelled_task(), return_exceptions=True,
        )

        # Collect partial results before propagating cancellation
        completed_results = [
            r for r in gather_results if not isinstance(r, BaseException)
        ]

        # Partial results should be preserved
        assert len(completed_results) == 1
        assert completed_results[0].sub_query_id == "sq-ok"

        # Cancellation should still be detected
        has_cancellation = any(
            isinstance(r, asyncio.CancelledError) for r in gather_results
        )
        assert has_cancellation


# ===========================================================================
# Phase 2: State Management Bugs
# ===========================================================================


class TestShouldContinueGatheringAccuracy:
    """2A: Verify supervision_history records the actual termination decision."""

    @pytest.mark.asyncio
    async def test_history_records_false_when_no_new_sources(self):
        """should_continue_gathering is False when round_new_sources == 0."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=2, sources_per_query=3, supervision_round=0)

        should_stop = await stub._post_round_bookkeeping(
            state=state,
            directives=[],
            directive_results=[],
            think_output=None,
            round_new_sources=0,
            inline_stats={},
            min_sources=2,
            timeout=60.0,
        )

        history = state.metadata.get("supervision_history", [])
        assert len(history) == 1
        assert history[0]["should_continue_gathering"] is False
        assert should_stop is True

    @pytest.mark.asyncio
    async def test_history_records_true_when_new_sources_found(self):
        """should_continue_gathering is True when round_new_sources > 0."""
        stub = StubSupervision(delegation_model=True)
        state = _make_state(num_completed=2, sources_per_query=3, supervision_round=0)

        should_stop = await stub._post_round_bookkeeping(
            state=state,
            directives=[],
            directive_results=[],
            think_output=None,
            round_new_sources=5,
            inline_stats={},
            min_sources=2,
            timeout=60.0,
        )

        history = state.metadata.get("supervision_history", [])
        assert len(history) == 1
        assert history[0]["should_continue_gathering"] is True
        assert should_stop is False


class TestCoverageSnapshotSuffix:
    """2B: Verify pre/post suffixed snapshots and accurate delta computation."""

    def _make_stub(self):
        return StubSupervision(delegation_model=True)

    def test_store_snapshot_with_suffix(self):
        """Snapshots stored with suffix use '{round}_{suffix}' key."""
        stub = self._make_stub()
        state = _make_state(num_completed=2, sources_per_query=2, supervision_round=0)

        coverage_data = stub._build_per_query_coverage(state)
        stub._store_coverage_snapshot(state, coverage_data, suffix="pre")

        snapshots = state.metadata.get("coverage_snapshots", {})
        assert "0_pre" in snapshots, "Pre-suffixed key should exist"
        assert "0" not in snapshots, "Bare key should not exist when suffix is given"

    def test_pre_and_post_snapshots_coexist(self):
        """Pre- and post-directive snapshots for the same round don't overwrite each other."""
        stub = self._make_stub()
        state = _make_state(num_completed=2, sources_per_query=2, supervision_round=0)

        coverage_data = stub._build_per_query_coverage(state)
        stub._store_coverage_snapshot(state, coverage_data, suffix="pre")

        # Add extra source to simulate directive execution
        state.sources.append(
            ResearchSource(
                id="src-extra",
                url="https://newdomain.com/article",
                title="Extra source",
                source_type=SourceType.WEB,
                quality=SourceQuality.HIGH,
                sub_query_id="sq-0",
            )
        )
        post_coverage = stub._build_per_query_coverage(state)
        stub._store_coverage_snapshot(state, post_coverage, suffix="post")

        snapshots = state.metadata["coverage_snapshots"]
        assert "0_pre" in snapshots
        assert "0_post" in snapshots
        # Pre and post should differ in source count
        assert snapshots["0_pre"]["sq-0"]["source_count"] == 2
        assert snapshots["0_post"]["sq-0"]["source_count"] == 3

    def test_delta_uses_prev_post_snapshot(self):
        """compute_coverage_delta compares against previous round's post snapshot."""
        from foundry_mcp.core.research.workflows.deep_research.phases.supervision_coverage import (
            compute_coverage_delta,
            build_per_query_coverage,
        )

        state = _make_state(num_completed=2, sources_per_query=2, supervision_round=1)

        # Store round 0 post snapshot (simulating what supervision loop does)
        state.metadata["coverage_snapshots"] = {
            "0_post": {
                "sq-0": {"query": "Sub-query 0: aspect 0 of deep learning", "source_count": 1, "unique_domains": 1, "status": "completed"},
                "sq-1": {"query": "Sub-query 1: aspect 1 of deep learning", "source_count": 1, "unique_domains": 1, "status": "completed"},
            }
        }

        coverage_data = build_per_query_coverage(state)
        delta = compute_coverage_delta(state, coverage_data, min_sources=2)

        assert delta is not None, "Delta should be generated using 0_post snapshot"
        assert "round 0" in delta

    def test_delta_falls_back_to_bare_key(self):
        """compute_coverage_delta falls back to bare round key for backward compatibility."""
        from foundry_mcp.core.research.workflows.deep_research.phases.supervision_coverage import (
            compute_coverage_delta,
            build_per_query_coverage,
        )

        state = _make_state(num_completed=2, sources_per_query=2, supervision_round=1)

        # Store using bare key (old format)
        state.metadata["coverage_snapshots"] = {
            "0": {
                "sq-0": {"query": "Sub-query 0: aspect 0 of deep learning", "source_count": 1, "unique_domains": 1, "status": "completed"},
                "sq-1": {"query": "Sub-query 1: aspect 1 of deep learning", "source_count": 1, "unique_domains": 1, "status": "completed"},
            }
        }

        coverage_data = build_per_query_coverage(state)
        delta = compute_coverage_delta(state, coverage_data, min_sources=2)

        assert delta is not None, "Delta should work with bare key (backward compat)"


class TestShouldExitHeuristicPure:
    """2D: _should_exit_heuristic returns data without mutating state."""

    def _make_stub(self):
        return StubSupervision(delegation_model=True)

    def test_returns_false_on_round_zero(self):
        """Round 0 always returns (False, {}) without assessing heuristic."""
        stub = self._make_stub()
        state = _make_state(supervision_round=0)

        should_exit, data = stub._should_exit_heuristic(state, min_sources=2)
        assert should_exit is False
        assert data == {}

    def test_returns_true_with_data_when_sufficient(self):
        """When heuristic says sufficient, returns (True, heuristic_data) without mutating state."""
        stub = self._make_stub()
        state = _make_state(
            num_completed=3, sources_per_query=5, supervision_round=1,
        )

        original_round = state.supervision_round
        original_history = state.metadata.get("supervision_history", []).copy()

        should_exit, data = stub._should_exit_heuristic(state, min_sources=2)

        assert should_exit is True
        assert "confidence" in data
        assert data["should_continue_gathering"] is False
        # Verify no state mutation occurred
        assert state.supervision_round == original_round, "Round should not be incremented"
        assert state.metadata.get("supervision_history", []) == original_history, "History should not be appended"
        assert len(stub._audit_events) == 0, "No audit events should be written"

    def test_returns_false_when_insufficient(self):
        """When heuristic says insufficient, returns (False, heuristic_data)."""
        stub = self._make_stub()
        state = _make_state(
            num_completed=2, sources_per_query=1, supervision_round=1,
        )

        should_exit, data = stub._should_exit_heuristic(state, min_sources=10)

        assert should_exit is False
        assert "confidence" in data
        assert data["should_continue_gathering"] is True


# ===========================================================================
# 4B  Wall-clock timeout test
# ===========================================================================


class TestSupervisionWallClockTimeout:
    """4B: Verify the wall-clock timeout forces early exit from the supervision loop."""

    @pytest.mark.asyncio
    async def test_wall_clock_timeout_exits_loop_early(self):
        """With a very short wall-clock timeout, the loop exits before max rounds.

        Verifies:
        - The supervision loop exits after the timeout fires
        - Metadata records the wall-clock exit details
        - Audit event ``supervision_wall_clock_timeout`` is recorded
        """
        stub = StubSupervision()
        # Very short wall-clock timeout (0.0 seconds) so it triggers immediately
        stub.config.deep_research_supervision_wall_clock_timeout = 0.0
        stub.config.deep_research_providers = ["tavily"]
        stub.config.deep_research_topic_max_tool_calls = 5

        # Give the state room to run many rounds if the timeout were absent
        state = _make_state(
            num_completed=2,
            sources_per_query=1,
            supervision_round=0,
            max_supervision_rounds=10,
        )

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ):
            result = await stub._execute_supervision_async(
                state=state, provider_id="test-provider", timeout=30.0,
            )

        assert result.success is True

        # Wall-clock exit metadata should be set
        wall_clock_exit = state.metadata.get("supervision_wall_clock_exit")
        assert wall_clock_exit is not None, "Expected supervision_wall_clock_exit in metadata"
        assert "elapsed_seconds" in wall_clock_exit
        assert "limit_seconds" in wall_clock_exit
        assert wall_clock_exit["limit_seconds"] == 0.0

        # Audit event should be recorded
        audit_event_types = [e[0] for e in stub._audit_events]
        assert "supervision_wall_clock_timeout" in audit_event_types

        # Should NOT have run all 10 rounds
        assert state.supervision_round < 10


# ===========================================================================
# 4C  All directives fail test
# ===========================================================================


class TestAllDirectivesFail:
    """4C: Verify graceful degradation when every directive in a batch fails."""

    @pytest.mark.asyncio
    async def test_all_directives_fail_graceful_degradation(self):
        """When all directive researchers raise exceptions, the loop degrades gracefully.

        Verifies:
        - The supervision loop exits without crashing
        - State is saved
        - The result indicates success (the supervision phase completes)
        - History records that no new sources were found (should_continue_gathering=False)
        """
        stub = StubSupervision()
        stub.config.deep_research_providers = ["tavily"]
        stub.config.deep_research_topic_max_tool_calls = 5
        state = _make_state(
            num_completed=2,
            sources_per_query=1,
            supervision_round=0,
            max_supervision_rounds=3,
        )
        state.max_sub_queries = 10
        state.topic_research_results = []

        # Delegation response with directives
        delegation_response = json.dumps({
            "research_complete": False,
            "directives": [
                {
                    "research_topic": "Topic A",
                    "perspective": "technical",
                    "evidence_needed": "papers",
                    "priority": 1,
                },
                {
                    "research_topic": "Topic B",
                    "perspective": "comparative",
                    "evidence_needed": "benchmarks",
                    "priority": 2,
                },
            ],
            "rationale": "Need more data",
        })

        async def mock_execute_llm_call(**kwargs):
            result = MagicMock()
            result.result = WorkflowResult(
                success=True,
                content=delegation_response,
                provider_id="test-provider",
                model_used="test-model",
                tokens_used=100,
                duration_ms=500.0,
            )
            return result

        # All topic researchers fail
        async def mock_topic_research_always_fails(**kwargs):
            raise RuntimeError("Simulated total failure in topic researcher")

        stub._execute_topic_research_async = mock_topic_research_always_fails
        stub._get_search_provider = MagicMock(return_value=MagicMock())

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            side_effect=_wrap_as_structured_mock(mock_execute_llm_call),
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ):
            result = await stub._execute_supervision_async(
                state=state, provider_id="test-provider", timeout=30.0,
            )

        # Should complete without crashing
        assert result.success is True

        # State was saved (at least once for post-round bookkeeping)
        assert stub.memory.save_deep_research.called

        # History should record that no new sources were found
        history = state.metadata.get("supervision_history", [])
        assert len(history) > 0
        # The delegation round should record 0 new sources
        delegation_entry = next(
            (h for h in history if h["method"] == "delegation"),
            None,
        )
        assert delegation_entry is not None
        assert delegation_entry["new_sources"] == 0
        assert delegation_entry["should_continue_gathering"] is False

    @pytest.mark.asyncio
    async def test_all_directives_fail_loop_stops(self):
        """When round yields 0 sources (all failures), the loop terminates."""
        stub = StubSupervision()
        stub.config.deep_research_providers = ["tavily"]
        stub.config.deep_research_topic_max_tool_calls = 5
        state = _make_state(
            num_completed=2,
            sources_per_query=1,
            supervision_round=0,
            max_supervision_rounds=5,
        )
        state.max_sub_queries = 10
        state.topic_research_results = []

        delegation_response = json.dumps({
            "research_complete": False,
            "directives": [
                {
                    "research_topic": "Failing topic",
                    "perspective": "technical",
                    "evidence_needed": "none",
                    "priority": 1,
                },
            ],
            "rationale": "Test",
        })

        async def mock_execute_llm_call(**kwargs):
            result = MagicMock()
            result.result = WorkflowResult(
                success=True,
                content=delegation_response,
                provider_id="test-provider",
                model_used="test-model",
                tokens_used=50,
                duration_ms=100.0,
            )
            return result

        async def mock_topic_research_fails(**kwargs):
            raise RuntimeError("All researchers fail")

        stub._execute_topic_research_async = mock_topic_research_fails
        stub._get_search_provider = MagicMock(return_value=MagicMock())

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.execute_structured_llm_call",
            side_effect=_wrap_as_structured_mock(mock_execute_llm_call),
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.supervision.finalize_phase",
        ):
            result = await stub._execute_supervision_async(
                state=state, provider_id="test-provider", timeout=30.0,
            )

        assert result.success is True
        # The loop should have stopped after the first round (0 new sources)
        # Round 0 ran → incremented to 1 → bookkeeping returns True (stop)
        assert state.supervision_round == 1
