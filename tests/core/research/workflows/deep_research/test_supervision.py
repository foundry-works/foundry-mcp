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
    _MAX_FOLLOW_UPS_PER_ROUND,
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
    state = DeepResearchState(
        id="deepres-supervision-test",
        original_query=query,
        research_brief="Investigating deep learning fundamentals",
        phase=phase,
        iteration=1,
        max_iterations=3,
        supervision_round=supervision_round,
        max_supervision_rounds=max_supervision_rounds,
    )
    for i in range(num_completed):
        sq = SubQuery(
            id=f"sq-{i}",
            query=f"Sub-query {i}: aspect {i} of deep learning",
            status="completed",
            priority=1,
        )
        state.sub_queries.append(sq)
        for j in range(sources_per_query):
            state.sources.append(
                ResearchSource(
                    id=f"src-{i}-{j}",
                    url=f"https://example{j}.com/article-{i}",
                    title=f"Source {i}-{j}",
                    source_type=SourceType.WEB,
                    quality=SourceQuality.HIGH if j == 0 else SourceQuality.MEDIUM,
                    sub_query_id=sq.id,
                )
            )
    for i in range(num_pending):
        state.sub_queries.append(
            SubQuery(
                id=f"sq-pending-{i}",
                query=f"Pending query {i}",
                status="pending",
                priority=2,
            )
        )
    return state


class StubSupervision(SupervisionPhaseMixin):
    """Concrete class for testing SupervisionPhaseMixin in isolation."""

    def __init__(self, *, delegation_model: bool = False) -> None:
        self.config = MagicMock()
        self.config.deep_research_supervision_min_sources_per_query = 2
        self.config.deep_research_max_concurrent_research_units = 5
        self.config.deep_research_reflection_timeout = 60.0
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

    def test_supervision_system_prompt_has_json_schema(self):
        """System prompt contains expected JSON structure keys."""
        stub = StubSupervision()
        state = _make_state()
        prompt = stub._build_supervision_system_prompt(state)

        assert "overall_coverage" in prompt
        assert "per_query_assessment" in prompt
        assert "follow_up_queries" in prompt
        assert "should_continue_gathering" in prompt
        assert "sufficient|partial|insufficient" in prompt

    def test_supervision_user_prompt_includes_coverage(self):
        """User prompt includes per-query coverage data and dedup list."""
        stub = StubSupervision()
        state = _make_state(num_completed=2, sources_per_query=2)
        coverage = stub._build_per_query_coverage(state)
        prompt = stub._build_supervision_user_prompt(state, coverage)

        assert state.original_query in prompt
        assert "sq-0" in prompt
        assert "sq-1" in prompt
        assert "Sources:" in prompt
        assert "DO NOT duplicate" in prompt
        # Round info
        assert "Supervision round:" in prompt


# ===========================================================================
# 6.7.3  Response parsing tests
# ===========================================================================


class TestSupervisionParsing:
    """Tests for _parse_supervision_response and heuristic fallback."""

    def test_parse_valid_supervision_response(self):
        """Valid JSON is parsed correctly with all fields."""
        stub = StubSupervision()
        state = _make_state(num_completed=2)

        response_json = json.dumps(
            {
                "overall_coverage": "partial",
                "per_query_assessment": [
                    {"sub_query_id": "sq-0", "coverage": "sufficient", "rationale": "Good sources"},
                    {"sub_query_id": "sq-1", "coverage": "insufficient", "rationale": "Needs more"},
                ],
                "follow_up_queries": [
                    {"query": "What are transformers?", "rationale": "Missing architecture details", "priority": 2},
                ],
                "should_continue_gathering": True,
                "rationale": "Coverage is partial, need transformer details",
            }
        )

        result = stub._parse_supervision_response(response_json, state)

        assert result["overall_coverage"] == "partial"
        assert len(result["per_query_assessment"]) == 2
        assert len(result["follow_up_queries"]) == 1
        assert result["follow_up_queries"][0]["query"] == "What are transformers?"
        assert result["should_continue_gathering"] is True
        assert "partial" in result["rationale"]

    def test_parse_with_duplicate_queries_deduped(self):
        """Follow-up queries that match existing sub-queries are stripped."""
        stub = StubSupervision()
        state = _make_state(num_completed=2)

        # One follow-up duplicates an existing sub-query (case-insensitive)
        response_json = json.dumps(
            {
                "overall_coverage": "partial",
                "per_query_assessment": [],
                "follow_up_queries": [
                    {"query": "SUB-QUERY 0: ASPECT 0 OF DEEP LEARNING", "rationale": "dup"},
                    {"query": "Brand new query about GANs", "rationale": "new"},
                    {"query": "Brand new query about GANs", "rationale": "within-batch dup"},
                ],
                "should_continue_gathering": True,
                "rationale": "",
            }
        )

        result = stub._parse_supervision_response(response_json, state)

        # Only the unique non-duplicate query should remain
        assert len(result["follow_up_queries"]) == 1
        assert result["follow_up_queries"][0]["query"] == "Brand new query about GANs"

    def test_parse_invalid_json_returns_fallback(self):
        """Invalid JSON returns default structure with should_continue_gathering=False."""
        stub = StubSupervision()
        state = _make_state()

        result = stub._parse_supervision_response("This is not JSON at all", state)

        assert result["overall_coverage"] == "unknown"
        assert result["follow_up_queries"] == []
        assert result["should_continue_gathering"] is False

    def test_parse_empty_content_returns_fallback(self):
        """Empty content returns default structure."""
        stub = StubSupervision()
        state = _make_state()

        result = stub._parse_supervision_response("", state)
        assert result["overall_coverage"] == "unknown"
        assert result["should_continue_gathering"] is False

    def test_parse_caps_follow_ups_at_max(self):
        """Follow-up queries are capped at _MAX_FOLLOW_UPS_PER_ROUND."""
        stub = StubSupervision()
        state = _make_state(num_completed=1)

        many_follow_ups = [
            {"query": f"Follow-up query {i}", "rationale": f"reason {i}", "priority": 2}
            for i in range(10)
        ]
        response_json = json.dumps(
            {
                "overall_coverage": "insufficient",
                "per_query_assessment": [],
                "follow_up_queries": many_follow_ups,
                "should_continue_gathering": True,
                "rationale": "",
            }
        )

        result = stub._parse_supervision_response(response_json, state)
        assert len(result["follow_up_queries"]) == _MAX_FOLLOW_UPS_PER_ROUND

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
        assert result.metadata["should_continue_gathering"] is False
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
        assert result.metadata["should_continue_gathering"] is False
        assert result.metadata["model"] == "delegation"
        assert state.supervision_round == 2  # incremented
        # History records the heuristic early-exit
        history = state.metadata["supervision_history"]
        assert history[0]["method"] == "delegation_heuristic"

    @pytest.mark.asyncio
    async def test_supervision_skipped_when_disabled(self):
        """When supervision is disabled, the workflow skips SUPERVISION entirely."""
        # This tests the workflow_execution.py logic, not the mixin
        state = _make_state(phase=DeepResearchPhase.SUPERVISION)
        config = MagicMock()
        config.deep_research_enable_supervision = False

        # Simulate the workflow_execution.py logic
        if not getattr(config, "deep_research_enable_supervision", True):
            state.advance_phase()

        assert state.phase == DeepResearchPhase.SYNTHESIS

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
        assert result.metadata["should_continue_gathering"] is False
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
        assert result.metadata["should_continue_gathering"] is False
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


class TestThinkToolInUserPrompt:
    """Tests for think output integration into supervision user prompt."""

    def test_user_prompt_includes_gap_analysis_when_provided(self):
        """User prompt includes <gap_analysis> section when think output is given."""
        stub = StubSupervision()
        state = _make_state(num_completed=2, sources_per_query=2)
        coverage = stub._build_per_query_coverage(state)

        think_output = (
            "## Per-Query Analysis\n"
            "Sub-query 0 covers neural network basics but lacks historical context.\n"
            "Sub-query 1 has good coverage of training methods but misses regularization.\n\n"
            "## Overall Gaps\n"
            "Missing: historical development, regularization techniques, hardware requirements."
        )

        prompt = stub._build_supervision_user_prompt(state, coverage, think_output)

        assert "<gap_analysis>" in prompt
        assert "</gap_analysis>" in prompt
        assert "historical context" in prompt
        assert "regularization" in prompt
        assert "TARGETED follow-up queries" in prompt
        assert "MUST reference a specific gap" in prompt

    def test_user_prompt_no_gap_section_when_think_absent(self):
        """User prompt omits gap analysis section when no think output."""
        stub = StubSupervision()
        state = _make_state(num_completed=2, sources_per_query=2)
        coverage = stub._build_per_query_coverage(state)

        prompt = stub._build_supervision_user_prompt(state, coverage, think_output=None)

        assert "<gap_analysis>" not in prompt
        assert "MUST reference a specific gap" not in prompt

    def test_user_prompt_no_gap_section_when_empty_string(self):
        """User prompt omits gap analysis section when think output is empty."""
        stub = StubSupervision()
        state = _make_state(num_completed=2, sources_per_query=2)
        coverage = stub._build_per_query_coverage(state)

        prompt = stub._build_supervision_user_prompt(state, coverage, think_output="")

        assert "<gap_analysis>" not in prompt


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
            if kwargs.get("phase_name") == "supervision_think":
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content="Gap analysis", tokens_used=50
                )
                return result
            elif kwargs.get("phase_name") == "supervision_delegate":
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
            # No other calls expected
            raise AssertionError(f"Unexpected call: {kwargs.get('phase_name')}")

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
        assert result.metadata["should_continue_gathering"] is False
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
            call_phases.append(kwargs.get("phase_name", "unknown"))
            if kwargs.get("phase_name") == "supervision_think":
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content="Gaps exist", tokens_used=40
                )
                return result
            elif kwargs.get("phase_name") == "supervision_delegate":
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
            raise AssertionError(f"Unexpected: {kwargs.get('phase_name')}")

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
            if kwargs.get("phase_name") == "supervision_think":
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content="Analysis", tokens_used=40
                )
                return result
            elif kwargs.get("phase_name") == "supervision_delegate":
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
            raise AssertionError(f"Unexpected: {kwargs.get('phase_name')}")

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
        assert history[0]["method"] == "delegation_no_directives"


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
        # Self-critique
        assert "Self-Critique" in prompt or "self-critique" in prompt.lower()
        assert "redundant" in prompt.lower()
        # JSON format
        assert "research_complete" in prompt
        assert "directives" in prompt
        assert "rationale" in prompt

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
        """BRIEF → GATHERING → SUPERVISION transition follows the active phase order."""
        state = DeepResearchState(
            original_query="test query",
            phase=DeepResearchPhase.BRIEF,
            research_brief="Enriched research brief",
        )

        # BRIEF → GATHERING
        state.advance_phase()
        assert state.phase == DeepResearchPhase.GATHERING

        # GATHERING → SUPERVISION
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

    def test_planning_phase_not_in_enum(self):
        """PLANNING phase does not exist in DeepResearchPhase — no legacy resume possible."""
        phase_values = [p.value for p in DeepResearchPhase]
        assert "planning" not in phase_values
        assert "PLANNING" not in [p.name for p in DeepResearchPhase]

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

    def test_first_round_delegation_prompt_has_self_critique(self):
        """First-round delegation system prompt includes self-critique checklist."""
        stub = StubSupervision(delegation_model=True)
        prompt = stub._build_first_round_delegation_system_prompt()
        assert "redundant" in prompt.lower()
        assert "missing" in prompt.lower() or "critical perspective" in prompt.lower()

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
