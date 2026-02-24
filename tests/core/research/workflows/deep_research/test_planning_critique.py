"""Tests for the planning self-critique phase (PLAN Phase 2).

Covers:
- 2.1: _build_decomposition_critique_prompt() and _build_critique_system_prompt()
- 2.3: _parse_critique_response() and _apply_critique_adjustments()
- 2.6: Integration tests for critique in _execute_planning_async()
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
from foundry_mcp.core.research.models.sources import SubQuery
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research.phases.planning import (
    PlanningPhaseMixin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    query: str = "What are the impacts of AI on healthcare?",
    num_sub_queries: int = 3,
    max_sub_queries: int = 7,
) -> DeepResearchState:
    """Create a DeepResearchState pre-populated for planning critique tests."""
    state = DeepResearchState(
        id="deepres-critique-test",
        original_query=query,
        research_brief="Investigate the impacts of AI on healthcare delivery, diagnostics, and policy",
        phase=DeepResearchPhase.PLANNING,
        iteration=1,
        max_iterations=3,
        max_sub_queries=max_sub_queries,
    )
    sample_queries = [
        ("AI safety regulations in healthcare", "Covers regulatory landscape", 1),
        ("AI governance policies for medical AI", "Covers governance frameworks", 1),
        ("AI diagnostic accuracy compared to human doctors", "Covers clinical impact", 2),
    ]
    for i in range(min(num_sub_queries, len(sample_queries))):
        q, r, p = sample_queries[i]
        state.sub_queries.append(
            SubQuery(id=f"sq-{i}", query=q, rationale=r, priority=p, status="pending")
        )
    return state


class StubPlanning(PlanningPhaseMixin):
    """Concrete class for testing PlanningPhaseMixin in isolation."""

    def __init__(self, enable_critique: bool = True) -> None:
        self.config = MagicMock()
        self.config.deep_research_enable_planning_critique = enable_critique
        self.memory = MagicMock()
        self._audit_events: list[tuple[str, dict]] = []

    def _write_audit_event(self, state: Any, event: str, **kwargs: Any) -> None:
        self._audit_events.append((event, kwargs))

    def _check_cancellation(self, state: Any) -> None:
        pass


# ===========================================================================
# 2.1  Prompt building tests
# ===========================================================================


class TestCritiquePrompts:
    """Tests for _build_decomposition_critique_prompt and _build_critique_system_prompt."""

    def test_critique_prompt_contains_sub_queries(self):
        """Critique prompt lists all generated sub-queries with indices."""
        stub = StubPlanning()
        state = _make_state(num_sub_queries=3)
        prompt = stub._build_decomposition_critique_prompt(state)

        assert "AI safety regulations" in prompt
        assert "AI governance policies" in prompt
        assert "AI diagnostic accuracy" in prompt
        # Indexed for redundancy references
        assert "0." in prompt
        assert "1." in prompt
        assert "2." in prompt

    def test_critique_prompt_includes_research_brief(self):
        """Critique prompt includes the research brief for context."""
        stub = StubPlanning()
        state = _make_state(num_sub_queries=2)
        prompt = stub._build_decomposition_critique_prompt(state)

        assert state.research_brief in prompt

    def test_critique_prompt_falls_back_to_original_query(self):
        """When no research brief, critique prompt uses original query."""
        stub = StubPlanning()
        state = _make_state(num_sub_queries=1)
        state.research_brief = None
        prompt = stub._build_decomposition_critique_prompt(state)

        assert state.original_query in prompt

    def test_critique_prompt_contains_evaluation_instructions(self):
        """Critique prompt asks for redundancy, perspective, and scope evaluation."""
        stub = StubPlanning()
        state = _make_state(num_sub_queries=2)
        prompt = stub._build_decomposition_critique_prompt(state)

        assert "Redundancies" in prompt
        assert "Missing perspectives" in prompt
        assert "Scope issues" in prompt

    def test_critique_system_prompt_has_json_schema(self):
        """System prompt contains expected JSON structure keys."""
        stub = StubPlanning()
        system = stub._build_critique_system_prompt()

        assert "redundancies" in system
        assert "gaps" in system
        assert "adjustments" in system
        assert "merged_query" in system
        assert "valid JSON" in system

    def test_critique_system_prompt_is_conservative(self):
        """System prompt instructs conservative critique (only TRUE redundancies)."""
        stub = StubPlanning()
        system = stub._build_critique_system_prompt()

        assert "TRUE redundancies" in system
        assert "GENUINELY missing" in system


# ===========================================================================
# 2.3  Response parsing tests
# ===========================================================================


class TestCritiqueParsing:
    """Tests for _parse_critique_response."""

    def test_parse_valid_critique_response(self):
        """Valid JSON is parsed correctly with all fields."""
        stub = StubPlanning()
        response = json.dumps({
            "redundancies": [
                {
                    "indices": [0, 1],
                    "reason": "Both cover AI regulations/governance",
                    "merged_query": "AI safety regulations and governance policies in healthcare",
                }
            ],
            "gaps": [
                {
                    "query": "Economic impact of AI adoption in hospitals",
                    "rationale": "Missing economic perspective",
                    "priority": 2,
                }
            ],
            "adjustments": [
                {
                    "index": 2,
                    "revised_query": "AI diagnostic accuracy vs human doctors in radiology and pathology",
                    "reason": "Original too broad, focusing on specific specialties",
                }
            ],
            "assessment": "Good decomposition but has one redundancy",
        })

        result = stub._parse_critique_response(response)

        assert result["has_changes"] is True
        assert len(result["redundancies"]) == 1
        assert result["redundancies"][0]["indices"] == [0, 1]
        assert result["redundancies"][0]["merged_query"] == "AI safety regulations and governance policies in healthcare"
        assert len(result["gaps"]) == 1
        assert result["gaps"][0]["query"] == "Economic impact of AI adoption in hospitals"
        assert len(result["adjustments"]) == 1
        assert result["adjustments"][0]["index"] == 2
        assert "radiology" in result["adjustments"][0]["revised_query"]
        assert result["assessment"] == "Good decomposition but has one redundancy"

    def test_parse_no_changes_response(self):
        """When critique finds no issues, has_changes is False."""
        stub = StubPlanning()
        response = json.dumps({
            "redundancies": [],
            "gaps": [],
            "adjustments": [],
            "assessment": "Decomposition is well-balanced",
        })

        result = stub._parse_critique_response(response)

        assert result["has_changes"] is False
        assert result["redundancies"] == []
        assert result["gaps"] == []
        assert result["adjustments"] == []

    def test_parse_empty_content(self):
        """Empty content returns default structure."""
        stub = StubPlanning()
        result = stub._parse_critique_response("")

        assert result["has_changes"] is False
        assert result["redundancies"] == []

    def test_parse_invalid_json(self):
        """Invalid JSON returns default structure."""
        stub = StubPlanning()
        result = stub._parse_critique_response("This is not JSON at all")

        assert result["has_changes"] is False

    def test_parse_skips_invalid_redundancies(self):
        """Redundancies with < 2 indices or missing merged_query are skipped."""
        stub = StubPlanning()
        response = json.dumps({
            "redundancies": [
                {"indices": [0], "reason": "only one index", "merged_query": "test"},
                {"indices": [0, 1], "reason": "no merged query", "merged_query": ""},
                {"indices": "not a list", "reason": "bad type", "merged_query": "test"},
            ],
            "gaps": [],
            "adjustments": [],
            "assessment": "",
        })

        result = stub._parse_critique_response(response)
        assert result["redundancies"] == []
        assert result["has_changes"] is False

    def test_parse_skips_invalid_gaps(self):
        """Gaps with empty query are skipped."""
        stub = StubPlanning()
        response = json.dumps({
            "redundancies": [],
            "gaps": [
                {"query": "", "rationale": "empty query"},
                {"query": "  ", "rationale": "whitespace only"},
                "not a dict",
            ],
            "adjustments": [],
            "assessment": "",
        })

        result = stub._parse_critique_response(response)
        assert result["gaps"] == []

    def test_parse_clamps_priority_range(self):
        """Gap priorities are clamped to [1, 10]."""
        stub = StubPlanning()
        response = json.dumps({
            "redundancies": [],
            "gaps": [
                {"query": "Test query", "rationale": "test", "priority": -5},
                {"query": "Another query", "rationale": "test", "priority": 999},
            ],
            "adjustments": [],
            "assessment": "",
        })

        result = stub._parse_critique_response(response)
        assert result["gaps"][0]["priority"] == 1
        assert result["gaps"][1]["priority"] == 10


# ===========================================================================
# 2.3  Adjustment application tests
# ===========================================================================


class TestCritiqueAdjustments:
    """Tests for _apply_critique_adjustments."""

    def test_merge_redundant_sub_queries(self):
        """Redundant sub-queries are merged into one."""
        stub = StubPlanning()
        state = _make_state(num_sub_queries=3, max_sub_queries=7)

        critique = {
            "redundancies": [
                {
                    "indices": [0, 1],
                    "reason": "Both cover AI regulations/governance",
                    "merged_query": "AI safety regulations and governance policies in healthcare",
                }
            ],
            "gaps": [],
            "adjustments": [],
        }

        stub._apply_critique_adjustments(state, critique)

        # 3 original - 2 removed + 1 merged = 2
        assert len(state.sub_queries) == 2
        # The merged query should exist
        queries = [sq.query for sq in state.sub_queries]
        assert "AI safety regulations and governance policies in healthcare" in queries
        # The original redundant queries should be gone
        assert "AI safety regulations in healthcare" not in queries
        assert "AI governance policies for medical AI" not in queries
        # The non-redundant query should remain
        assert "AI diagnostic accuracy compared to human doctors" in queries

    def test_add_gap_sub_queries(self):
        """Missing perspectives are added as new sub-queries."""
        stub = StubPlanning()
        state = _make_state(num_sub_queries=2, max_sub_queries=5)

        critique = {
            "redundancies": [],
            "gaps": [
                {
                    "query": "Historical evolution of AI in medicine",
                    "rationale": "Missing historical context",
                    "priority": 2,
                },
                {
                    "query": "Patient privacy concerns with AI diagnostics",
                    "rationale": "Missing privacy angle",
                    "priority": 1,
                },
            ],
            "adjustments": [],
        }

        stub._apply_critique_adjustments(state, critique)

        assert len(state.sub_queries) == 4  # 2 original + 2 gaps
        queries = [sq.query for sq in state.sub_queries]
        assert "Historical evolution of AI in medicine" in queries
        assert "Patient privacy concerns with AI diagnostics" in queries

    def test_apply_scope_adjustments(self):
        """Scope adjustments revise query text in-place."""
        stub = StubPlanning()
        state = _make_state(num_sub_queries=3, max_sub_queries=7)

        critique = {
            "redundancies": [],
            "gaps": [],
            "adjustments": [
                {
                    "index": 2,
                    "revised_query": "AI vs human accuracy in radiology and pathology",
                    "reason": "More focused",
                }
            ],
        }

        stub._apply_critique_adjustments(state, critique)

        assert len(state.sub_queries) == 3  # No change in count
        assert state.sub_queries[2].query == "AI vs human accuracy in radiology and pathology"

    def test_respects_max_sub_queries_for_gaps(self):
        """Gap additions stop at max_sub_queries bound."""
        stub = StubPlanning()
        state = _make_state(num_sub_queries=3, max_sub_queries=4)

        critique = {
            "redundancies": [],
            "gaps": [
                {"query": "Gap 1", "rationale": "r1", "priority": 1},
                {"query": "Gap 2", "rationale": "r2", "priority": 1},
                {"query": "Gap 3", "rationale": "r3", "priority": 1},
            ],
            "adjustments": [],
        }

        stub._apply_critique_adjustments(state, critique)

        # 3 original + only 1 gap (at max_sub_queries=4)
        assert len(state.sub_queries) == 4

    def test_final_safety_cap(self):
        """If somehow over max_sub_queries, truncate to highest priority."""
        stub = StubPlanning()
        state = _make_state(num_sub_queries=3, max_sub_queries=2)

        critique = {
            "redundancies": [],
            "gaps": [],
            "adjustments": [],
        }

        stub._apply_critique_adjustments(state, critique)

        # Should be capped at 2 (highest priority kept)
        assert len(state.sub_queries) == 2

    def test_combined_merge_and_gap(self):
        """Redundancy merge + gap addition in same critique."""
        stub = StubPlanning()
        state = _make_state(num_sub_queries=3, max_sub_queries=5)

        critique = {
            "redundancies": [
                {
                    "indices": [0, 1],
                    "reason": "Overlap",
                    "merged_query": "AI regulation and governance in healthcare",
                }
            ],
            "gaps": [
                {"query": "Economic costs of AI in hospitals", "rationale": "Missing economic angle", "priority": 1},
            ],
            "adjustments": [],
        }

        stub._apply_critique_adjustments(state, critique)

        # 3 - 2 + 1 merged + 1 gap = 3
        assert len(state.sub_queries) == 3
        queries = [sq.query for sq in state.sub_queries]
        assert "AI regulation and governance in healthcare" in queries
        assert "Economic costs of AI in hospitals" in queries
        assert "AI diagnostic accuracy compared to human doctors" in queries

    def test_out_of_range_indices_ignored(self):
        """Redundancy indices out of range are handled gracefully."""
        stub = StubPlanning()
        state = _make_state(num_sub_queries=2, max_sub_queries=5)

        critique = {
            "redundancies": [
                {
                    "indices": [0, 99],  # 99 is out of range
                    "reason": "Invalid",
                    "merged_query": "Merged",
                }
            ],
            "gaps": [],
            "adjustments": [],
        }

        stub._apply_critique_adjustments(state, critique)

        # Should not merge — only one valid index
        assert len(state.sub_queries) == 2

    def test_merged_query_preserves_highest_priority(self):
        """Merged query gets the best (lowest number) priority from its sources."""
        stub = StubPlanning()
        state = _make_state(num_sub_queries=3, max_sub_queries=7)
        # Set different priorities
        state.sub_queries[0].priority = 3
        state.sub_queries[1].priority = 1

        critique = {
            "redundancies": [
                {
                    "indices": [0, 1],
                    "reason": "Overlap",
                    "merged_query": "Merged regulation query",
                }
            ],
            "gaps": [],
            "adjustments": [],
        }

        stub._apply_critique_adjustments(state, critique)

        merged = [sq for sq in state.sub_queries if sq.query == "Merged regulation query"]
        assert len(merged) == 1
        assert merged[0].priority == 1  # Best priority from the merged pair


# ===========================================================================
# 2.6  Integration tests
# ===========================================================================


class TestCritiqueIntegration:
    """Integration tests for critique in the full planning execution flow."""

    @pytest.mark.asyncio
    async def test_critique_merges_redundant_queries(self):
        """Full flow: planning generates queries, critique merges redundancies."""
        stub = StubPlanning(enable_critique=True)

        planning_response = json.dumps({
            "research_brief": "AI impact on healthcare",
            "sub_queries": [
                {"query": "AI safety regulations", "rationale": "Regulatory", "priority": 1},
                {"query": "AI governance policies", "rationale": "Governance", "priority": 1},
                {"query": "AI diagnostic accuracy", "rationale": "Clinical", "priority": 2},
            ],
        })

        critique_response = json.dumps({
            "redundancies": [
                {
                    "indices": [0, 1],
                    "reason": "Both cover regulatory/governance",
                    "merged_query": "AI regulation and governance in healthcare",
                }
            ],
            "gaps": [],
            "adjustments": [],
            "assessment": "One redundancy found",
        })

        call_count = 0

        async def mock_execute_llm_call(**kwargs):
            nonlocal call_count
            call_count += 1

            phase = kwargs.get("phase_name", "")

            if phase == "planning":
                # Sub-query generation
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True,
                    content=planning_response,
                    provider_id="test-provider",
                    model_used="test-model",
                    tokens_used=100,
                    duration_ms=500.0,
                )
                return result
            elif phase == "planning_critique":
                # Critique step
                assert kwargs.get("role") == "reflection"
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True,
                    content=critique_response,
                    provider_id="reflection-provider",
                    model_used="cheap-model",
                    tokens_used=60,
                    duration_ms=200.0,
                )
                return result
            else:
                # Brief refinement
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True,
                    content="AI impact on healthcare delivery and regulation",
                    provider_id="test-provider",
                    model_used="test-model",
                    tokens_used=30,
                    duration_ms=100.0,
                )
                return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.planning.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.planning.finalize_phase",
        ):
            state = DeepResearchState(
                id="test-critique-flow",
                original_query="AI impact on healthcare",
                phase=DeepResearchPhase.PLANNING,
                max_sub_queries=7,
            )
            result = await stub._execute_planning_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        assert result.success is True
        # Should have 3 calls: brief refinement + planning + critique
        assert call_count == 3

        # After critique: 3 - 2 + 1 merged = 2 sub-queries
        assert len(state.sub_queries) == 2
        queries = [sq.query for sq in state.sub_queries]
        assert "AI regulation and governance in healthcare" in queries
        assert "AI diagnostic accuracy" in queries

        # Critique recorded in metadata
        assert "planning_critique" in state.metadata
        critique_meta = state.metadata["planning_critique"]
        assert critique_meta["applied"] is True
        assert len(critique_meta["original_sub_queries"]) == 3
        assert len(critique_meta["adjusted_sub_queries"]) == 2

    @pytest.mark.asyncio
    async def test_critique_adds_missing_perspectives(self):
        """Full flow: critique adds missing perspectives."""
        stub = StubPlanning(enable_critique=True)

        planning_response = json.dumps({
            "research_brief": "AI in healthcare",
            "sub_queries": [
                {"query": "AI diagnostic tools", "rationale": "Clinical", "priority": 1},
            ],
        })

        critique_response = json.dumps({
            "redundancies": [],
            "gaps": [
                {"query": "Economic impact of AI in hospitals", "rationale": "Missing economic angle", "priority": 2},
                {"query": "Patient privacy with AI systems", "rationale": "Missing privacy angle", "priority": 1},
            ],
            "adjustments": [],
            "assessment": "Missing economic and privacy perspectives",
        })

        async def mock_execute_llm_call(**kwargs):
            phase = kwargs.get("phase_name", "")
            if phase == "planning":
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content=planning_response,
                    provider_id="p", model_used="m", tokens_used=50, duration_ms=200.0,
                )
                return result
            elif phase == "planning_critique":
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content=critique_response,
                    provider_id="p", model_used="m", tokens_used=40, duration_ms=150.0,
                )
                return result
            else:
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content="Brief", provider_id="p",
                    model_used="m", tokens_used=20, duration_ms=80.0,
                )
                return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.planning.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.planning.finalize_phase",
        ):
            state = DeepResearchState(
                id="test-gaps-flow",
                original_query="AI in healthcare",
                phase=DeepResearchPhase.PLANNING,
                max_sub_queries=5,
            )
            result = await stub._execute_planning_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        assert result.success is True
        assert len(state.sub_queries) == 3  # 1 original + 2 gaps
        queries = [sq.query for sq in state.sub_queries]
        assert "Economic impact of AI in hospitals" in queries
        assert "Patient privacy with AI systems" in queries

    @pytest.mark.asyncio
    async def test_critique_skipped_when_config_disabled(self):
        """Critique does not run when config flag is False."""
        stub = StubPlanning(enable_critique=False)

        planning_response = json.dumps({
            "research_brief": "AI in healthcare",
            "sub_queries": [
                {"query": "AI tools", "rationale": "Tools", "priority": 1},
                {"query": "AI policy", "rationale": "Policy", "priority": 1},
            ],
        })

        call_phases: list[str] = []

        async def mock_execute_llm_call(**kwargs):
            phase = kwargs.get("phase_name", "")
            call_phases.append(phase)
            result = MagicMock()
            result.result = WorkflowResult(
                success=True, content=planning_response if phase == "planning" else "Brief",
                provider_id="p", model_used="m", tokens_used=50, duration_ms=200.0,
            )
            return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.planning.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.planning.finalize_phase",
        ):
            state = DeepResearchState(
                id="test-disabled",
                original_query="AI in healthcare",
                phase=DeepResearchPhase.PLANNING,
                max_sub_queries=5,
            )
            result = await stub._execute_planning_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        assert result.success is True
        # Only brief refinement + planning, NO critique
        assert "planning_critique" not in call_phases
        assert len(state.sub_queries) == 2  # Unchanged
        assert "planning_critique" not in state.metadata

    @pytest.mark.asyncio
    async def test_critique_respects_max_sub_queries_bound(self):
        """After critique adjustments, sub-query count stays within bounds."""
        stub = StubPlanning(enable_critique=True)

        planning_response = json.dumps({
            "research_brief": "AI in healthcare",
            "sub_queries": [
                {"query": "Query A", "rationale": "r", "priority": 1},
                {"query": "Query B", "rationale": "r", "priority": 1},
                {"query": "Query C", "rationale": "r", "priority": 2},
            ],
        })

        # Critique wants to add 5 gaps — should be capped
        critique_response = json.dumps({
            "redundancies": [],
            "gaps": [
                {"query": f"Gap {i}", "rationale": f"reason {i}", "priority": 2}
                for i in range(5)
            ],
            "adjustments": [],
            "assessment": "Many gaps",
        })

        async def mock_execute_llm_call(**kwargs):
            phase = kwargs.get("phase_name", "")
            if phase == "planning":
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content=planning_response,
                    provider_id="p", model_used="m", tokens_used=50, duration_ms=200.0,
                )
                return result
            elif phase == "planning_critique":
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content=critique_response,
                    provider_id="p", model_used="m", tokens_used=40, duration_ms=150.0,
                )
                return result
            else:
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content="Brief", provider_id="p",
                    model_used="m", tokens_used=20, duration_ms=80.0,
                )
                return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.planning.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.planning.finalize_phase",
        ):
            state = DeepResearchState(
                id="test-bounds",
                original_query="AI in healthcare",
                phase=DeepResearchPhase.PLANNING,
                max_sub_queries=5,  # Tight bound
            )
            result = await stub._execute_planning_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        assert result.success is True
        assert len(state.sub_queries) <= 5

    @pytest.mark.asyncio
    async def test_critique_failure_is_non_fatal(self):
        """When critique LLM call fails, planning proceeds without critique."""
        stub = StubPlanning(enable_critique=True)

        planning_response = json.dumps({
            "research_brief": "AI in healthcare",
            "sub_queries": [
                {"query": "AI tools", "rationale": "Tools", "priority": 1},
                {"query": "AI policy", "rationale": "Policy", "priority": 2},
            ],
        })

        async def mock_execute_llm_call(**kwargs):
            phase = kwargs.get("phase_name", "")
            if phase == "planning_critique":
                # Critique fails
                return WorkflowResult(
                    success=False, content="", error="Provider timeout"
                )
            else:
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content=planning_response if phase == "planning" else "Brief",
                    provider_id="p", model_used="m", tokens_used=50, duration_ms=200.0,
                )
                return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.planning.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.planning.finalize_phase",
        ):
            state = DeepResearchState(
                id="test-failure",
                original_query="AI in healthcare",
                phase=DeepResearchPhase.PLANNING,
                max_sub_queries=5,
            )
            result = await stub._execute_planning_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        assert result.success is True
        # Sub-queries unchanged (no critique applied)
        assert len(state.sub_queries) == 2
        # Critique recorded as failed
        assert "planning_critique" in state.metadata
        critique_meta = state.metadata["planning_critique"]
        assert critique_meta["applied"] is False
        assert "Provider timeout" in critique_meta["error"]

    @pytest.mark.asyncio
    async def test_critique_no_changes_records_metadata(self):
        """When critique finds no issues, metadata records applied=False."""
        stub = StubPlanning(enable_critique=True)

        planning_response = json.dumps({
            "research_brief": "AI in healthcare",
            "sub_queries": [
                {"query": "AI tools", "rationale": "Tools", "priority": 1},
                {"query": "AI policy", "rationale": "Policy", "priority": 2},
            ],
        })

        critique_response = json.dumps({
            "redundancies": [],
            "gaps": [],
            "adjustments": [],
            "assessment": "Decomposition looks good, no changes needed",
        })

        async def mock_execute_llm_call(**kwargs):
            phase = kwargs.get("phase_name", "")
            if phase == "planning":
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content=planning_response,
                    provider_id="p", model_used="m", tokens_used=50, duration_ms=200.0,
                )
                return result
            elif phase == "planning_critique":
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content=critique_response,
                    provider_id="p", model_used="m", tokens_used=40, duration_ms=150.0,
                )
                return result
            else:
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content="Brief", provider_id="p",
                    model_used="m", tokens_used=20, duration_ms=80.0,
                )
                return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.planning.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.planning.finalize_phase",
        ):
            state = DeepResearchState(
                id="test-no-changes",
                original_query="AI in healthcare",
                phase=DeepResearchPhase.PLANNING,
                max_sub_queries=5,
            )
            result = await stub._execute_planning_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        assert result.success is True
        assert len(state.sub_queries) == 2  # Unchanged
        assert "planning_critique" in state.metadata
        critique_meta = state.metadata["planning_critique"]
        assert critique_meta["applied"] is False

    @pytest.mark.asyncio
    async def test_critique_uses_reflection_role(self):
        """Critique LLM call uses the 'reflection' role for cheap model routing."""
        stub = StubPlanning(enable_critique=True)

        planning_response = json.dumps({
            "research_brief": "AI",
            "sub_queries": [
                {"query": "Query A", "rationale": "r", "priority": 1},
            ],
        })

        captured_kwargs: list[dict] = []

        async def mock_execute_llm_call(**kwargs):
            captured_kwargs.append(dict(kwargs))
            phase = kwargs.get("phase_name", "")
            if phase == "planning_critique":
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True,
                    content=json.dumps({"redundancies": [], "gaps": [], "adjustments": [], "assessment": "OK"}),
                    provider_id="p", model_used="m", tokens_used=30, duration_ms=100.0,
                )
                return result
            else:
                result = MagicMock()
                result.result = WorkflowResult(
                    success=True, content=planning_response if phase == "planning" else "Brief",
                    provider_id="p", model_used="m", tokens_used=50, duration_ms=200.0,
                )
                return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.planning.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.planning.finalize_phase",
        ):
            state = DeepResearchState(
                id="test-role",
                original_query="AI",
                phase=DeepResearchPhase.PLANNING,
                max_sub_queries=5,
            )
            await stub._execute_planning_async(
                state=state, provider_id="test-provider", timeout=30.0
            )

        # Find the critique call
        critique_calls = [kw for kw in captured_kwargs if kw.get("phase_name") == "planning_critique"]
        assert len(critique_calls) == 1
        assert critique_calls[0]["role"] == "reflection"
        assert critique_calls[0]["temperature"] == 0.2
        assert critique_calls[0]["provider_id"] is None  # resolved by role
        assert critique_calls[0]["model"] is None  # resolved by role
