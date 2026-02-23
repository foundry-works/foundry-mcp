"""Cross-phase integration test for deep research pipeline (PT.5).

Verifies state propagation and data consistency across all phases:
  CLARIFICATION → PLANNING → (simulated GATHERING) → ANALYSIS → SYNTHESIS

Uses mocked LLM providers but exercises real phase logic:
- Real prompt building (system + user)
- Real response parsing
- Real state mutation
- Real constraint/data propagation between phases
- Compressed findings flow from gathering → analysis → synthesis

Gathering is simulated (populated manually) because it uses search
providers rather than LLM calls.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
    TopicResearchResult,
)
from foundry_mcp.core.research.models.enums import ConfidenceLevel
from foundry_mcp.core.research.models.sources import SourceType, SubQuery
from foundry_mcp.core.research.workflows.deep_research._helpers import (
    ClarificationDecision,
)
from foundry_mcp.core.research.workflows.deep_research.phases._analysis_prompts import (
    AnalysisPromptsMixin,
)
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    LLMCallResult,
    StructuredLLMCallResult,
)
from foundry_mcp.core.research.workflows.deep_research.phases.clarification import (
    ClarificationPhaseMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases.planning import (
    PlanningPhaseMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases.synthesis import (
    SynthesisPhaseMixin,
)

# =============================================================================
# Stub workflow that composes phase mixins
# =============================================================================


class StubWorkflow(ClarificationPhaseMixin, PlanningPhaseMixin, SynthesisPhaseMixin):
    """Minimal composite mixin for cross-phase testing.

    Provides the common attributes and methods that each phase mixin
    expects on ``self`` at runtime.
    """

    def __init__(self) -> None:
        self.config = MagicMock()
        self.config.audit_verbosity = "minimal"
        self.memory = MagicMock()
        self.memory.save_deep_research = MagicMock()
        self._audit_events: list[tuple[str, dict]] = []

    def _write_audit_event(self, _state: Any, event: str, **kwargs: Any) -> None:
        self._audit_events.append((event, kwargs))

    def _check_cancellation(self, _state: Any) -> None:
        pass


# =============================================================================
# Mock LLM response helpers
# =============================================================================


def _make_llm_result(content: str, **overrides: Any) -> MagicMock:
    """Create a mock WorkflowResult with LLM response content."""
    result = MagicMock()
    result.content = content
    result.provider_id = overrides.get("provider_id", "test-provider")
    result.model_used = overrides.get("model_used", "test-model")
    result.tokens_used = overrides.get("tokens_used", 100)
    result.duration_ms = overrides.get("duration_ms", 500.0)
    result.input_tokens = overrides.get("input_tokens", 50)
    result.output_tokens = overrides.get("output_tokens", 50)
    result.cached_tokens = overrides.get("cached_tokens", 0)
    result.success = True
    return result


# =============================================================================
# Test: cross-phase state propagation
# =============================================================================


class TestCrossPhaseIntegration:
    """End-to-end integration test verifying state propagation across phases."""

    @pytest.fixture
    def workflow(self) -> StubWorkflow:
        return StubWorkflow()

    @pytest.fixture
    def state(self) -> DeepResearchState:
        return DeepResearchState(
            id="deepres-integration-test",
            original_query="Compare PostgreSQL vs MySQL for OLTP workloads in 2024",
            phase=DeepResearchPhase.CLARIFICATION,
            iteration=1,
            max_iterations=3,
        )

    @pytest.mark.asyncio
    async def test_full_pipeline_state_propagation(
        self,
        workflow: StubWorkflow,
        state: DeepResearchState,
    ) -> None:
        """Verify state flows correctly: clarification → planning → gathering → synthesis.

        This test exercises real phase logic with mocked LLM responses:
        1. Clarification sets constraints on state
        2. Planning reads constraints and creates sub-queries
        3. Gathering is simulated (sources added manually)
        4. Synthesis reads findings/sources and produces a report
        """
        # ------------------------------------------------------------------
        # Phase 1: CLARIFICATION
        # ------------------------------------------------------------------
        clarification_decision = ClarificationDecision(
            need_clarification=False,
            question="",
            verification="User wants a comparison of PostgreSQL and MySQL for high-write OLTP workloads, focusing on 2024 benchmarks and pricing.",
        )

        clarification_content = json.dumps(clarification_decision.to_dict())
        clarification_result = StructuredLLMCallResult(
            result=_make_llm_result(clarification_content),
            llm_call_duration_ms=500.0,
            parsed=clarification_decision,
            parse_retries=0,
        )

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.clarification.execute_structured_llm_call",
                return_value=clarification_result,
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.clarification.finalize_phase",
            ),
        ):
            result = await workflow._execute_clarification_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        assert result.success is True
        # Verification should be stored in clarification_constraints
        assert "verification" in state.clarification_constraints
        assert "PostgreSQL" in state.clarification_constraints["verification"]

        # ------------------------------------------------------------------
        # Phase 2: PLANNING — should see clarification constraints
        # ------------------------------------------------------------------
        state.phase = DeepResearchPhase.PLANNING

        planning_response = json.dumps({
            "research_brief": "Comparing PostgreSQL and MySQL for OLTP with focus on write performance, cost, and 2024 benchmarks.",
            "sub_queries": [
                {
                    "query": "PostgreSQL OLTP write performance benchmarks 2024",
                    "rationale": "Latest write-heavy workload benchmarks",
                    "priority": 1,
                },
                {
                    "query": "MySQL 8.0 OLTP performance comparison",
                    "rationale": "MySQL-side performance data",
                    "priority": 1,
                },
                {
                    "query": "PostgreSQL vs MySQL cloud hosting cost 2024",
                    "rationale": "Cost comparison for managed database services",
                    "priority": 2,
                },
            ],
        })

        planning_llm_result = _make_llm_result(planning_response)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.planning.execute_llm_call",
                return_value=LLMCallResult(result=planning_llm_result, llm_call_duration_ms=500.0),
            ) as mock_planning_llm,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.planning.finalize_phase",
            ),
        ):
            result = await workflow._execute_planning_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        assert result.success is True

        # Sub-queries should be populated
        assert len(state.sub_queries) == 3
        assert "PostgreSQL" in state.sub_queries[0].query

        # Research brief should be set
        assert state.research_brief is not None
        assert "OLTP" in state.research_brief

        # Clarification constraints should have been included in the planning prompt
        planning_user_prompt = mock_planning_llm.call_args.kwargs["user_prompt"]
        assert "Clarification constraints" in planning_user_prompt
        assert "verification" in planning_user_prompt

        # ------------------------------------------------------------------
        # Phase 3: GATHERING (simulated — add sources manually)
        # ------------------------------------------------------------------
        state.phase = DeepResearchPhase.GATHERING

        # Simulate source gathering — add sources with citation numbers
        source1 = state.add_source(
            title="PostgreSQL 16 OLTP Benchmark Results",
            url="https://example.com/pg16-benchmark",
            source_type=SourceType.WEB,
            snippet="PostgreSQL 16 achieves 150K TPS on write-heavy OLTP workloads.",
            sub_query_id=state.sub_queries[0].id,
        )
        source2 = state.add_source(
            title="MySQL 8.0 vs PostgreSQL: 2024 Performance Comparison",
            url="https://example.com/mysql-vs-pg",
            source_type=SourceType.WEB,
            snippet="MySQL 8.0 shows 120K TPS on similar OLTP benchmarks.",
            sub_query_id=state.sub_queries[1].id,
        )
        source3 = state.add_source(
            title="Cloud Database Pricing Comparison 2024",
            url="https://example.com/cloud-pricing",
            source_type=SourceType.WEB,
            snippet="PostgreSQL managed hosting averages $150/mo; MySQL $120/mo for equivalent workloads.",
            sub_query_id=state.sub_queries[2].id,
        )

        # Mark sub-queries as completed
        for sq in state.sub_queries:
            sq.status = "completed"

        assert len(state.sources) == 3
        assert source1.citation_number == 1
        assert source2.citation_number == 2
        assert source3.citation_number == 3

        # ------------------------------------------------------------------
        # Phase 4: ANALYSIS (simulated — add findings manually)
        # ------------------------------------------------------------------
        state.phase = DeepResearchPhase.ANALYSIS

        finding1 = state.add_finding(
            content="PostgreSQL 16 achieves 25% higher TPS than MySQL 8.0 on write-heavy OLTP workloads.",
            confidence=ConfidenceLevel.HIGH,
            category="Performance",
            source_ids=[source1.id, source2.id],
        )
        state.add_finding(
            content="MySQL managed hosting is approximately 20% cheaper than PostgreSQL for equivalent configurations.",
            confidence=ConfidenceLevel.MEDIUM,
            category="Cost",
            source_ids=[source3.id],
        )

        assert len(state.findings) == 2

        # ------------------------------------------------------------------
        # Phase 5: SYNTHESIS — should see all prior state
        # ------------------------------------------------------------------
        state.phase = DeepResearchPhase.SYNTHESIS

        # The synthesis LLM will receive all findings and sources in its prompt
        synthesis_response = """# Research Report: PostgreSQL vs MySQL for OLTP Workloads

## Executive Summary

Based on 2024 benchmarks, PostgreSQL 16 outperforms MySQL 8.0 by approximately 25% in write-heavy OLTP workloads. However, MySQL offers a roughly 20% cost advantage in managed hosting.

## Key Findings

### Performance
- PostgreSQL 16 achieves 25% higher TPS than MySQL 8.0 on write-heavy OLTP workloads [1], [2].

### Cost
- MySQL managed hosting is approximately 20% cheaper than PostgreSQL [3].

## Conclusions

For write-intensive OLTP applications, PostgreSQL offers superior performance at a modest cost premium.
"""

        synthesis_llm_result = _make_llm_result(synthesis_response)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.synthesis.execute_llm_call",
                return_value=LLMCallResult(result=synthesis_llm_result, llm_call_duration_ms=800.0),
            ) as mock_synthesis_llm,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.synthesis.finalize_phase",
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.synthesis.allocate_synthesis_budget",
            ) as mock_budget,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.synthesis.final_fit_validate",
                return_value=(True, {}, "system prompt", "user prompt"),
            ),
        ):
            # Configure budget allocation mock
            mock_allocation = MagicMock()
            mock_allocation.dropped_ids = []
            mock_allocation.items = []
            mock_allocation.fidelity = 1.0
            mock_allocation.to_dict.return_value = {"fidelity": 1.0}
            mock_budget.return_value = mock_allocation

            result = await workflow._execute_synthesis_async(
                state=state,
                provider_id="test-provider",
                timeout=120.0,
            )

        assert result.success is True

        # Report should be stored in state
        assert state.report is not None
        assert "PostgreSQL" in state.report
        assert "MySQL" in state.report

        # Synthesis LLM call should have been made
        assert mock_synthesis_llm.call_count == 1

        # ------------------------------------------------------------------
        # Cross-phase consistency checks
        # ------------------------------------------------------------------
        # All sub-queries should be completed
        assert all(sq.status == "completed" for sq in state.sub_queries)

        # Sources should still be accessible
        assert len(state.sources) == 3
        assert state.get_source(source1.id) is not None

        # Findings should still be accessible
        assert len(state.findings) == 2

        # Source-finding linkage: finding1 references source1 and source2
        assert source1.id in finding1.source_ids
        assert source2.id in finding1.source_ids

        # Citation numbers are consistent
        assert state.sources[0].citation_number == 1
        assert state.sources[1].citation_number == 2
        assert state.sources[2].citation_number == 3

        # Token tracking: total_tokens_used should reflect LLM calls
        # (only direct state.total_tokens_used increments from execute_llm_call)
        # Since we mocked execute_llm_call, the lifecycle code didn't run,
        # but we can verify state is in a consistent final configuration
        assert state.original_query == "Compare PostgreSQL vs MySQL for OLTP workloads in 2024"
        assert state.research_brief is not None
        assert state.report is not None

    @pytest.mark.asyncio
    async def test_clarification_constraints_propagate_to_planning(
        self,
        workflow: StubWorkflow,
    ) -> None:
        """Focused test: constraints from clarification appear in planning prompt."""
        state = DeepResearchState(
            id="deepres-propagation-test",
            original_query="How does caching work?",
            phase=DeepResearchPhase.PLANNING,
            iteration=1,
            max_iterations=3,
        )

        # Simulate clarification having set constraints
        state.clarification_constraints = {
            "verification": "User wants to understand web caching mechanisms.",
            "scope": "web application caching",
            "depth": "comprehensive",
        }

        planning_response = json.dumps({
            "research_brief": "Research on web application caching mechanisms.",
            "sub_queries": [
                {"query": "HTTP caching headers and CDN strategies", "rationale": "Core caching mechanism", "priority": 1},
            ],
        })

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.planning.execute_llm_call",
                return_value=LLMCallResult(
                    result=_make_llm_result(planning_response),
                    llm_call_duration_ms=500.0,
                ),
            ) as mock_llm,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.planning.finalize_phase",
            ),
        ):
            result = await workflow._execute_planning_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        assert result.success is True

        # Constraints should appear in the user prompt sent to the LLM
        user_prompt = mock_llm.call_args.kwargs["user_prompt"]
        assert "web application caching" in user_prompt
        assert "comprehensive" in user_prompt
        assert "Clarification constraints" in user_prompt

    @pytest.mark.asyncio
    async def test_no_constraints_no_constraint_section_in_planning(
        self,
        workflow: StubWorkflow,
    ) -> None:
        """When clarification had no constraints, planning prompt omits constraint section."""
        state = DeepResearchState(
            id="deepres-no-constraints",
            original_query="Compare PostgreSQL vs MySQL",
            phase=DeepResearchPhase.PLANNING,
            iteration=1,
            max_iterations=3,
        )
        state.clarification_constraints = {}

        planning_response = json.dumps({
            "research_brief": "Compare the two databases.",
            "sub_queries": [
                {"query": "PostgreSQL vs MySQL comparison", "rationale": "Main query", "priority": 1},
            ],
        })

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.planning.execute_llm_call",
                return_value=LLMCallResult(
                    result=_make_llm_result(planning_response),
                    llm_call_duration_ms=500.0,
                ),
            ) as mock_llm,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.planning.finalize_phase",
            ),
        ):
            await workflow._execute_planning_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        user_prompt = mock_llm.call_args.kwargs["user_prompt"]
        assert "Clarification constraints" not in user_prompt

    @pytest.mark.asyncio
    async def test_synthesis_sees_all_findings_and_sources(
        self,
        workflow: StubWorkflow,
    ) -> None:
        """Synthesis prompt includes findings from analysis and sources from gathering."""
        state = DeepResearchState(
            id="deepres-synthesis-sees-all",
            original_query="Test query",
            phase=DeepResearchPhase.SYNTHESIS,
            iteration=1,
            max_iterations=3,
            research_brief="Test research brief",
        )

        # Add sources
        src = state.add_source(
            title="Test Source",
            url="https://example.com",
            source_type=SourceType.WEB,
            snippet="Test snippet content",
        )

        # Add findings referencing the source
        state.add_finding(
            content="Test finding with important data",
            confidence=ConfidenceLevel.HIGH,
            category="TestCategory",
            source_ids=[src.id],
        )

        # Build the synthesis prompt (not calling the full phase)
        user_prompt = workflow._build_synthesis_user_prompt(state)

        # Findings should appear in the prompt
        assert "Test finding with important data" in user_prompt
        assert "TestCategory" in user_prompt
        assert "HIGH" in user_prompt

        # Source should appear in the prompt
        assert "Test Source" in user_prompt
        assert "[1]" in user_prompt

        # Research brief should appear
        assert "Test research brief" in user_prompt

    @pytest.mark.asyncio
    async def test_empty_findings_generates_empty_report(
        self,
        workflow: StubWorkflow,
    ) -> None:
        """Synthesis with no findings produces a minimal report without LLM call."""
        state = DeepResearchState(
            id="deepres-empty-findings",
            original_query="Some query with no findings",
            phase=DeepResearchPhase.SYNTHESIS,
            iteration=1,
            max_iterations=3,
            research_brief="Test brief",
        )

        # No findings added — synthesis should produce an empty report
        result = await workflow._execute_synthesis_async(
            state=state,
            provider_id="test-provider",
            timeout=60.0,
        )

        assert result.success is True
        assert state.report is not None
        assert "no extractable findings" in state.report.lower() or "did not yield" in state.report.lower()
        assert result.metadata.get("empty_report") is True


# =============================================================================
# Test: compressed findings flow through analysis/synthesis
# =============================================================================


class StubAnalysisWorkflow(AnalysisPromptsMixin, SynthesisPhaseMixin):
    """Minimal composite for testing compressed findings in analysis/synthesis."""

    def __init__(self) -> None:
        self.config = MagicMock()
        self.config.audit_verbosity = "minimal"
        self.memory = MagicMock()
        self.memory.save_deep_research = MagicMock()
        self._audit_events: list[tuple[str, dict]] = []

    def _write_audit_event(self, _state: Any, event: str, **kwargs: Any) -> None:
        self._audit_events.append((event, kwargs))

    def _check_cancellation(self, _state: Any) -> None:
        pass


class TestCompressedFindingsCrossPhase:
    """Verify compressed findings (new format) flow correctly through analysis/synthesis."""

    @pytest.fixture
    def workflow(self) -> StubAnalysisWorkflow:
        return StubAnalysisWorkflow()

    @pytest.fixture
    def state_with_compressed_findings(self) -> DeepResearchState:
        """Build state simulating gathering + compression output."""
        state = DeepResearchState(
            id="deepres-compressed-flow",
            original_query="Compare cloud storage providers for enterprise backup",
            phase=DeepResearchPhase.ANALYSIS,
            iteration=1,
            max_iterations=3,
            research_brief="Research cloud storage providers focusing on enterprise backup pricing, reliability, and compliance.",
        )

        # Sub-queries from planning
        sq0 = SubQuery(
            id="sq-pricing",
            query="Enterprise cloud storage backup pricing comparison 2024",
            rationale="Cost analysis across providers",
            priority=1,
        )
        sq1 = SubQuery(
            id="sq-reliability",
            query="Cloud storage reliability SLA comparison enterprise",
            rationale="Uptime and durability guarantees",
            priority=1,
        )
        state.sub_queries.extend([sq0, sq1])

        # Sources from gathering
        src_aws = state.add_source(
            title="AWS S3 Glacier Enterprise Pricing Guide",
            url="https://example.com/aws-pricing",
            source_type=SourceType.WEB,
            snippet="AWS S3 Glacier offers $0.004/GB/month for archive storage.",
            sub_query_id="sq-pricing",
        )
        src_azure = state.add_source(
            title="Azure Blob Storage Enterprise Tier",
            url="https://example.com/azure-pricing",
            source_type=SourceType.WEB,
            snippet="Azure Archive Storage costs $0.002/GB/month with cool tier at $0.01/GB/month.",
            sub_query_id="sq-pricing",
        )
        src_aws_sla = state.add_source(
            title="AWS S3 SLA and Durability Guarantees",
            url="https://example.com/aws-sla",
            source_type=SourceType.WEB,
            snippet="AWS S3 offers 99.999999999% durability and 99.99% availability SLA.",
            sub_query_id="sq-reliability",
        )
        src_azure_sla = state.add_source(
            title="Azure Storage Reliability Whitepaper",
            url="https://example.com/azure-sla",
            source_type=SourceType.WEB,
            snippet="Azure LRS provides 99.999999999% durability; ZRS adds zone redundancy.",
            sub_query_id="sq-reliability",
        )

        # Topic results WITH compressed findings (new format from Phase 1 compression)
        tr_pricing = TopicResearchResult(
            sub_query_id="sq-pricing",
            searches_performed=2,
            sources_found=2,
            source_ids=[src_aws.id, src_azure.id],
            reflection_notes=[
                "Found general pricing data; need tier-specific comparisons.",
                "Refined search yielded archive-tier pricing details.",
            ],
            refined_queries=["enterprise archive storage pricing per GB 2024"],
            compressed_findings=(
                "## Queries Made\n"
                "- Enterprise cloud storage backup pricing comparison 2024\n"
                "- Enterprise archive storage pricing per GB 2024\n\n"
                "## Comprehensive Findings\n"
                "AWS S3 Glacier offers archive storage at $0.004/GB/month [1]. "
                "Azure Archive Storage is cheaper at $0.002/GB/month, with "
                "cool tier at $0.01/GB/month [2]. Both providers offer volume "
                "discounts for enterprise contracts exceeding 500TB [1][2].\n\n"
                "## Source List\n"
                "- [1] AWS S3 Glacier Enterprise Pricing Guide — https://example.com/aws-pricing\n"
                "- [2] Azure Blob Storage Enterprise Tier — https://example.com/azure-pricing\n"
            ),
        )

        tr_reliability = TopicResearchResult(
            sub_query_id="sq-reliability",
            searches_performed=1,
            sources_found=2,
            source_ids=[src_aws_sla.id, src_azure_sla.id],
            reflection_notes=["Both providers have similar durability guarantees."],
            early_completion=True,
            completion_rationale="Sufficient SLA data collected from both providers.",
            compressed_findings=(
                "## Queries Made\n"
                "- Cloud storage reliability SLA comparison enterprise\n\n"
                "## Comprehensive Findings\n"
                "AWS S3 provides 99.999999999% (11 nines) durability and "
                "99.99% availability SLA [1]. Azure LRS provides equivalent "
                "11-nines durability, and ZRS adds zone redundancy for "
                "higher availability [2]. Both providers meet SOC 2 and "
                "HIPAA compliance requirements [1][2].\n\n"
                "## Source List\n"
                "- [1] AWS S3 SLA and Durability Guarantees — https://example.com/aws-sla\n"
                "- [2] Azure Storage Reliability Whitepaper — https://example.com/azure-sla\n"
            ),
        )

        state.topic_research_results.extend([tr_pricing, tr_reliability])
        return state

    def test_analysis_prompt_includes_compressed_findings_format(
        self,
        workflow: StubAnalysisWorkflow,
        state_with_compressed_findings: DeepResearchState,
    ) -> None:
        """Analysis prompt uses compressed findings with new format sections."""
        prompt = workflow._build_analysis_user_prompt(state_with_compressed_findings)

        # Should use compressed findings path
        assert "Per-Topic Research Summaries" in prompt
        assert "Sources to Analyze:" not in prompt

        # Should contain topic headers
        assert "Topic 1:" in prompt
        assert "Topic 2:" in prompt

        # Should contain compressed findings content (from both topics)
        assert "## Queries Made" in prompt
        assert "## Comprehensive Findings" in prompt
        assert "## Source List" in prompt

        # Pricing topic content
        assert "$0.004/GB/month" in prompt
        assert "$0.002/GB/month" in prompt

        # Reliability topic content
        assert "99.999999999%" in prompt
        assert "SOC 2" in prompt

    def test_analysis_prompt_preserves_inline_citations(
        self,
        workflow: StubAnalysisWorkflow,
        state_with_compressed_findings: DeepResearchState,
    ) -> None:
        """Compressed findings' inline citations [1], [2] are preserved in analysis prompt."""
        prompt = workflow._build_analysis_user_prompt(state_with_compressed_findings)

        # Citation references from compressed findings should be preserved
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[1][2]" in prompt

    def test_analysis_prompt_includes_source_id_mapping(
        self,
        workflow: StubAnalysisWorkflow,
        state_with_compressed_findings: DeepResearchState,
    ) -> None:
        """Analysis prompt includes source ID → citation mapping for each topic."""
        prompt = workflow._build_analysis_user_prompt(state_with_compressed_findings)

        # Source ID mapping section should exist
        assert "Source ID mapping" in prompt

        # All source IDs should be referenced
        for src in state_with_compressed_findings.sources:
            assert src.id in prompt

    def test_analysis_prompt_contains_research_query_and_brief(
        self,
        workflow: StubAnalysisWorkflow,
        state_with_compressed_findings: DeepResearchState,
    ) -> None:
        """Analysis prompt includes the original query and research brief."""
        prompt = workflow._build_analysis_user_prompt(state_with_compressed_findings)

        assert "Compare cloud storage providers for enterprise backup" in prompt
        assert "enterprise backup pricing, reliability, and compliance" in prompt

    @pytest.mark.asyncio
    async def test_synthesis_sees_findings_from_compressed_analysis(
        self,
        workflow: StubAnalysisWorkflow,
        state_with_compressed_findings: DeepResearchState,
    ) -> None:
        """Synthesis prompt includes findings that were extracted from compressed input."""
        state = state_with_compressed_findings
        state.phase = DeepResearchPhase.SYNTHESIS

        # Simulate analysis having extracted findings from compressed input
        state.add_finding(
            content="Azure Archive Storage is 50% cheaper than AWS S3 Glacier at $0.002/GB/month vs $0.004/GB/month.",
            confidence=ConfidenceLevel.HIGH,
            category="Pricing",
            source_ids=[state.sources[0].id, state.sources[1].id],
        )
        state.add_finding(
            content="Both AWS and Azure provide 11-nines durability and meet SOC 2 / HIPAA compliance.",
            confidence=ConfidenceLevel.HIGH,
            category="Reliability",
            source_ids=[state.sources[2].id, state.sources[3].id],
        )

        # Build synthesis prompt
        synthesis_prompt = workflow._build_synthesis_user_prompt(state)

        # Findings should appear in synthesis
        assert "$0.002/GB/month" in synthesis_prompt
        assert "11-nines durability" in synthesis_prompt
        assert "Pricing" in synthesis_prompt
        assert "Reliability" in synthesis_prompt

        # Sources should be referenced
        assert "[1]" in synthesis_prompt
        assert "[2]" in synthesis_prompt

        # Research brief should be included
        assert state.research_brief is not None
        assert state.research_brief in synthesis_prompt


# =============================================================================
# Test: brief refinement (Phase 2 alignment — steps 2.6 & 2.7)
# =============================================================================


class TestBriefRefinement:
    """Tests for the brief-refinement step added to the planning phase.

    Verifies that an ambiguous query is refined into a more specific brief
    and that sub-query decomposition operates on the refined brief.
    """

    @pytest.fixture
    def workflow(self) -> StubWorkflow:
        return StubWorkflow()

    @pytest.mark.asyncio
    async def test_ambiguous_query_produces_specific_brief(
        self,
        workflow: StubWorkflow,
    ) -> None:
        """Brief refinement should store a more specific brief than the raw query."""
        state = DeepResearchState(
            id="deepres-brief-specificity",
            original_query="tell me about databases",  # Deliberately vague
            phase=DeepResearchPhase.PLANNING,
            iteration=1,
            max_iterations=3,
        )

        # Mock: the brief-refinement LLM returns a specific brief
        refined_brief = (
            "An investigation into the major categories of modern database "
            "systems — relational (PostgreSQL, MySQL), document-oriented "
            "(MongoDB), graph (Neo4j), and time-series (TimescaleDB) — "
            "examining their architectural trade-offs, typical use cases, "
            "and current market adoption."
        )
        # Mock: the planning LLM returns sub-queries
        planning_response = json.dumps({
            "research_brief": "Databases overview",  # Should be ignored
            "sub_queries": [
                {
                    "query": "relational vs document database trade-offs",
                    "rationale": "Core architectural comparison",
                    "priority": 1,
                },
            ],
        })

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.planning.execute_llm_call",
                side_effect=[
                    LLMCallResult(
                        result=_make_llm_result(refined_brief),
                        llm_call_duration_ms=200.0,
                    ),
                    LLMCallResult(
                        result=_make_llm_result(planning_response),
                        llm_call_duration_ms=500.0,
                    ),
                ],
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.planning.finalize_phase",
            ),
        ):
            result = await workflow._execute_planning_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        assert result.success is True

        # The refined brief — not the raw query — should be stored
        assert state.research_brief == refined_brief
        assert len(state.research_brief) > len(state.original_query)

        # The brief should contain specifics absent from the vague query
        assert "PostgreSQL" in state.research_brief
        assert "MongoDB" in state.research_brief

        # The planning response's research_brief should NOT have overwritten it
        assert state.research_brief != "Databases overview"

    @pytest.mark.asyncio
    async def test_sub_queries_grounded_in_refined_brief(
        self,
        workflow: StubWorkflow,
    ) -> None:
        """Sub-query decomposition prompt should use the refined brief, not the raw query."""
        state = DeepResearchState(
            id="deepres-brief-wiring",
            original_query="tell me about caching",  # Vague
            phase=DeepResearchPhase.PLANNING,
            iteration=1,
            max_iterations=3,
        )

        refined_brief = (
            "An investigation into web application caching strategies "
            "including HTTP cache headers, CDN edge caching, and "
            "in-memory stores such as Redis and Memcached."
        )
        planning_response = json.dumps({
            "research_brief": "Caching overview",
            "sub_queries": [
                {
                    "query": "HTTP caching headers best practices",
                    "rationale": "Core mechanism",
                    "priority": 1,
                },
                {
                    "query": "Redis vs Memcached comparison 2024",
                    "rationale": "In-memory store trade-offs",
                    "priority": 2,
                },
            ],
        })

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.planning.execute_llm_call",
                side_effect=[
                    LLMCallResult(
                        result=_make_llm_result(refined_brief),
                        llm_call_duration_ms=200.0,
                    ),
                    LLMCallResult(
                        result=_make_llm_result(planning_response),
                        llm_call_duration_ms=500.0,
                    ),
                ],
            ) as mock_llm,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.planning.finalize_phase",
            ),
        ):
            result = await workflow._execute_planning_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        assert result.success is True

        # Two LLM calls: brief refinement, then planning decomposition
        assert mock_llm.call_count == 2

        # The brief-refinement call (first) should use the "summarization" role
        brief_call = mock_llm.call_args_list[0]
        assert brief_call.kwargs["role"] == "summarization"

        # The planning call (second) should contain the refined brief
        planning_call = mock_llm.call_args_list[1]
        planning_user_prompt = planning_call.kwargs["user_prompt"]

        # Refined brief content should appear in the planning prompt
        assert "web application caching strategies" in planning_user_prompt
        assert "Redis" in planning_user_prompt

        # The raw vague query should NOT be the research query input
        assert not planning_user_prompt.startswith("Research Query: tell me about caching")

    @pytest.mark.asyncio
    async def test_brief_refinement_failure_falls_back_to_raw_query(
        self,
        workflow: StubWorkflow,
    ) -> None:
        """When brief refinement fails, planning should proceed with the raw query."""
        state = DeepResearchState(
            id="deepres-brief-fallback",
            original_query="Compare React and Vue",
            phase=DeepResearchPhase.PLANNING,
            iteration=1,
            max_iterations=3,
        )

        planning_response = json.dumps({
            "research_brief": "React vs Vue comparison",
            "sub_queries": [
                {
                    "query": "React vs Vue performance 2024",
                    "rationale": "Performance comparison",
                    "priority": 1,
                },
            ],
        })

        # Brief refinement returns a WorkflowResult (error), planning succeeds
        brief_error = WorkflowResult(
            success=False,
            content="",
            error="Provider timeout",
        )

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.planning.execute_llm_call",
                side_effect=[
                    brief_error,
                    LLMCallResult(
                        result=_make_llm_result(planning_response),
                        llm_call_duration_ms=500.0,
                    ),
                ],
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.planning.finalize_phase",
            ),
        ):
            result = await workflow._execute_planning_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        assert result.success is True

        # Falls back to original query as the brief
        assert state.research_brief == state.original_query

        # Sub-queries should still be extracted from the planning response
        assert len(state.sub_queries) == 1
        assert "React vs Vue" in state.sub_queries[0].query
