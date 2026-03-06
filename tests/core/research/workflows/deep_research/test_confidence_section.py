"""Tests for the LLM-interpreted Research Confidence section.

Covers:
- Context assembly (deterministic, no LLM)
- LLM interpretation with mock
- Deterministic fallback on LLM failure
- Edge cases (no verification data, empty details)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from tests.core.research.workflows.deep_research.conftest import make_test_state

from foundry_mcp.core.research.models.deep_research import (
    ClaimVerdict,
    ClaimVerificationResult,
    DeepResearchState,
)
from foundry_mcp.core.research.models.sources import SubQuery
from foundry_mcp.core.research.workflows.deep_research.phases._confidence_section import (
    build_confidence_context,
    generate_confidence_section,
    _build_deterministic_fallback,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_verification_result(
    *,
    supported: int = 5,
    partial: int = 2,
    unsupported: int = 3,
    contradicted: int = 1,
    corrections: int = 1,
    details: list[ClaimVerdict] | None = None,
) -> ClaimVerificationResult:
    """Build a ClaimVerificationResult with specified verdict counts."""
    if details is None:
        details = []
        for i in range(supported):
            details.append(
                ClaimVerdict(
                    claim=f"Supported claim {i}",
                    claim_type="quantitative",
                    verdict="SUPPORTED",
                    report_section="Findings",
                )
            )
        for i in range(partial):
            details.append(
                ClaimVerdict(
                    claim=f"Partial claim {i}",
                    claim_type="positive",
                    verdict="PARTIALLY_SUPPORTED",
                    report_section="Analysis",
                )
            )
        for i in range(unsupported):
            details.append(
                ClaimVerdict(
                    claim=f"Unsupported claim {i}",
                    claim_type="comparative",
                    verdict="UNSUPPORTED",
                    report_section="Recommendations",
                )
            )
        for i in range(contradicted):
            details.append(
                ClaimVerdict(
                    claim=f"Contradicted claim {i}",
                    claim_type="quantitative",
                    verdict="CONTRADICTED",
                    report_section="Findings",
                    correction_applied=True,
                    corrected_text=f"Corrected text {i}",
                )
            )

    total = supported + partial + unsupported + contradicted
    return ClaimVerificationResult(
        claims_extracted=total + 2,  # Some filtered out
        claims_filtered=total,
        claims_verified=total,
        claims_supported=supported,
        claims_partially_supported=partial,
        claims_unsupported=unsupported,
        claims_contradicted=contradicted,
        corrections_applied=corrections,
        details=details,
    )


def _make_state_with_verification(**kwargs: Any) -> DeepResearchState:
    """Create a test state with claim verification data."""
    state = make_test_state(iteration=2, max_iterations=3)
    state.claim_verification = _make_verification_result(**kwargs)
    state.fidelity_scores = [0.33, 0.42]
    state.metadata["_query_type"] = "comparison"
    return state


# ---------------------------------------------------------------------------
# Tests: build_confidence_context
# ---------------------------------------------------------------------------


class TestBuildConfidenceContext:
    """Step 1: deterministic context assembly."""

    def test_basic_context_structure(self):
        """Known verdict distribution produces expected context structure."""
        state = _make_state_with_verification()

        ctx = build_confidence_context(state)

        assert ctx is not None
        assert ctx["original_query"] == state.original_query
        assert ctx["query_type"] == "comparison"
        assert ctx["verdict_distribution"]["supported"] == 5
        assert ctx["verdict_distribution"]["unsupported"] == 3
        assert ctx["verdict_distribution"]["contradicted"] == 1
        assert ctx["verdict_distribution"]["total_verified"] == 11
        assert ctx["iteration_count"] == 2
        assert ctx["research_iteration_progress"] == "improving over 2 iterations"
        assert ctx["corrections_applied"] == 1
        assert len(ctx["correction_summaries"]) == 1

    def test_no_verification_returns_none(self):
        """None claim_verification handled gracefully — returns None."""
        state = make_test_state()
        state.claim_verification = None

        ctx = build_confidence_context(state)

        assert ctx is None

    def test_failed_subqueries_appear_in_context(self):
        """Failed sub-queries are included in context output."""
        state = _make_state_with_verification()
        state.sub_queries = [
            SubQuery(query="Transfer partner data for Turkish Airlines", status="failed"),
            SubQuery(query="Credit card annual fees comparison", status="completed"),
            SubQuery(query="Aeroplan redemption rates", status="failed"),
        ]

        ctx = build_confidence_context(state)

        assert ctx is not None
        assert len(ctx["failed_sub_queries"]) == 2
        assert ctx["failed_sub_queries"][0]["query"] == "Transfer partner data for Turkish Airlines"

    def test_unsupported_claims_grouped_by_type(self):
        """Unsupported claims are grouped by claim_type."""
        state = _make_state_with_verification(unsupported=3)

        ctx = build_confidence_context(state)

        assert ctx is not None
        assert "comparative" in ctx["unsupported_by_claim_type"]
        assert ctx["unsupported_by_claim_type"]["comparative"] == 3

    def test_per_section_breakdown(self):
        """Sections with unsupported claims are counted."""
        state = _make_state_with_verification(unsupported=3)

        ctx = build_confidence_context(state)

        assert ctx is not None
        assert "Recommendations" in ctx["sections_with_unsupported_claims"]
        assert ctx["sections_with_unsupported_claims"]["Recommendations"] == 3

    def test_source_count_included(self):
        """Source count from state is included."""
        state = _make_state_with_verification()
        # make_test_state has 0 sources by default
        assert build_confidence_context(state)["source_count"] == 0

    def test_empty_details_handled(self):
        """Empty details list produces zero counts."""
        state = make_test_state()
        state.claim_verification = ClaimVerificationResult(
            claims_extracted=0,
            claims_verified=0,
            details=[],
        )

        ctx = build_confidence_context(state)

        assert ctx is not None
        assert ctx["verdict_distribution"]["total_verified"] == 0
        assert ctx["corrections_applied"] == 0
        assert ctx["sections_with_unsupported_claims"] == {}


# ---------------------------------------------------------------------------
# Tests: generate_confidence_section
# ---------------------------------------------------------------------------


class TestGenerateConfidenceSection:
    """Step 2: LLM interpretation with fallback."""

    @pytest.mark.asyncio
    async def test_success_returns_markdown(self):
        """Mock LLM returns valid markdown; section starts with ## Research Confidence."""
        state = _make_state_with_verification()

        mock_llm = AsyncMock(
            return_value="## Research Confidence\n\nThis report was verified against 11 claims..."
        )

        section = await generate_confidence_section(state, mock_llm)

        assert section is not None
        assert section.startswith("## Research Confidence")
        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back(self):
        """Mock LLM raises; deterministic fallback is returned."""
        state = _make_state_with_verification()

        mock_llm = AsyncMock(side_effect=RuntimeError("LLM timeout"))

        section = await generate_confidence_section(state, mock_llm)

        assert section is not None
        assert section.startswith("## Research Confidence")
        assert "11 claims" in section  # total_verified
        assert "1 correction" in section

    @pytest.mark.asyncio
    async def test_skipped_when_no_verification(self):
        """No verification data → section is None."""
        state = make_test_state()
        state.claim_verification = None

        mock_llm = AsyncMock()

        section = await generate_confidence_section(state, mock_llm)

        assert section is None
        mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_empty_result_uses_fallback(self):
        """LLM returns empty string; fallback is used."""
        state = _make_state_with_verification()

        mock_llm = AsyncMock(return_value="")

        section = await generate_confidence_section(state, mock_llm)

        assert section is not None
        assert section.startswith("## Research Confidence")

    @pytest.mark.asyncio
    async def test_heading_prepended_if_missing(self):
        """If LLM response doesn't start with heading, it's prepended."""
        state = _make_state_with_verification()

        mock_llm = AsyncMock(
            return_value="This report covers a comparison of credit cards..."
        )

        section = await generate_confidence_section(state, mock_llm)

        assert section is not None
        assert section.startswith("## Research Confidence\n\nThis report covers")

    @pytest.mark.asyncio
    async def test_query_type_override(self):
        """query_type kwarg overrides state metadata."""
        state = _make_state_with_verification()

        captured_prompts: list[str] = []

        async def capturing_llm(system: str, user: str) -> str:
            captured_prompts.append(user)
            return "## Research Confidence\n\nOverridden."

        section = await generate_confidence_section(
            state, capturing_llm, query_type="literature_review"
        )

        assert section is not None
        user_data = json.loads(captured_prompts[0])
        assert user_data["query_type"] == "literature_review"

    @pytest.mark.asyncio
    async def test_integration_realistic_state(self):
        """End-to-end with realistic mock state."""
        from foundry_mcp.core.research.models.sources import ResearchSource

        state = _make_state_with_verification(
            supported=8, partial=3, unsupported=4, contradicted=1, corrections=1
        )
        state.fidelity_scores = [0.33, 0.42, 0.58]
        state.metadata["_query_type"] = "comparison"

        # Add some sources
        for i in range(10):
            state.sources.append(
                ResearchSource(
                    title=f"Source {i}",
                    url=f"https://example.com/source{i}",
                    citation_number=i + 1,
                )
            )

        # Add failed sub-queries
        state.sub_queries = [
            SubQuery(query="Turkish Airlines transfer data", status="failed"),
            SubQuery(query="Credit card comparison", status="completed"),
        ]

        mock_llm = AsyncMock(
            return_value=(
                "## Research Confidence\n\n"
                "This comparison report drew on 10 sources across 3 research cycles. "
                "Of 16 verifiable claims, 8 were fully supported by source material "
                "and 3 received partial support. Four claims — primarily comparative "
                "judgments like card rankings — are synthesis conclusions that no "
                "single source would state directly; this is expected for comparison "
                "queries. One factual error regarding annual fees was caught and "
                "corrected against source data. Data on Turkish Airlines transfer "
                "partners could not be retrieved from any source."
            )
        )

        section = await generate_confidence_section(state, mock_llm)

        assert section is not None
        assert "## Research Confidence" in section
        assert len(section) > 100  # Meaningful content


# ---------------------------------------------------------------------------
# Tests: deterministic fallback
# ---------------------------------------------------------------------------


class TestDeterministicFallback:
    """Test the fallback bullet-point summary."""

    def test_basic_fallback(self):
        """Fallback produces readable bullet points."""
        context = {
            "verdict_distribution": {
                "supported": 5,
                "partially_supported": 2,
                "unsupported": 3,
                "contradicted": 1,
                "total_verified": 11,
                "total_extracted": 13,
            },
            "corrections_applied": 1,
            "failed_sub_queries": [
                {"query": "Test query", "rationale": None},
            ],
            "iteration_count": 2,
            "source_count": 15,
        }

        result = _build_deterministic_fallback(context)

        assert result.startswith("## Research Confidence")
        assert "11 claims" in result
        assert "5 fully supported" in result
        assert "1 correction" in result
        assert "Test query" in result
        assert "2 research cycles" in result
        assert "15 sources" in result

    def test_fallback_no_verification(self):
        """Fallback with zero verified claims."""
        context = {
            "verdict_distribution": {"total_verified": 0},
            "corrections_applied": 0,
            "failed_sub_queries": [],
            "iteration_count": 1,
            "source_count": 0,
        }

        result = _build_deterministic_fallback(context)

        assert "No claims were verified" in result

    def test_fallback_single_iteration_omits_cycle_note(self):
        """Single iteration doesn't mention 'research cycles'."""
        context = {
            "verdict_distribution": {"total_verified": 5, "supported": 5,
                                      "partially_supported": 0, "unsupported": 0,
                                      "contradicted": 0},
            "corrections_applied": 0,
            "failed_sub_queries": [],
            "iteration_count": 1,
            "source_count": 3,
        }

        result = _build_deterministic_fallback(context)

        assert "research cycles" not in result
