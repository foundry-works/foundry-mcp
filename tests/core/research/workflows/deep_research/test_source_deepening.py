"""Tests for source-deepening verification strategy.

Covers:
- UNSUPPORTED claim classification (inferential / deepen_window / deepen_extract / widen)
- Expanded-window re-verification (verdict upgrades)
- Thin source re-extraction
- Integration with build_gap_queries (inferential exclusion)
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from foundry_mcp.core.research.models.deep_research import (
    ClaimVerdict,
    ClaimVerificationResult,
)
from foundry_mcp.core.research.models.sources import ResearchSource
from foundry_mcp.core.research.workflows.deep_research.phases._source_deepening import (
    DeepClassification,
    classify_unsupported_claims,
    deepen_thin_sources,
    reverify_with_expanded_window,
)
from foundry_mcp.core.research.workflows.deep_research.phases.claim_verification import (
    build_gap_queries,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source(
    citation_number: int,
    *,
    raw_content: str | None = None,
    content: str | None = None,
    snippet: str | None = None,
    url: str | None = None,
    title: str = "Test Source",
) -> ResearchSource:
    """Create a ResearchSource for testing."""
    return ResearchSource(
        title=f"{title} [{citation_number}]",
        citation_number=citation_number,
        raw_content=raw_content,
        content=content,
        snippet=snippet,
        url=url or f"https://example.com/source{citation_number}",
    )


def _make_claim(
    claim_text: str,
    *,
    claim_type: str = "quantitative",
    verdict: str = "UNSUPPORTED",
    cited_sources: list[int] | None = None,
    report_section: str = "Findings",
) -> ClaimVerdict:
    """Create a ClaimVerdict for testing."""
    return ClaimVerdict(
        claim=claim_text,
        claim_type=claim_type,
        verdict=verdict,
        cited_sources=cited_sources or [],
        report_section=report_section,
    )


def _make_verification_result(
    claims: list[ClaimVerdict],
) -> ClaimVerificationResult:
    """Build a ClaimVerificationResult from a list of claims."""
    supported = sum(1 for c in claims if c.verdict == "SUPPORTED")
    unsupported = sum(1 for c in claims if c.verdict == "UNSUPPORTED")
    contradicted = sum(1 for c in claims if c.verdict == "CONTRADICTED")
    partial = sum(1 for c in claims if c.verdict == "PARTIALLY_SUPPORTED")

    return ClaimVerificationResult(
        claims_extracted=len(claims),
        claims_filtered=len(claims),
        claims_verified=len(claims),
        claims_supported=supported,
        claims_unsupported=unsupported,
        claims_contradicted=contradicted,
        claims_partially_supported=partial,
        details=claims,
    )


# ---------------------------------------------------------------------------
# Tests: classify_unsupported_claims
# ---------------------------------------------------------------------------


class TestClassifyUnsupportedClaims:
    """Step 1: Classification of UNSUPPORTED claims."""

    def test_classify_inferential_claims(self):
        """Comparative claims with recommendation language → inferential."""
        claims = [
            _make_claim(
                "Citi Double Cash is the best card for dining-heavy spenders",
                claim_type="comparative",
                cited_sources=[1],
            ),
            _make_claim(
                "We recommend the Chase Sapphire Preferred overall",
                claim_type="positive",
                cited_sources=[2],
            ),
        ]
        vr = _make_verification_result(claims)
        citation_map = {
            1: _make_source(1, raw_content="x" * 20000),
            2: _make_source(2, raw_content="x" * 20000),
        }

        result = classify_unsupported_claims(vr, citation_map)

        assert len(result.inferential) == 2
        assert len(result.deepen_window) == 0
        assert len(result.deepen_extract) == 0
        assert len(result.widen) == 0

    def test_classify_deepen_window(self):
        """Quantitative claim with rich raw_content → deepen_window."""
        claims = [
            _make_claim(
                "The annual fee is $95 per year",
                claim_type="quantitative",
                cited_sources=[1],
            ),
        ]
        vr = _make_verification_result(claims)
        citation_map = {
            1: _make_source(1, raw_content="x" * 20000),
        }

        result = classify_unsupported_claims(vr, citation_map)

        assert len(result.deepen_window) == 1
        assert len(result.inferential) == 0

    def test_classify_deepen_extract(self):
        """Quantitative claim with thin source → deepen_extract."""
        claims = [
            _make_claim(
                "Aeroplan awards 60,000 points for Europe flights",
                claim_type="quantitative",
                cited_sources=[1],
            ),
        ]
        vr = _make_verification_result(claims)
        citation_map = {
            1: _make_source(1, raw_content="Short snippet", snippet="Short snippet"),
        }

        result = classify_unsupported_claims(vr, citation_map)

        assert len(result.deepen_extract) == 1
        assert len(result.deepen_window) == 0

    def test_classify_widen(self):
        """Factual claim with no existing source → widen."""
        claims = [
            _make_claim(
                "Transfer processing takes 24-48 hours",
                claim_type="quantitative",
                cited_sources=[99],  # Not in citation_map
            ),
        ]
        vr = _make_verification_result(claims)
        citation_map: dict[int, ResearchSource] = {}

        result = classify_unsupported_claims(vr, citation_map)

        assert len(result.widen) == 1

    def test_classify_mixed(self):
        """Mixed claims are classified into correct buckets."""
        claims = [
            # Inferential
            _make_claim(
                "Chase is better for travel rewards",
                claim_type="comparative",
                cited_sources=[1],
            ),
            # Deepen window (quantitative + rich source)
            _make_claim(
                "The signup bonus is 80,000 points",
                claim_type="quantitative",
                cited_sources=[2],
            ),
            # Deepen extract (quantitative + thin source)
            _make_claim(
                "Annual fee waived first year",
                claim_type="negative",
                cited_sources=[3],
            ),
            # Widen (no source)
            _make_claim(
                "Processing time is 3-5 days",
                claim_type="quantitative",
                cited_sources=[],
            ),
            # SUPPORTED claim should be skipped entirely
            _make_claim(
                "Card was launched in 2020",
                claim_type="quantitative",
                verdict="SUPPORTED",
                cited_sources=[1],
            ),
        ]
        vr = _make_verification_result(claims)
        citation_map = {
            1: _make_source(1, raw_content="x" * 20000),
            2: _make_source(2, raw_content="x" * 20000),
            3: _make_source(3, raw_content="tiny"),
        }

        result = classify_unsupported_claims(vr, citation_map)

        assert len(result.inferential) == 1
        assert len(result.deepen_window) == 1
        assert len(result.deepen_extract) == 1
        assert len(result.widen) == 1

    def test_comparative_without_recommendation_language_still_inferential(self):
        """All comparative claims are inferential — they are cross-item judgments."""
        claims = [
            _make_claim(
                "Card A has a higher annual fee than Card B",
                claim_type="comparative",
                cited_sources=[1],
            ),
        ]
        vr = _make_verification_result(claims)
        citation_map = {
            1: _make_source(1, raw_content="x" * 20000),
        }

        result = classify_unsupported_claims(vr, citation_map)

        assert len(result.inferential) == 1
        assert len(result.deepen_window) == 0


# ---------------------------------------------------------------------------
# Tests: reverify_with_expanded_window
# ---------------------------------------------------------------------------


class TestReverifyExpandedWindow:
    """Step 2: Expanded-window re-verification."""

    @pytest.mark.asyncio
    async def test_upgrades_verdict(self):
        """Claim goes from UNSUPPORTED to SUPPORTED with larger window."""
        claim = _make_claim(
            "The annual fee is $95",
            claim_type="quantitative",
            cited_sources=[1],
        )
        citation_map = {
            1: _make_source(
                1,
                raw_content="The annual fee for this card is $95 per year. " + "x" * 20000,
            ),
        }

        mock_llm = AsyncMock(
            return_value='{"verdict": "SUPPORTED", "evidence_quote": "annual fee is $95", "explanation": "Found in source"}'
        )

        result = await reverify_with_expanded_window(
            [claim], citation_map, mock_llm
        )

        assert len(result) == 1
        assert result[0].verdict == "SUPPORTED"
        assert "Upgraded from UNSUPPORTED" in (result[0].explanation or "")
        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_unchanged_verdict(self):
        """Claim stays UNSUPPORTED with larger window (evidence genuinely absent)."""
        claim = _make_claim(
            "The annual fee is $95",
            claim_type="quantitative",
            cited_sources=[1],
        )
        citation_map = {
            1: _make_source(
                1,
                raw_content="This card has many benefits. " + "x" * 20000,
            ),
        }

        mock_llm = AsyncMock(
            return_value='{"verdict": "UNSUPPORTED", "evidence_quote": null, "explanation": "No evidence found"}'
        )

        result = await reverify_with_expanded_window(
            [claim], citation_map, mock_llm
        )

        assert result[0].verdict == "UNSUPPORTED"

    @pytest.mark.asyncio
    async def test_llm_failure_keeps_original_verdict(self):
        """LLM failure leaves verdict unchanged."""
        claim = _make_claim(
            "The fee is $95",
            claim_type="quantitative",
            cited_sources=[1],
        )
        citation_map = {
            1: _make_source(1, raw_content="x" * 20000),
        }

        mock_llm = AsyncMock(side_effect=RuntimeError("LLM timeout"))

        result = await reverify_with_expanded_window(
            [claim], citation_map, mock_llm
        )

        assert result[0].verdict == "UNSUPPORTED"

    @pytest.mark.asyncio
    async def test_empty_claims_list(self):
        """Empty claims list returns empty."""
        result = await reverify_with_expanded_window(
            [], {}, AsyncMock()
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_no_source_in_map_skips(self):
        """Claim citing a source not in the map is skipped."""
        claim = _make_claim(
            "Some fact",
            claim_type="quantitative",
            cited_sources=[99],
        )

        mock_llm = AsyncMock()

        result = await reverify_with_expanded_window(
            [claim], {}, mock_llm
        )

        assert result[0].verdict == "UNSUPPORTED"
        mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: deepen_thin_sources
# ---------------------------------------------------------------------------


class TestDeepenThinSources:
    """Step 3: Re-extract thin sources."""

    @pytest.mark.asyncio
    async def test_deepen_with_url(self):
        """Source with URL triggers re-extraction and updates raw_content."""
        claim = _make_claim(
            "Transfer takes 24 hours",
            claim_type="quantitative",
            cited_sources=[1],
        )
        source = _make_source(
            1,
            raw_content="Short",
            url="https://example.com/article",
        )
        citation_map = {1: source}

        # Mock extract provider
        mock_provider = AsyncMock()
        extracted_source = ResearchSource(
            title="Extracted",
            url="https://example.com/article",
            raw_content="Much richer content with transfer timing details. " + "x" * 10000,
        )
        mock_provider.extract = AsyncMock(return_value=[extracted_source])

        count = await deepen_thin_sources(
            [claim], citation_map, mock_provider
        )

        assert count == 1
        assert len(source.raw_content or "") > 100
        assert source.metadata.get("_pre_deepen_content") == "Short"

    @pytest.mark.asyncio
    async def test_deepen_no_provider_returns_zero(self):
        """No extract provider returns 0."""
        claim = _make_claim("Some fact", cited_sources=[1])

        count = await deepen_thin_sources(
            [claim], {}, None
        )

        assert count == 0

    @pytest.mark.asyncio
    async def test_deepen_provider_failure_returns_zero(self):
        """Provider failure returns 0 (non-fatal)."""
        claim = _make_claim(
            "Some fact",
            claim_type="quantitative",
            cited_sources=[1],
        )
        source = _make_source(1, raw_content="Short", url="https://example.com")
        citation_map = {1: source}

        mock_provider = AsyncMock()
        mock_provider.extract = AsyncMock(side_effect=RuntimeError("Network error"))

        count = await deepen_thin_sources(
            [claim], citation_map, mock_provider
        )

        assert count == 0
        # Original content should be unchanged
        assert source.raw_content == "Short"

    @pytest.mark.asyncio
    async def test_deepen_skips_if_extracted_not_richer(self):
        """If extracted content isn't longer than original, skip update."""
        claim = _make_claim(
            "Some fact",
            claim_type="quantitative",
            cited_sources=[1],
        )
        source = _make_source(
            1,
            raw_content="Original content that is moderately long",
            url="https://example.com",
        )
        citation_map = {1: source}

        mock_provider = AsyncMock()
        # Extracted content is shorter
        extracted_source = ResearchSource(
            title="Extracted",
            url="https://example.com",
            raw_content="Short",
        )
        mock_provider.extract = AsyncMock(return_value=[extracted_source])

        count = await deepen_thin_sources(
            [claim], citation_map, mock_provider
        )

        assert count == 0
        assert "_pre_deepen_content" not in source.metadata


# ---------------------------------------------------------------------------
# Tests: build_gap_queries with exclusions
# ---------------------------------------------------------------------------


class TestBuildGapQueriesExclusions:
    """Integration: inferential claims excluded from gap queries."""

    def test_excludes_inferential_claims(self):
        """Inferential claims are not included in gap queries."""
        claims = [
            _make_claim(
                "Chase is the best overall card",
                claim_type="comparative",
                report_section="Recommendations",
            ),
            _make_claim(
                "Annual fee is $95",
                claim_type="quantitative",
                report_section="Pricing",
            ),
        ]
        vr = _make_verification_result(claims)

        # Exclude the inferential claim
        queries = build_gap_queries(
            vr,
            exclude_claims={"Chase is the best overall card"},
        )

        # Only the non-excluded unsupported claim should generate a query
        assert len(queries) == 1
        assert "Pricing" in queries[0]
        assert "best overall" not in queries[0]

    def test_no_exclusions_backward_compatible(self):
        """Without exclude_claims, all UNSUPPORTED claims generate queries."""
        claims = [
            _make_claim(
                "Claim A",
                report_section="Section A",
            ),
            _make_claim(
                "Claim B",
                report_section="Section B",
            ),
        ]
        vr = _make_verification_result(claims)

        queries = build_gap_queries(vr)

        assert len(queries) == 2

    def test_all_excluded_returns_empty(self):
        """If all claims are excluded, returns empty list."""
        claims = [
            _make_claim("Claim A"),
            _make_claim("Claim B"),
        ]
        vr = _make_verification_result(claims)

        queries = build_gap_queries(
            vr,
            exclude_claims={"Claim A", "Claim B"},
        )

        assert queries == []


# ---------------------------------------------------------------------------
# Tests: End-to-end fidelity improvement
# ---------------------------------------------------------------------------


class TestFidelityImprovesAfterDeepening:
    """Integration: deepening step improves fidelity score."""

    @pytest.mark.asyncio
    async def test_fidelity_improves(self):
        """End-to-end: expanded-window re-verification upgrades verdicts,
        improving the fidelity score."""
        # Initial state: 5 supported, 5 unsupported → fidelity 0.50
        supported_claims = [
            _make_claim(
                f"Supported claim {i}",
                verdict="SUPPORTED",
                cited_sources=[i],
            )
            for i in range(1, 6)
        ]
        unsupported_claims = [
            _make_claim(
                f"Unsupported claim {i}",
                claim_type="quantitative",
                verdict="UNSUPPORTED",
                cited_sources=[i + 5],
            )
            for i in range(1, 6)
        ]
        all_claims = supported_claims + unsupported_claims
        vr = _make_verification_result(all_claims)

        initial_fidelity = vr.fidelity_score
        assert initial_fidelity == pytest.approx(0.5)

        # Rich sources for the unsupported claims
        citation_map = {
            i + 5: _make_source(i + 5, raw_content="x" * 20000)
            for i in range(1, 6)
        }

        # Classify → all 5 unsupported should be deepen_window
        classification = classify_unsupported_claims(vr, citation_map)
        assert len(classification.deepen_window) == 5

        # Mock LLM: upgrade 3 of 5 to SUPPORTED
        call_count = 0

        async def mock_llm(sys: str, user: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return '{"verdict": "SUPPORTED", "evidence_quote": "found it", "explanation": "evidence present"}'
            return '{"verdict": "UNSUPPORTED", "evidence_quote": null, "explanation": "not found"}'

        await reverify_with_expanded_window(
            classification.deepen_window, citation_map, mock_llm
        )

        # Recompute aggregates
        vr.claims_supported = sum(1 for c in vr.details if c.verdict == "SUPPORTED")
        vr.claims_unsupported = sum(1 for c in vr.details if c.verdict == "UNSUPPORTED")

        # Now: 8 supported, 2 unsupported → fidelity 0.80
        assert vr.claims_supported == 8
        assert vr.claims_unsupported == 2
        assert vr.fidelity_score == pytest.approx(0.8)
        assert vr.fidelity_score > initial_fidelity
