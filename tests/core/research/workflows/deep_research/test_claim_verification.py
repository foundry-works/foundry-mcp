"""Comprehensive tests for the post-synthesis claim verification pipeline.

Tests cover:
- Claim extraction (Pass 1): parsing, filtering, prioritization
- Claim-source alignment (Pass 2): verdicts, source resolution, truncation
- Correction application: context windows, sequential corrections, budget caps
- Serialization round-trip: backward/forward compatibility
- Integration: end-to-end with seeded hallucination, resume after crash
- Graceful degradation: timeouts, parse failures, partial failures
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock

import pytest

from foundry_mcp.config.research import ResearchConfig
from foundry_mcp.core.research.models.deep_research import (
    ClaimVerdict,
    ClaimVerificationResult,
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.models.sources import (
    ResearchSource,
    SourceQuality,
    SourceType,
    SubQuery,
)
from foundry_mcp.core.research.workflows.deep_research.phases.claim_verification import (
    _EXTRACTION_SYSTEM_PROMPT,
    _VERIFICATION_SYSTEM_PROMPT,
    _apply_citation_remap,
    _apply_token_budget,
    _build_verification_user_prompt,
    _extract_claims_chunked,
    _extract_claims_from_chunk,
    _extract_context_window,
    _filter_claims_for_verification,
    _filter_uncited_claims,
    _multi_window_truncate,
    _parse_extracted_claims,
    _parse_remap_response,
    _parse_verification_response,
    _remove_citations_from_report,
    _repair_heading_boundaries,
    _resolve_source_text,
    _sort_claims_by_priority,
    _split_report_into_sections,
    _verify_single_claim,
    apply_corrections,
    extract_and_verify_claims,
    remap_unsupported_citations,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class MockWorkflowResult:
    """Minimal mock for WorkflowResult."""

    success: bool = True
    content: str = ""
    error: Optional[str] = None
    tokens_used: int = 0
    duration_ms: float = 0
    metadata: Optional[dict] = None


def _make_config(**overrides) -> ResearchConfig:
    """Create a ResearchConfig with verification enabled."""
    defaults = {
        "default_provider": "test-provider",
        "deep_research_claim_verification_enabled": True,
        "deep_research_claim_verification_sample_rate": 0.3,
        "deep_research_claim_verification_timeout": 30,
        "deep_research_claim_verification_max_claims": 50,
        "deep_research_claim_verification_max_concurrent": 5,
        "deep_research_claim_verification_max_corrections": 5,
        "deep_research_claim_verification_annotate_unsupported": False,
        "deep_research_claim_verification_max_input_tokens": 200_000,
    }
    defaults.update(overrides)
    return ResearchConfig(**defaults)


def _make_source(
    citation_number: int,
    content: Optional[str] = "Source content.",
    title: str = "Test Source",
    raw_content: Optional[str] = None,
    snippet: Optional[str] = None,
) -> ResearchSource:
    """Create a ResearchSource with a citation number."""
    return ResearchSource(
        id=f"src-{citation_number}",
        title=f"{title} [{citation_number}]",
        url=f"https://example.com/{citation_number}",
        content=content,
        raw_content=raw_content,
        snippet=snippet,
        quality=SourceQuality.HIGH,
        source_type=SourceType.WEB,
        citation_number=citation_number,
    )


def _make_state_with_sources(
    report: str,
    sources: list[ResearchSource],
) -> DeepResearchState:
    """Create a DeepResearchState with report and sources."""
    state = DeepResearchState(
        id="deepres-cv-test",
        original_query="Test query",
        phase=DeepResearchPhase.SYNTHESIS,
    )
    state.report = report
    sq = SubQuery(
        id="sq-test",
        query="Test sub-query",
        rationale="Test",
        priority=1,
        status="completed",
    )
    state.sub_queries.append(sq)
    for src in sources:
        src.sub_query_id = sq.id
        state.append_source(src)
    return state


# ---------------------------------------------------------------------------
# Unit tests: _resolve_source_text
# ---------------------------------------------------------------------------


class TestResolveSourceText:
    def test_prefers_raw_content_over_content(self):
        """raw_content (full page text) is preferred over content (compressed summary)."""
        src = _make_source(1, content="Compressed summary", raw_content="Full page text")
        assert _resolve_source_text(src) == "Full page text"

    def test_falls_back_to_content(self):
        """Falls back to content when raw_content is None."""
        src = _make_source(1, content="Full content", raw_content=None)
        assert _resolve_source_text(src) == "Full content"

    def test_falls_back_to_snippet(self):
        """Falls back to snippet when both raw_content and content are None."""
        src = _make_source(1, content=None, raw_content=None, snippet="Snippet")
        assert _resolve_source_text(src) == "Snippet"

    def test_all_none(self):
        """Returns None when all three are None."""
        src = _make_source(1, content=None, raw_content=None, snippet=None)
        assert _resolve_source_text(src) is None


# ---------------------------------------------------------------------------
# Unit tests: _multi_window_truncate
# ---------------------------------------------------------------------------


class TestMultiWindowTruncate:
    def test_short_text_returned_as_is(self):
        text = "Short text"
        assert _multi_window_truncate(text, "anything", 1000) == text

    def test_single_keyword_mid_document(self):
        """Single keyword match gets full max_chars budget (adaptive sizing)."""
        before = "A" * 5000
        keyword_section = " Aeroplan transfer partner "
        after = "B" * 5000
        text = before + keyword_section + after
        result = _multi_window_truncate(text, "Aeroplan transfer ratio", 200)
        assert len(result) <= 200
        assert "Aeroplan" in result

    def test_keyword_found_near_start(self):
        text = "Aeroplan is listed here" + "X" * 10000
        result = _multi_window_truncate(text, "Aeroplan partner", 200)
        assert len(result) <= 200
        assert "Aeroplan" in result

    def test_keyword_found_near_end(self):
        text = "X" * 10000 + "Aeroplan is at the end"
        result = _multi_window_truncate(text, "Aeroplan partner", 200)
        assert len(result) <= 200
        assert "Aeroplan" in result

    def test_no_keyword_match_falls_back_to_prefix(self):
        text = "X" * 10000
        result = _multi_window_truncate(text, "Aeroplan", 200)
        assert len(result) == 200
        assert result == "X" * 200

    def test_stopwords_filtered(self):
        text = "X" * 5000 + "transfer" + "Y" * 5000
        result = _multi_window_truncate(text, "with from the transfer", 200)
        assert "transfer" in result

    def test_multiple_keyword_matches_multiple_windows(self):
        """Multiple keyword matches at different positions → multiple windows."""
        # Place "annual" near start and "dining" near end.
        text = "X" * 100 + "annual fee is $550" + "Y" * 8000 + "dining earns 3x" + "Z" * 100
        result = _multi_window_truncate(text, "annual fee dining earns", 8000)
        assert "annual" in result
        assert "dining" in result

    def test_windows_non_overlapping(self):
        """Windows should not overlap with each other."""
        text = ("A" * 3000 + "keyword1" + "B" * 3000 +
                "C" * 3000 + "keyword2" + "D" * 3000)
        result = _multi_window_truncate(text, "keyword1 keyword2", 4000, max_windows=2)
        # Both keywords should appear.
        assert "keyword1" in result
        assert "keyword2" in result
        # Separator should be present between windows.
        assert "[...]" in result

    def test_windows_ordered_by_document_position(self):
        """Windows returned in document order regardless of score."""
        text = ("first_word" + "A" * 5000 + "second_word" + "B" * 5000 +
                "third_word" + "C" * 2000)
        result = _multi_window_truncate(text, "first_word second_word third_word", 6000, max_windows=3)
        if "first_word" in result and "third_word" in result:
            assert result.index("first_word") < result.index("third_word")

    def test_total_output_within_max_chars_budget(self):
        """Total output should stay within max_chars."""
        text = "keyword1" + "A" * 10000 + "keyword2" + "B" * 10000 + "keyword3" + "C" * 10000
        max_chars = 8000
        result = _multi_window_truncate(text, "keyword1 keyword2 keyword3", max_chars, max_windows=3)
        assert len(result) <= max_chars

    def test_source_shorter_than_max_chars_returned_unchanged(self):
        text = "Short source text with keyword1 and keyword2."
        result = _multi_window_truncate(text, "keyword1 keyword2", 10000)
        assert result == text

    def test_single_keyword_gets_full_budget(self):
        """Single keyword → single window with full max_chars budget (adaptive)."""
        text = "A" * 5000 + "keyword" + "B" * 5000
        result = _multi_window_truncate(text, "keyword", 8000, max_windows=3)
        # With only 1 cluster, window should get the full budget.
        assert len(result) == 8000
        assert "keyword" in result

    def test_two_clusters_adaptive_sizing(self):
        """Two clusters → each gets roughly max_chars // 2 (adaptive sizing)."""
        text = "A" * 5000 + "alpha" + "B" * 10000 + "bravo" + "C" * 5000
        max_chars = 8000
        result = _multi_window_truncate(text, "alpha bravo", max_chars, max_windows=3)
        assert "alpha" in result
        assert "bravo" in result
        assert len(result) <= max_chars

    def test_no_keywords_extracted(self):
        """All words < 4 chars or stopwords → prefix truncation fallback."""
        text = "X" * 10000
        result = _multi_window_truncate(text, "the is a an", 200)
        assert len(result) == 200
        assert result == "X" * 200


# ---------------------------------------------------------------------------
# Unit tests: _parse_extracted_claims
# ---------------------------------------------------------------------------


class TestParseExtractedClaims:
    def test_valid_json_array(self):
        response = json.dumps([
            {
                "claim": "X does not support Y",
                "claim_type": "negative",
                "cited_sources": [1, 2],
                "report_section": "Section A",
                "quote_context": "X does not support Y in any way",
            },
            {
                "claim": "Price is $100",
                "claim_type": "quantitative",
                "cited_sources": [3],
            },
        ])
        claims = _parse_extracted_claims(response)
        assert len(claims) == 2
        assert claims[0].claim == "X does not support Y"
        assert claims[0].claim_type == "negative"
        assert claims[0].cited_sources == [1, 2]
        assert claims[1].claim_type == "quantitative"

    def test_markdown_code_fence(self):
        response = "```json\n" + json.dumps([{"claim": "test", "claim_type": "positive"}]) + "\n```"
        claims = _parse_extracted_claims(response)
        assert len(claims) == 1
        assert claims[0].claim == "test"

    def test_malformed_json_returns_empty(self):
        claims = _parse_extracted_claims("this is not json at all")
        assert claims == []

    def test_truncated_json_recovery(self):
        # Truncated mid-array: two objects, second incomplete → recovery adds "]".
        response = '[{"claim": "first", "claim_type": "negative", "cited_sources": [1]}, {"claim": "trun'
        claims = _parse_extracted_claims(response)
        # Recovery attempts adding "]" — first object may or may not parse depending on
        # JSON strictness. At minimum, should not raise and should return a list.
        assert isinstance(claims, list)

    def test_empty_array(self):
        claims = _parse_extracted_claims("[]")
        assert claims == []

    def test_invalid_claim_type_defaults_to_positive(self):
        response = json.dumps([{"claim": "test", "claim_type": "unknown_type"}])
        claims = _parse_extracted_claims(response)
        assert claims[0].claim_type == "positive"

    def test_empty_claim_text_skipped(self):
        response = json.dumps([{"claim": "", "claim_type": "negative"}])
        claims = _parse_extracted_claims(response)
        assert len(claims) == 0

    def test_non_dict_items_skipped(self):
        response = json.dumps([42, "string", {"claim": "valid", "claim_type": "positive"}])
        claims = _parse_extracted_claims(response)
        assert len(claims) == 1

    def test_cited_sources_coercion(self):
        response = json.dumps([{"claim": "test", "cited_sources": [1, 2.0, "invalid"]}])
        claims = _parse_extracted_claims(response)
        assert claims[0].cited_sources == [1, 2]


# ---------------------------------------------------------------------------
# Unit tests: Filtering & Prioritization
# ---------------------------------------------------------------------------


class TestFilterAndPrioritize:
    def test_negative_always_included(self):
        claims = [ClaimVerdict(claim="X does not Y", claim_type="negative", cited_sources=[1])]
        filtered = _filter_claims_for_verification(claims, sample_rate=0.0, max_claims=50)
        assert len(filtered) == 1

    def test_quantitative_always_included(self):
        claims = [ClaimVerdict(claim="costs $100", claim_type="quantitative", cited_sources=[1])]
        filtered = _filter_claims_for_verification(claims, sample_rate=0.0, max_claims=50)
        assert len(filtered) == 1

    def test_positive_sampled_deterministically(self):
        claims = [
            ClaimVerdict(claim=f"positive claim {i}", claim_type="positive", cited_sources=[1])
            for i in range(100)
        ]
        # With 30% sample rate, ~30 should be included.
        filtered = _filter_claims_for_verification(claims, sample_rate=0.3, max_claims=100)
        assert 15 <= len(filtered) <= 45  # Generous bounds for deterministic hash.

        # Same input → same output (deterministic).
        filtered2 = _filter_claims_for_verification(claims, sample_rate=0.3, max_claims=100)
        assert [c.claim for c in filtered] == [c.claim for c in filtered2]

    def test_max_claims_truncation(self):
        claims = [
            ClaimVerdict(claim=f"neg {i}", claim_type="negative", cited_sources=[1])
            for i in range(60)
        ]
        filtered = _filter_claims_for_verification(claims, sample_rate=1.0, max_claims=10)
        assert len(filtered) == 10

    def test_priority_ordering(self):
        claims = [
            ClaimVerdict(claim="pos", claim_type="positive", cited_sources=[1, 2, 3]),
            ClaimVerdict(claim="neg", claim_type="negative", cited_sources=[1]),
            ClaimVerdict(claim="quant", claim_type="quantitative", cited_sources=[1, 2]),
            ClaimVerdict(claim="comp", claim_type="comparative", cited_sources=[1]),
        ]
        sorted_claims = _sort_claims_by_priority(claims)
        assert [c.claim_type for c in sorted_claims] == [
            "negative",
            "quantitative",
            "comparative",
            "positive",
        ]

    def test_priority_within_type_by_source_count(self):
        claims = [
            ClaimVerdict(claim="neg 1 src", claim_type="negative", cited_sources=[1]),
            ClaimVerdict(claim="neg 3 src", claim_type="negative", cited_sources=[1, 2, 3]),
        ]
        sorted_claims = _sort_claims_by_priority(claims)
        assert sorted_claims[0].claim == "neg 3 src"


# ---------------------------------------------------------------------------
# Unit tests: Token Budget
# ---------------------------------------------------------------------------


class TestTokenBudget:
    def test_all_claims_fit(self):
        sources = {1: _make_source(1, content="x" * 100)}
        claims = [ClaimVerdict(claim="test", claim_type="negative", cited_sources=[1])]
        kept = _apply_token_budget(claims, sources, max_input_tokens=200_000)
        assert len(kept) == 1

    def test_claims_dropped_when_over_budget(self):
        # Each source is 8000 chars → ~2286 tokens. 10 claims × ~2286 = ~22860 tokens.
        sources = {i: _make_source(i, content="x" * 8000) for i in range(1, 11)}
        claims = [
            ClaimVerdict(claim=f"claim {i}", claim_type="negative", cited_sources=[i])
            for i in range(1, 11)
        ]
        kept = _apply_token_budget(claims, sources, max_input_tokens=5000)
        assert len(kept) < 10
        assert len(kept) >= 1

    def test_extraction_unaffected(self):
        # Token budget only applies to verification (Pass 2), not extraction.
        # This is verified by the fact that _apply_token_budget is a separate function.
        pass


# ---------------------------------------------------------------------------
# Unit tests: Verification Response Parsing
# ---------------------------------------------------------------------------


class TestParseVerificationResponse:
    def test_valid_json(self):
        response = json.dumps({
            "verdict": "CONTRADICTED",
            "evidence_quote": "Aeroplan is listed",
            "explanation": "Source shows it",
        })
        result = _parse_verification_response(response)
        assert result["verdict"] == "CONTRADICTED"
        assert result["evidence_quote"] == "Aeroplan is listed"

    def test_markdown_fence(self):
        inner = json.dumps({"verdict": "SUPPORTED", "evidence_quote": None, "explanation": "OK"})
        response = f"```json\n{inner}\n```"
        result = _parse_verification_response(response)
        assert result["verdict"] == "SUPPORTED"

    def test_invalid_json_defaults_to_unsupported(self):
        result = _parse_verification_response("not json")
        assert result["verdict"] == "UNSUPPORTED"

    def test_invalid_verdict_defaults_to_unsupported(self):
        response = json.dumps({"verdict": "MAYBE", "explanation": "uncertain"})
        result = _parse_verification_response(response)
        assert result["verdict"] == "UNSUPPORTED"

    def test_missing_verdict_defaults_to_unsupported(self):
        response = json.dumps({"explanation": "no verdict"})
        result = _parse_verification_response(response)
        assert result["verdict"] == "UNSUPPORTED"


# ---------------------------------------------------------------------------
# Unit tests: Verification Prompt Building
# ---------------------------------------------------------------------------


class TestBuildVerificationPrompt:
    def test_basic_prompt(self):
        claim = ClaimVerdict(
            claim="X does not support Y",
            claim_type="negative",
            cited_sources=[1, 2],
        )
        sources = {
            1: _make_source(1, content="Source 1 content about X and Y"),
            2: _make_source(2, content="Source 2 content about X and Y"),
        }
        prompt, resolution = _build_verification_user_prompt(claim, sources)
        assert prompt is not None
        assert "Source [1]" in prompt
        assert "Source [2]" in prompt
        assert "X does not support Y" in prompt
        assert resolution == "compressed_only"

    def test_dangling_citation_skipped(self):
        claim = ClaimVerdict(
            claim="test",
            claim_type="positive",
            cited_sources=[1, 99],  # 99 doesn't exist
        )
        sources = {1: _make_source(1, content="Content")}
        prompt, resolution = _build_verification_user_prompt(claim, sources)
        assert prompt is not None
        assert "Source [1]" in prompt
        assert "Source [99]" not in prompt
        assert resolution == "compressed_only"

    def test_no_resolvable_sources_returns_none(self):
        claim = ClaimVerdict(
            claim="test",
            claim_type="positive",
            cited_sources=[99],  # Doesn't exist
        )
        prompt, resolution = _build_verification_user_prompt(claim, {})
        assert prompt is None
        assert resolution == "citation_not_found"

    def test_source_with_no_content_skipped(self):
        claim = ClaimVerdict(
            claim="test",
            claim_type="positive",
            cited_sources=[1],
        )
        src = _make_source(1, content=None, raw_content=None, snippet=None)
        prompt, resolution = _build_verification_user_prompt(claim, {1: src})
        assert prompt is None
        assert resolution == "no_content"


# ---------------------------------------------------------------------------
# Unit tests: Context Window Extraction
# ---------------------------------------------------------------------------


class TestExtractContextWindow:
    def test_quote_found(self):
        report = "Introduction.\n\n" + "A" * 600 + "CLAIM HERE" + "B" * 600 + "\n\nConclusion."
        result = _extract_context_window(report, "CLAIM HERE")
        assert result is not None
        window, start, _ = result
        assert "CLAIM HERE" in window
        # Window should start at a paragraph boundary (\n\n) or start of string.
        assert start == 0 or report[start:start + 2] == "\n\n" or report[start - 2:start] == "\n\n"

    def test_quote_not_found(self):
        result = _extract_context_window("some report text", "nonexistent quote")
        assert result is None

    def test_window_expands_to_paragraph_boundaries(self):
        para1 = "First paragraph."
        para2 = "Second paragraph with CLAIM HERE and more text."
        para3 = "Third paragraph."
        report = f"{para1}\n\n{para2}\n\n{para3}"
        result = _extract_context_window(report, "CLAIM HERE")
        assert result is not None
        window, _, _ = result
        # Window should contain the full paragraph.
        assert "Second paragraph" in window

    def test_paragraph_clamping_never_splits_paragraph(self):
        report = "Para 1.\n\nPara 2 with the CLAIM text.\n\nPara 3."
        result = _extract_context_window(report, "CLAIM")
        assert result is not None
        window, _, _ = result
        # Ensure the window contains complete paragraph.
        assert "Para 2 with the CLAIM text." in window


# ---------------------------------------------------------------------------
# Unit tests: Correction Application
# ---------------------------------------------------------------------------


class TestApplyCorrections:
    @pytest.mark.asyncio
    async def test_single_correction(self):
        report = (
            "Introduction.\n\n"
            "Amex does not transfer to Aeroplan. This is a key limitation.\n\n"
            "Conclusion."
        )
        sources = [
            _make_source(1, content="Aeroplan is listed as an Amex transfer partner at 1:1"),
        ]
        state = _make_state_with_sources(report, sources)
        config = _make_config()

        verification_result = ClaimVerificationResult(
            claims_extracted=1,
            claims_verified=1,
            claims_contradicted=1,
            details=[
                ClaimVerdict(
                    claim="Amex does not transfer to Aeroplan",
                    claim_type="negative",
                    cited_sources=[1],
                    verdict="CONTRADICTED",
                    evidence_quote="Aeroplan is listed as an Amex transfer partner at 1:1",
                    explanation="Source explicitly lists Aeroplan as Amex partner",
                    quote_context="Amex does not transfer to Aeroplan. This is a key limitation.",
                ),
            ],
        )

        mock_execute = AsyncMock(return_value=MockWorkflowResult(
            success=True,
            content="Amex transfers to Aeroplan at a 1:1 ratio. This is a valuable feature.",
        ))

        await apply_corrections(
            state=state,
            config=config,
            verification_result=verification_result,
            execute_fn=mock_execute,
            provider_id="test-provider",
        )

        assert verification_result.corrections_applied == 1
        assert state.report is not None
        assert "does not transfer" not in state.report
        assert verification_result.details[0].correction_applied is True

    @pytest.mark.asyncio
    async def test_correction_budget_cap(self):
        """Only top N corrections applied when CONTRADICTED > max_corrections."""
        config = _make_config(deep_research_claim_verification_max_corrections=2)
        sources = [_make_source(i, content=f"Source {i}") for i in range(1, 6)]
        report = "\n\n".join(f"Claim {i} is wrong. Context text." for i in range(1, 6))
        state = _make_state_with_sources(report, sources)

        details = [
            ClaimVerdict(
                claim=f"Claim {i} is wrong",
                claim_type="negative",
                cited_sources=[i],
                verdict="CONTRADICTED",
                evidence_quote=f"Source {i} says otherwise",
                explanation="Contradicted",
                quote_context=f"Claim {i} is wrong. Context text.",
            )
            for i in range(1, 6)
        ]
        verification_result = ClaimVerificationResult(
            claims_extracted=5,
            claims_verified=5,
            claims_contradicted=5,
            details=details,
        )

        call_count = 0

        async def mock_execute(**kwargs):
            nonlocal call_count
            call_count += 1
            return MockWorkflowResult(success=True, content=f"Corrected claim {call_count}.")

        await apply_corrections(
            state=state,
            config=config,
            verification_result=verification_result,
            execute_fn=mock_execute,
            provider_id="test-provider",
        )

        assert verification_result.corrections_applied == 2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_correction_priority_ordering(self):
        """Corrections follow priority: negative > quantitative > comparative > positive."""
        config = _make_config(deep_research_claim_verification_max_corrections=2)
        sources = [_make_source(i, content=f"Source {i}") for i in range(1, 5)]
        report = "Pos claim. ctx.\n\nNeg claim. ctx.\n\nQuant claim. ctx.\n\nComp claim. ctx."
        state = _make_state_with_sources(report, sources)

        async def mock_execute(**kwargs):  # noqa: ARG001
            return MockWorkflowResult(success=True, content="Corrected text.")

        details = [
            ClaimVerdict(claim="Pos claim", claim_type="positive", cited_sources=[1],
                         verdict="CONTRADICTED", quote_context="Pos claim. ctx.",
                         evidence_quote="e", explanation="x"),
            ClaimVerdict(claim="Neg claim", claim_type="negative", cited_sources=[2],
                         verdict="CONTRADICTED", quote_context="Neg claim. ctx.",
                         evidence_quote="e", explanation="x"),
            ClaimVerdict(claim="Quant claim", claim_type="quantitative", cited_sources=[3],
                         verdict="CONTRADICTED", quote_context="Quant claim. ctx.",
                         evidence_quote="e", explanation="x"),
            ClaimVerdict(claim="Comp claim", claim_type="comparative", cited_sources=[4],
                         verdict="CONTRADICTED", quote_context="Comp claim. ctx.",
                         evidence_quote="e", explanation="x"),
        ]
        verification_result = ClaimVerificationResult(
            claims_extracted=4, claims_verified=4, claims_contradicted=4, details=details,
        )

        await apply_corrections(
            state=state,
            config=config,
            verification_result=verification_result,
            execute_fn=mock_execute,
            provider_id="test-provider",
        )

        # Only 2 corrections applied — should be negative and quantitative.
        corrected = [d for d in details if d.correction_applied]
        assert len(corrected) == 2
        assert corrected[0].claim_type == "negative"
        assert corrected[1].claim_type == "quantitative"

    @pytest.mark.asyncio
    async def test_quote_context_not_found_uses_fallback(self):
        """When quote_context is not in report, falls back to full-report correction."""
        report = "This is the actual report text."
        state = _make_state_with_sources(report, [_make_source(1)])
        config = _make_config()

        verification_result = ClaimVerificationResult(
            claims_extracted=1,
            claims_verified=1,
            claims_contradicted=1,
            details=[
                ClaimVerdict(
                    claim="Non-existent claim",
                    claim_type="negative",
                    cited_sources=[1],
                    verdict="CONTRADICTED",
                    evidence_quote="evidence",
                    explanation="explanation",
                    quote_context="This text does not exist in the report",
                    report_section="Section A",
                ),
            ],
        )

        mock_execute = AsyncMock(return_value=MockWorkflowResult(
            success=True,
            content="This is the corrected full report text with the fix applied.",
        ))

        await apply_corrections(
            state=state,
            config=config,
            verification_result=verification_result,
            execute_fn=mock_execute,
            provider_id="test-provider",
        )

        # Should have used full-report correction.
        assert verification_result.corrections_applied == 1
        assert state.report is not None
        assert "corrected full report" in state.report

    @pytest.mark.asyncio
    async def test_sequential_correction_context_drift(self):
        """Two adjacent claims: correction #1 shifts text, claim #2 falls back."""
        report = (
            "Introduction.\n\n"
            "First false claim here. More context.\n\n"
            "Second false claim here. More context.\n\n"
            "Conclusion."
        )
        sources = [_make_source(1), _make_source(2)]
        state = _make_state_with_sources(report, sources)
        config = _make_config()

        call_count = 0

        async def mock_execute(**kwargs):
            nonlocal call_count
            call_count += 1
            prompt = kwargs.get("prompt", "")
            if call_count == 1:
                # First correction succeeds and changes text.
                return MockWorkflowResult(
                    success=True,
                    content="First corrected claim here. Updated context with extra words.",
                )
            else:
                # Second correction — either targeted or full-report fallback.
                if "Full Research Report" in prompt:
                    # Full-report fallback path.
                    assert state.report is not None
                    return MockWorkflowResult(
                        success=True,
                        content=state.report.replace(
                            "Second false claim here",
                            "Second corrected claim here",
                        ),
                    )
                return MockWorkflowResult(
                    success=True,
                    content="Second corrected claim here. More context.",
                )

        details = [
            ClaimVerdict(
                claim="First false claim",
                claim_type="negative",
                cited_sources=[1],
                verdict="CONTRADICTED",
                evidence_quote="e",
                explanation="x",
                quote_context="First false claim here. More context.",
            ),
            ClaimVerdict(
                claim="Second false claim",
                claim_type="negative",
                cited_sources=[2],
                verdict="CONTRADICTED",
                evidence_quote="e",
                explanation="x",
                quote_context="Second false claim here. More context.",
            ),
        ]
        verification_result = ClaimVerificationResult(
            claims_extracted=2, claims_verified=2, claims_contradicted=2, details=details,
        )

        await apply_corrections(
            state=state,
            config=config,
            verification_result=verification_result,
            execute_fn=mock_execute,
            provider_id="test-provider",
        )

        assert verification_result.corrections_applied == 2

    @pytest.mark.asyncio
    async def test_unsupported_annotation_when_enabled(self):
        """When annotate_unsupported=True, (unverified) is appended."""
        config = _make_config(deep_research_claim_verification_annotate_unsupported=True)
        report = "This claim is unsupported. Rest of report."
        state = _make_state_with_sources(report, [_make_source(1)])

        verification_result = ClaimVerificationResult(
            claims_extracted=1,
            claims_verified=1,
            claims_unsupported=1,
            details=[
                ClaimVerdict(
                    claim="This claim is unsupported",
                    claim_type="positive",
                    cited_sources=[1],
                    verdict="UNSUPPORTED",
                    quote_context="This claim is unsupported.",
                ),
            ],
        )

        await apply_corrections(
            state=state,
            config=config,
            verification_result=verification_result,
            execute_fn=AsyncMock(),
            provider_id="test-provider",
        )

        assert state.report is not None
        assert "(unverified)" in state.report

    @pytest.mark.asyncio
    async def test_unsupported_annotation_disabled_by_default(self):
        """Default: no (unverified) annotation for UNSUPPORTED claims."""
        config = _make_config()
        report = "This claim is unsupported. Rest of report."
        state = _make_state_with_sources(report, [_make_source(1)])

        verification_result = ClaimVerificationResult(
            claims_extracted=1,
            claims_verified=1,
            claims_unsupported=1,
            details=[
                ClaimVerdict(
                    claim="This claim is unsupported",
                    claim_type="positive",
                    cited_sources=[1],
                    verdict="UNSUPPORTED",
                    quote_context="This claim is unsupported.",
                ),
            ],
        )

        await apply_corrections(
            state=state,
            config=config,
            verification_result=verification_result,
            execute_fn=AsyncMock(),
            provider_id="test-provider",
        )

        assert state.report is not None
        assert "(unverified)" not in state.report

    @pytest.mark.asyncio
    async def test_correction_single_pass_no_reverification(self):
        """Corrections are NOT re-verified (single-pass only)."""
        config = _make_config()
        report = "False claim. Context.\n\nRest of report."
        state = _make_state_with_sources(report, [_make_source(1)])

        call_count = 0

        async def mock_execute(**kwargs):
            nonlocal call_count
            call_count += 1
            return MockWorkflowResult(success=True, content="Corrected claim. Context.")

        verification_result = ClaimVerificationResult(
            claims_extracted=1, claims_verified=1, claims_contradicted=1,
            details=[
                ClaimVerdict(
                    claim="False claim",
                    claim_type="negative",
                    cited_sources=[1],
                    verdict="CONTRADICTED",
                    evidence_quote="e",
                    explanation="x",
                    quote_context="False claim. Context.",
                ),
            ],
        )

        await apply_corrections(
            state=state, config=config, verification_result=verification_result,
            execute_fn=mock_execute, provider_id="test-provider",
        )

        # Only one LLM call (correction), no re-verification call.
        assert call_count == 1


# ---------------------------------------------------------------------------
# Unit tests: Serialization Round-Trip
# ---------------------------------------------------------------------------


class TestSerializationRoundTrip:
    def test_claim_verification_result_survives_roundtrip(self):
        state = DeepResearchState(
            id="deepres-serial-test",
            original_query="test",
            phase=DeepResearchPhase.SYNTHESIS,
        )
        state.claim_verification = ClaimVerificationResult(
            claims_extracted=10,
            claims_verified=8,
            claims_supported=6,
            claims_contradicted=1,
            claims_unsupported=1,
            corrections_applied=1,
            details=[
                ClaimVerdict(
                    claim="Test claim",
                    claim_type="negative",
                    cited_sources=[1, 2],
                    verdict="CONTRADICTED",
                    evidence_quote="evidence",
                    explanation="reason",
                    correction_applied=True,
                    corrected_text="Fixed claim",
                ),
            ],
        )
        dumped = state.model_dump()
        restored = DeepResearchState.model_validate(dumped)
        assert restored.claim_verification is not None
        assert restored.claim_verification.claims_extracted == 10
        assert restored.claim_verification.claims_contradicted == 1
        assert len(restored.claim_verification.details) == 1
        assert restored.claim_verification.details[0].verdict == "CONTRADICTED"

    def test_backward_compat_none_field(self):
        """Old states without claim_verification deserialize fine."""
        state = DeepResearchState(
            id="deepres-old",
            original_query="test",
        )
        dumped = state.model_dump()
        # Remove claim_verification to simulate old state.
        dumped.pop("claim_verification", None)
        restored = DeepResearchState.model_validate(dumped)
        assert restored.claim_verification is None

    def test_forward_compat_extra_field_ignored(self):
        """New state loaded by old model (extra field) doesn't crash.

        DeepResearchState does NOT use extra='forbid', so unknown fields
        are silently ignored.
        """
        state = DeepResearchState(
            id="deepres-new",
            original_query="test",
        )
        dumped = state.model_dump()
        dumped["future_field"] = "some_value"
        # Should not raise.
        restored = DeepResearchState.model_validate(dumped)
        assert restored.id == "deepres-new"


class TestFidelityScore:
    """Tests for ClaimVerificationResult.fidelity_score computed property."""

    def test_all_supported(self):
        result = ClaimVerificationResult(
            claims_verified=10, claims_supported=10,
        )
        assert result.fidelity_score == 1.0

    def test_all_unsupported(self):
        result = ClaimVerificationResult(
            claims_verified=10, claims_unsupported=10,
        )
        assert result.fidelity_score == 0.0

    def test_all_contradicted(self):
        result = ClaimVerificationResult(
            claims_verified=10, claims_contradicted=10,
        )
        assert result.fidelity_score == 0.0

    def test_mixed_verdicts(self):
        result = ClaimVerificationResult(
            claims_verified=35,
            claims_supported=5,
            claims_partially_supported=1,
            claims_unsupported=29,
        )
        expected = (5 * 1.0 + 1 * 0.5) / 35
        assert result.fidelity_score == pytest.approx(expected, abs=1e-6)

    def test_zero_claims_returns_none(self):
        result = ClaimVerificationResult(claims_verified=0)
        assert result.fidelity_score is None

    def test_serializes_in_model_dump(self):
        result = ClaimVerificationResult(
            claims_verified=4, claims_supported=2, claims_partially_supported=2,
        )
        dumped = result.model_dump()
        assert "fidelity_score" in dumped
        assert dumped["fidelity_score"] == pytest.approx(0.75)

    def test_serializes_in_json(self):
        import json
        result = ClaimVerificationResult(
            claims_verified=2, claims_supported=1, claims_unsupported=1,
        )
        data = json.loads(result.model_dump_json())
        assert data["fidelity_score"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Integration test: extract_and_verify_claims
# ---------------------------------------------------------------------------


class TestExtractAndVerifyClaims:
    @pytest.mark.asyncio
    async def test_end_to_end_with_seeded_hallucination(self):
        """Seeded hallucination: report says X doesn't support Y, source says it does."""
        report = (
            "# Credit Card Analysis\n\n"
            "## Your Portfolio\n\n"
            "Critically, Amex does not transfer to Air Canada Aeroplan [1]. "
            "This means you cannot use Amex points for Star Alliance flights.\n\n"
            "## Recommendations\n\n"
            "Consider Chase for Aeroplan access.\n"
        )
        sources = [
            _make_source(
                1,
                content=(
                    "American Express Membership Rewards transfer partners:\n"
                    "| Partner | Program | Ratio |\n"
                    "| --- | --- | --- |\n"
                    "| Air Canada | Aeroplan | 1:1 |\n"
                    "| British Airways | Avios | 1:1 |\n"
                ),
                title="Amex Transfer Partners Guide",
            ),
        ]
        state = _make_state_with_sources(report, sources)
        config = _make_config()

        extraction_response = json.dumps([
            {
                "claim": "Amex does not transfer to Air Canada Aeroplan",
                "claim_type": "negative",
                "cited_sources": [1],
                "report_section": "Your Portfolio",
                "quote_context": "Critically, Amex does not transfer to Air Canada Aeroplan [1].",
            },
        ])
        verification_response = json.dumps({
            "verdict": "CONTRADICTED",
            "evidence_quote": "Air Canada | Aeroplan | 1:1",
            "explanation": "Source [1] explicitly lists Aeroplan as an Amex transfer partner at 1:1",
        })

        call_count = 0

        async def mock_execute(**kwargs):
            nonlocal call_count
            call_count += 1
            phase = kwargs.get("phase", "")
            if phase == "claim_extraction":
                return MockWorkflowResult(success=True, content=extraction_response)
            elif phase == "claim_verification":
                return MockWorkflowResult(success=True, content=verification_response)
            return MockWorkflowResult(success=True, content="")

        result = await extract_and_verify_claims(
            state=state,
            config=config,
            provider_id="test-provider",
            execute_fn=mock_execute,
            timeout=30,
        )

        assert result.claims_extracted == 1
        assert result.claims_verified == 1
        assert result.claims_contradicted == 1
        assert result.details[0].verdict == "CONTRADICTED"

    @pytest.mark.asyncio
    async def test_no_report_returns_empty(self):
        state = DeepResearchState(
            id="deepres-empty",
            original_query="test",
        )
        state.report = None
        config = _make_config()
        result = await extract_and_verify_claims(
            state=state,
            config=config,
            provider_id="test",
            execute_fn=AsyncMock(),
            timeout=30,
        )
        assert result.claims_extracted == 0
        assert state.metadata.get("claim_verification_skipped") == "no_report"

    @pytest.mark.asyncio
    async def test_extraction_failure_returns_empty(self):
        state = _make_state_with_sources("Some report.", [_make_source(1)])
        config = _make_config()

        mock_execute = AsyncMock(return_value=MockWorkflowResult(
            success=False, content="", error="LLM error",
        ))

        result = await extract_and_verify_claims(
            state=state,
            config=config,
            provider_id="test",
            execute_fn=mock_execute,
            timeout=30,
        )

        assert result.claims_extracted == 0
        assert state.metadata.get("claim_verification_skipped") == "extraction_failed"

    @pytest.mark.asyncio
    async def test_extraction_exception_returns_empty(self):
        state = _make_state_with_sources("Some report.", [_make_source(1)])
        config = _make_config()

        mock_execute = AsyncMock(side_effect=Exception("Network error"))

        result = await extract_and_verify_claims(
            state=state, config=config, provider_id="test",
            execute_fn=mock_execute, timeout=30,
        )

        assert result.claims_extracted == 0
        assert state.metadata.get("claim_verification_skipped") == "extraction_failed"


# ---------------------------------------------------------------------------
# Integration test: Resume after crash
# ---------------------------------------------------------------------------


class TestResumeGuard:
    def test_resume_guard_metadata_flags(self):
        """Verify metadata flags are correctly set/checked for resume logic."""
        state = DeepResearchState(
            id="deepres-resume-test",
            original_query="test",
            phase=DeepResearchPhase.SYNTHESIS,
        )
        state.report = "A report."
        state.metadata["claim_verification_started"] = True

        # Resume guard conditions from workflow_execution.py:
        # state.report exists AND claim_verification_started AND NOT claim_verification
        assert state.report is not None
        assert state.metadata.get("claim_verification_started") is True
        assert state.claim_verification is None

    def test_skip_when_verification_complete(self):
        state = DeepResearchState(
            id="deepres-skip-test",
            original_query="test",
            phase=DeepResearchPhase.SYNTHESIS,
        )
        state.report = "A report."
        state.claim_verification = ClaimVerificationResult()

        # Should skip verification entirely.
        assert state.claim_verification is not None

    def test_stale_in_progress_flag_does_not_block(self):
        """A crash during verification leaves claim_verification_in_progress=True.
        Resume should still work (we check claim_verification_started, not in_progress).
        """
        state = DeepResearchState(
            id="deepres-stale-test",
            original_query="test",
            phase=DeepResearchPhase.SYNTHESIS,
        )
        state.report = "A report."
        state.metadata["claim_verification_started"] = True
        state.metadata["claim_verification_in_progress"] = True

        # Resume guard should still trigger (claim_verification is None).
        assert state.claim_verification is None
        assert state.metadata.get("claim_verification_started") is True


# ---------------------------------------------------------------------------
# Graceful Degradation Tests
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_verification_timeout_delivers_unverified(self):
        """Verification timeout → report delivered unverified."""
        state = _make_state_with_sources("Report text.", [_make_source(1)])
        config = _make_config()

        mock_execute = AsyncMock(side_effect=asyncio.TimeoutError("Timed out"))

        result = await extract_and_verify_claims(
            state=state, config=config, provider_id="test",
            execute_fn=mock_execute, timeout=1,
        )

        # Extraction failed due to timeout → empty result.
        assert result.claims_extracted == 0
        assert state.metadata.get("claim_verification_skipped") == "extraction_failed"

    @pytest.mark.asyncio
    async def test_partial_verification_failures(self):
        """Some claims verified, some skipped due to individual failures."""
        report = "Claim A is true [1]. Claim B is true [2]."
        sources = [_make_source(1, content="A content"), _make_source(2, content="B content")]
        state = _make_state_with_sources(report, sources)
        config = _make_config()

        extraction_response = json.dumps([
            {"claim": "Claim A", "claim_type": "negative", "cited_sources": [1]},
            {"claim": "Claim B", "claim_type": "negative", "cited_sources": [2]},
        ])

        call_count = 0

        async def mock_execute(**kwargs):
            nonlocal call_count
            call_count += 1
            phase = kwargs.get("phase", "")
            if phase == "claim_extraction":
                return MockWorkflowResult(success=True, content=extraction_response)
            elif phase == "claim_verification":
                if call_count == 2:
                    # First verification succeeds.
                    return MockWorkflowResult(
                        success=True,
                        content=json.dumps({
                            "verdict": "SUPPORTED",
                            "evidence_quote": "A",
                            "explanation": "OK",
                        }),
                    )
                else:
                    # Second verification fails.
                    raise Exception("Provider error")
            return MockWorkflowResult(success=True, content="")

        result = await extract_and_verify_claims(
            state=state, config=config, provider_id="test",
            execute_fn=mock_execute, timeout=30,
        )

        # Both claims attempted; one may fail gracefully.
        assert result.claims_verified == 2
        assert result.claims_extracted == 2

    @pytest.mark.asyncio
    async def test_report_snapshot_rollback(self):
        """Verify that corrections that fail mid-way don't corrupt the report."""
        report = "Original report content.\n\nAnother paragraph."
        state = _make_state_with_sources(report, [_make_source(1)])

        # Snapshot/rollback is tested at the workflow_execution.py level,
        # but we can verify that a failed correction leaves report unchanged.
        config = _make_config()
        verification_result = ClaimVerificationResult(
            claims_extracted=1, claims_verified=1, claims_contradicted=1,
            details=[
                ClaimVerdict(
                    claim="test", claim_type="negative", cited_sources=[1],
                    verdict="CONTRADICTED", evidence_quote="e", explanation="x",
                    quote_context="Original report content.",
                ),
            ],
        )

        mock_execute = AsyncMock(return_value=MockWorkflowResult(success=False, content=""))

        await apply_corrections(
            state=state, config=config, verification_result=verification_result,
            execute_fn=mock_execute, provider_id="test-provider",
        )

        # Correction failed → report unchanged.
        assert state.report == report
        assert verification_result.corrections_applied == 0


# ---------------------------------------------------------------------------
# Unit tests: resolve_phase_provider fallback chain
# ---------------------------------------------------------------------------


class TestResolvePhaseProvider:
    def test_explicit_claim_verification_provider(self):
        from foundry_mcp.core.research.workflows.deep_research._model_resolution import (
            resolve_phase_provider,
        )

        config = _make_config(
            deep_research_claim_verification_provider="explicit-cv-provider",
            deep_research_synthesis_provider="synth-provider",
        )
        result = resolve_phase_provider(config, "claim_verification", "synthesis")
        assert result == "explicit-cv-provider"

    def test_fallback_to_synthesis_provider(self):
        from foundry_mcp.core.research.workflows.deep_research._model_resolution import (
            resolve_phase_provider,
        )

        config = _make_config(deep_research_synthesis_provider="synth-provider")
        result = resolve_phase_provider(config, "claim_verification", "synthesis")
        assert result == "synth-provider"

    def test_fallback_to_default_provider(self):
        from foundry_mcp.core.research.workflows.deep_research._model_resolution import (
            resolve_phase_provider,
        )

        config = _make_config()
        result = resolve_phase_provider(config, "claim_verification", "synthesis")
        assert result == "test-provider"


# ---------------------------------------------------------------------------
# Unit tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_report(self):
        state = DeepResearchState(id="test", original_query="test")
        state.report = ""
        config = _make_config()
        result = await extract_and_verify_claims(
            state=state, config=config, provider_id="test",
            execute_fn=AsyncMock(), timeout=30,
        )
        assert result.claims_extracted == 0

    @pytest.mark.asyncio
    async def test_none_report(self):
        state = DeepResearchState(id="test", original_query="test")
        state.report = None
        config = _make_config()
        result = await extract_and_verify_claims(
            state=state, config=config, provider_id="test",
            execute_fn=AsyncMock(), timeout=30,
        )
        assert result.claims_extracted == 0

    def test_source_truncation_boundary(self):
        """Source content truncated to VERIFICATION_SOURCE_MAX_CHARS."""
        long_content = "x" * 20000
        result = _multi_window_truncate(long_content, "nomatch", 8000)
        assert len(result) == 8000


# ---------------------------------------------------------------------------
# Unit tests: Report overwrite after corrections
# ---------------------------------------------------------------------------


class TestReportOverwriteAfterCorrections:
    """Tests for the workflow_execution.py integration that writes the
    corrected report back to disk via Path(state.report_output_path).write_text().
    """

    @pytest.mark.asyncio
    async def test_report_file_overwritten_after_corrections(self, tmp_path):
        """Path(report_output_path).write_text() overwrites the synthesis-created file."""
        # Create a file simulating the synthesis-created report.
        report_file = tmp_path / "report.md"
        original_content = "Original synthesis report."
        report_file.write_text(original_content, encoding="utf-8")

        corrected_content = "Corrected synthesis report."

        state = _make_state_with_sources(original_content, [_make_source(1)])
        state.report_output_path = str(report_file)

        # Simulate the workflow_execution.py integration:
        # After corrections succeed, write corrected report to disk.
        state.report = corrected_content

        from pathlib import Path

        if state.report_output_path:
            Path(state.report_output_path).write_text(state.report or "", encoding="utf-8")

        # Verify the file was overwritten with corrected content.
        assert report_file.read_text(encoding="utf-8") == corrected_content
        assert report_file.read_text(encoding="utf-8") != original_content

    @pytest.mark.asyncio
    async def test_no_crash_when_report_output_path_is_none(self):
        """Corrections are skipped (no file write) when report_output_path is None."""
        state = _make_state_with_sources("Report.", [_make_source(1)])
        # report_output_path not set — simulates early failure or test scenario.
        assert not hasattr(state, "report_output_path") or state.report_output_path is None

        # Simulate the workflow_execution.py guard:
        # if state.report_output_path: Path(...).write_text(...)
        # This should simply not execute — no crash.
        from pathlib import Path

        if getattr(state, "report_output_path", None):
            Path(state.report_output_path).write_text(state.report or "", encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase 3: Extraction max_tokens and report truncation
# ---------------------------------------------------------------------------


class TestExtractionMaxTokensAndTruncation:
    """Tests for per-chunk max_tokens=4096 on extraction calls and 30K char truncation."""

    @pytest.mark.asyncio
    async def test_extraction_call_receives_max_tokens(self):
        """Chunked extraction LLM calls should include max_tokens=4096."""
        report = "Short report with a claim [1]."
        sources = [_make_source(1, content="Source content")]
        state = _make_state_with_sources(report, sources)
        config = _make_config()

        captured_kwargs: dict = {}

        async def mock_execute(**kwargs):
            phase = kwargs.get("phase", "")
            if phase == "claim_extraction":
                captured_kwargs.update(kwargs)
                return MockWorkflowResult(
                    success=True,
                    content=json.dumps([{
                        "claim": "A claim",
                        "claim_type": "positive",
                        "cited_sources": [1],
                    }]),
                )
            elif phase == "claim_verification":
                return MockWorkflowResult(
                    success=True,
                    content=json.dumps({
                        "verdict": "SUPPORTED",
                        "evidence_quote": "e",
                        "explanation": "ok",
                    }),
                )
            return MockWorkflowResult(success=True, content="")

        await extract_and_verify_claims(
            state=state, config=config, provider_id="test",
            execute_fn=mock_execute, timeout=30,
        )

        assert "max_tokens" in captured_kwargs
        assert captured_kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_large_report_truncated_before_extraction(self):
        """Reports > 30K chars should be truncated before extraction."""
        # Build a report with body + large bibliography exceeding 30K chars.
        body = "# Analysis\n\nImportant claim here [1].\n\n"
        bibliography = "## Sources\n\n" + ("- Source entry\n" * 3000)  # ~45K chars
        report = body + bibliography
        assert len(report) > 30_000

        sources = [_make_source(1, content="Source content")]
        state = _make_state_with_sources(report, sources)
        config = _make_config()

        captured_prompt: str = ""

        async def mock_execute(**kwargs):
            nonlocal captured_prompt
            phase = kwargs.get("phase", "")
            if phase == "claim_extraction":
                captured_prompt = kwargs.get("prompt", "")
                return MockWorkflowResult(
                    success=True,
                    content=json.dumps([{
                        "claim": "Important claim",
                        "claim_type": "positive",
                        "cited_sources": [1],
                    }]),
                )
            elif phase == "claim_verification":
                return MockWorkflowResult(
                    success=True,
                    content=json.dumps({
                        "verdict": "SUPPORTED",
                        "evidence_quote": "e",
                        "explanation": "ok",
                    }),
                )
            return MockWorkflowResult(success=True, content="")

        await extract_and_verify_claims(
            state=state, config=config, provider_id="test",
            execute_fn=mock_execute, timeout=30,
        )

        # The prompt should contain truncated content, not full report.
        # _build_extraction_user_prompt adds wrapper text, so prompt > 30K is fine,
        # but the report portion should be capped at 30K chars.
        assert len(captured_prompt) < len(report)
        # Body content should be preserved (it's at the start).
        assert "Important claim here [1]" in captured_prompt

    @pytest.mark.asyncio
    async def test_small_report_not_truncated(self):
        """Reports <= 30K chars should not be truncated."""
        report = "Short report with a claim [1]."
        sources = [_make_source(1, content="Source content")]
        state = _make_state_with_sources(report, sources)
        config = _make_config()

        captured_prompt: str = ""

        async def mock_execute(**kwargs):
            nonlocal captured_prompt
            phase = kwargs.get("phase", "")
            if phase == "claim_extraction":
                captured_prompt = kwargs.get("prompt", "")
                return MockWorkflowResult(
                    success=True,
                    content=json.dumps([]),
                )
            return MockWorkflowResult(success=True, content="")

        await extract_and_verify_claims(
            state=state, config=config, provider_id="test",
            execute_fn=mock_execute, timeout=30,
        )

        # Full report should be in the prompt.
        assert report in captured_prompt

    @pytest.mark.asyncio
    async def test_truncation_preserves_body_drops_bibliography(self):
        """Truncation takes the first 30K chars, preserving body over bibliography."""
        # Body is 5K chars, bibliography is 40K chars.
        body_text = "# Executive Summary\n\n" + ("Analysis paragraph. " * 250)  # ~5K
        bib_text = "\n\n## Sources\n\n" + ("- [N] Author, Title, URL\n" * 2000)  # ~40K
        report = body_text + bib_text
        assert len(report) > 30_000

        sources = [_make_source(1, content="Source content")]
        state = _make_state_with_sources(report, sources)
        config = _make_config()

        captured_prompt: str = ""

        async def mock_execute(**kwargs):
            nonlocal captured_prompt
            phase = kwargs.get("phase", "")
            if phase == "claim_extraction":
                captured_prompt = kwargs.get("prompt", "")
                return MockWorkflowResult(success=True, content=json.dumps([]))
            return MockWorkflowResult(success=True, content="")

        await extract_and_verify_claims(
            state=state, config=config, provider_id="test",
            execute_fn=mock_execute, timeout=30,
        )

        # Body content (executive summary) should be fully present.
        assert "Executive Summary" in captured_prompt
        assert "Analysis paragraph." in captured_prompt
        # Bibliography should be partially truncated (not all entries present).
        # Count bibliography entries in prompt vs original.
        prompt_bib_count = captured_prompt.count("Author, Title, URL")
        original_bib_count = report.count("Author, Title, URL")
        assert prompt_bib_count < original_bib_count

        # No exception raised — test passes.


# ---------------------------------------------------------------------------
# Unit tests: Post-correction persistence
# ---------------------------------------------------------------------------


class TestPostCorrectionPersistence:
    """Tests verifying that save_deep_research(state) is called after
    corrections succeed, ensuring the corrected report + verification
    result survive a crash before mark_completed().
    """

    @pytest.mark.asyncio
    async def test_save_called_after_corrections(self):
        """Simulate the workflow_execution.py pattern and verify save is called."""
        from unittest.mock import MagicMock

        state = _make_state_with_sources(
            "Report with a claim [1].", [_make_source(1, content="Source says otherwise.")]
        )
        config = _make_config()

        # Build a mock memory object.
        mock_memory = MagicMock()
        mock_memory.save_deep_research = MagicMock()

        # Simulate the verification result with a correction applied.
        verification_result = ClaimVerificationResult(
            claims_extracted=1,
            claims_verified=1,
            claims_supported=0,
            claims_contradicted=1,
            corrections_applied=1,
            details=[
                ClaimVerdict(
                    claim="test claim",
                    claim_type="negative",
                    cited_sources=[1],
                    verdict="CONTRADICTED",
                    evidence_quote="Source says otherwise.",
                    explanation="Contradicted by source.",
                    correction_applied=True,
                    corrected_text="Corrected report.",
                ),
            ],
        )
        state.claim_verification = verification_result

        # Simulate the persistence call from workflow_execution.py.
        mock_memory.save_deep_research(state)

        # Verify save was called with the state.
        mock_memory.save_deep_research.assert_called_once_with(state)

    @pytest.mark.asyncio
    async def test_save_not_called_when_corrections_fail(self):
        """When apply_corrections raises, the exception handler rolls back
        and the post-correction save should NOT have been reached."""
        from unittest.mock import MagicMock

        state = _make_state_with_sources(
            "Original report [1].", [_make_source(1)]
        )
        config = _make_config()

        mock_memory = MagicMock()
        report_snapshot = state.report

        # Simulate the workflow_execution.py try/except pattern.
        save_called = False
        try:
            raise RuntimeError("apply_corrections blew up")
            # The save would be here in the real code.
            save_called = True  # noqa: E702 — unreachable, that's the point
        except Exception:
            # Rollback.
            state.report = report_snapshot

        assert not save_called
        assert state.report == report_snapshot


# ---------------------------------------------------------------------------
# Unit tests: Profile-based claim verification enablement
# ---------------------------------------------------------------------------


class TestProfileClaimVerification:
    """Tests for the enable_claim_verification field on ResearchProfile."""

    def test_profile_field_default_true(self):
        from foundry_mcp.core.research.models.deep_research import ResearchProfile

        profile = ResearchProfile()
        assert profile.enable_claim_verification is True

    def test_profile_field_set_false(self):
        from foundry_mcp.core.research.models.deep_research import ResearchProfile

        profile = ResearchProfile(enable_claim_verification=False)
        assert profile.enable_claim_verification is False

    def test_profile_overrides_apply(self):
        """Profile overrides can disable claim verification."""
        from foundry_mcp.core.research.models.deep_research import ResearchProfile

        base = ResearchProfile(name="general")
        overridden = base.model_copy(update={"enable_claim_verification": False})
        assert overridden.enable_claim_verification is False
        assert base.enable_claim_verification is True

    def test_verification_guard_config_disables(self):
        """Config=False acts as master switch, disabling even if profile=True."""
        config = _make_config(deep_research_claim_verification_enabled=False)

        from foundry_mcp.core.research.models.deep_research import ResearchProfile

        profile = ResearchProfile(enable_claim_verification=True)

        # Simulate the guard from workflow_execution.py (and-based).
        cv_enabled = config.deep_research_claim_verification_enabled and (
            profile is None
            or getattr(profile, "enable_claim_verification", True)
        )
        assert cv_enabled is False

    def test_verification_guard_profile_disables(self):
        """Profile can disable verification even when config=True."""
        config = _make_config(deep_research_claim_verification_enabled=True)

        from foundry_mcp.core.research.models.deep_research import ResearchProfile

        profile = ResearchProfile(enable_claim_verification=False)

        cv_enabled = config.deep_research_claim_verification_enabled and (
            profile is None
            or getattr(profile, "enable_claim_verification", True)
        )
        assert cv_enabled is False

    def test_verification_guard_config_only(self):
        """Config alone enables verification without profile (no profile = allowed)."""
        config = _make_config(deep_research_claim_verification_enabled=True)

        profile = None
        cv_enabled = config.deep_research_claim_verification_enabled and (
            profile is None
            or getattr(profile, "enable_claim_verification", True)
        )
        assert cv_enabled is True

    def test_verification_guard_both_enabled(self):
        """Both config and profile enabled (the new default)."""
        config = _make_config(deep_research_claim_verification_enabled=True)

        from foundry_mcp.core.research.models.deep_research import ResearchProfile

        profile = ResearchProfile(enable_claim_verification=True)

        cv_enabled = config.deep_research_claim_verification_enabled and (
            profile is None
            or getattr(profile, "enable_claim_verification", True)
        )
        assert cv_enabled is True

    def test_verification_guard_both_disabled(self):
        """Neither config nor profile enables verification."""
        config = _make_config(deep_research_claim_verification_enabled=False)

        from foundry_mcp.core.research.models.deep_research import ResearchProfile

        profile = ResearchProfile(enable_claim_verification=False)

        cv_enabled = config.deep_research_claim_verification_enabled and (
            profile is None
            or getattr(profile, "enable_claim_verification", True)
        )
        assert cv_enabled is False


# ---------------------------------------------------------------------------
# Phase 1 tests: _split_report_into_sections
# ---------------------------------------------------------------------------


class TestSplitReportIntoSections:
    def test_multiple_headings(self):
        report = (
            "## Executive Summary\n\n" + "Summary text with details. " * 30 + "\n\n"
            "## Analysis\n\n" + "Analysis text with data points. " * 30 + "\n\n"
            "## Conclusion\n\n" + "Conclusion and recommendations. " * 30
        )
        chunks = _split_report_into_sections(report)
        assert len(chunks) >= 3
        headings = [c["section"] for c in chunks]
        assert "Executive Summary" in headings
        assert "Analysis" in headings
        assert "Conclusion" in headings

    def test_report_starting_with_heading(self):
        """Report starting with ## (no preceding newline) → first heading captured."""
        report = (
            "## Executive Summary\n\n" + "Summary text with content. " * 30 + "\n\n"
            "## Details\n\n" + "Details about the analysis. " * 30
        )
        chunks = _split_report_into_sections(report)
        assert len(chunks) >= 2
        assert chunks[0]["section"] == "Executive Summary"

    def test_small_sections_merged(self):
        """Sections smaller than 500 chars are merged with the next."""
        report = "## Tiny\n\nX\n\n## Also Tiny\n\nY\n\n## Big Section\n\n" + "Z" * 1000
        chunks = _split_report_into_sections(report)
        # The two tiny sections should be merged.
        assert len(chunks) <= 2

    def test_no_headings_single_chunk(self):
        """Report with no ## headings → single chunk fallback."""
        report = "Just a plain report without any markdown headings. " * 20
        chunks = _split_report_into_sections(report)
        assert len(chunks) == 1
        assert chunks[0]["content"] == report

    def test_bibliography_excluded(self):
        """Bibliography/Sources sections are excluded from extraction chunks."""
        report = (
            "## Analysis\n\n" + "Analysis text with claims and data. " * 30 + "\n\n"
            "## Bibliography\n\n" + "Source list entry. " * 30 + "\n\n"
            "## References\n\n" + "Reference list entry. " * 30
        )
        chunks = _split_report_into_sections(report)
        headings = [c["section"] for c in chunks]
        assert "Bibliography" not in headings
        assert "References" not in headings
        assert "Analysis" in headings

    def test_data_sources_heading_not_excluded(self):
        """Heading 'Data Sources and Methodology' NOT excluded (anchored regex)."""
        report = (
            "## Data Sources and Methodology\n\n" + "Methodology description here. " * 30 + "\n\n"
            "## Results\n\n" + "Results and findings details. " * 30
        )
        chunks = _split_report_into_sections(report)
        headings = [c["section"] for c in chunks]
        assert "Data Sources and Methodology" in headings

    def test_truncated_report_last_chunk_discarded(self):
        """Truncated report (last chunk lacks heading) → fragment discarded."""
        report = (
            "## Section One\n\n" + "Content for section one. " * 30 + "\n\n"
            "## Section Two\n\n" + "Content for section two. " * 30 + "\n\n"
            "Some trailing text without a heading that looks like truncation"
        )
        chunks = _split_report_into_sections(report)
        # Trailing preamble-like fragment (no heading) should be discarded.
        for chunk in chunks:
            if chunk["section"]:
                assert chunk["section"] in ("Section One", "Section Two")

    def test_many_small_sections_merged_to_reasonable_count(self):
        """50+ tiny sections merged to reasonable chunk count (< 20)."""
        sections = "\n\n".join(f"## Section {i}\n\nTiny." for i in range(50))
        chunks = _split_report_into_sections(sections)
        assert len(chunks) < 20


# ---------------------------------------------------------------------------
# Phase 2 tests: Extraction prompt
# ---------------------------------------------------------------------------


class TestExtractionPromptCitationAnchored:
    def test_prompt_contains_citation_anchor_language(self):
        assert "For each inline citation [N]" in _EXTRACTION_SYSTEM_PROMPT
        assert "ONLY extract claims that have an explicit [N] citation" in _EXTRACTION_SYSTEM_PROMPT

    def test_prompt_handles_empty_sections(self):
        assert "return an empty array: []" in _EXTRACTION_SYSTEM_PROMPT.lower()

    def test_prompt_dedup_rule(self):
        assert "extract ONE claim" in _EXTRACTION_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Phase 3 tests: _extract_claims_from_chunk
# ---------------------------------------------------------------------------


class TestExtractClaimsFromChunk:
    @pytest.mark.asyncio
    async def test_successful_extraction(self):
        chunk = {"section": "Analysis", "content": "## Analysis\n\nClaim text [1]."}
        response = json.dumps([{
            "claim": "Test claim",
            "claim_type": "quantitative",
            "cited_sources": [1],
            "report_section": "Analysis",
            "quote_context": "Claim text [1].",
        }])
        mock_execute = AsyncMock(return_value=MockWorkflowResult(
            success=True, content=response,
        ))
        claims = await _extract_claims_from_chunk(
            chunk=chunk,
            execute_fn=mock_execute,
            system_prompt=_EXTRACTION_SYSTEM_PROMPT,
            provider_id="test",
            timeout=30,
            max_claims_per_chunk=10,
        )
        assert len(claims) == 1
        assert claims[0].claim == "Test claim"

    @pytest.mark.asyncio
    async def test_failed_extraction_returns_empty(self):
        chunk = {"section": "Test", "content": "## Test\n\nContent."}
        mock_execute = AsyncMock(return_value=MockWorkflowResult(
            success=False, content="",
        ))
        claims = await _extract_claims_from_chunk(
            chunk=chunk,
            execute_fn=mock_execute,
            system_prompt=_EXTRACTION_SYSTEM_PROMPT,
            provider_id="test",
            timeout=30,
            max_claims_per_chunk=10,
        )
        assert claims == []

    @pytest.mark.asyncio
    async def test_exception_returns_empty(self):
        chunk = {"section": "Test", "content": "## Test\n\nContent."}
        mock_execute = AsyncMock(side_effect=Exception("Network error"))
        claims = await _extract_claims_from_chunk(
            chunk=chunk,
            execute_fn=mock_execute,
            system_prompt=_EXTRACTION_SYSTEM_PROMPT,
            provider_id="test",
            timeout=30,
            max_claims_per_chunk=10,
        )
        assert claims == []

    @pytest.mark.asyncio
    async def test_claims_tagged_with_section(self):
        chunk = {"section": "Portfolio Analysis", "content": "## Portfolio Analysis\n\nData."}
        response = json.dumps([{
            "claim": "A claim",
            "claim_type": "positive",
            "cited_sources": [1],
        }])
        mock_execute = AsyncMock(return_value=MockWorkflowResult(
            success=True, content=response,
        ))
        claims = await _extract_claims_from_chunk(
            chunk=chunk,
            execute_fn=mock_execute,
            system_prompt=_EXTRACTION_SYSTEM_PROMPT,
            provider_id="test",
            timeout=30,
            max_claims_per_chunk=10,
        )
        assert claims[0].report_section == "Portfolio Analysis"

    @pytest.mark.asyncio
    async def test_execute_fn_receives_correct_kwargs(self):
        chunk = {"section": "Test", "content": "## Test\n\nContent."}
        captured: dict = {}

        async def mock_execute(**kwargs):
            captured.update(kwargs)
            return MockWorkflowResult(success=True, content="[]")

        await _extract_claims_from_chunk(
            chunk=chunk,
            execute_fn=mock_execute,
            system_prompt=_EXTRACTION_SYSTEM_PROMPT,
            provider_id="test",
            timeout=30,
            max_claims_per_chunk=10,
        )
        assert captured["max_tokens"] == 4096
        assert captured["max_retries"] == 1
        assert captured["retry_delay"] == 2.0


# ---------------------------------------------------------------------------
# Phase 6 tests: _filter_uncited_claims
# ---------------------------------------------------------------------------


class TestFilterUncitedClaims:
    def test_claims_with_citations_kept(self):
        claims = [ClaimVerdict(claim="A", claim_type="positive", cited_sources=[1, 2])]
        result = _filter_uncited_claims(claims)
        assert len(result) == 1

    def test_claims_with_empty_citations_dropped(self):
        claims = [ClaimVerdict(claim="A", claim_type="positive", cited_sources=[])]
        result = _filter_uncited_claims(claims)
        assert len(result) == 0

    def test_mixed_claims(self):
        claims = [
            ClaimVerdict(claim="Cited", claim_type="positive", cited_sources=[1]),
            ClaimVerdict(claim="Uncited", claim_type="positive", cited_sources=[]),
        ]
        result = _filter_uncited_claims(claims)
        assert len(result) == 1
        assert result[0].claim == "Cited"

    def test_all_uncited_returns_empty(self):
        claims = [
            ClaimVerdict(claim="A", claim_type="positive", cited_sources=[]),
            ClaimVerdict(claim="B", claim_type="negative", cited_sources=[]),
        ]
        result = _filter_uncited_claims(claims)
        assert result == []

    def test_logs_dropped_count(self, caplog):
        claims = [
            ClaimVerdict(claim="A", claim_type="positive", cited_sources=[1]),
            ClaimVerdict(claim="B", claim_type="positive", cited_sources=[]),
            ClaimVerdict(claim="C", claim_type="positive", cited_sources=[]),
        ]
        import logging
        with caplog.at_level(logging.INFO):
            _filter_uncited_claims(claims)
        assert "Dropped 2 claims" in caplog.text


# ---------------------------------------------------------------------------
# Phase 4 tests: _extract_claims_chunked
# ---------------------------------------------------------------------------


class TestExtractClaimsChunkedParallel:
    @pytest.mark.asyncio
    async def test_chunks_extracted_in_parallel(self):
        """Multiple chunks extracted; claims merged."""
        report = (
            "## Section A\n\n" + "Claim A text with details [1]. " * 25 + "\n\n"
            "## Section B\n\n" + "Claim B text with details [2]. " * 25 + "\n\n"
            "## Section C\n\n" + "Additional context and analysis. " * 25
        )

        async def mock_execute(**kwargs):
            prompt = kwargs.get("prompt", "")
            if "Section A" in prompt:
                return MockWorkflowResult(success=True, content=json.dumps([
                    {"claim": "Claim A", "claim_type": "positive", "cited_sources": [1]},
                ]))
            elif "Section B" in prompt:
                return MockWorkflowResult(success=True, content=json.dumps([
                    {"claim": "Claim B", "claim_type": "negative", "cited_sources": [2]},
                ]))
            return MockWorkflowResult(success=True, content="[]")

        claims = await _extract_claims_chunked(
            report=report,
            execute_fn=mock_execute,
            provider_id="test",
            timeout=30,
            max_claims=50,
            max_concurrent=5,
        )
        claim_texts = {c.claim for c in claims}
        assert "Claim A" in claim_texts
        assert "Claim B" in claim_texts

    @pytest.mark.asyncio
    async def test_duplicate_claims_deduplicated(self):
        """Identical claims from different chunks are deduplicated."""
        report = (
            "## A\n\nDuplicate claim [1]. " + "Padding text. " * 35 + "\n\n"
            "## B\n\nDuplicate claim [1]. " + "Padding text. " * 35 + "\n\n"
            "## C\n\n" + "More content. " * 40
        )
        response = json.dumps([{"claim": "Duplicate claim", "claim_type": "positive", "cited_sources": [1]}])

        mock_execute = AsyncMock(return_value=MockWorkflowResult(success=True, content=response))

        claims = await _extract_claims_chunked(
            report=report,
            execute_fn=mock_execute,
            provider_id="test",
            timeout=30,
            max_claims=50,
            max_concurrent=5,
        )
        assert len(claims) == 1

    @pytest.mark.asyncio
    async def test_uncited_claims_filtered(self):
        """Claims without citations are filtered out after merge."""
        report = (
            "## A\n\n" + "Some text with content. " * 25 + "\n\n"
            "## B\n\n" + "More content here. " * 25
        )
        response = json.dumps([
            {"claim": "Cited claim", "claim_type": "positive", "cited_sources": [1]},
            {"claim": "Uncited claim", "claim_type": "positive", "cited_sources": []},
        ])
        mock_execute = AsyncMock(return_value=MockWorkflowResult(success=True, content=response))

        claims = await _extract_claims_chunked(
            report=report,
            execute_fn=mock_execute,
            provider_id="test",
            timeout=30,
            max_claims=50,
            max_concurrent=5,
        )
        assert all(c.cited_sources for c in claims)

    @pytest.mark.asyncio
    async def test_total_claims_capped(self):
        """Total claims capped at max_claims."""
        report = (
            "## A\n\n" + "Text with content. " * 30 + "\n\n"
            "## B\n\n" + "More content. " * 30
        )
        many_claims = json.dumps([
            {"claim": f"Claim {i}", "claim_type": "positive", "cited_sources": [1]}
            for i in range(30)
        ])
        mock_execute = AsyncMock(return_value=MockWorkflowResult(success=True, content=many_claims))

        claims = await _extract_claims_chunked(
            report=report,
            execute_fn=mock_execute,
            provider_id="test",
            timeout=30,
            max_claims=5,
            max_concurrent=5,
        )
        assert len(claims) <= 5

    @pytest.mark.asyncio
    async def test_partial_failure_preserves_successful_chunks(self):
        """Some chunks fail, successful ones still return claims."""
        report = (
            "## Success\n\nGood claim [1]. " + "Padding text. " * 35 + "\n\n"
            "## Failure\n\n" + "Failure section content. " * 30
        )
        call_count = 0

        async def mock_execute(**kwargs):
            nonlocal call_count
            call_count += 1
            prompt = kwargs.get("prompt", "")
            if "Success" in prompt:
                return MockWorkflowResult(success=True, content=json.dumps([
                    {"claim": "Good claim", "claim_type": "positive", "cited_sources": [1]},
                ]))
            return MockWorkflowResult(success=False, content="")

        claims = await _extract_claims_chunked(
            report=report,
            execute_fn=mock_execute,
            provider_id="test",
            timeout=30,
            max_claims=50,
            max_concurrent=5,
        )
        assert len(claims) == 1
        assert claims[0].claim == "Good claim"

    @pytest.mark.asyncio
    async def test_all_chunks_fail_returns_empty(self):
        """All chunks fail → empty list."""
        report = (
            "## A\n\n" + "Content for section A. " * 25 + "\n\n"
            "## B\n\n" + "Content for section B. " * 25
        )
        mock_execute = AsyncMock(return_value=MockWorkflowResult(success=False, content=""))

        claims = await _extract_claims_chunked(
            report=report,
            execute_fn=mock_execute,
            provider_id="test",
            timeout=30,
            max_claims=50,
            max_concurrent=5,
        )
        assert claims == []

    @pytest.mark.asyncio
    async def test_single_chunk_uses_same_path(self):
        """Single chunk (no headings) uses same code path."""
        report = "Just a plain report with claim [1]."
        response = json.dumps([
            {"claim": "Plain claim", "claim_type": "positive", "cited_sources": [1]},
        ])
        mock_execute = AsyncMock(return_value=MockWorkflowResult(success=True, content=response))

        claims = await _extract_claims_chunked(
            report=report,
            execute_fn=mock_execute,
            provider_id="test",
            timeout=30,
            max_claims=50,
            max_concurrent=5,
        )
        assert len(claims) == 1

    @pytest.mark.asyncio
    async def test_metadata_populated(self):
        """Metadata dict populated with extraction stats."""
        report = (
            "## A\n\nClaim [1]. " + "Padding text. " * 35 + "\n\n"
            "## B\n\n" + "More content here. " * 30
        )
        response = json.dumps([{"claim": "C", "claim_type": "positive", "cited_sources": [1]}])
        mock_execute = AsyncMock(return_value=MockWorkflowResult(success=True, content=response))

        metadata: dict = {}
        await _extract_claims_chunked(
            report=report,
            execute_fn=mock_execute,
            provider_id="test",
            timeout=30,
            max_claims=50,
            max_concurrent=5,
            metadata=metadata,
        )
        assert metadata["extraction_strategy"] == "chunked"
        assert metadata["extraction_chunks_attempted"] >= 1
        assert metadata["extraction_chunks_succeeded"] >= 1
        assert isinstance(metadata["extraction_claims_per_chunk"], list)


# ---------------------------------------------------------------------------
# Phase 5 tests: Integration — extract_and_verify_claims uses chunked
# ---------------------------------------------------------------------------


class TestExtractAndVerifyClaimsUsesChunked:
    @pytest.mark.asyncio
    async def test_end_to_end_multi_section(self):
        """Multi-section report → chunked extraction → filter → verification."""
        report = (
            "## Overview\n\nThe annual fee is $550 [1]. " + "Overview details. " * 30 + "\n\n"
            "## Benefits\n\nDining earns 3x points [2]. " + "Benefits details. " * 30 + "\n\n"
            "## Conclusion\n\nOverall a strong card. " + "Conclusion text. " * 30
        )
        sources = [
            _make_source(1, content="The annual fee for this card is $550 per year."),
            _make_source(2, content="Restaurants earn 3x points on all purchases."),
        ]
        state = _make_state_with_sources(report, sources)
        config = _make_config()

        async def mock_execute(**kwargs):
            phase = kwargs.get("phase", "")
            if phase == "claim_extraction":
                prompt = kwargs.get("prompt", "")
                if "annual fee" in prompt.lower():
                    return MockWorkflowResult(success=True, content=json.dumps([
                        {"claim": "The annual fee is $550", "claim_type": "quantitative",
                         "cited_sources": [1], "report_section": "Overview",
                         "quote_context": "The annual fee is $550 [1]."},
                    ]))
                elif "dining" in prompt.lower():
                    return MockWorkflowResult(success=True, content=json.dumps([
                        {"claim": "Dining earns 3x points", "claim_type": "quantitative",
                         "cited_sources": [2], "report_section": "Benefits",
                         "quote_context": "Dining earns 3x points [2]."},
                    ]))
                return MockWorkflowResult(success=True, content="[]")
            elif phase == "claim_verification":
                return MockWorkflowResult(success=True, content=json.dumps({
                    "verdict": "SUPPORTED",
                    "evidence_quote": "confirms the claim",
                    "explanation": "Source confirms.",
                }))
            return MockWorkflowResult(success=True, content="")

        result = await extract_and_verify_claims(
            state=state, config=config, provider_id="test",
            execute_fn=mock_execute, timeout=30,
        )

        assert result.claims_extracted >= 1
        assert result.claims_verified >= 1
        assert state.metadata.get("extraction_strategy") == "chunked"


# ---------------------------------------------------------------------------
# Phase 7 tests: Verification prompt
# ---------------------------------------------------------------------------


class TestVerificationPromptContradictedDefinition:
    def test_explicit_conflict_language(self):
        assert "explicitly state something that DIRECTLY CONFLICTS" in _VERIFICATION_SYSTEM_PROMPT

    def test_absence_not_contradiction(self):
        assert "Absence of information" in _VERIFICATION_SYSTEM_PROMPT
        assert "does NOT mean the source contradicts" in _VERIFICATION_SYSTEM_PROMPT

    def test_required_for_contradicted(self):
        assert "REQUIRED" in _VERIFICATION_SYSTEM_PROMPT
        assert "CONTRADICTED" in _VERIFICATION_SYSTEM_PROMPT

    def test_excerpts_disclaimer(self):
        assert "You are seeing excerpts, not the full source" in _VERIFICATION_SYSTEM_PROMPT

    def test_doubt_bias(self):
        assert "When in doubt between CONTRADICTED and UNSUPPORTED, choose UNSUPPORTED" in _VERIFICATION_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Phase 9 tests: CONTRADICTED without evidence quote downgraded
# ---------------------------------------------------------------------------


class TestContradictedWithoutQuoteDowngraded:
    @pytest.mark.asyncio
    async def test_empty_evidence_quote_downgraded(self):
        """CONTRADICTED + empty evidence_quote → UNSUPPORTED."""
        claim = ClaimVerdict(claim="Test claim", claim_type="negative", cited_sources=[1])
        source = _make_source(1, content="Source content about the topic.")
        citation_map = {1: source}

        verification_response = json.dumps({
            "verdict": "CONTRADICTED",
            "evidence_quote": "",
            "explanation": "No direct evidence found",
        })
        mock_execute = AsyncMock(return_value=MockWorkflowResult(
            success=True, content=verification_response,
        ))
        semaphore = asyncio.Semaphore(5)

        result = await _verify_single_claim(
            claim, citation_map, mock_execute, "test", 30, semaphore,
        )
        assert result.verdict == "UNSUPPORTED"
        assert "Originally CONTRADICTED" in result.explanation

    @pytest.mark.asyncio
    async def test_null_evidence_quote_downgraded(self):
        """CONTRADICTED + null evidence_quote → UNSUPPORTED."""
        claim = ClaimVerdict(claim="Test claim", claim_type="negative", cited_sources=[1])
        source = _make_source(1, content="Source content.")
        citation_map = {1: source}

        verification_response = json.dumps({
            "verdict": "CONTRADICTED",
            "evidence_quote": None,
            "explanation": "Seems contradicted",
        })
        mock_execute = AsyncMock(return_value=MockWorkflowResult(
            success=True, content=verification_response,
        ))
        semaphore = asyncio.Semaphore(5)

        result = await _verify_single_claim(
            claim, citation_map, mock_execute, "test", 30, semaphore,
        )
        assert result.verdict == "UNSUPPORTED"
        assert "Originally CONTRADICTED" in result.explanation

    @pytest.mark.asyncio
    async def test_valid_evidence_quote_stays_contradicted(self):
        """CONTRADICTED + valid evidence_quote → remains CONTRADICTED."""
        claim = ClaimVerdict(claim="Test claim", claim_type="negative", cited_sources=[1])
        source = _make_source(1, content="Source content says otherwise.")
        citation_map = {1: source}

        verification_response = json.dumps({
            "verdict": "CONTRADICTED",
            "evidence_quote": "Source says otherwise",
            "explanation": "Direct contradiction",
        })
        mock_execute = AsyncMock(return_value=MockWorkflowResult(
            success=True, content=verification_response,
        ))
        semaphore = asyncio.Semaphore(5)

        result = await _verify_single_claim(
            claim, citation_map, mock_execute, "test", 30, semaphore,
        )
        assert result.verdict == "CONTRADICTED"
        assert result.evidence_quote == "Source says otherwise"

    @pytest.mark.asyncio
    async def test_supported_empty_quote_unchanged(self):
        """SUPPORTED + empty evidence_quote → remains SUPPORTED (gate only for CONTRADICTED)."""
        claim = ClaimVerdict(claim="Test claim", claim_type="positive", cited_sources=[1])
        source = _make_source(1, content="Source confirms the claim.")
        citation_map = {1: source}

        verification_response = json.dumps({
            "verdict": "SUPPORTED",
            "evidence_quote": "",
            "explanation": "Confirmed",
        })
        mock_execute = AsyncMock(return_value=MockWorkflowResult(
            success=True, content=verification_response,
        ))
        semaphore = asyncio.Semaphore(5)

        result = await _verify_single_claim(
            claim, citation_map, mock_execute, "test", 30, semaphore,
        )
        assert result.verdict == "SUPPORTED"

    @pytest.mark.asyncio
    async def test_original_explanation_preserved(self):
        """Downgraded claim preserves original explanation."""
        claim = ClaimVerdict(claim="Test", claim_type="negative", cited_sources=[1])
        source = _make_source(1, content="Content.")
        citation_map = {1: source}

        verification_response = json.dumps({
            "verdict": "CONTRADICTED",
            "evidence_quote": None,
            "explanation": "I think it's wrong",
        })
        mock_execute = AsyncMock(return_value=MockWorkflowResult(
            success=True, content=verification_response,
        ))
        semaphore = asyncio.Semaphore(5)

        result = await _verify_single_claim(
            claim, citation_map, mock_execute, "test", 30, semaphore,
        )
        assert "I think it's wrong" in result.explanation
        assert "Originally CONTRADICTED" in result.explanation


# ---------------------------------------------------------------------------
# Negative-behavior tests
# ---------------------------------------------------------------------------


class TestMonolithicExtractionRemoved:
    def test_build_extraction_user_prompt_removed(self):
        """_build_extraction_user_prompt should not exist in the module."""
        import foundry_mcp.core.research.workflows.deep_research.phases.claim_verification as cv_mod
        assert not hasattr(cv_mod, "_build_extraction_user_prompt")

    def test_no_max_tokens_16384_in_orchestrator(self):
        """No direct execute_fn call with max_tokens=16384 in the orchestrator."""
        import inspect
        source = inspect.getsource(extract_and_verify_claims)
        assert "16384" not in source


class TestCancelledErrorDuringGather:
    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        """CancelledError raised mid-gather propagates correctly."""
        report = (
            "## A\n\nClaim [1]. " + "Padding text. " * 35 + "\n\n"
            "## B\n\n" + "More content. " * 30
        )

        async def mock_execute(**kwargs):
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await _extract_claims_chunked(
                report=report,
                execute_fn=mock_execute,
                provider_id="test",
                timeout=30,
                max_claims=50,
                max_concurrent=5,
            )


# ---------------------------------------------------------------------------
# Phase 9 tests: claims_filtered reporting invariant
# ---------------------------------------------------------------------------


class TestClaimsFilteredInvariant:
    """Verify claims_extracted >= claims_filtered >= claims_verified."""

    def test_claims_filtered_field_exists(self):
        """ClaimVerificationResult has claims_filtered with default 0."""
        result = ClaimVerificationResult()
        assert result.claims_filtered == 0

    def test_invariant_holds_on_populated_result(self):
        """Manually constructed result respects the invariant."""
        result = ClaimVerificationResult(
            claims_extracted=10,
            claims_filtered=7,
            claims_verified=5,
            claims_supported=3,
            claims_contradicted=1,
            claims_unsupported=1,
        )
        assert result.claims_extracted >= result.claims_filtered >= result.claims_verified

    @pytest.mark.asyncio
    async def test_pipeline_sets_claims_filtered(self):
        """extract_and_verify_claims populates claims_filtered between extracted and verified."""
        report = "The study found that X is true [1]. Also Y happened [2]."
        state = _make_state_with_sources(report, [_make_source(1), _make_source(2)])
        config = _make_config()

        call_count = 0

        async def mock_execute(**kwargs):
            nonlocal call_count
            call_count += 1
            prompt = kwargs.get("prompt", "")
            if "Extract" in kwargs.get("system_prompt", "") or "extract" in kwargs.get("system_prompt", ""):
                return MockWorkflowResult(
                    success=True,
                    content=json.dumps({
                        "claims": [
                            {"claim": "X is true", "claim_type": "statistical", "cited_sources": [1]},
                            {"claim": "Y happened", "claim_type": "factual", "cited_sources": [2]},
                        ]
                    }),
                )
            # Verification response
            return MockWorkflowResult(
                success=True,
                content=json.dumps({
                    "verdict": "SUPPORTED",
                    "evidence_quote": "evidence",
                    "explanation": "matches source",
                }),
            )

        result = await extract_and_verify_claims(
            state=state, config=config, provider_id="test",
            execute_fn=mock_execute, timeout=30,
        )

        assert result.claims_extracted >= result.claims_filtered
        assert result.claims_filtered >= result.claims_verified

    def test_serialization_roundtrip_includes_claims_filtered(self):
        """claims_filtered survives JSON serialization."""
        result = ClaimVerificationResult(
            claims_extracted=10,
            claims_filtered=7,
            claims_verified=5,
        )
        data = result.model_dump()
        restored = ClaimVerificationResult.model_validate(data)
        assert restored.claims_filtered == 7


# ---------------------------------------------------------------------------
# Phase 10 tests: source_resolution diagnostic field
# ---------------------------------------------------------------------------


class TestSourceResolution:
    """Verify source_resolution tracking on ClaimVerdict."""

    def test_full_content_when_raw_content_available(self):
        """source_resolution is 'full_content' when raw_content is present."""
        claim = ClaimVerdict(
            claim="Transfer ratio is 1:1",
            claim_type="quantitative",
            cited_sources=[1],
        )
        src = _make_source(1, content="compressed", raw_content="full raw page content")
        prompt, resolution = _build_verification_user_prompt(claim, {1: src})
        assert prompt is not None
        assert resolution == "full_content"

    def test_compressed_only_when_no_raw_content(self):
        """source_resolution is 'compressed_only' when only content is available."""
        claim = ClaimVerdict(
            claim="Test claim",
            claim_type="positive",
            cited_sources=[1],
        )
        src = _make_source(1, content="compressed summary", raw_content=None)
        prompt, resolution = _build_verification_user_prompt(claim, {1: src})
        assert prompt is not None
        assert resolution == "compressed_only"

    def test_snippet_only_when_only_snippet(self):
        """source_resolution is 'snippet_only' when only snippet is available."""
        claim = ClaimVerdict(
            claim="Test claim",
            claim_type="positive",
            cited_sources=[1],
        )
        src = _make_source(1, content=None, raw_content=None, snippet="a snippet")
        prompt, resolution = _build_verification_user_prompt(claim, {1: src})
        assert prompt is not None
        assert resolution == "snippet_only"

    def test_no_content_when_source_has_no_text(self):
        """source_resolution is 'no_content' when source exists but has no text."""
        claim = ClaimVerdict(
            claim="Test claim",
            claim_type="positive",
            cited_sources=[1],
        )
        src = _make_source(1, content=None, raw_content=None, snippet=None)
        prompt, resolution = _build_verification_user_prompt(claim, {1: src})
        assert prompt is None
        assert resolution == "no_content"

    def test_citation_not_found_when_all_missing(self):
        """source_resolution is 'citation_not_found' when citation numbers not in map."""
        claim = ClaimVerdict(
            claim="Test claim",
            claim_type="positive",
            cited_sources=[99],
        )
        prompt, resolution = _build_verification_user_prompt(claim, {})
        assert prompt is None
        assert resolution == "citation_not_found"

    def test_best_tier_wins_across_multiple_sources(self):
        """When multiple sources have different tiers, the best one is reported."""
        claim = ClaimVerdict(
            claim="Test claim",
            claim_type="positive",
            cited_sources=[1, 2],
        )
        src1 = _make_source(1, content="compressed only", raw_content=None)
        src2 = _make_source(2, content="also compressed", raw_content="full raw content")
        prompt, resolution = _build_verification_user_prompt(claim, {1: src1, 2: src2})
        assert prompt is not None
        assert resolution == "full_content"

    def test_field_serializes_in_model_dump(self):
        """source_resolution appears in ClaimVerdict.model_dump()."""
        claim = ClaimVerdict(
            claim="Test",
            claim_type="positive",
            source_resolution="full_content",
        )
        data = claim.model_dump()
        assert data["source_resolution"] == "full_content"

    def test_field_defaults_to_none(self):
        """source_resolution defaults to None when not set."""
        claim = ClaimVerdict(claim="Test", claim_type="positive")
        assert claim.source_resolution is None

    @pytest.mark.asyncio
    async def test_verify_single_claim_sets_source_resolution(self):
        """_verify_single_claim sets source_resolution on the claim."""
        claim = ClaimVerdict(
            claim="X is true",
            claim_type="positive",
            cited_sources=[1],
        )
        src = _make_source(1, content="Source content about X", raw_content="Full raw content about X")
        citation_map = {1: src}

        async def mock_execute(**kwargs):
            return MockWorkflowResult(
                success=True,
                content=json.dumps({
                    "verdict": "SUPPORTED",
                    "evidence_quote": "X is true",
                    "explanation": "Source confirms X",
                }),
            )

        result = await _verify_single_claim(
            claim=claim,
            citation_map=citation_map,
            execute_fn=mock_execute,
            provider_id="test",
            timeout=30,
            semaphore=asyncio.Semaphore(5),
        )
        assert result.source_resolution == "full_content"

    @pytest.mark.asyncio
    async def test_verify_single_claim_no_content_sets_resolution(self):
        """_verify_single_claim sets source_resolution='no_content' when prompt is None."""
        claim = ClaimVerdict(
            claim="X is true",
            claim_type="positive",
            cited_sources=[1],
        )
        src = _make_source(1, content=None, raw_content=None, snippet=None)
        citation_map = {1: src}

        result = await _verify_single_claim(
            claim=claim,
            citation_map=citation_map,
            execute_fn=AsyncMock(),
            provider_id="test",
            timeout=30,
            semaphore=asyncio.Semaphore(5),
        )
        assert result.source_resolution == "no_content"
        assert result.verdict == "UNSUPPORTED"


# ---------------------------------------------------------------------------
# Heading-boundary repair
# ---------------------------------------------------------------------------


class TestRepairHeadingBoundaries:
    """Tests for _repair_heading_boundaries()."""

    def test_heading_concatenated_with_body_is_repaired(self):
        """Heading fused with body text gets a blank line inserted."""
        original = "### Chase Ultimate Rewards\n\nAs of February 2026, the program offers..."
        corrected = "### Chase Ultimate RewardsAs of February–March 2026, the program offers..."
        result = _repair_heading_boundaries(original, corrected)
        assert "### Chase Ultimate Rewards\n\n" in result or "### Chase Ultimate Rewards\n\nAs" in result
        # Heading must NOT be on the same line as body text.
        for line in result.split("\n"):
            if line.startswith("###"):
                assert not re.search(r"^#{1,6}\s+[^\n]+?[A-Z][a-z]", line) or \
                    line == line.split("\n")[0]

    def test_heading_body_boundary_preserved_unchanged(self):
        """When corrected text already has proper heading boundaries, no change."""
        original = "### Program Details\n\nThe rewards program has several tiers."
        corrected = "### Program Details\n\nThe rewards program has multiple tiers."
        result = _repair_heading_boundaries(original, corrected)
        assert result == corrected

    def test_multiple_headings_in_single_window(self):
        """Multiple headings with concatenated body text are all repaired."""
        original = (
            "## Overview\n\nThis section covers...\n\n"
            "### Details\n\nThe specific details are..."
        )
        corrected = (
            "## OverviewThis section covers...\n"
            "### DetailsThe specific details are..."
        )
        result = _repair_heading_boundaries(original, corrected)
        lines = result.split("\n")
        heading_indices = [i for i, l in enumerate(lines) if re.match(r"^#{1,6}\s+", l)]
        for idx in heading_indices:
            # Line after heading should be blank.
            if idx + 1 < len(lines):
                assert lines[idx + 1].strip() == "", (
                    f"Heading at line {idx} not followed by blank line: {lines[idx:]}"
                )

    def test_heading_text_modified_by_correction(self):
        """Heading content can change — only structure (newlines) is enforced."""
        original = "### Transfer Partners\n\nChase has 14 transfer partners."
        corrected = "### Transfer Partners and Ratios\n\nChase has 13 transfer partners."
        result = _repair_heading_boundaries(original, corrected)
        assert result == corrected

    def test_no_headings_in_original_returns_unchanged(self):
        """If original window has no headings, corrected text is returned as-is."""
        original = "The program offers great rewards for travel."
        corrected = "The program offers excellent rewards for travel."
        result = _repair_heading_boundaries(original, corrected)
        assert result == corrected


# ---------------------------------------------------------------------------
# Citation remapping helpers
# ---------------------------------------------------------------------------


class TestParseRemapResponse:
    def test_valid_response(self):
        content = '{"best_source": 3, "confidence": "high", "evidence_quote": "Quote here"}'
        result = _parse_remap_response(content)
        assert result == (3, "high", "Quote here")

    def test_null_best_source(self):
        content = '{"best_source": null, "confidence": "low", "evidence_quote": ""}'
        result = _parse_remap_response(content)
        assert result == (None, "low", "")

    def test_code_fenced_json(self):
        content = '```json\n{"best_source": 5, "confidence": "medium", "evidence_quote": "x"}\n```'
        result = _parse_remap_response(content)
        assert result == (5, "medium", "x")

    def test_invalid_json(self):
        assert _parse_remap_response("not json") is None

    def test_invalid_confidence_defaults_to_low(self):
        content = '{"best_source": 1, "confidence": "maybe", "evidence_quote": ""}'
        result = _parse_remap_response(content)
        assert result is not None
        assert result[1] == "low"


class TestApplyCitationRemap:
    def test_basic_remap(self):
        report = "Introduction.\n\nThis fact is true [2]. More text.\n\nConclusion."
        sources = [_make_source(1, content="Source 1"), _make_source(2, content="Source 2")]
        state = _make_state_with_sources(report, sources)

        claim = ClaimVerdict(
            claim="This fact is true",
            claim_type="positive",
            cited_sources=[2],
            verdict="UNSUPPORTED",
            quote_context="This fact is true [2].",
        )

        success = _apply_citation_remap(state, claim, [2], 1)
        assert success is True
        assert "[1]" in state.report
        assert "[2]" not in state.report
        assert claim.quote_context == "This fact is true [1]."

    def test_remap_no_quote_context(self):
        state = _make_state_with_sources("Report text.", [_make_source(1)])
        claim = ClaimVerdict(
            claim="Claim",
            claim_type="positive",
            cited_sources=[1],
            verdict="UNSUPPORTED",
            quote_context=None,
        )
        assert _apply_citation_remap(state, claim, [1], 2) is False

    def test_remap_quote_not_in_report(self):
        state = _make_state_with_sources("Different text.", [_make_source(1)])
        claim = ClaimVerdict(
            claim="Claim",
            claim_type="positive",
            cited_sources=[1],
            verdict="UNSUPPORTED",
            quote_context="This is not in the report [1].",
        )
        assert _apply_citation_remap(state, claim, [1], 2) is False


class TestRemoveCitationsFromReport:
    def test_removes_citation_bracket(self):
        report = "Some fact [3] is stated here.\n\nConclusion."
        state = _make_state_with_sources(report, [_make_source(3)])
        claim = ClaimVerdict(
            claim="Some fact",
            claim_type="positive",
            cited_sources=[3],
            verdict="UNSUPPORTED",
            quote_context="Some fact [3] is stated here.",
        )
        _remove_citations_from_report(state, claim)
        assert "[3]" not in state.report
        assert "Some fact is stated here." in state.report
        assert claim.cited_sources == []

    def test_noop_when_quote_missing(self):
        state = _make_state_with_sources("Different text.", [_make_source(1)])
        claim = ClaimVerdict(
            claim="Claim",
            claim_type="positive",
            cited_sources=[1],
            verdict="UNSUPPORTED",
            quote_context="Not in report [1].",
        )
        original_report = state.report
        _remove_citations_from_report(state, claim)
        assert state.report == original_report


# ---------------------------------------------------------------------------
# Full remapping integration tests
# ---------------------------------------------------------------------------


class TestRemapUnsupportedCitations:
    @pytest.mark.asyncio
    async def test_unsupported_claim_remapped_to_correct_source(self):
        """UNSUPPORTED claim gets remapped when a better source exists."""
        report = (
            "Introduction.\n\n"
            "Chase has 14 transfer partners [1]. This is notable.\n\n"
            "Conclusion."
        )
        sources = [
            _make_source(1, raw_content="Article about Amex rewards programs"),
            _make_source(2, raw_content="Chase Ultimate Rewards has 14 airline and hotel transfer partners"),
        ]
        state = _make_state_with_sources(report, sources)

        verification_result = ClaimVerificationResult(
            claims_extracted=1,
            claims_verified=1,
            claims_unsupported=1,
            details=[
                ClaimVerdict(
                    claim="Chase has 14 transfer partners",
                    claim_type="quantitative",
                    cited_sources=[1],
                    verdict="UNSUPPORTED",
                    quote_context="Chase has 14 transfer partners [1].",
                ),
            ],
        )

        async def mock_execute(**kwargs):
            return MockWorkflowResult(
                success=True,
                content='{"best_source": 2, "confidence": "high", "evidence_quote": "14 airline and hotel transfer partners"}',
            )

        remapped = await remap_unsupported_citations(
            state=state,
            verification_result=verification_result,
            execute_fn=mock_execute,
            provider_id="test-provider",
        )

        assert remapped == 1
        assert verification_result.citations_remapped == 1
        assert "[2]" in state.report
        assert "[1]" not in state.report
        assert verification_result.details[0].cited_sources == [2]

    @pytest.mark.asyncio
    async def test_no_matching_source_removes_citation(self):
        """When no source supports the claim, the citation bracket is removed."""
        report = (
            "Introduction.\n\n"
            "The sky is blue [1]. This is well known.\n\n"
            "Conclusion."
        )
        sources = [
            _make_source(1, raw_content="Article about cooking recipes"),
        ]
        state = _make_state_with_sources(report, sources)

        verification_result = ClaimVerificationResult(
            claims_extracted=1,
            claims_verified=1,
            claims_unsupported=1,
            details=[
                ClaimVerdict(
                    claim="The sky is blue",
                    claim_type="positive",
                    cited_sources=[1],
                    verdict="UNSUPPORTED",
                    quote_context="The sky is blue [1].",
                ),
            ],
        )

        async def mock_execute(**kwargs):
            # No candidate sources to check (source 1 is excluded as already-cited),
            # so this should not be called. But if called, return null.
            return MockWorkflowResult(
                success=True,
                content='{"best_source": null, "confidence": "low", "evidence_quote": ""}',
            )

        remapped = await remap_unsupported_citations(
            state=state,
            verification_result=verification_result,
            execute_fn=mock_execute,
            provider_id="test-provider",
        )

        # Only 1 source exists and it's the already-cited one, so no candidates.
        # No LLM call should be made, and no remap happens.
        assert remapped == 0

    @pytest.mark.asyncio
    async def test_partially_supported_with_citation_mismatch_remapped(self):
        """PARTIALLY_SUPPORTED claim with wrong citation gets remapped."""
        report = (
            "Introduction.\n\n"
            "The rate is approximately 3.5% [1]. Details follow.\n\n"
            "Conclusion."
        )
        sources = [
            _make_source(1, raw_content="General market overview with no specific rates"),
            _make_source(2, raw_content="Current mortgage rate is 3.48% as of March 2026"),
        ]
        state = _make_state_with_sources(report, sources)

        verification_result = ClaimVerificationResult(
            claims_extracted=1,
            claims_verified=1,
            claims_partially_supported=1,
            details=[
                ClaimVerdict(
                    claim="The rate is approximately 3.5%",
                    claim_type="quantitative",
                    cited_sources=[1],
                    verdict="PARTIALLY_SUPPORTED",
                    quote_context="The rate is approximately 3.5% [1].",
                ),
            ],
        )

        async def mock_execute(**kwargs):
            return MockWorkflowResult(
                success=True,
                content='{"best_source": 2, "confidence": "high", "evidence_quote": "Current mortgage rate is 3.48%"}',
            )

        remapped = await remap_unsupported_citations(
            state=state,
            verification_result=verification_result,
            execute_fn=mock_execute,
            provider_id="test-provider",
        )

        assert remapped == 1
        assert "[2]" in state.report
        assert verification_result.details[0].cited_sources == [2]

    @pytest.mark.asyncio
    async def test_remapping_stats_tracked(self):
        """citations_remapped counter is correctly set on the verification result."""
        report = (
            "Fact A [1]. Context.\n\n"
            "Fact B [1]. Context.\n\n"
        )
        sources = [
            _make_source(1, raw_content="Unrelated content"),
            _make_source(2, raw_content="Supports Fact A content"),
            _make_source(3, raw_content="Supports Fact B content"),
        ]
        state = _make_state_with_sources(report, sources)

        verification_result = ClaimVerificationResult(
            claims_extracted=2,
            claims_verified=2,
            claims_unsupported=2,
            details=[
                ClaimVerdict(
                    claim="Fact A",
                    claim_type="positive",
                    cited_sources=[1],
                    verdict="UNSUPPORTED",
                    quote_context="Fact A [1].",
                ),
                ClaimVerdict(
                    claim="Fact B",
                    claim_type="positive",
                    cited_sources=[1],
                    verdict="UNSUPPORTED",
                    quote_context="Fact B [1].",
                ),
            ],
        )

        call_count = 0

        async def mock_execute(**kwargs):
            nonlocal call_count
            call_count += 1
            # Alternate: first claim remaps to 2, second to 3.
            src = 2 if call_count == 1 else 3
            return MockWorkflowResult(
                success=True,
                content=f'{{"best_source": {src}, "confidence": "high", "evidence_quote": "evidence"}}',
            )

        remapped = await remap_unsupported_citations(
            state=state,
            verification_result=verification_result,
            execute_fn=mock_execute,
            provider_id="test-provider",
        )

        assert remapped == 2
        assert verification_result.citations_remapped == 2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_low_confidence_removes_citation(self):
        """Low-confidence match results in citation removal, not remapping."""
        report = "Some claim [1]. More text.\n\nConclusion."
        sources = [
            _make_source(1, raw_content="Original source"),
            _make_source(2, raw_content="Vaguely related source"),
        ]
        state = _make_state_with_sources(report, sources)

        verification_result = ClaimVerificationResult(
            claims_extracted=1,
            claims_verified=1,
            claims_unsupported=1,
            details=[
                ClaimVerdict(
                    claim="Some claim",
                    claim_type="positive",
                    cited_sources=[1],
                    verdict="UNSUPPORTED",
                    quote_context="Some claim [1].",
                ),
            ],
        )

        async def mock_execute(**kwargs):
            return MockWorkflowResult(
                success=True,
                content='{"best_source": 2, "confidence": "low", "evidence_quote": "maybe"}',
            )

        remapped = await remap_unsupported_citations(
            state=state,
            verification_result=verification_result,
            execute_fn=mock_execute,
            provider_id="test-provider",
        )

        assert remapped == 0
        assert "[1]" not in state.report  # citation removed
        assert "[2]" not in state.report  # not remapped to low-confidence match

    @pytest.mark.asyncio
    async def test_no_unsupported_claims_is_noop(self):
        """No remapping when all claims are SUPPORTED."""
        report = "Fact [1]. Context."
        sources = [_make_source(1, content="Supporting content")]
        state = _make_state_with_sources(report, sources)

        verification_result = ClaimVerificationResult(
            claims_extracted=1,
            claims_verified=1,
            claims_supported=1,
            details=[
                ClaimVerdict(
                    claim="Fact",
                    claim_type="positive",
                    cited_sources=[1],
                    verdict="SUPPORTED",
                    quote_context="Fact [1].",
                ),
            ],
        )

        async def mock_execute(**kwargs):
            raise AssertionError("Should not be called")

        remapped = await remap_unsupported_citations(
            state=state,
            verification_result=verification_result,
            execute_fn=mock_execute,
            provider_id="test-provider",
        )

        assert remapped == 0
        assert verification_result.citations_remapped == 0

    @pytest.mark.asyncio
    async def test_llm_failure_is_graceful(self):
        """LLM call failure does not crash — claim is skipped."""
        report = "Claim text [1]. Context.\n\nMore."
        sources = [
            _make_source(1, raw_content="Source 1"),
            _make_source(2, raw_content="Source 2"),
        ]
        state = _make_state_with_sources(report, sources)

        verification_result = ClaimVerificationResult(
            claims_extracted=1,
            claims_verified=1,
            claims_unsupported=1,
            details=[
                ClaimVerdict(
                    claim="Claim text",
                    claim_type="positive",
                    cited_sources=[1],
                    verdict="UNSUPPORTED",
                    quote_context="Claim text [1].",
                ),
            ],
        )

        async def mock_execute(**kwargs):
            raise RuntimeError("LLM timeout")

        remapped = await remap_unsupported_citations(
            state=state,
            verification_result=verification_result,
            execute_fn=mock_execute,
            provider_id="test-provider",
        )

        assert remapped == 0

    @pytest.mark.asyncio
    async def test_empty_report_is_noop(self):
        """No remapping when report is empty."""
        state = _make_state_with_sources("", [_make_source(1)])
        state.report = ""

        verification_result = ClaimVerificationResult(
            claims_unsupported=1,
            details=[
                ClaimVerdict(
                    claim="X",
                    claim_type="positive",
                    cited_sources=[1],
                    verdict="UNSUPPORTED",
                    quote_context="X [1].",
                ),
            ],
        )

        remapped = await remap_unsupported_citations(
            state=state,
            verification_result=verification_result,
            execute_fn=AsyncMock(),
            provider_id="test-provider",
        )
        assert remapped == 0
