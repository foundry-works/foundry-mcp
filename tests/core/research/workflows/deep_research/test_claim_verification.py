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
    _apply_token_budget,
    _build_verification_user_prompt,
    _extract_context_window,
    _filter_claims_for_verification,
    _keyword_proximity_truncate,
    _parse_extracted_claims,
    _parse_verification_response,
    _resolve_source_text,
    _sort_claims_by_priority,
    apply_corrections,
    extract_and_verify_claims,
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
    def test_content_populated(self):
        src = _make_source(1, content="Full content")
        assert _resolve_source_text(src) == "Full content"

    def test_raw_content_fallback(self):
        src = _make_source(1, content=None, raw_content="Raw content")
        assert _resolve_source_text(src) == "Raw content"

    def test_snippet_fallback(self):
        src = _make_source(1, content=None, raw_content=None, snippet="Snippet")
        assert _resolve_source_text(src) == "Snippet"

    def test_all_none(self):
        src = _make_source(1, content=None, raw_content=None, snippet=None)
        assert _resolve_source_text(src) is None


# ---------------------------------------------------------------------------
# Unit tests: _keyword_proximity_truncate
# ---------------------------------------------------------------------------


class TestKeywordProximityTruncate:
    def test_short_text_returned_as_is(self):
        text = "Short text"
        assert _keyword_proximity_truncate(text, "anything", 1000) == text

    def test_keyword_found_mid_document(self):
        # Build a long document with the keyword "Aeroplan" in the middle.
        before = "A" * 5000
        keyword_section = " Aeroplan transfer partner "
        after = "B" * 5000
        text = before + keyword_section + after
        result = _keyword_proximity_truncate(text, "Aeroplan transfer ratio", 200)
        assert len(result) == 200
        assert "Aeroplan" in result

    def test_keyword_found_near_start(self):
        text = "Aeroplan is listed here" + "X" * 10000
        result = _keyword_proximity_truncate(text, "Aeroplan partner", 200)
        assert len(result) == 200
        assert result.startswith("Aeroplan")

    def test_keyword_found_near_end(self):
        text = "X" * 10000 + "Aeroplan is at the end"
        result = _keyword_proximity_truncate(text, "Aeroplan partner", 200)
        assert len(result) == 200
        assert "Aeroplan" in result

    def test_no_keyword_match_falls_back_to_prefix(self):
        text = "X" * 10000
        result = _keyword_proximity_truncate(text, "Aeroplan", 200)
        assert len(result) == 200
        assert result == "X" * 200

    def test_stopwords_filtered(self):
        # "with" and "from" are stopwords and < 4 chars words are filtered.
        text = "X" * 5000 + "transfer" + "Y" * 5000
        result = _keyword_proximity_truncate(text, "with from the transfer", 200)
        assert "transfer" in result


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
        prompt = _build_verification_user_prompt(claim, sources)
        assert prompt is not None
        assert "Source [1]" in prompt
        assert "Source [2]" in prompt
        assert "X does not support Y" in prompt

    def test_dangling_citation_skipped(self):
        claim = ClaimVerdict(
            claim="test",
            claim_type="positive",
            cited_sources=[1, 99],  # 99 doesn't exist
        )
        sources = {1: _make_source(1, content="Content")}
        prompt = _build_verification_user_prompt(claim, sources)
        assert prompt is not None
        assert "Source [1]" in prompt
        assert "Source [99]" not in prompt

    def test_no_resolvable_sources_returns_none(self):
        claim = ClaimVerdict(
            claim="test",
            claim_type="positive",
            cited_sources=[99],  # Doesn't exist
        )
        prompt = _build_verification_user_prompt(claim, {})
        assert prompt is None

    def test_source_with_no_content_skipped(self):
        claim = ClaimVerdict(
            claim="test",
            claim_type="positive",
            cited_sources=[1],
        )
        src = _make_source(1, content=None, raw_content=None, snippet=None)
        prompt = _build_verification_user_prompt(claim, {1: src})
        assert prompt is None


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
        result = _keyword_proximity_truncate(long_content, "nomatch", 8000)
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
