"""Integration tests for deep research digest flow.

Tests cover:
1. End-to-end digest pipeline within _execute_digest_step_async
2. Ranking uses raw content (content length boosts score)
3. Budget allocation uses compressed digest_chars size
4. Citations use evidence snippets from digest payload
5. Multi-iteration skips already-digested sources (no re-digest)
"""

import asyncio
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import pytest

from foundry_mcp.config import ResearchConfig
from foundry_mcp.core.research.document_digest import (
    deserialize_payload,
    serialize_payload,
)
from foundry_mcp.core.research.context_budget import AllocationResult
from foundry_mcp.core.research.pdf_extractor import PDFExtractionResult
from foundry_mcp.core.research.models import (
    DeepResearchState,
    DigestPayload,
    EvidenceSnippet,
    FidelityLevel,
    ResearchSource,
    SourceQuality,
)
from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow


# =============================================================================
# Helpers
# =============================================================================


def _make_source(
    source_id: str,
    content: Optional[str] = None,
    snippet: Optional[str] = None,
    quality: SourceQuality = SourceQuality.HIGH,
    content_type: str = "text/plain",
    url: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> ResearchSource:
    """Create a ResearchSource with sensible defaults."""
    return ResearchSource(
        id=source_id,
        title=f"Source {source_id}",
        content=content,
        snippet=snippet,
        quality=quality,
        content_type=content_type,
        url=url,
        metadata=metadata or {},
    )


def _make_digest_payload(
    summary: str = "Test summary of document.",
    key_points: Optional[list[str]] = None,
    evidence_snippets: Optional[list[EvidenceSnippet]] = None,
    original_chars: int = 10000,
    digest_chars: int = 2000,
) -> DigestPayload:
    """Create a DigestPayload for testing."""
    return DigestPayload(
        version="1.0",
        content_type="digest/v1",
        query_hash="ab12cd34",
        summary=summary,
        key_points=key_points or ["Key point 1", "Key point 2"],
        evidence_snippets=evidence_snippets
        or [
            EvidenceSnippet(
                text="Evidence from source.",
                locator="char:100-120",
                relevance_score=0.9,
            )
        ],
        original_chars=original_chars,
        digest_chars=digest_chars,
        compression_ratio=digest_chars / original_chars if original_chars else 0.0,
        source_text_hash="sha256:" + "a" * 64,
    )


def _make_config(**overrides: Any) -> ResearchConfig:
    """Create a ResearchConfig with digest defaults suitable for testing."""
    defaults = {
        "deep_research_digest_policy": "auto",
        "deep_research_digest_min_chars": 500,
        "deep_research_digest_max_sources": 8,
        "deep_research_digest_timeout": 30.0,
        "deep_research_digest_max_concurrent": 3,
        "deep_research_digest_include_evidence": True,
        "deep_research_digest_evidence_max_chars": 400,
        "deep_research_digest_max_evidence_snippets": 5,
        "deep_research_digest_fetch_pdfs": False,
    }
    defaults.update(overrides)
    return ResearchConfig(**defaults)


def _make_state(
    sources: Optional[list[ResearchSource]] = None,
    query: str = "test research query",
) -> DeepResearchState:
    """Create a DeepResearchState with sources."""
    state = DeepResearchState(original_query=query)
    if sources:
        state.sources = sources
    state.analysis_provider = "test-provider"
    return state


def _make_workflow(config: Optional[ResearchConfig] = None) -> DeepResearchWorkflow:
    """Create a DeepResearchWorkflow with test config."""
    cfg = config or _make_config()
    return DeepResearchWorkflow(config=cfg)


# =============================================================================
# Test: End-to-end digest flow
# =============================================================================


class TestEndToEndDigestFlow:
    """Test that the full digest pipeline works end-to-end."""

    @pytest.mark.asyncio
    async def test_eligible_source_gets_digested(self):
        """Source with enough content is digested and content replaced."""
        content = "A" * 1000  # Above min_chars=500
        source = _make_source("src-1", content=content, quality=SourceQuality.HIGH)
        state = _make_state(sources=[source])
        workflow = _make_workflow()

        payload = _make_digest_payload(original_chars=1000, digest_chars=200)

        # Mock the digestor's digest method
        from foundry_mcp.core.research.document_digest import DigestResult

        mock_result = DigestResult(
            payload=payload,
            cache_hit=False,
            duration_ms=50.0,
        )

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.DocumentDigestor"
            ) as MockDigestor,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.ContentSummarizer"
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.PDFExtractor"
            ),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = AsyncMock(return_value=mock_result)

            stats = await workflow._execute_digest_step_async(state, "test query")

        assert stats["sources_digested"] == 1
        assert stats["sources_ranked"] == 1
        assert stats["sources_selected"] == 1
        assert source.content_type == "digest/v1"
        # Content should be the serialized payload
        deserialized = deserialize_payload(source.content)
        assert deserialized.summary == payload.summary
        # Raw content should be cleaned up
        assert "_raw_content" not in source.metadata

    @pytest.mark.asyncio
    async def test_source_below_min_chars_not_digested(self):
        """Source with content below min_chars is not selected for digest."""
        content = "A" * 100  # Below min_chars=500
        source = _make_source("src-1", content=content, quality=SourceQuality.HIGH)
        state = _make_state(sources=[source])
        workflow = _make_workflow()

        stats = await workflow._execute_digest_step_async(state, "test query")

        assert stats["sources_selected"] == 0
        assert stats["sources_digested"] == 0
        assert source.content_type == "text/plain"
        assert "below minimum" in (source.metadata.get("_digest_skip_reason") or "")

    @pytest.mark.asyncio
    async def test_source_without_content_not_digested(self):
        """Source with no content is not selected for digest."""
        source = _make_source(
            "src-1", content=None, snippet="A snippet", quality=SourceQuality.HIGH
        )
        state = _make_state(sources=[source])
        workflow = _make_workflow()

        stats = await workflow._execute_digest_step_async(state, "test query")

        assert stats["sources_selected"] == 0
        assert source.metadata.get("_digest_skip_reason") == "no_content"

    @pytest.mark.asyncio
    async def test_policy_off_skips_all(self):
        """When policy is off, no sources are digested."""
        content = "A" * 1000
        source = _make_source("src-1", content=content)
        state = _make_state(sources=[source])
        workflow = _make_workflow(_make_config(deep_research_digest_policy="off"))

        stats = await workflow._execute_digest_step_async(state, "test query")

        assert stats["sources_ranked"] == 0
        assert stats["sources_digested"] == 0
        assert source.content_type == "text/plain"

    @pytest.mark.asyncio
    async def test_pdf_fetch_populates_content_when_enabled(self):
        """PDF extraction populates content when fetch_pdfs is enabled."""
        source = _make_source(
            "src-1",
            content=None,
            snippet=None,
            quality=SourceQuality.HIGH,
            url="https://example.com/doc.pdf",
        )
        state = _make_state(sources=[source])
        workflow = _make_workflow(_make_config(deep_research_digest_fetch_pdfs=True))

        pdf_result = PDFExtractionResult(
            text="PDF text",
            page_offsets=[(0, 8)],
            warnings=[],
            page_count=1,
            extracted_page_count=1,
        )

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.PDFExtractor"
        ) as MockPDFExtractor:
            mock_instance = MockPDFExtractor.return_value
            mock_instance.extract_from_url = AsyncMock(return_value=pdf_result)

            stats = await workflow._execute_digest_step_async(state, "test query")

        assert stats["sources_extracted"] == 1
        assert source.content == "PDF text"
        assert source.metadata.get("_pdf_extracted") is True
        assert source.metadata.get("_pdf_page_count") == 1
        assert source.metadata.get("_pdf_page_offsets") == [(0, 8)]


# =============================================================================
# Test: Ranking uses raw content
# =============================================================================


class TestRankingUsesRawContent:
    """Verify that ranking scores are based on raw content length."""

    @pytest.mark.asyncio
    async def test_longer_content_ranks_higher(self):
        """Source with more content ranks higher than one with less."""
        short_source = _make_source(
            "src-short", content="B" * 600, quality=SourceQuality.HIGH
        )
        long_source = _make_source(
            "src-long", content="A" * 5000, quality=SourceQuality.HIGH
        )
        state = _make_state(sources=[short_source, long_source])

        # Limit to 1 source to verify ranking order
        workflow = _make_workflow(
            _make_config(deep_research_digest_max_sources=1)
        )

        from foundry_mcp.core.research.document_digest import DigestResult

        payload = _make_digest_payload()
        mock_result = DigestResult(payload=payload, cache_hit=False, duration_ms=10.0)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.DocumentDigestor"
            ) as MockDigestor,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.ContentSummarizer"
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.PDFExtractor"
            ),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = AsyncMock(return_value=mock_result)

            stats = await workflow._execute_digest_step_async(state, "test query")

        # Only 1 source selected (the longer one should win)
        assert stats["sources_selected"] == 1
        # The long source should have been digested
        assert long_source.content_type == "digest/v1"
        assert short_source.content_type == "text/plain"

    @pytest.mark.asyncio
    async def test_snippet_only_ranks_lower_than_content(self):
        """Snippet-only source ranks lower than source with full content."""
        snippet_source = _make_source(
            "src-snippet",
            content=None,
            snippet="A brief snippet",
            quality=SourceQuality.HIGH,
        )
        content_source = _make_source(
            "src-content", content="A" * 600, quality=SourceQuality.MEDIUM
        )
        state = _make_state(sources=[snippet_source, content_source])
        workflow = _make_workflow()

        from foundry_mcp.core.research.document_digest import DigestResult

        payload = _make_digest_payload()
        mock_result = DigestResult(payload=payload, cache_hit=False, duration_ms=10.0)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.DocumentDigestor"
            ) as MockDigestor,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.ContentSummarizer"
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.PDFExtractor"
            ),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = AsyncMock(return_value=mock_result)

            stats = await workflow._execute_digest_step_async(state, "test query")

        # Content source eligible, snippet source not (no content)
        assert stats["sources_selected"] == 1
        assert content_source.content_type == "digest/v1"

    @pytest.mark.asyncio
    async def test_quality_contributes_to_ranking(self):
        """Higher quality sources rank above lower quality with same content."""
        low_q = _make_source(
            "src-low", content="A" * 600, quality=SourceQuality.LOW
        )
        high_q = _make_source(
            "src-high", content="A" * 600, quality=SourceQuality.HIGH
        )
        state = _make_state(sources=[low_q, high_q])

        workflow = _make_workflow(
            _make_config(deep_research_digest_max_sources=1)
        )

        from foundry_mcp.core.research.document_digest import DigestResult

        payload = _make_digest_payload()
        mock_result = DigestResult(payload=payload, cache_hit=False, duration_ms=10.0)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.DocumentDigestor"
            ) as MockDigestor,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.ContentSummarizer"
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.PDFExtractor"
            ),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = AsyncMock(return_value=mock_result)

            stats = await workflow._execute_digest_step_async(state, "test query")

        assert stats["sources_selected"] == 1
        assert high_q.content_type == "digest/v1"
        assert low_q.content_type == "text/plain"


# =============================================================================
# Test: Budget uses compressed (digest) size
# =============================================================================


class TestBudgetUsesCompressedSize:
    """Verify that fidelity tracking uses digest_chars for token estimation."""

    @pytest.mark.asyncio
    async def test_fidelity_records_compressed_tokens(self):
        """Budget fidelity uses digest_chars // 4 for final_tokens."""
        content = "A" * 2000
        source = _make_source("src-1", content=content, quality=SourceQuality.HIGH)
        state = _make_state(sources=[source])
        workflow = _make_workflow()

        payload = _make_digest_payload(original_chars=2000, digest_chars=400)

        from foundry_mcp.core.research.document_digest import DigestResult

        mock_result = DigestResult(payload=payload, cache_hit=False, duration_ms=10.0)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.DocumentDigestor"
            ) as MockDigestor,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.ContentSummarizer"
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.PDFExtractor"
            ),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = AsyncMock(return_value=mock_result)

            await workflow._execute_digest_step_async(state, "test query")

        # Check fidelity was recorded with compressed size
        assert "src-1" in state.content_fidelity
        record = state.content_fidelity["src-1"]
        phase_record = record.phases["digest"]
        assert phase_record.level == FidelityLevel.DIGEST
        assert phase_record.original_tokens == 2000 // 4  # 500
        assert phase_record.final_tokens == 400 // 4  # 100
        assert phase_record.reason == "digest_compression"

    @pytest.mark.asyncio
    async def test_skipped_digest_records_full_fidelity(self):
        """When digest is skipped, fidelity is recorded as FULL."""
        content = "A" * 2000
        source = _make_source("src-1", content=content, quality=SourceQuality.HIGH)
        state = _make_state(sources=[source])
        workflow = _make_workflow()

        from foundry_mcp.core.research.document_digest import DigestResult

        mock_result = DigestResult(
            payload=None,
            skipped=True,
            skip_reason="test_skip",
            duration_ms=1.0,
        )

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.DocumentDigestor"
            ) as MockDigestor,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.ContentSummarizer"
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.PDFExtractor"
            ),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = AsyncMock(return_value=mock_result)

            await workflow._execute_digest_step_async(state, "test query")

        assert "src-1" in state.content_fidelity
        phase_record = state.content_fidelity["src-1"].phases["digest"]
        assert phase_record.level == FidelityLevel.FULL
        assert phase_record.reason == "digest_skipped"

    def test_allocate_budget_uses_digest_text(self):
        """Allocation uses digest payload text rather than raw JSON."""
        payload = _make_digest_payload(
            summary="Budget summary",
            key_points=["Point A", "Point B"],
            evidence_snippets=[
                EvidenceSnippet(
                    text="Evidence snippet text.",
                    locator="char:10-30",
                    relevance_score=0.8,
                )
            ],
        )
        digest_json = serialize_payload(payload)
        source = _make_source(
            "src-1",
            content=digest_json,
            content_type="digest/v1",
            quality=SourceQuality.HIGH,
        )
        state = _make_state(sources=[source])
        workflow = _make_workflow()

        captured: dict[str, Any] = {}

        def fake_allocate(self, items, budget, strategy):
            captured["content"] = items[0].content
            return AllocationResult(items=items, tokens_used=0, tokens_available=budget)

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.ContextBudgetManager.allocate_budget",
            new=fake_allocate,
        ):
            workflow._allocate_source_budget(state, provider_id=None)

        assert "Budget summary" in captured["content"]
        assert "Point A" in captured["content"]
        assert "Evidence snippet text." in captured["content"]
        assert "\"content_type\"" not in captured["content"]


# =============================================================================
# Test: Citations use evidence snippets
# =============================================================================


class TestCitationsUseEvidenceSnippets:
    """Verify that analysis prompt uses evidence snippets from digested sources."""

    def test_digest_source_renders_summary_and_evidence(self):
        """Digested source renders summary, key points, and evidence in prompt."""
        payload = _make_digest_payload(
            summary="Document discusses machine learning advances.",
            key_points=["ML models improved", "New architectures emerged"],
            evidence_snippets=[
                EvidenceSnippet(
                    text="Transformer models outperform RNNs.",
                    locator="char:500-535",
                    relevance_score=0.95,
                ),
                EvidenceSnippet(
                    text="Attention mechanism is key innovation.",
                    locator="char:1200-1238",
                    relevance_score=0.88,
                ),
            ],
        )
        source = _make_source(
            "src-1",
            content=serialize_payload(payload),
            content_type="digest/v1",
            quality=SourceQuality.HIGH,
        )
        state = _make_state(sources=[source])
        workflow = _make_workflow()

        # Call _build_analysis_user_prompt
        prompt = workflow._build_analysis_user_prompt(state)

        # Prompt should contain summary, key points, and evidence with locators
        assert "Document discusses machine learning advances." in prompt
        assert "ML models improved" in prompt
        assert "Transformer models outperform RNNs." in prompt
        assert "char:500-535" in prompt
        assert "Attention mechanism is key innovation." in prompt
        assert "char:1200-1238" in prompt

    def test_non_digest_source_renders_raw_content(self):
        """Non-digested source renders raw content in prompt."""
        source = _make_source(
            "src-1",
            content="Raw text content about research.",
            quality=SourceQuality.HIGH,
        )
        state = _make_state(sources=[source])
        workflow = _make_workflow()

        prompt = workflow._build_analysis_user_prompt(state)

        assert "Raw text content about research." in prompt
        assert "Evidence:" not in prompt

    def test_invalid_digest_falls_back_to_raw(self):
        """If digest payload is invalid JSON, falls back to raw content display."""
        source = _make_source(
            "src-1",
            content="not valid json {{{",
            content_type="digest/v1",
            quality=SourceQuality.HIGH,
        )
        state = _make_state(sources=[source])
        workflow = _make_workflow()

        prompt = workflow._build_analysis_user_prompt(state)

        # Should fall back to showing content as-is
        assert "not valid json {{{" in prompt


# =============================================================================
# Test: Multi-iteration does not re-digest
# =============================================================================


class TestMultiIterationNoReDigest:
    """Verify that already-digested sources are skipped in subsequent iterations."""

    @pytest.mark.asyncio
    async def test_already_digested_source_skipped(self):
        """Source with content_type=digest/v1 is not re-digested."""
        payload = _make_digest_payload()
        source = _make_source(
            "src-1",
            content=serialize_payload(payload),
            content_type="digest/v1",
            quality=SourceQuality.HIGH,
        )
        state = _make_state(sources=[source])
        workflow = _make_workflow()

        stats = await workflow._execute_digest_step_async(state, "test query")

        assert stats["sources_selected"] == 0
        assert stats["sources_digested"] == 0
        assert source.metadata.get("_digest_skip_reason") == "already_digested"
        # Content should remain unchanged
        assert source.content_type == "digest/v1"

    @pytest.mark.asyncio
    async def test_mix_of_digested_and_new_sources(self):
        """Only new sources are digested when mixed with already-digested ones."""
        payload = _make_digest_payload()
        digested_source = _make_source(
            "src-digested",
            content=serialize_payload(payload),
            content_type="digest/v1",
            quality=SourceQuality.HIGH,
        )
        new_source = _make_source(
            "src-new",
            content="A" * 1000,
            quality=SourceQuality.HIGH,
        )
        state = _make_state(sources=[digested_source, new_source])
        workflow = _make_workflow()

        from foundry_mcp.core.research.document_digest import DigestResult

        new_payload = _make_digest_payload(original_chars=1000, digest_chars=200)
        mock_result = DigestResult(
            payload=new_payload, cache_hit=False, duration_ms=10.0
        )

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.DocumentDigestor"
            ) as MockDigestor,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.ContentSummarizer"
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.PDFExtractor"
            ),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = AsyncMock(return_value=mock_result)

            stats = await workflow._execute_digest_step_async(state, "test query")

        assert stats["sources_ranked"] == 2
        assert stats["sources_selected"] == 1  # Only new source
        assert stats["sources_digested"] == 1
        assert new_source.content_type == "digest/v1"
        assert digested_source.metadata.get("_digest_skip_reason") == "already_digested"

    @pytest.mark.asyncio
    async def test_raw_content_cleaned_up_after_digest(self):
        """_raw_content metadata is removed after digest completes."""
        content = "A" * 1000
        source = _make_source("src-1", content=content, quality=SourceQuality.HIGH)
        state = _make_state(sources=[source])
        workflow = _make_workflow()

        from foundry_mcp.core.research.document_digest import DigestResult

        payload = _make_digest_payload()
        mock_result = DigestResult(payload=payload, cache_hit=False, duration_ms=10.0)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.DocumentDigestor"
            ) as MockDigestor,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.ContentSummarizer"
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.PDFExtractor"
            ),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = AsyncMock(return_value=mock_result)

            await workflow._execute_digest_step_async(state, "test query")

        assert "_raw_content" not in source.metadata

    @pytest.mark.asyncio
    async def test_raw_content_cleaned_up_on_error(self):
        """_raw_content metadata is removed even when digest fails."""
        content = "A" * 1000
        source = _make_source("src-1", content=content, quality=SourceQuality.HIGH)
        state = _make_state(sources=[source])
        workflow = _make_workflow()

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.DocumentDigestor"
            ) as MockDigestor,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.ContentSummarizer"
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.PDFExtractor"
            ),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = AsyncMock(side_effect=RuntimeError("LLM failed"))

            await workflow._execute_digest_step_async(state, "test query")

        assert "_raw_content" not in source.metadata
        assert source.metadata.get("_digest_error") == "LLM failed"

    @pytest.mark.asyncio
    async def test_raw_content_cleaned_up_on_timeout(self):
        """_raw_content metadata is removed when digest times out."""
        content = "A" * 1000
        source = _make_source("src-1", content=content, quality=SourceQuality.HIGH)
        state = _make_state(sources=[source])
        workflow = _make_workflow(
            _make_config(deep_research_digest_timeout=0.01, deep_research_digest_max_concurrent=1)
        )

        async def slow_digest(**kwargs):
            await asyncio.sleep(10)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.DocumentDigestor"
            ) as MockDigestor,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.ContentSummarizer"
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.PDFExtractor"
            ),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = slow_digest

            stats = await workflow._execute_digest_step_async(state, "test query")

        assert "_raw_content" not in source.metadata
        assert source.metadata.get("_digest_timeout") is True
        assert len(stats["digest_errors"]) == 1


# =============================================================================
# Test: Timeout budgeting with concurrency
# =============================================================================


class TestTimeoutBudgeting:
    """Ensure digest timeout budgets are not divided by concurrency."""

    @pytest.mark.asyncio
    async def test_timeout_not_divided_by_concurrency(self):
        """Per-source timeout applies even when max_concurrent > 1."""
        sources = [
            _make_source("src-1", content="A" * 1000, quality=SourceQuality.HIGH),
            _make_source("src-2", content="B" * 1000, quality=SourceQuality.HIGH),
        ]
        state = _make_state(sources=sources)
        workflow = _make_workflow(
            _make_config(deep_research_digest_timeout=0.3, deep_research_digest_max_concurrent=2)
        )

        from foundry_mcp.core.research.document_digest import DigestResult

        async def delayed_digest(**kwargs):
            await asyncio.sleep(0.2)
            payload = _make_digest_payload(original_chars=1000, digest_chars=200)
            return DigestResult(payload=payload, cache_hit=False, duration_ms=10.0)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.DocumentDigestor"
            ) as MockDigestor,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.ContentSummarizer"
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.PDFExtractor"
            ),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = delayed_digest
            mock_instance._is_eligible.return_value = True

            stats = await workflow._execute_digest_step_async(state, "test query")

        assert stats["sources_digested"] == 2
        assert stats["digest_errors"] == []
        for source in sources:
            assert source.metadata.get("_digest_timeout") is None


# =============================================================================
# Test: Max sources limit
# =============================================================================


class TestMaxSourcesLimit:
    """Verify that max_sources config limits the number of digested sources."""

    @pytest.mark.asyncio
    async def test_respects_max_sources(self):
        """Only max_sources number of sources are selected for digest."""
        sources = [
            _make_source(f"src-{i}", content="A" * 1000, quality=SourceQuality.HIGH)
            for i in range(5)
        ]
        state = _make_state(sources=sources)
        workflow = _make_workflow(
            _make_config(deep_research_digest_max_sources=2)
        )

        from foundry_mcp.core.research.document_digest import DigestResult

        payload = _make_digest_payload()
        mock_result = DigestResult(payload=payload, cache_hit=False, duration_ms=10.0)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.DocumentDigestor"
            ) as MockDigestor,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.ContentSummarizer"
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.PDFExtractor"
            ),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = AsyncMock(return_value=mock_result)

            stats = await workflow._execute_digest_step_async(state, "test query")

        assert stats["sources_selected"] == 2
        assert stats["sources_digested"] == 2
        assert stats["sources_ranked"] == 5


# =============================================================================
# Test: Fidelity tracking on errors
# =============================================================================


class TestFidelityTrackingOnErrors:
    """Verify fidelity is recorded correctly for error and timeout cases."""

    @pytest.mark.asyncio
    async def test_timeout_records_full_fidelity(self):
        """Timeout records FULL fidelity since content is unchanged."""
        content = "A" * 1000
        source = _make_source("src-1", content=content, quality=SourceQuality.HIGH)
        state = _make_state(sources=[source])
        workflow = _make_workflow(
            _make_config(deep_research_digest_timeout=0.01, deep_research_digest_max_concurrent=1)
        )

        async def slow_digest(**kwargs):
            await asyncio.sleep(10)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.DocumentDigestor"
            ) as MockDigestor,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.ContentSummarizer"
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.PDFExtractor"
            ),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = slow_digest

            await workflow._execute_digest_step_async(state, "test query")

        assert "src-1" in state.content_fidelity
        phase_record = state.content_fidelity["src-1"].phases["digest"]
        assert phase_record.level == FidelityLevel.FULL
        assert phase_record.reason == "digest_timeout"

    @pytest.mark.asyncio
    async def test_error_records_full_fidelity(self):
        """Errors record FULL fidelity since content is unchanged."""
        content = "A" * 1000
        source = _make_source("src-1", content=content, quality=SourceQuality.HIGH)
        state = _make_state(sources=[source])
        workflow = _make_workflow()

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.DocumentDigestor"
            ) as MockDigestor,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.ContentSummarizer"
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.PDFExtractor"
            ),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = AsyncMock(
                side_effect=ValueError("Summarization failed")
            )

            await workflow._execute_digest_step_async(state, "test query")

        assert "src-1" in state.content_fidelity
        phase_record = state.content_fidelity["src-1"].phases["digest"]
        assert phase_record.level == FidelityLevel.FULL
        assert phase_record.reason == "digest_error"


# =============================================================================
# Test: Digest archive safety
# =============================================================================


class TestDigestArchiveSafety:
    """Verify archive path safety checks."""

    @pytest.mark.asyncio
    async def test_archive_rejects_unsafe_source_id(self, tmp_path, monkeypatch):
        """Unsafe source IDs should not be used as archive paths."""
        source = _make_source("../evil", content="A" * 1000, quality=SourceQuality.HIGH)
        state = _make_state(sources=[source])
        workflow = _make_workflow(
            _make_config(
                deep_research_archive_content=True,
                deep_research_archive_retention_days=1,
            )
        )

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        from foundry_mcp.core.research.document_digest import DigestResult

        payload = _make_digest_payload(original_chars=1000, digest_chars=200)
        mock_result = DigestResult(payload=payload, cache_hit=False, duration_ms=10.0)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.DocumentDigestor"
            ) as MockDigestor,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.ContentSummarizer"
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.PDFExtractor"
            ),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = AsyncMock(return_value=mock_result)
            mock_instance._is_eligible.return_value = True
            mock_instance._normalize_text.return_value = "canonical text"
            mock_instance._compute_source_hash.return_value = payload.source_text_hash

            stats = await workflow._execute_digest_step_async(state, "test query")

        assert stats["sources_digested"] == 1
        assert source.metadata.get("_digest_archive_error")
        assert source.metadata.get("_digest_archive_hash") is None

    @pytest.mark.asyncio
    async def test_archive_writes_canonical_text(self, tmp_path, monkeypatch):
        """Successful archive writes canonical text to disk."""
        source = _make_source("src-1", content="A" * 1000, quality=SourceQuality.HIGH)
        state = _make_state(sources=[source])
        workflow = _make_workflow(
            _make_config(
                deep_research_archive_content=True,
                deep_research_archive_retention_days=1,
            )
        )

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        from foundry_mcp.core.research.document_digest import DigestResult

        payload = _make_digest_payload(original_chars=1000, digest_chars=200)
        mock_result = DigestResult(payload=payload, cache_hit=False, duration_ms=10.0)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.DocumentDigestor"
            ) as MockDigestor,
            patch(
                "foundry_mcp.core.research.workflows.deep_research.ContentSummarizer"
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.PDFExtractor"
            ),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = AsyncMock(return_value=mock_result)
            mock_instance._is_eligible.return_value = True
            mock_instance._normalize_text.return_value = "canonical text"
            mock_instance._compute_source_hash.return_value = payload.source_text_hash

            stats = await workflow._execute_digest_step_async(state, "test query")

        assert stats["sources_digested"] == 1
        assert source.metadata.get("_digest_archive_hash") == payload.source_text_hash
        archive_path = (
            tmp_path
            / ".foundry-mcp"
            / "research_archives"
            / source.id
            / f"{payload.source_text_hash}.txt"
        )
        assert archive_path.exists()
        assert archive_path.read_text(encoding="utf-8") == "canonical text"
