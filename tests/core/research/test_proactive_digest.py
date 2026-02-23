"""Tests for proactive content digest (digest_policy='proactive').

Tests cover:
1. DigestPolicy.PROACTIVE enum value and eligibility behavior
2. Proactive digest runs after gathering (in workflow_execution)
3. Analysis phase skips already-digested sources from proactive digest
4. Token counting uses digested content length after proactive digest
"""

from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.config.research import ResearchConfig
from foundry_mcp.core.research.document_digest import (
    DigestResult,
    serialize_payload,
)
from foundry_mcp.core.research.document_digest.config import DigestConfig, DigestPolicy
from foundry_mcp.core.research.document_digest.digestor import DocumentDigestor
from foundry_mcp.core.research.models.deep_research import DeepResearchState
from foundry_mcp.core.research.models.digest import DigestPayload, EvidenceSnippet
from foundry_mcp.core.research.models.sources import ResearchSource, SourceQuality
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
    """Create a ResearchConfig with proactive digest defaults for testing."""
    defaults = {
        "deep_research_digest_policy": "proactive",
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
# Test: DigestPolicy.PROACTIVE enum
# =============================================================================


class TestDigestPolicyProactiveEnum:
    """Test that PROACTIVE is a valid DigestPolicy value."""

    def test_proactive_enum_exists(self):
        """DigestPolicy.PROACTIVE should be a valid enum member."""
        assert DigestPolicy.PROACTIVE == "proactive"
        assert DigestPolicy("proactive") == DigestPolicy.PROACTIVE

    def test_proactive_config_validation(self):
        """ResearchConfig should accept 'proactive' as a digest policy."""
        config = ResearchConfig(deep_research_digest_policy="proactive")
        config._validate_digest_config()  # Should not raise


# =============================================================================
# Test: PROACTIVE eligibility (behaves like ALWAYS)
# =============================================================================


class TestProactiveEligibility:
    """Test that PROACTIVE policy behaves like ALWAYS for eligibility."""

    def test_proactive_eligible_with_content(self):
        """Source with content is eligible under PROACTIVE policy."""
        config = DigestConfig(policy=DigestPolicy.PROACTIVE)
        summarizer = MagicMock()
        pdf_extractor = MagicMock()
        digestor = DocumentDigestor(summarizer=summarizer, pdf_extractor=pdf_extractor, config=config)

        assert digestor._is_eligible("Some content", SourceQuality.HIGH) is True
        assert digestor._is_eligible("Some content", SourceQuality.LOW) is True
        assert digestor._is_eligible("Some content", SourceQuality.UNKNOWN) is True

    def test_proactive_not_eligible_empty_content(self):
        """Empty content is not eligible under PROACTIVE policy."""
        config = DigestConfig(policy=DigestPolicy.PROACTIVE)
        summarizer = MagicMock()
        pdf_extractor = MagicMock()
        digestor = DocumentDigestor(summarizer=summarizer, pdf_extractor=pdf_extractor, config=config)

        assert digestor._is_eligible("", SourceQuality.HIGH) is False
        assert digestor._is_eligible("   ", SourceQuality.HIGH) is False

    def test_proactive_skip_reason_for_empty(self):
        """Skip reason for empty content under PROACTIVE is 'Content is empty'."""
        config = DigestConfig(policy=DigestPolicy.PROACTIVE)
        summarizer = MagicMock()
        pdf_extractor = MagicMock()
        digestor = DocumentDigestor(summarizer=summarizer, pdf_extractor=pdf_extractor, config=config)

        assert digestor._get_skip_reason("", SourceQuality.HIGH) == "Content is empty"

    def test_proactive_ignores_min_content_length(self):
        """PROACTIVE policy ignores min_content_length threshold."""
        config = DigestConfig(policy=DigestPolicy.PROACTIVE, min_content_length=10000)
        summarizer = MagicMock()
        pdf_extractor = MagicMock()
        digestor = DocumentDigestor(summarizer=summarizer, pdf_extractor=pdf_extractor, config=config)

        # Short content is still eligible under PROACTIVE
        assert digestor._is_eligible("Short text", SourceQuality.LOW) is True


# =============================================================================
# Test: Proactive digest in workflow digest step
# =============================================================================


class TestProactiveDigestStep:
    """Test that _execute_digest_step_async works with proactive policy."""

    @pytest.mark.asyncio
    async def test_proactive_digests_all_content_sources(self):
        """Under proactive policy, all sources with content are digested."""
        sources = [
            _make_source("src-1", content="A" * 600, quality=SourceQuality.HIGH),
            _make_source("src-2", content="B" * 300, quality=SourceQuality.LOW),
            _make_source("src-3", content=None, snippet="only snippet"),
        ]
        state = _make_state(sources=sources)
        workflow = _make_workflow()

        payload = _make_digest_payload(original_chars=600, digest_chars=150)
        mock_result = DigestResult(payload=payload, cache_hit=False, duration_ms=10.0)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases._analysis_digest.DocumentDigestor"
            ) as MockDigestor,
            patch("foundry_mcp.core.research.workflows.deep_research.phases._analysis_digest.ContentSummarizer"),
            patch("foundry_mcp.core.research.workflows.deep_research.phases._analysis_digest.PDFExtractor"),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = AsyncMock(return_value=mock_result)

            stats = await workflow._execute_digest_step_async(state, "test query")

        # Both content sources should be digested (PROACTIVE = ALWAYS for eligibility)
        assert stats["sources_digested"] == 2
        assert stats["sources_ranked"] == 3
        assert sources[0].content_type == "digest/v1"
        assert sources[1].content_type == "digest/v1"
        # Snippet-only source should not be digested
        assert sources[2].content_type == "text/plain"

    @pytest.mark.asyncio
    async def test_proactive_policy_not_skipped_like_off(self):
        """Proactive policy should NOT return early like 'off' does."""
        source = _make_source("src-1", content="A" * 1000, quality=SourceQuality.HIGH)
        state = _make_state(sources=[source])
        workflow = _make_workflow()

        payload = _make_digest_payload(original_chars=1000, digest_chars=200)
        mock_result = DigestResult(payload=payload, cache_hit=False, duration_ms=10.0)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases._analysis_digest.DocumentDigestor"
            ) as MockDigestor,
            patch("foundry_mcp.core.research.workflows.deep_research.phases._analysis_digest.ContentSummarizer"),
            patch("foundry_mcp.core.research.workflows.deep_research.phases._analysis_digest.PDFExtractor"),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = AsyncMock(return_value=mock_result)

            stats = await workflow._execute_digest_step_async(state, "test query")

        assert stats["sources_digested"] == 1
        assert stats["sources_ranked"] == 1


# =============================================================================
# Test: Analysis skips proactively-digested sources
# =============================================================================


class TestAnalysisSkipsProactivelyDigested:
    """Verify the analysis digest step skips sources already digested proactively."""

    @pytest.mark.asyncio
    async def test_already_digested_skipped_in_analysis(self):
        """Sources digested proactively are skipped when analysis runs digest."""
        payload = _make_digest_payload()
        # Simulate a proactively-digested source
        source = _make_source(
            "src-1",
            content=serialize_payload(payload),
            content_type="digest/v1",
            quality=SourceQuality.HIGH,
        )
        state = _make_state(sources=[source])
        workflow = _make_workflow()

        stats = await workflow._execute_digest_step_async(state, "test query")

        # Source should be skipped (already digested)
        assert stats["sources_selected"] == 0
        assert stats["sources_digested"] == 0
        assert source.metadata.get("_digest_skip_reason") == "already_digested"
        assert source.content_type == "digest/v1"

    @pytest.mark.asyncio
    async def test_mix_proactive_and_new_sources(self):
        """New sources from refinement are digested; proactively-digested ones are skipped."""
        payload = _make_digest_payload()
        proactive_source = _make_source(
            "src-proactive",
            content=serialize_payload(payload),
            content_type="digest/v1",
            quality=SourceQuality.HIGH,
        )
        new_source = _make_source(
            "src-new",
            content="C" * 1000,
            quality=SourceQuality.MEDIUM,
        )
        state = _make_state(sources=[proactive_source, new_source])
        workflow = _make_workflow()

        new_payload = _make_digest_payload(original_chars=1000, digest_chars=200)
        mock_result = DigestResult(payload=new_payload, cache_hit=False, duration_ms=10.0)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases._analysis_digest.DocumentDigestor"
            ) as MockDigestor,
            patch("foundry_mcp.core.research.workflows.deep_research.phases._analysis_digest.ContentSummarizer"),
            patch("foundry_mcp.core.research.workflows.deep_research.phases._analysis_digest.PDFExtractor"),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = AsyncMock(return_value=mock_result)

            stats = await workflow._execute_digest_step_async(state, "test query")

        assert stats["sources_ranked"] == 2
        assert stats["sources_selected"] == 1  # Only the new source
        assert stats["sources_digested"] == 1
        assert new_source.content_type == "digest/v1"
        assert proactive_source.metadata.get("_digest_skip_reason") == "already_digested"


# =============================================================================
# Test: Token counting uses digested content
# =============================================================================


class TestTokenCountingUsesDigestedContent:
    """Verify that proactively-digested sources use digest_chars for token estimation."""

    @pytest.mark.asyncio
    async def test_fidelity_records_compressed_tokens(self):
        """Fidelity tracking uses digest_chars for token estimation."""
        content = "A" * 2000
        source = _make_source("src-1", content=content, quality=SourceQuality.HIGH)
        state = _make_state(sources=[source])
        workflow = _make_workflow()

        payload = _make_digest_payload(original_chars=2000, digest_chars=400)
        mock_result = DigestResult(payload=payload, cache_hit=False, duration_ms=10.0)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases._analysis_digest.DocumentDigestor"
            ) as MockDigestor,
            patch("foundry_mcp.core.research.workflows.deep_research.phases._analysis_digest.ContentSummarizer"),
            patch("foundry_mcp.core.research.workflows.deep_research.phases._analysis_digest.PDFExtractor"),
        ):
            mock_instance = MockDigestor.return_value
            mock_instance.digest = AsyncMock(return_value=mock_result)

            await workflow._execute_digest_step_async(state, "test query")

        from foundry_mcp.core.research.models.fidelity import FidelityLevel

        assert "src-1" in state.content_fidelity
        record = state.content_fidelity["src-1"]
        phase_record = record.phases["digest"]
        assert phase_record.level == FidelityLevel.DIGEST
        # Token estimation: chars // 4
        assert phase_record.original_tokens == 2000 // 4  # 500
        assert phase_record.final_tokens == 400 // 4  # 100
        assert phase_record.reason == "digest_compression"

    def test_digested_source_content_is_serialized_payload(self):
        """After proactive digest, source content is a serialized DigestPayload."""
        payload = _make_digest_payload(
            summary="Proactive summary",
            key_points=["Point A"],
        )
        serialized = serialize_payload(payload)
        source = _make_source(
            "src-1",
            content=serialized,
            content_type="digest/v1",
        )

        # The source's content should be parseable as a digest payload
        from foundry_mcp.core.research.document_digest import deserialize_payload

        assert source.content is not None
        deserialized = deserialize_payload(source.content)
        assert deserialized.summary == "Proactive summary"
        assert deserialized.key_points == ["Point A"]
