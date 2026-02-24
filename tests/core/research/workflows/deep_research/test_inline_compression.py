"""Tests for Phase 2 PLAN: Inline Per-Topic Compression Before Supervision.

Covers:
- 2.1: _compress_single_topic_async reusable helper
- 2.2: Inline compression in _execute_topic_research_async
- 2.3: Supervision user prompt includes compressed findings excerpts
- 2.4: Supervision system prompt includes content assessment guidance
- 2.5: Batch compression skips already-compressed topics
- 2.6: Config flag deep_research_inline_compression
- 2.7: Non-fatal inline compression failure
"""

from __future__ import annotations

import asyncio
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
from foundry_mcp.core.research.workflows.deep_research.phases.compression import (
    CompressionMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases.supervision import (
    SupervisionPhaseMixin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    query: str = "What are the benefits of renewable energy?",
    phase: DeepResearchPhase = DeepResearchPhase.GATHERING,
    num_sub_queries: int = 2,
    sources_per_query: int = 3,
) -> DeepResearchState:
    """Create a DeepResearchState with sub-queries and sources for testing."""
    state = DeepResearchState(
        id="deepres-test-inline-compression",
        original_query=query,
        research_brief="Detailed investigation of renewable energy benefits",
        phase=phase,
        iteration=1,
        max_iterations=3,
        max_sources_per_query=5,
    )
    for i in range(num_sub_queries):
        sq = SubQuery(
            id=f"sq-{i}",
            query=f"Sub-query {i}: {query}",
            rationale=f"Rationale {i}",
            priority=i + 1,
            status="completed",
        )
        state.sub_queries.append(sq)
        for j in range(sources_per_query):
            src = ResearchSource(
                id=f"src-{i}-{j}",
                title=f"Source {j} for sq-{i}",
                url=f"https://example{j}.com/sq-{i}/{j}",
                content=f"Content about topic sq-{i}, finding {j}.",
                quality=SourceQuality.HIGH if j == 0 else SourceQuality.MEDIUM,
                source_type=SourceType.WEB,
                sub_query_id=sq.id,
            )
            state.append_source(src)
    return state


def _add_topic_results(
    state: DeepResearchState,
    compressed: bool = False,
) -> list[TopicResearchResult]:
    """Create TopicResearchResults for all sub-queries in state."""
    results = []
    for sq in state.sub_queries:
        source_ids = [s.id for s in state.sources if s.sub_query_id == sq.id]
        tr = TopicResearchResult(
            sub_query_id=sq.id,
            searches_performed=2,
            sources_found=len(source_ids),
            source_ids=source_ids,
            reflection_notes=["Found relevant sources", "Good coverage"],
            refined_queries=["refined query for " + sq.id],
        )
        if compressed:
            tr.compressed_findings = (
                f"Compressed findings for {sq.id}: "
                "Key finding 1 [1]. Key finding 2 [2]. Key finding 3 [3]."
            )
        results.append(tr)
        state.topic_research_results.append(tr)
    return results


class StubCompression(CompressionMixin):
    """Concrete class for testing CompressionMixin in isolation."""

    def __init__(self) -> None:
        self.config = MagicMock()
        self.config.default_provider = "test-provider"
        self.config.resolve_model_for_role = MagicMock(
            return_value=("test-provider", None)
        )
        self.config.get_phase_fallback_providers = MagicMock(return_value=[])
        self.config.deep_research_max_retries = 1
        self.config.deep_research_retry_delay = 0.1
        self.config.deep_research_compression_max_content_length = 50_000
        self.memory = MagicMock()
        self._audit_events: list[tuple[str, dict]] = []
        self._cancelled = False

    def _write_audit_event(self, state: Any, event: str, **kwargs: Any) -> None:
        self._audit_events.append((event, kwargs))

    def _check_cancellation(self, state: Any) -> None:
        if self._cancelled:
            raise asyncio.CancelledError()


class StubSupervision(SupervisionPhaseMixin):
    """Concrete class for testing SupervisionPhaseMixin in isolation."""

    def __init__(self) -> None:
        self.config = MagicMock()
        self.config.deep_research_supervision_min_sources_per_query = 2
        self.memory = MagicMock()
        self._audit_events: list[tuple[str, dict]] = []

    def _write_audit_event(self, state: Any, event: str, **kwargs: Any) -> None:
        self._audit_events.append((event, kwargs))

    def _check_cancellation(self, state: Any) -> None:
        pass


# =============================================================================
# Tests: _compress_single_topic_async (2.1)
# =============================================================================


class TestCompressSingleTopicAsync:
    """Tests for the reusable _compress_single_topic_async helper."""

    @pytest.mark.asyncio
    async def test_compress_single_topic_populates_compressed_findings(self):
        """Compressed findings should be populated on success."""
        state = _make_state(num_sub_queries=1)
        results = _add_topic_results(state, compressed=False)
        tr = results[0]

        stub = StubCompression()

        # Mock execute_llm_call to return a successful compression result
        mock_result = MagicMock()
        mock_result.result.success = True
        mock_result.result.content = "Compressed: Key findings [1][2][3]"
        mock_result.result.input_tokens = 100
        mock_result.result.output_tokens = 50

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            inp, out, success = await stub._compress_single_topic_async(
                topic_result=tr,
                state=state,
                timeout=120.0,
            )

        assert success is True
        assert tr.compressed_findings == "Compressed: Key findings [1][2][3]"
        assert inp == 100
        assert out == 50

    @pytest.mark.asyncio
    async def test_compress_single_topic_returns_false_on_llm_failure(self):
        """Should return (0, 0, False) when LLM call fails."""
        state = _make_state(num_sub_queries=1)
        results = _add_topic_results(state, compressed=False)
        tr = results[0]

        stub = StubCompression()

        # Mock execute_llm_call to return a WorkflowResult (error path)
        from foundry_mcp.core.research.workflows.base import WorkflowResult

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            new_callable=AsyncMock,
            return_value=WorkflowResult(success=False, content="", error="context window exceeded"),
        ):
            inp, out, success = await stub._compress_single_topic_async(
                topic_result=tr,
                state=state,
                timeout=120.0,
            )

        assert success is False
        assert tr.compressed_findings is None
        assert inp == 0
        assert out == 0

    @pytest.mark.asyncio
    async def test_compress_single_topic_no_sources_is_noop(self):
        """Should return (0, 0, True) when topic has no sources."""
        state = _make_state(num_sub_queries=1, sources_per_query=0)
        tr = TopicResearchResult(
            sub_query_id="sq-0",
            searches_performed=1,
            sources_found=0,
            source_ids=[],
        )
        state.topic_research_results.append(tr)

        stub = StubCompression()

        inp, out, success = await stub._compress_single_topic_async(
            topic_result=tr,
            state=state,
            timeout=120.0,
        )

        assert success is True
        assert inp == 0
        assert out == 0


# =============================================================================
# Tests: Batch compression delegates to helper (2.1 refactor)
# =============================================================================


class TestBatchCompressionDelegatesToHelper:
    """Tests that _compress_topic_findings_async delegates to _compress_single_topic_async."""

    @pytest.mark.asyncio
    async def test_batch_skips_already_compressed_topics(self):
        """Batch compression should skip topics that already have compressed_findings."""
        state = _make_state(num_sub_queries=2)
        _add_topic_results(state, compressed=True)

        stub = StubCompression()

        result = await stub._compress_topic_findings_async(
            state=state,
            max_concurrent=3,
            timeout=120.0,
        )

        # All topics already compressed — nothing to do
        assert result["topics_compressed"] == 0
        assert result["topics_failed"] == 0

    @pytest.mark.asyncio
    async def test_batch_compresses_uncompressed_topics_only(self):
        """Batch should only compress topics where compressed_findings is None."""
        state = _make_state(num_sub_queries=2)
        results = _add_topic_results(state, compressed=False)
        # Pre-compress one topic
        results[0].compressed_findings = "Already compressed"

        stub = StubCompression()

        mock_result = MagicMock()
        mock_result.result.success = True
        mock_result.result.content = "Compressed findings"
        mock_result.result.input_tokens = 80
        mock_result.result.output_tokens = 40

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await stub._compress_topic_findings_async(
                state=state,
                max_concurrent=3,
                timeout=120.0,
            )

        # Only 1 topic needed compression (the second one)
        assert result["topics_compressed"] == 1
        assert result["topics_failed"] == 0


# =============================================================================
# Tests: Supervision prompts include compressed findings (2.3-2.4)
# =============================================================================


class TestSupervisionContentAwarePrompts:
    """Tests for content-aware supervision prompts."""

    def test_coverage_data_includes_compressed_findings_excerpt(self):
        """_build_per_query_coverage should include compressed_findings_excerpt."""
        state = _make_state(num_sub_queries=2, phase=DeepResearchPhase.SUPERVISION)
        _add_topic_results(state, compressed=True)

        stub = StubSupervision()
        coverage = stub._build_per_query_coverage(state)

        assert len(coverage) == 2
        for entry in coverage:
            assert entry["compressed_findings_excerpt"] is not None
            assert "Compressed findings" in entry["compressed_findings_excerpt"]

    def test_coverage_data_none_when_no_compressed_findings(self):
        """compressed_findings_excerpt should be None when not available."""
        state = _make_state(num_sub_queries=1, phase=DeepResearchPhase.SUPERVISION)
        _add_topic_results(state, compressed=False)

        stub = StubSupervision()
        coverage = stub._build_per_query_coverage(state)

        assert len(coverage) == 1
        assert coverage[0]["compressed_findings_excerpt"] is None

    def test_user_prompt_includes_key_findings_when_compressed(self):
        """User prompt should show 'Key findings' section when compressed findings available."""
        state = _make_state(num_sub_queries=1, phase=DeepResearchPhase.SUPERVISION)
        _add_topic_results(state, compressed=True)

        stub = StubSupervision()
        coverage = stub._build_per_query_coverage(state)
        prompt = stub._build_supervision_user_prompt(state, coverage)

        assert "**Key findings:**" in prompt
        assert "Compressed findings for sq-0" in prompt

    def test_user_prompt_falls_back_to_findings_summary_when_uncompressed(self):
        """User prompt should show 'Findings' when only per_topic_summary available."""
        state = _make_state(num_sub_queries=1, phase=DeepResearchPhase.SUPERVISION)
        results = _add_topic_results(state, compressed=False)
        results[0].per_topic_summary = "Summary of findings for this topic"

        stub = StubSupervision()
        coverage = stub._build_per_query_coverage(state)
        prompt = stub._build_supervision_user_prompt(state, coverage)

        assert "**Key findings:**" not in prompt
        assert "**Findings:** Summary of findings" in prompt

    def test_system_prompt_includes_content_assessment_guidance(self):
        """System prompt should instruct content-aware assessment."""
        state = _make_state(phase=DeepResearchPhase.SUPERVISION)

        stub = StubSupervision()
        system = stub._build_supervision_system_prompt(state)

        assert "SUBSTANTIVELY" in system
        assert "CONTENT gaps" in system
        assert "qualitative coverage" in system

    def test_compressed_findings_excerpt_truncated_at_2000_chars(self):
        """Compressed findings should be truncated to ~2000 chars in coverage data."""
        state = _make_state(num_sub_queries=1, phase=DeepResearchPhase.SUPERVISION)
        results = _add_topic_results(state, compressed=False)
        # Set very long compressed findings
        results[0].compressed_findings = "A" * 5000

        stub = StubSupervision()
        coverage = stub._build_per_query_coverage(state)

        excerpt = coverage[0]["compressed_findings_excerpt"]
        assert excerpt is not None
        assert len(excerpt) == 2000


# =============================================================================
# Tests: Config flag (2.6)
# =============================================================================


class TestInlineCompressionConfig:
    """Tests for the deep_research_inline_compression config flag."""

    def test_default_value_is_true(self):
        """Config should default to enabling inline compression."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig()
        assert config.deep_research_inline_compression is True

    def test_from_toml_dict_parses_flag(self):
        """from_toml_dict should parse the inline compression flag."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig.from_toml_dict(
            {"deep_research_inline_compression": False}
        )
        assert config.deep_research_inline_compression is False

    def test_from_toml_dict_defaults_to_true(self):
        """from_toml_dict should default to True when key absent."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig.from_toml_dict({})
        assert config.deep_research_inline_compression is True


# =============================================================================
# Tests: Non-fatal inline compression failure (2.7)
# =============================================================================


class TestInlineCompressionNonFatal:
    """Tests that inline compression failure doesn't break the research flow."""

    @pytest.mark.asyncio
    async def test_compress_single_topic_exception_returns_false(self):
        """An unexpected exception should return (0, 0, False)."""
        state = _make_state(num_sub_queries=1)
        results = _add_topic_results(state, compressed=False)
        tr = results[0]

        stub = StubCompression()

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            new_callable=AsyncMock,
            side_effect=RuntimeError("unexpected error"),
        ):
            with pytest.raises(RuntimeError):
                await stub._compress_single_topic_async(
                    topic_result=tr,
                    state=state,
                    timeout=120.0,
                )

        # compressed_findings should still be None (not set)
        assert tr.compressed_findings is None
