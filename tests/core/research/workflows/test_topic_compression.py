"""Unit tests for Phase 3: Per-Topic Compression Before Aggregation.

Tests cover:
1. TopicResearchResult.compressed_findings field — default, set/get, backward compat
2. Compression prompt construction — correct sources per topic, citation formatting
3. Progressive truncation on token limit error (reuses Phase 5 infra)
4. Fallback to raw sources when compression fails
5. Analysis phase uses compressed findings when available
6. Analysis phase falls back when compressed findings absent
7. Parallel compression across multiple topics
8. Citation numbering consistency between compression and analysis
9. Config fields — compression_provider, compression_model
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

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
from foundry_mcp.core.research.workflows.deep_research.phases.gathering import (
    GatheringPhaseMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases._analysis_prompts import (
    AnalysisPromptsMixin,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_state(
    query: str = "What are the benefits of renewable energy?",
    phase: DeepResearchPhase = DeepResearchPhase.GATHERING,
    num_sub_queries: int = 2,
) -> DeepResearchState:
    """Create a DeepResearchState with sub-queries and sources for testing."""
    state = DeepResearchState(
        id="deepres-test-compression",
        original_query=query,
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
        )
        state.sub_queries.append(sq)
    return state


def _make_source(
    source_id: str = "src-1",
    url: str = "https://example.com/1",
    title: str = "Test Source",
    content: str = "Test content about renewable energy benefits.",
    quality: SourceQuality = SourceQuality.MEDIUM,
    sub_query_id: str | None = None,
) -> ResearchSource:
    return ResearchSource(
        id=source_id,
        title=title,
        url=url,
        content=content,
        quality=quality,
        sub_query_id=sub_query_id,
        source_type=SourceType.WEB,
    )


def _add_sources_for_topic(
    state: DeepResearchState,
    sub_query_id: str,
    num_sources: int = 3,
) -> TopicResearchResult:
    """Add sources to state and create a TopicResearchResult for them."""
    source_ids = []
    for i in range(num_sources):
        src_id = f"src-{sub_query_id}-{i}"
        source = _make_source(
            source_id=src_id,
            url=f"https://example.com/{sub_query_id}/{i}",
            title=f"Source {i} for {sub_query_id}",
            content=f"Detailed content about topic {sub_query_id}, finding {i}.",
            sub_query_id=sub_query_id,
        )
        state.append_source(source)
        source_ids.append(src_id)

    return TopicResearchResult(
        sub_query_id=sub_query_id,
        searches_performed=1,
        sources_found=num_sources,
        source_ids=source_ids,
    )


class StubGatheringMixin(GatheringPhaseMixin):
    """Concrete class inheriting GatheringPhaseMixin for compression testing."""

    def __init__(self) -> None:
        self.config = MagicMock()
        self.config.default_provider = "test-provider"
        self.config.get_compression_provider = MagicMock(return_value="test-provider")
        self.config.get_compression_model = MagicMock(return_value=None)
        self.config.get_phase_fallback_providers = MagicMock(return_value=[])
        self.config.deep_research_max_retries = 1
        self.config.deep_research_retry_delay = 0.1
        self.memory = MagicMock()
        self._search_providers: dict[str, Any] = {}
        self._audit_events: list[tuple[str, dict]] = []
        self._cancelled = False
        self._provider_async_fn: Any = None

    def _write_audit_event(self, state: Any, event: str, **kwargs: Any) -> None:
        self._audit_events.append((event, kwargs))

    def _check_cancellation(self, state: Any) -> None:
        if self._cancelled:
            raise asyncio.CancelledError()

    async def _execute_provider_async(self, **kwargs: Any) -> MagicMock:
        """Mock provider async execution for compression calls."""
        if self._provider_async_fn:
            return await self._provider_async_fn(**kwargs)
        # Default: return a compressed findings response
        result = MagicMock()
        result.success = True
        result.content = (
            "## Queries Made\n"
            "- Sub-query about renewable energy\n\n"
            "## Comprehensive Findings\n"
            "Renewable energy reduces carbon emissions [1]. "
            "Solar power costs have decreased by 90% [2]. "
            "Wind energy is now cost-competitive with fossil fuels [3].\n\n"
            "## Source List\n"
            "- [1] Source 0 — https://example.com/sq-0/0\n"
            "- [2] Source 1 — https://example.com/sq-0/1\n"
            "- [3] Source 2 — https://example.com/sq-0/2\n"
        )
        result.tokens_used = 200
        result.input_tokens = 150
        result.output_tokens = 50
        result.provider_id = "test-provider"
        result.model_used = "test-model"
        return result


class StubAnalysisPrompts(AnalysisPromptsMixin):
    """Concrete class inheriting AnalysisPromptsMixin for testing."""
    pass


# =============================================================================
# Unit tests: TopicResearchResult.compressed_findings field
# =============================================================================


class TestCompressedFindingsField:
    """Tests for the compressed_findings field on TopicResearchResult."""

    def test_default_is_none(self) -> None:
        """compressed_findings defaults to None."""
        result = TopicResearchResult(sub_query_id="sq-1")
        assert result.compressed_findings is None

    def test_set_and_get(self) -> None:
        """compressed_findings can be set and retrieved."""
        result = TopicResearchResult(
            sub_query_id="sq-1",
            compressed_findings="## Findings\nSome findings [1].",
        )
        assert result.compressed_findings == "## Findings\nSome findings [1]."

    def test_backward_compat_deserialization(self) -> None:
        """TopicResearchResult without compressed_findings deserializes cleanly."""
        data = {
            "sub_query_id": "sq-1",
            "searches_performed": 2,
            "sources_found": 3,
            "source_ids": ["src-1", "src-2"],
        }
        result = TopicResearchResult(**data)
        assert result.compressed_findings is None
        assert result.searches_performed == 2

    def test_model_dump_includes_field(self) -> None:
        """compressed_findings is included in model_dump."""
        result = TopicResearchResult(
            sub_query_id="sq-1",
            compressed_findings="Test findings",
        )
        dumped = result.model_dump()
        assert "compressed_findings" in dumped
        assert dumped["compressed_findings"] == "Test findings"

    def test_model_dump_none_when_unset(self) -> None:
        """compressed_findings is None in model_dump when not set."""
        result = TopicResearchResult(sub_query_id="sq-1")
        dumped = result.model_dump()
        assert dumped["compressed_findings"] is None


# =============================================================================
# Unit tests: Per-topic compression execution
# =============================================================================


class TestCompressTopicFindings:
    """Tests for _compress_topic_findings_async."""

    @pytest.mark.asyncio
    async def test_compresses_topics_with_sources(self) -> None:
        """Topics with sources get compressed findings."""
        mixin = StubGatheringMixin()
        state = _make_state(num_sub_queries=1)

        topic_result = _add_sources_for_topic(state, "sq-0", num_sources=3)
        state.topic_research_results.append(topic_result)

        stats = await mixin._compress_topic_findings_async(
            state=state,
            max_concurrent=3,
            timeout=60.0,
        )

        assert stats["topics_compressed"] == 1
        assert stats["topics_failed"] == 0
        assert stats["total_compression_tokens"] == 200
        assert topic_result.compressed_findings is not None
        assert "Renewable energy" in topic_result.compressed_findings

    @pytest.mark.asyncio
    async def test_skips_topics_without_sources(self) -> None:
        """Topics with no source_ids are skipped."""
        mixin = StubGatheringMixin()
        state = _make_state(num_sub_queries=1)

        topic_result = TopicResearchResult(
            sub_query_id="sq-0",
            searches_performed=1,
            sources_found=0,
            source_ids=[],
        )
        state.topic_research_results.append(topic_result)

        stats = await mixin._compress_topic_findings_async(
            state=state,
            max_concurrent=3,
            timeout=60.0,
        )

        assert stats["topics_compressed"] == 0
        assert stats["topics_failed"] == 0
        assert topic_result.compressed_findings is None

    @pytest.mark.asyncio
    async def test_skips_already_compressed_topics(self) -> None:
        """Topics that already have compressed_findings are not re-compressed."""
        mixin = StubGatheringMixin()
        state = _make_state(num_sub_queries=1)

        topic_result = _add_sources_for_topic(state, "sq-0", num_sources=2)
        topic_result.compressed_findings = "Already compressed."
        state.topic_research_results.append(topic_result)

        stats = await mixin._compress_topic_findings_async(
            state=state,
            max_concurrent=3,
            timeout=60.0,
        )

        assert stats["topics_compressed"] == 0
        assert topic_result.compressed_findings == "Already compressed."

    @pytest.mark.asyncio
    async def test_compression_prompt_includes_sources(self) -> None:
        """Compression prompt contains source titles, URLs, and content."""
        mixin = StubGatheringMixin()
        state = _make_state(num_sub_queries=1)

        topic_result = _add_sources_for_topic(state, "sq-0", num_sources=2)
        state.topic_research_results.append(topic_result)

        captured_prompts: list[str] = []

        async def capture_prompt(**kwargs: Any) -> MagicMock:
            captured_prompts.append(kwargs.get("prompt", ""))
            result = MagicMock()
            result.success = True
            result.content = "## Findings\nCompressed [1]."
            result.tokens_used = 100
            result.input_tokens = 75
            result.output_tokens = 25
            return result

        mixin._provider_async_fn = capture_prompt

        await mixin._compress_topic_findings_async(
            state=state,
            max_concurrent=3,
            timeout=60.0,
        )

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # Should contain source titles
        assert "Source 0 for sq-0" in prompt
        assert "Source 1 for sq-0" in prompt
        # Should contain URLs
        assert "https://example.com/sq-0/0" in prompt
        assert "https://example.com/sq-0/1" in prompt
        # Should contain content
        assert "Detailed content about topic sq-0" in prompt
        # Should reference the sub-query
        assert "Sub-query 0" in prompt

    @pytest.mark.asyncio
    async def test_compression_system_prompt_preserves_info(self) -> None:
        """System prompt instructs NOT to summarize, only reformat."""
        mixin = StubGatheringMixin()
        state = _make_state(num_sub_queries=1)

        topic_result = _add_sources_for_topic(state, "sq-0", num_sources=1)
        state.topic_research_results.append(topic_result)

        captured_system: list[str] = []

        async def capture_system(**kwargs: Any) -> MagicMock:
            captured_system.append(kwargs.get("system_prompt", ""))
            result = MagicMock()
            result.success = True
            result.content = "Compressed."
            result.tokens_used = 50
            result.input_tokens = 35
            result.output_tokens = 15
            return result

        mixin._provider_async_fn = capture_system

        await mixin._compress_topic_findings_async(
            state=state,
            max_concurrent=3,
            timeout=60.0,
        )

        assert len(captured_system) == 1
        sys_prompt = captured_system[0]
        assert "DO NOT summarize" in sys_prompt
        assert "inline citations" in sys_prompt.lower() or "inline citations" in sys_prompt
        assert "[1]" in sys_prompt

    @pytest.mark.asyncio
    async def test_fallback_on_provider_failure(self) -> None:
        """When compression LLM call fails, compressed_findings stays None."""
        mixin = StubGatheringMixin()
        state = _make_state(num_sub_queries=1)

        topic_result = _add_sources_for_topic(state, "sq-0", num_sources=2)
        state.topic_research_results.append(topic_result)

        async def failing_provider(**kwargs: Any) -> MagicMock:
            result = MagicMock()
            result.success = False
            result.content = ""
            result.tokens_used = 0
            return result

        mixin._provider_async_fn = failing_provider

        stats = await mixin._compress_topic_findings_async(
            state=state,
            max_concurrent=3,
            timeout=60.0,
        )

        assert stats["topics_compressed"] == 0
        assert stats["topics_failed"] == 1
        assert topic_result.compressed_findings is None

    @pytest.mark.asyncio
    async def test_fallback_on_exception(self) -> None:
        """When compression raises an exception, compressed_findings stays None."""
        mixin = StubGatheringMixin()
        state = _make_state(num_sub_queries=1)

        topic_result = _add_sources_for_topic(state, "sq-0", num_sources=2)
        state.topic_research_results.append(topic_result)

        async def exception_provider(**kwargs: Any) -> MagicMock:
            raise RuntimeError("LLM provider unavailable")

        mixin._provider_async_fn = exception_provider

        stats = await mixin._compress_topic_findings_async(
            state=state,
            max_concurrent=3,
            timeout=60.0,
        )

        assert stats["topics_compressed"] == 0
        assert stats["topics_failed"] == 1
        assert topic_result.compressed_findings is None

    @pytest.mark.asyncio
    async def test_progressive_truncation_on_context_window_error(self) -> None:
        """Context window errors trigger progressive truncation retries."""
        from foundry_mcp.core.errors.provider import ContextWindowError

        mixin = StubGatheringMixin()
        state = _make_state(num_sub_queries=1)

        topic_result = _add_sources_for_topic(state, "sq-0", num_sources=2)
        state.topic_research_results.append(topic_result)

        call_count = 0

        async def retry_then_succeed(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ContextWindowError(
                    message="Prompt too long",
                    prompt_tokens=200000,
                    max_tokens=128000,
                )
            result = MagicMock()
            result.success = True
            result.content = "## Compressed after retry\nFindings [1]."
            result.tokens_used = 150
            result.input_tokens = 110
            result.output_tokens = 40
            return result

        mixin._provider_async_fn = retry_then_succeed

        stats = await mixin._compress_topic_findings_async(
            state=state,
            max_concurrent=3,
            timeout=60.0,
        )

        assert call_count == 3  # 2 failures + 1 success
        assert stats["topics_compressed"] == 1
        assert topic_result.compressed_findings is not None
        assert "after retry" in topic_result.compressed_findings

    @pytest.mark.asyncio
    async def test_progressive_truncation_all_retries_exhausted(self) -> None:
        """When all truncation retries are exhausted, topic fails gracefully."""
        from foundry_mcp.core.errors.provider import ContextWindowError

        mixin = StubGatheringMixin()
        state = _make_state(num_sub_queries=1)

        topic_result = _add_sources_for_topic(state, "sq-0", num_sources=2)
        state.topic_research_results.append(topic_result)

        async def always_fail(**kwargs: Any) -> MagicMock:
            raise ContextWindowError(
                message="Prompt too long",
                prompt_tokens=200000,
                max_tokens=128000,
            )

        mixin._provider_async_fn = always_fail

        stats = await mixin._compress_topic_findings_async(
            state=state,
            max_concurrent=3,
            timeout=60.0,
        )

        assert stats["topics_compressed"] == 0
        assert stats["topics_failed"] == 1
        assert topic_result.compressed_findings is None


# =============================================================================
# Unit tests: Parallel compression across multiple topics
# =============================================================================


class TestParallelCompression:
    """Tests for parallel compression across multiple topic researchers."""

    @pytest.mark.asyncio
    async def test_parallel_compression_multiple_topics(self) -> None:
        """Multiple topics are compressed in parallel."""
        mixin = StubGatheringMixin()
        state = _make_state(num_sub_queries=3)

        for i in range(3):
            tr = _add_sources_for_topic(state, f"sq-{i}", num_sources=2)
            state.topic_research_results.append(tr)

        topic_ids_compressed: list[str] = []

        async def track_topics(**kwargs: Any) -> MagicMock:
            # Extract topic from prompt to track which topics were compressed
            prompt = kwargs.get("prompt", "")
            for i in range(3):
                if f"sq-{i}" in prompt:
                    topic_ids_compressed.append(f"sq-{i}")
                    break
            result = MagicMock()
            result.success = True
            result.content = f"## Findings for topic\nCompressed [1] [2]."
            result.tokens_used = 100
            result.input_tokens = 75
            result.output_tokens = 25
            return result

        mixin._provider_async_fn = track_topics

        stats = await mixin._compress_topic_findings_async(
            state=state,
            max_concurrent=3,
            timeout=60.0,
        )

        assert stats["topics_compressed"] == 3
        assert stats["topics_failed"] == 0
        assert stats["total_compression_tokens"] == 300
        # All topics should have compressed findings
        for tr in state.topic_research_results:
            assert tr.compressed_findings is not None

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self) -> None:
        """Some topics compress successfully, others fail."""
        mixin = StubGatheringMixin()
        state = _make_state(num_sub_queries=3)

        for i in range(3):
            tr = _add_sources_for_topic(state, f"sq-{i}", num_sources=2)
            state.topic_research_results.append(tr)

        call_idx = 0

        async def mixed_results(**kwargs: Any) -> MagicMock:
            nonlocal call_idx
            call_idx += 1
            result = MagicMock()
            if call_idx == 2:
                # Second topic fails
                result.success = False
                result.content = ""
                result.tokens_used = 0
                result.input_tokens = 0
                result.output_tokens = 0
            else:
                result.success = True
                result.content = "## Findings\nCompressed [1]."
                result.tokens_used = 100
                result.input_tokens = 75
                result.output_tokens = 25
            return result

        mixin._provider_async_fn = mixed_results

        stats = await mixin._compress_topic_findings_async(
            state=state,
            max_concurrent=3,
            timeout=60.0,
        )

        assert stats["topics_compressed"] == 2
        assert stats["topics_failed"] == 1

    @pytest.mark.asyncio
    async def test_concurrency_bounded_by_semaphore(self) -> None:
        """Compression respects max_concurrent semaphore."""
        mixin = StubGatheringMixin()
        state = _make_state(num_sub_queries=5)

        for i in range(5):
            tr = _add_sources_for_topic(state, f"sq-{i}", num_sources=1)
            state.topic_research_results.append(tr)

        max_concurrent_observed = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def track_concurrency(**kwargs: Any) -> MagicMock:
            nonlocal max_concurrent_observed, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent_observed:
                    max_concurrent_observed = current_concurrent
            await asyncio.sleep(0.05)  # Simulate work
            async with lock:
                current_concurrent -= 1
            result = MagicMock()
            result.success = True
            result.content = "Compressed."
            result.tokens_used = 50
            result.input_tokens = 35
            result.output_tokens = 15
            return result

        mixin._provider_async_fn = track_concurrency

        # Limit to 2 concurrent
        stats = await mixin._compress_topic_findings_async(
            state=state,
            max_concurrent=2,
            timeout=60.0,
        )

        assert stats["topics_compressed"] == 5
        assert max_concurrent_observed <= 2


# =============================================================================
# Unit tests: Token tracking
# =============================================================================


class TestCompressionTokenTracking:
    """Tests for compression token tracking in PhaseMetrics."""

    @pytest.mark.asyncio
    async def test_compression_tokens_tracked_in_state(self) -> None:
        """Compression tokens are added to state.total_tokens_used."""
        mixin = StubGatheringMixin()
        state = _make_state(num_sub_queries=1)
        initial_tokens = state.total_tokens_used

        tr = _add_sources_for_topic(state, "sq-0", num_sources=2)
        state.topic_research_results.append(tr)

        stats = await mixin._compress_topic_findings_async(
            state=state,
            max_concurrent=3,
            timeout=60.0,
        )

        assert state.total_tokens_used == initial_tokens + stats["total_compression_tokens"]

    @pytest.mark.asyncio
    async def test_compression_phase_metrics_recorded(self) -> None:
        """A PhaseMetrics entry is recorded for compression."""
        mixin = StubGatheringMixin()
        state = _make_state(num_sub_queries=1)

        tr = _add_sources_for_topic(state, "sq-0", num_sources=2)
        state.topic_research_results.append(tr)

        initial_metrics_count = len(state.phase_metrics)

        await mixin._compress_topic_findings_async(
            state=state,
            max_concurrent=3,
            timeout=60.0,
        )

        assert len(state.phase_metrics) == initial_metrics_count + 1
        compression_metric = state.phase_metrics[-1]
        assert compression_metric.phase == "compression"
        assert compression_metric.input_tokens == 150
        assert compression_metric.output_tokens == 50
        assert compression_metric.metadata["topics_compressed"] == 1

    @pytest.mark.asyncio
    async def test_no_metrics_when_no_tokens(self) -> None:
        """No PhaseMetrics entry when no compression occurs."""
        mixin = StubGatheringMixin()
        state = _make_state(num_sub_queries=1)

        # Topic with no sources — won't compress
        tr = TopicResearchResult(sub_query_id="sq-0", source_ids=[])
        state.topic_research_results.append(tr)

        initial_count = len(state.phase_metrics)

        await mixin._compress_topic_findings_async(
            state=state,
            max_concurrent=3,
            timeout=60.0,
        )

        assert len(state.phase_metrics) == initial_count


# =============================================================================
# Unit tests: Audit events
# =============================================================================


class TestCompressionAuditEvents:
    """Tests for compression audit event emission."""

    @pytest.mark.asyncio
    async def test_audit_event_emitted_on_completion(self) -> None:
        """topic_compression_complete audit event is emitted."""
        mixin = StubGatheringMixin()
        state = _make_state(num_sub_queries=1)

        tr = _add_sources_for_topic(state, "sq-0", num_sources=2)
        state.topic_research_results.append(tr)

        await mixin._compress_topic_findings_async(
            state=state,
            max_concurrent=3,
            timeout=60.0,
        )

        events = [e[0] for e in mixin._audit_events]
        assert "topic_compression_complete" in events

        # Check event data
        compression_event = next(
            e for e in mixin._audit_events if e[0] == "topic_compression_complete"
        )
        event_data = compression_event[1]["data"]
        assert event_data["topics_compressed"] == 1
        assert event_data["topics_failed"] == 0
        assert event_data["compression_provider"] == "test-provider"


# =============================================================================
# Unit tests: Analysis phase uses compressed findings
# =============================================================================


class TestAnalysisUsesCompressedFindings:
    """Tests for the analysis prompt preferring compressed_findings."""

    def test_prompt_uses_compressed_findings_when_available(self) -> None:
        """When all topics have compressed_findings, analysis prompt includes them."""
        mixin = StubAnalysisPrompts()
        state = _make_state(num_sub_queries=2)

        # Add sources and topic results with compressed findings
        for i in range(2):
            tr = _add_sources_for_topic(state, f"sq-{i}", num_sources=2)
            tr.compressed_findings = (
                f"## Findings for topic {i}\n"
                f"Key insight from topic {i} [1] [2].\n"
                f"## Source List\n"
                f"- [1] Source 0 — https://example.com/sq-{i}/0\n"
                f"- [2] Source 1 — https://example.com/sq-{i}/1\n"
            )
            state.topic_research_results.append(tr)

        prompt = mixin._build_analysis_user_prompt(state)

        # Should contain compressed findings
        assert "Per-Topic Research Summaries" in prompt
        assert "Key insight from topic 0" in prompt
        assert "Key insight from topic 1" in prompt
        # Should contain topic headers
        assert "Topic 1:" in prompt
        assert "Topic 2:" in prompt

    def test_prompt_falls_back_to_raw_sources_when_no_compression(self) -> None:
        """When no topics have compressed_findings, prompt uses raw sources."""
        mixin = StubAnalysisPrompts()
        state = _make_state(num_sub_queries=1)

        tr = _add_sources_for_topic(state, "sq-0", num_sources=2)
        # No compressed_findings set
        state.topic_research_results.append(tr)

        prompt = mixin._build_analysis_user_prompt(state)

        # Should use raw source listing
        assert "Sources to Analyze:" in prompt
        assert "Per-Topic Research Summaries" not in prompt
        # Should contain individual source details
        assert "Source 0 for sq-0" in prompt
        assert "Source 1 for sq-0" in prompt

    def test_mixed_compressed_and_uncompressed_topics(self) -> None:
        """When some topics have compressed findings and others don't."""
        mixin = StubAnalysisPrompts()
        state = _make_state(num_sub_queries=2)

        # Topic 0: has compressed findings
        tr0 = _add_sources_for_topic(state, "sq-0", num_sources=2)
        tr0.compressed_findings = "## Findings\nCompressed topic 0 [1] [2]."
        state.topic_research_results.append(tr0)

        # Topic 1: no compressed findings
        tr1 = _add_sources_for_topic(state, "sq-1", num_sources=2)
        state.topic_research_results.append(tr1)

        prompt = mixin._build_analysis_user_prompt(state)

        # Should have compressed section for topic 0
        assert "Per-Topic Research Summaries" in prompt
        assert "Compressed topic 0" in prompt
        # Should have raw sources section for topic 1's uncompressed sources
        assert "Additional Sources (not yet compressed)" in prompt

    def test_source_id_mapping_in_compressed_prompt(self) -> None:
        """Compressed findings prompt includes source ID → citation mapping."""
        mixin = StubAnalysisPrompts()
        state = _make_state(num_sub_queries=1)

        tr = _add_sources_for_topic(state, "sq-0", num_sources=2)
        tr.compressed_findings = "## Findings\nSome findings [1] [2]."
        state.topic_research_results.append(tr)

        prompt = mixin._build_analysis_user_prompt(state)

        # Should contain source ID mapping
        assert "Source ID mapping" in prompt
        assert "src-sq-0-0" in prompt
        assert "src-sq-0-1" in prompt

    def test_no_topic_results_uses_raw_sources(self) -> None:
        """When there are no topic_research_results at all, uses raw sources."""
        mixin = StubAnalysisPrompts()
        state = _make_state(num_sub_queries=0)

        # Add sources directly (no topic research results)
        for i in range(3):
            state.append_source(_make_source(
                source_id=f"src-{i}",
                url=f"https://example.com/{i}",
                title=f"Direct Source {i}",
            ))

        prompt = mixin._build_analysis_user_prompt(state)

        assert "Sources to Analyze:" in prompt
        assert "Per-Topic Research Summaries" not in prompt
        assert "Direct Source 0" in prompt


# =============================================================================
# Unit tests: Config fields
# =============================================================================


class TestCompressionConfig:
    """Tests for compression configuration fields."""

    def test_default_compression_provider(self) -> None:
        """Default compression provider falls back to default_provider."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig(default_provider="gemini")
        assert config.deep_research_compression_provider is None
        assert config.get_compression_provider() == "gemini"

    def test_explicit_compression_provider(self) -> None:
        """Explicit compression provider is used when set."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig(
            default_provider="gemini",
            deep_research_compression_provider="claude",
        )
        assert config.get_compression_provider() == "claude"

    def test_default_compression_model(self) -> None:
        """Default compression model is None."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig()
        assert config.get_compression_model() is None

    def test_explicit_compression_model(self) -> None:
        """Explicit compression model is used when set."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig(deep_research_compression_model="haiku")
        assert config.get_compression_model() == "haiku"

    def test_compression_provider_from_provider_spec(self) -> None:
        """Compression model extracted from ProviderSpec bracket format."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig(
            deep_research_compression_provider="[cli]claude:haiku",
        )
        assert config.get_compression_provider() == "claude"
        # Model from provider spec
        assert config.get_compression_model() == "haiku"

    def test_from_toml_dict_parses_compression_fields(self) -> None:
        """from_toml_dict correctly parses compression config fields."""
        from foundry_mcp.config.research import ResearchConfig

        data = {
            "deep_research_compression_provider": "openai",
            "deep_research_compression_model": "gpt-4o-mini",
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_compression_provider == "openai"
        assert config.deep_research_compression_model == "gpt-4o-mini"

    def test_from_toml_dict_defaults_when_absent(self) -> None:
        """from_toml_dict uses None defaults when compression fields absent."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig.from_toml_dict({})
        assert config.deep_research_compression_provider is None
        assert config.deep_research_compression_model is None
