"""Stress tests for concurrent state mutation in deep research.

Tests cover:
1. state_lock correctness with multiple topic agents mutating state concurrently
2. total_tokens_used consistency after parallel updates
3. Source deduplication under concurrent appends
4. Citation counter consistency under concurrent add_source / append_source calls
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

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


# =============================================================================
# Helpers
# =============================================================================


def _make_state(
    num_sub_queries: int = 5,
) -> DeepResearchState:
    """Create a DeepResearchState with sub-queries for concurrent testing."""
    state = DeepResearchState(
        id="deepres-concurrent-test",
        original_query="concurrent mutation test",
        phase=DeepResearchPhase.GATHERING,
        iteration=1,
        max_iterations=3,
        max_sources_per_query=10,
    )
    for i in range(num_sub_queries):
        state.sub_queries.append(
            SubQuery(
                id=f"sq-{i}",
                query=f"Sub-query {i}",
                rationale=f"Rationale {i}",
                priority=i + 1,
            )
        )
    return state


def _make_source(
    source_id: str,
    url: str = "",
    title: str = "Test Source",
) -> ResearchSource:
    return ResearchSource(
        id=source_id,
        title=title,
        url=url or f"https://example.com/{source_id}",
        content="Test content",
        source_type=SourceType.WEB,
        quality=SourceQuality.MEDIUM,
    )


# =============================================================================
# Tests: Concurrent token updates
# =============================================================================


class TestConcurrentTokenTracking:
    """Stress tests for total_tokens_used under concurrent mutation."""

    @pytest.mark.asyncio
    async def test_concurrent_token_increments_are_consistent(self) -> None:
        """Multiple concurrent tasks incrementing total_tokens_used produce correct total.

        Under cooperative scheduling (asyncio), simple integer increments are
        safe because there's no preemption between read and write.  This test
        validates the invariant holds even with many concurrent coroutines.
        """
        state = _make_state()
        num_tasks = 50
        tokens_per_task = 100

        async def increment_tokens(task_id: int) -> None:
            # Simulate some async work before mutation
            await asyncio.sleep(0)
            state.total_tokens_used += tokens_per_task

        tasks = [increment_tokens(i) for i in range(num_tasks)]
        await asyncio.gather(*tasks)

        assert state.total_tokens_used == num_tasks * tokens_per_task

    @pytest.mark.asyncio
    async def test_concurrent_token_updates_with_lock(self) -> None:
        """Token updates using a lock remain consistent under contention."""
        state = _make_state()
        state_lock = asyncio.Lock()
        num_tasks = 100
        tokens_per_task = 37  # Odd number to catch off-by-one

        async def locked_increment(task_id: int) -> None:
            await asyncio.sleep(0)  # Yield to event loop
            async with state_lock:
                state.total_tokens_used += tokens_per_task

        tasks = [locked_increment(i) for i in range(num_tasks)]
        await asyncio.gather(*tasks)

        assert state.total_tokens_used == num_tasks * tokens_per_task


# =============================================================================
# Tests: Concurrent source appends
# =============================================================================


class TestConcurrentSourceAppends:
    """Stress tests for source dedup and citation numbering under concurrency."""

    @pytest.mark.asyncio
    async def test_concurrent_append_source_unique_citations(self) -> None:
        """Each append_source() call gets a unique citation number even under concurrency.

        Under cooperative scheduling, the citation counter increment in
        append_source() is atomic (no preemption), so all citation numbers
        should be unique and sequential.
        """
        state = _make_state()
        num_sources = 50

        async def append_one(idx: int) -> None:
            await asyncio.sleep(0)
            source = _make_source(f"src-{idx}", f"https://example.com/{idx}")
            state.append_source(source)

        tasks = [append_one(i) for i in range(num_sources)]
        await asyncio.gather(*tasks)

        assert len(state.sources) == num_sources

        # All citation numbers should be unique
        citation_numbers = [s.citation_number for s in state.sources]
        assert len(set(citation_numbers)) == num_sources

        # Citation counter should be ready for the next one
        assert state.next_citation_number == num_sources + 1

    @pytest.mark.asyncio
    async def test_concurrent_add_source_unique_citations(self) -> None:
        """Each add_source() call gets a unique citation number under concurrency."""
        state = _make_state()
        num_sources = 30

        async def add_one(idx: int) -> None:
            await asyncio.sleep(0)
            state.add_source(
                title=f"Source {idx}",
                url=f"https://example.com/{idx}",
                source_type=SourceType.WEB,
            )

        tasks = [add_one(i) for i in range(num_sources)]
        await asyncio.gather(*tasks)

        assert len(state.sources) == num_sources

        citation_numbers = [s.citation_number for s in state.sources]
        assert len(set(citation_numbers)) == num_sources
        assert state.next_citation_number == num_sources + 1

    @pytest.mark.asyncio
    async def test_concurrent_dedup_with_lock(self) -> None:
        """URL-based deduplication under concurrent appends with a lock."""
        state = _make_state()
        state_lock = asyncio.Lock()
        seen_urls: set[str] = set()

        # 20 coroutines each try to add sources with overlapping URLs
        # Sources 0-9 are unique, 10-19 duplicate 0-9
        num_unique = 10
        num_total = 20

        async def dedup_append(idx: int) -> None:
            url = f"https://example.com/{idx % num_unique}"
            await asyncio.sleep(0)
            async with state_lock:
                if url in seen_urls:
                    return
                seen_urls.add(url)
                source = _make_source(f"src-{idx}", url)
                state.append_source(source)

        tasks = [dedup_append(i) for i in range(num_total)]
        await asyncio.gather(*tasks)

        # Only unique URLs should be added
        assert len(state.sources) == num_unique
        urls = [s.url for s in state.sources]
        assert len(set(urls)) == num_unique


# =============================================================================
# Tests: Simulated multi-topic-agent state mutation
# =============================================================================


class TestMultiTopicAgentConcurrency:
    """Simulate multiple topic agents mutating shared state concurrently."""

    @pytest.mark.asyncio
    async def test_parallel_topic_agents_consistent_state(self) -> None:
        """Multiple topic agents adding sources and tokens in parallel.

        Simulates the gathering phase where each topic agent:
        1. Searches and finds sources
        2. Appends sources to shared state (under lock)
        3. Updates token counts
        4. Creates a TopicResearchResult
        """
        state = _make_state(num_sub_queries=5)
        state_lock = asyncio.Lock()
        seen_urls: set[str] = set()
        sources_per_topic = 4
        tokens_per_topic = 150

        async def topic_agent(topic_idx: int) -> TopicResearchResult:
            sq_id = f"sq-{topic_idx}"
            source_ids: list[str] = []

            # Simulate search
            await asyncio.sleep(0.01)

            # Add sources under lock
            for src_idx in range(sources_per_topic):
                src_id = f"src-{topic_idx}-{src_idx}"
                url = f"https://example.com/{topic_idx}/{src_idx}"
                async with state_lock:
                    if url not in seen_urls:
                        seen_urls.add(url)
                        source = _make_source(src_id, url, f"Source {src_idx} for topic {topic_idx}")
                        state.append_source(source)
                        source_ids.append(src_id)

            # Update tokens
            async with state_lock:
                state.total_tokens_used += tokens_per_topic

            return TopicResearchResult(
                sub_query_id=sq_id,
                searches_performed=1,
                sources_found=len(source_ids),
                source_ids=source_ids,
            )

        tasks = [topic_agent(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # Verify state consistency
        expected_sources = 5 * sources_per_topic
        assert len(state.sources) == expected_sources
        assert state.total_tokens_used == 5 * tokens_per_topic

        # All citation numbers unique
        citations = [s.citation_number for s in state.sources]
        assert len(set(citations)) == expected_sources

        # All topic results have correct source counts
        total_sources_from_results = sum(r.sources_found for r in results)
        assert total_sources_from_results == expected_sources

    @pytest.mark.asyncio
    async def test_parallel_topic_agents_with_overlapping_urls(self) -> None:
        """Topic agents finding the same URL don't create duplicate sources."""
        state = _make_state(num_sub_queries=3)
        state_lock = asyncio.Lock()
        seen_urls: set[str] = set()

        # Each topic agent finds 3 sources, but 1 URL is shared across all
        shared_url = "https://example.com/shared-paper"

        async def topic_agent(topic_idx: int) -> int:
            added = 0
            urls = [
                shared_url,
                f"https://example.com/{topic_idx}/unique-1",
                f"https://example.com/{topic_idx}/unique-2",
            ]
            for i, url in enumerate(urls):
                async with state_lock:
                    if url not in seen_urls:
                        seen_urls.add(url)
                        source = _make_source(
                            f"src-{topic_idx}-{i}", url,
                        )
                        state.append_source(source)
                        added += 1
            return added

        tasks = [topic_agent(i) for i in range(3)]
        results = await asyncio.gather(*tasks)

        # 1 shared + 2 unique per topic * 3 topics = 7 total
        assert len(state.sources) == 7
        total_added = sum(results)
        assert total_added == 7

        # Verify no duplicate URLs
        urls = [s.url for s in state.sources]
        assert len(set(urls)) == 7

    @pytest.mark.asyncio
    async def test_high_contention_token_tracking(self) -> None:
        """Many concurrent updates to token tracking under high contention."""
        state = _make_state()
        state_lock = asyncio.Lock()
        num_agents = 20
        updates_per_agent = 10
        tokens_per_update = 50

        async def agent_work(agent_id: int) -> None:
            for _ in range(updates_per_agent):
                await asyncio.sleep(0)  # Yield frequently
                async with state_lock:
                    state.total_tokens_used += tokens_per_update

        tasks = [agent_work(i) for i in range(num_agents)]
        await asyncio.gather(*tasks)

        expected = num_agents * updates_per_agent * tokens_per_update
        assert state.total_tokens_used == expected
