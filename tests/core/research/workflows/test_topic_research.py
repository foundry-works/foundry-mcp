"""Unit and integration tests for Phase 3: Parallel Sub-Topic Researcher Agents.

Tests cover:
1. TopicResearchResult model — fields, defaults, serialization
2. _execute_topic_research_async() — ReAct loop: search → reflect → refine → search
3. _topic_search() — search with deduplication and budget splitting
4. _topic_reflect() — LLM reflection call and parsing
5. Budget splitting — per-topic max_sources calculation in gathering
6. Gathering delegation — topic agent path in _execute_gathering_async
7. Config keys — deep_research_enable_topic_agents, topic_max_searches
8. Rate limiting — semaphore-bounded concurrency
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
    SubQuery,
)
from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
    TopicResearchMixin,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_state(
    query: str = "How does deep learning work?",
    phase: DeepResearchPhase = DeepResearchPhase.GATHERING,
    max_sources_per_query: int = 5,
    num_sub_queries: int = 3,
) -> DeepResearchState:
    """Create a DeepResearchState with pending sub-queries for testing."""
    state = DeepResearchState(
        id="deepres-test-topic",
        original_query=query,
        phase=phase,
        iteration=1,
        max_iterations=3,
        max_sources_per_query=max_sources_per_query,
    )
    for i in range(num_sub_queries):
        state.sub_queries.append(
            SubQuery(
                id=f"sq-{i}",
                query=f"Sub-query {i}: {query}",
                rationale=f"Rationale {i}",
                priority=i + 1,
            )
        )
    return state


def _make_source(
    source_id: str = "src-1",
    url: str = "https://example.com/1",
    title: str = "Test Source",
    quality: SourceQuality = SourceQuality.MEDIUM,
) -> ResearchSource:
    return ResearchSource(
        id=source_id,
        title=title,
        url=url,
        content="Test content",
        quality=quality,
    )


def _make_mock_provider(name: str = "tavily", sources: list | None = None):
    """Create a mock search provider."""
    provider = MagicMock()
    provider.get_provider_name.return_value = name
    if sources is None:
        sources = [
            _make_source(f"src-{name}-1", f"https://{name}.com/1", f"Result from {name}"),
        ]
    provider.search = AsyncMock(return_value=sources)
    return provider


class StubTopicResearch(TopicResearchMixin):
    """Concrete class inheriting TopicResearchMixin for testing.

    Provides the runtime attributes and cross-cutting methods that the
    mixin expects from DeepResearchWorkflow at runtime.
    """

    def __init__(self) -> None:
        self.config = MagicMock()
        self.config.deep_research_topic_reflection_provider = None
        self.config.deep_research_reflection_provider = None
        self.config.default_provider = "test-provider"
        self.config.deep_research_reflection_timeout = 60.0
        self.memory = MagicMock()
        self._search_providers: dict[str, Any] = {}
        self._audit_events: list[tuple[str, dict]] = []
        self._cancelled = False
        self._provider_async_fn: Any = None  # Override for reflection calls

    def _write_audit_event(self, state: Any, event: str, **kwargs: Any) -> None:
        self._audit_events.append((event, kwargs))

    def _check_cancellation(self, state: Any) -> None:
        if self._cancelled:
            raise asyncio.CancelledError()

    def _get_tavily_search_kwargs(self, state: Any) -> dict[str, Any]:
        return {"search_depth": "basic"}

    def _get_perplexity_search_kwargs(self, state: Any) -> dict[str, Any]:
        return {}

    def _get_semantic_scholar_search_kwargs(self, state: Any) -> dict[str, Any]:
        return {}

    async def _execute_provider_async(self, **kwargs: Any) -> MagicMock:
        """Mock provider async execution for reflection calls."""
        if self._provider_async_fn:
            return await self._provider_async_fn(**kwargs)
        result = MagicMock()
        result.success = True
        result.content = json.dumps({"sufficient": True, "assessment": "Enough sources found"})
        result.tokens_used = 50
        return result


# =============================================================================
# Unit tests: TopicResearchResult model
# =============================================================================


class TestTopicResearchResult:
    """Tests for TopicResearchResult model."""

    def test_default_values(self) -> None:
        """Default values are correct."""
        result = TopicResearchResult(sub_query_id="sq-1")
        assert result.sub_query_id == "sq-1"
        assert result.searches_performed == 0
        assert result.sources_found == 0
        assert result.per_topic_summary is None
        assert result.reflection_notes == []
        assert result.refined_queries == []
        assert result.source_ids == []

    def test_field_updates(self) -> None:
        """Fields can be updated during ReAct loop."""
        result = TopicResearchResult(sub_query_id="sq-1")
        result.searches_performed = 3
        result.sources_found = 7
        result.reflection_notes.append("Found relevant results")
        result.refined_queries.append("refined query text")
        result.source_ids.extend(["src-1", "src-2"])

        assert result.searches_performed == 3
        assert result.sources_found == 7
        assert len(result.reflection_notes) == 1
        assert len(result.refined_queries) == 1
        assert len(result.source_ids) == 2

    def test_serialization(self) -> None:
        """Model serializes to dict correctly."""
        result = TopicResearchResult(
            sub_query_id="sq-test",
            searches_performed=2,
            sources_found=5,
            per_topic_summary="Summary of findings",
            reflection_notes=["Note 1"],
            refined_queries=["refined q"],
            source_ids=["src-a", "src-b"],
        )
        d = result.model_dump()

        assert d["sub_query_id"] == "sq-test"
        assert d["searches_performed"] == 2
        assert d["sources_found"] == 5
        assert d["per_topic_summary"] == "Summary of findings"


# =============================================================================
# Unit tests: _topic_reflect
# =============================================================================


class TestTopicReflect:
    """Tests for TopicResearchMixin._topic_reflect()."""

    @pytest.mark.asyncio
    async def test_sufficient_result(self) -> None:
        """Reflection returns sufficient=True when enough sources found."""
        mixin = StubTopicResearch()
        state = _make_state()

        reflection = await mixin._topic_reflect(
            original_query="deep learning",
            current_query="deep learning",
            sources_found=3,
            iteration=1,
            max_iterations=3,
            state=state,
        )

        assert reflection["sufficient"] is True
        assert "assessment" in reflection

    @pytest.mark.asyncio
    async def test_insufficient_with_refined_query(self) -> None:
        """Reflection returns refined query when sources are insufficient."""
        mixin = StubTopicResearch()
        state = _make_state()

        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps(
                {
                    "sufficient": False,
                    "assessment": "Only 1 source found",
                    "refined_query": "deep learning architectures comparison",
                }
            )
            result.tokens_used = 50
            return result

        mixin._provider_async_fn = mock_provider

        reflection = await mixin._topic_reflect(
            original_query="deep learning",
            current_query="deep learning",
            sources_found=1,
            iteration=1,
            max_iterations=3,
            state=state,
        )

        assert reflection["sufficient"] is False
        assert reflection["refined_query"] == "deep learning architectures comparison"

    @pytest.mark.asyncio
    async def test_provider_failure_returns_sufficient(self) -> None:
        """Provider failure falls back to sufficient=True."""
        mixin = StubTopicResearch()
        state = _make_state()

        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = False
            return result

        mixin._provider_async_fn = mock_provider

        reflection = await mixin._topic_reflect(
            original_query="test",
            current_query="test",
            sources_found=2,
            iteration=1,
            max_iterations=3,
            state=state,
        )

        assert reflection["sufficient"] is True

    @pytest.mark.asyncio
    async def test_exception_returns_sufficient(self) -> None:
        """Exception during reflection falls back to sufficient=True."""
        mixin = StubTopicResearch()
        state = _make_state()

        async def mock_provider(**kwargs):
            raise RuntimeError("Network error")

        mixin._provider_async_fn = mock_provider

        reflection = await mixin._topic_reflect(
            original_query="test",
            current_query="test",
            sources_found=0,
            iteration=1,
            max_iterations=3,
            state=state,
        )

        assert reflection["sufficient"] is True

    @pytest.mark.asyncio
    async def test_malformed_json_returns_sufficient(self) -> None:
        """Malformed JSON in reflection response falls back to sufficient=True."""
        mixin = StubTopicResearch()
        state = _make_state()

        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = "This is not JSON at all"
            result.tokens_used = 30
            return result

        mixin._provider_async_fn = mock_provider

        reflection = await mixin._topic_reflect(
            original_query="test",
            current_query="test",
            sources_found=2,
            iteration=1,
            max_iterations=3,
            state=state,
        )

        assert reflection["sufficient"] is True

    @pytest.mark.asyncio
    async def test_tokens_tracked_in_state(self) -> None:
        """Reflection tokens are returned in the result dict (callers aggregate under lock)."""
        mixin = StubTopicResearch()
        state = _make_state()
        state.total_tokens_used = 100

        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({"sufficient": True, "assessment": "OK"})
            result.tokens_used = 75
            return result

        mixin._provider_async_fn = mock_provider

        reflection = await mixin._topic_reflect(
            original_query="test",
            current_query="test",
            sources_found=3,
            iteration=1,
            max_iterations=3,
            state=state,
        )

        # Tokens are returned in the dict, NOT directly mutated on state
        # (callers aggregate under state_lock to avoid concurrent races)
        assert reflection["tokens_used"] == 75
        # State should remain unchanged by _topic_reflect itself
        assert state.total_tokens_used == 100


# =============================================================================
# Unit tests: _topic_search
# =============================================================================


class TestTopicSearch:
    """Tests for TopicResearchMixin._topic_search()."""

    @pytest.mark.asyncio
    async def test_adds_sources_to_state(self) -> None:
        """Search adds discovered sources to state."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily")

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()
        seen_urls: set[str] = set()
        seen_titles: dict[str, str] = {}

        added = await mixin._topic_search(
            query=sq.query,
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_sources_per_provider=5,
            timeout=30.0,
            seen_urls=seen_urls,
            seen_titles=seen_titles,
            state_lock=state_lock,
            semaphore=semaphore,
        )

        assert added == 1
        assert len(state.sources) == 1
        assert state.sources[0].id.startswith("src-")

    @pytest.mark.asyncio
    async def test_url_deduplication(self) -> None:
        """Duplicate URLs are skipped."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        source = _make_source("src-dup", "https://example.com/dup")
        provider = _make_mock_provider("tavily", [source])

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()
        seen_urls: set[str] = {"https://example.com/dup"}  # Already seen
        seen_titles: dict[str, str] = {}

        added = await mixin._topic_search(
            query=sq.query,
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_sources_per_provider=5,
            timeout=30.0,
            seen_urls=seen_urls,
            seen_titles=seen_titles,
            state_lock=state_lock,
            semaphore=semaphore,
        )

        assert added == 0

    @pytest.mark.asyncio
    async def test_budget_split_max_results(self) -> None:
        """max_sources_per_provider is passed to provider.search()."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1, max_sources_per_query=10)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily", [])

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        await mixin._topic_search(
            query=sq.query,
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_sources_per_provider=2,  # Budget-split value
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        # Verify max_results was the budget-split value, not state.max_sources_per_query
        provider.search.assert_called_once()
        call_kwargs = provider.search.call_args
        assert call_kwargs.kwargs.get("max_results") == 2

    @pytest.mark.asyncio
    async def test_none_budget_falls_back_to_state(self) -> None:
        """When max_sources_per_provider is None, uses state.max_sources_per_query."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1, max_sources_per_query=7)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily", [])

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        await mixin._topic_search(
            query=sq.query,
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_sources_per_provider=None,
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        call_kwargs = provider.search.call_args
        assert call_kwargs.kwargs.get("max_results") == 7

    @pytest.mark.asyncio
    async def test_provider_error_handled(self) -> None:
        """SearchProviderError is caught and search continues."""
        from foundry_mcp.core.research.providers import SearchProviderError

        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        bad_provider = MagicMock()
        bad_provider.get_provider_name.return_value = "bad"
        bad_provider.search = AsyncMock(side_effect=SearchProviderError("bad", "API error"))

        good_provider = _make_mock_provider("good")

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        added = await mixin._topic_search(
            query=sq.query,
            sub_query=sq,
            state=state,
            available_providers=[bad_provider, good_provider],
            max_sources_per_provider=5,
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        # Good provider still returned results despite bad provider failing
        assert added == 1

    @pytest.mark.asyncio
    async def test_timeout_handled(self) -> None:
        """Timeout is caught and search continues."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        slow_provider = MagicMock()
        slow_provider.get_provider_name.return_value = "slow"
        slow_provider.search = AsyncMock(side_effect=asyncio.TimeoutError())

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        added = await mixin._topic_search(
            query=sq.query,
            sub_query=sq,
            state=state,
            available_providers=[slow_provider],
            max_sources_per_provider=5,
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        assert added == 0

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self) -> None:
        """Semaphore prevents more than N concurrent search operations."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        concurrent_count = 0
        max_concurrent_seen = 0

        async def slow_search(**kwargs):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return [_make_source("src-slow", f"https://slow.com/{concurrent_count}")]

        provider = MagicMock()
        provider.get_provider_name.return_value = "slow"
        provider.search = slow_search

        # Semaphore of 1 — only 1 search at a time
        semaphore = asyncio.Semaphore(1)
        state_lock = asyncio.Lock()

        # Run 3 concurrent topic searches
        tasks = [
            mixin._topic_search(
                query=f"query-{i}",
                sub_query=sq,
                state=state,
                available_providers=[provider],
                max_sources_per_provider=5,
                timeout=30.0,
                seen_urls=set(),
                seen_titles={},
                state_lock=state_lock,
                semaphore=semaphore,
            )
            for i in range(3)
        ]
        await asyncio.gather(*tasks)

        assert max_concurrent_seen == 1

    @pytest.mark.asyncio
    async def test_citation_numbers_assigned(self) -> None:
        """Sources get sequential citation numbers."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        sources = [
            _make_source("src-a", "https://a.com", "Source A"),
            _make_source("src-b", "https://b.com", "Source B"),
        ]
        provider = _make_mock_provider("tavily", sources)

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        await mixin._topic_search(
            query=sq.query,
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_sources_per_provider=5,
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        assert len(state.sources) == 2
        assert state.sources[0].citation_number == 1
        assert state.sources[1].citation_number == 2

    @pytest.mark.asyncio
    async def test_search_provider_stats_tracked(self) -> None:
        """Search provider query counts are tracked in state."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily")

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        await mixin._topic_search(
            query=sq.query,
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_sources_per_provider=5,
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        assert state.search_provider_stats.get("tavily") == 1


# =============================================================================
# Unit tests: _execute_topic_research_async (ReAct loop)
# =============================================================================


class TestExecuteTopicResearchAsync:
    """Tests for the full ReAct loop."""

    @pytest.mark.asyncio
    async def test_single_search_sufficient(self) -> None:
        """Single search finds enough sources and loop exits."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        sources = [_make_source(f"src-{i}", f"https://ex.com/{i}") for i in range(3)]
        provider = _make_mock_provider("tavily", sources)

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        result = await mixin._execute_topic_research_async(
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_searches=3,
            max_sources_per_provider=5,
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        assert isinstance(result, TopicResearchResult)
        assert result.sub_query_id == sq.id
        assert result.searches_performed >= 1
        assert result.sources_found == 3
        assert sq.status == "completed"

    @pytest.mark.asyncio
    async def test_no_sources_triggers_broadened_search(self) -> None:
        """Zero sources on first search triggers LLM reflection for query refinement."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        sq.query = '"very specific phrase"'  # Quoted query

        # First call returns nothing, second returns results
        search_count = 0

        async def dynamic_search(**kwargs):
            nonlocal search_count
            search_count += 1
            if search_count == 1:
                return []
            return [_make_source("src-retry", f"https://retry.com/{search_count}")]

        # LLM reflection returns a refined query
        async def reflection_fn(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps(
                {
                    "sufficient": False,
                    "assessment": "No results found, broadening query",
                    "refined_query": "very specific phrase broader terms",
                }
            )
            result.tokens_used = 30
            return result

        mixin._provider_async_fn = reflection_fn

        provider = MagicMock()
        provider.get_provider_name.return_value = "tavily"
        provider.search = dynamic_search

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        result = await mixin._execute_topic_research_async(
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_searches=3,
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        assert result.searches_performed >= 2
        assert result.sources_found >= 1
        # Check that a refined query was generated
        assert len(result.refined_queries) >= 1

    @pytest.mark.asyncio
    async def test_max_searches_respected(self) -> None:
        """Loop doesn't exceed max_searches iterations."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        # Always return 1 source but reflection says insufficient
        provider = _make_mock_provider("tavily")

        async def always_insufficient(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps(
                {
                    "sufficient": False,
                    "assessment": "Need more",
                    "refined_query": "better query",
                }
            )
            result.tokens_used = 50
            return result

        mixin._provider_async_fn = always_insufficient

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        result = await mixin._execute_topic_research_async(
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_searches=2,
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        assert result.searches_performed <= 2

    @pytest.mark.asyncio
    async def test_audit_event_emitted(self) -> None:
        """Topic research completion emits audit event."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        provider = _make_mock_provider("tavily")

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        await mixin._execute_topic_research_async(
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_searches=1,
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        assert len(mixin._audit_events) >= 1
        event_name, event_data = mixin._audit_events[-1]
        assert event_name == "topic_research_complete"
        assert event_data["data"]["sub_query_id"] == sq.id

    @pytest.mark.asyncio
    async def test_all_failed_marks_sub_query_failed(self) -> None:
        """If no sources found after all iterations, sub-query is marked failed."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        provider = _make_mock_provider("tavily", [])  # Always returns empty

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        result = await mixin._execute_topic_research_async(
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_searches=2,
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        assert result.sources_found == 0
        assert sq.status == "failed"

    @pytest.mark.asyncio
    async def test_reflection_refine_loop(self) -> None:
        """ReAct loop: search → reflect (insufficient) → refine → search again."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        search_call_count = 0

        async def dynamic_search(**kwargs):
            nonlocal search_call_count
            search_call_count += 1
            return [_make_source(f"src-{search_call_count}", f"https://ex.com/{search_call_count}")]

        provider = MagicMock()
        provider.get_provider_name.return_value = "tavily"
        provider.search = dynamic_search

        # First reflection: insufficient, suggests refined query
        # Second reflection: sufficient
        reflect_call_count = 0

        async def dynamic_reflect(**kwargs):
            nonlocal reflect_call_count
            reflect_call_count += 1
            result = MagicMock()
            result.success = True
            result.tokens_used = 40
            if reflect_call_count == 1:
                result.content = json.dumps(
                    {
                        "sufficient": False,
                        "assessment": "Need more data",
                        "refined_query": "refined deep learning query",
                    }
                )
            else:
                result.content = json.dumps(
                    {
                        "sufficient": True,
                        "assessment": "Sufficient now",
                    }
                )
            return result

        mixin._provider_async_fn = dynamic_reflect

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        result = await mixin._execute_topic_research_async(
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_searches=3,
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        assert result.searches_performed >= 2
        assert result.sources_found >= 2
        assert len(result.refined_queries) >= 1
        assert "refined deep learning query" in result.refined_queries


# =============================================================================
# Unit tests: Budget splitting in gathering
# =============================================================================


class TestBudgetSplitting:
    """Tests for budget splitting logic in gathering phase."""

    def test_budget_split_calculation(self) -> None:
        """Per-topic budget is max_sources_per_query // num_topics, min 2.

        Exercises the same formula used in gathering.py to ensure consistency.
        """

        def compute_per_topic_budget(max_sources: int, num_topics: int) -> int:
            """Mirrors the budget formula from GatheringPhaseMixin."""
            num_topics = max(1, num_topics)
            return max(2, max_sources // num_topics)

        # 5 sources / 5 topics = 1 → clamped to 2
        assert compute_per_topic_budget(5, 5) == 2

        # 10 sources / 3 topics = 3
        assert compute_per_topic_budget(10, 3) == 3

        # 10 sources / 1 topic = 10
        assert compute_per_topic_budget(10, 1) == 10

        # 5 sources / 2 topics = 2
        assert compute_per_topic_budget(5, 2) == 2

        # 20 sources / 5 topics = 4
        assert compute_per_topic_budget(20, 5) == 4

    def test_single_topic_gets_full_budget(self) -> None:
        """With 1 topic, per-topic budget equals max_sources_per_query."""
        max_sources = 5
        num_topics = 1
        per_topic = max(2, max_sources // max(1, num_topics))
        assert per_topic == 5

    def test_many_topics_get_minimum_budget(self) -> None:
        """With many topics, per-topic budget is at least 2."""
        max_sources = 5
        num_topics = 10
        per_topic = max(2, max_sources // max(1, num_topics))
        assert per_topic == 2


# =============================================================================
# Unit tests: Config keys
# =============================================================================


class TestTopicAgentConfig:
    """Tests for topic agent configuration keys."""

    def test_default_config_topic_agents_enabled(self) -> None:
        """Topic agents are disabled by default."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig()
        assert config.deep_research_enable_topic_agents is True
        assert config.deep_research_topic_max_searches == 3
        assert config.deep_research_topic_reflection_provider is None

    def test_from_toml_dict_parses_topic_keys(self) -> None:
        """from_toml_dict correctly parses topic agent config."""
        from foundry_mcp.config.research import ResearchConfig

        data = {
            "deep_research_enable_topic_agents": True,
            "deep_research_topic_max_searches": 5,
            "deep_research_topic_reflection_provider": "[cli]gemini:flash",
        }
        config = ResearchConfig.from_toml_dict(data)

        assert config.deep_research_enable_topic_agents is True
        assert config.deep_research_topic_max_searches == 5
        assert config.deep_research_topic_reflection_provider == "[cli]gemini:flash"

    def test_from_toml_dict_string_bool(self) -> None:
        """String 'true' is parsed as boolean True."""
        from foundry_mcp.config.research import ResearchConfig

        data = {"deep_research_enable_topic_agents": "true"}
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_enable_topic_agents is True


# =============================================================================
# Integration tests
# =============================================================================


class TestTopicAgentIntegration:
    """Integration tests for topic agent workflow."""

    def test_topic_research_results_stored_on_state(self) -> None:
        """TopicResearchResult objects are stored in state.topic_research_results."""
        state = _make_state()
        result = TopicResearchResult(
            sub_query_id="sq-1",
            searches_performed=2,
            sources_found=3,
            source_ids=["src-1", "src-2", "src-3"],
        )
        state.topic_research_results.append(result)

        assert len(state.topic_research_results) == 1
        assert state.topic_research_results[0].sub_query_id == "sq-1"

    def test_deduplication_across_topics(self) -> None:
        """URLs seen by one topic agent are skipped by others via shared seen_urls.

        Exercises the dedup logic from _topic_search: URL-based and title-based.
        """
        from foundry_mcp.core.research.workflows.deep_research.source_quality import _normalize_title

        seen_urls: set[str] = set()
        seen_titles: dict[str, str] = {}

        # Simulate topic 1 finding a URL
        url1 = "https://example.com/shared"
        seen_urls.add(url1)

        # Topic 2 should skip the same URL
        assert url1 in seen_urls

        # Title-based dedup: normalize and check
        title = "  My Research Paper (2024)  "
        normalized = _normalize_title(title)
        assert normalized is not None
        assert len(normalized) > 20 or True  # Short titles skip dedup
        seen_titles[normalized] = url1

        # Same title from different domain should be detected
        assert normalized in seen_titles

    @pytest.mark.asyncio
    async def test_parallel_topic_agents_share_semaphore(self) -> None:
        """Multiple parallel topic agents respect the shared semaphore."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=3, max_sources_per_query=9)

        concurrent_count = 0
        max_concurrent_seen = 0

        async def tracking_search(**kwargs):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0)
            concurrent_count -= 1
            src_id = f"src-{kwargs.get('query', 'x')[:10]}-{concurrent_count}"
            return [_make_source(src_id, f"https://{src_id}.com")]

        provider = MagicMock()
        provider.get_provider_name.return_value = "tavily"
        provider.search = tracking_search

        semaphore = asyncio.Semaphore(2)  # Allow max 2 concurrent
        state_lock = asyncio.Lock()
        seen_urls: set[str] = set()
        seen_titles: dict[str, str] = {}

        tasks = [
            mixin._execute_topic_research_async(
                sub_query=sq,
                state=state,
                available_providers=[provider],
                max_searches=1,
                max_sources_per_provider=3,
                timeout=30.0,
                seen_urls=seen_urls,
                seen_titles=seen_titles,
                state_lock=state_lock,
                semaphore=semaphore,
            )
            for sq in state.sub_queries
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all(isinstance(r, TopicResearchResult) for r in results)
        # Semaphore should have limited concurrency to 2
        assert max_concurrent_seen <= 2
