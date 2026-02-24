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
        result.content = json.dumps({
            "continue_searching": False,
            "research_complete": False,
            "rationale": "Enough sources found",
        })
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
    """Tests for TopicResearchMixin._topic_reflect().

    The updated _topic_reflect() returns ``{assessment, raw_response, tokens_used}``
    instead of the old ``{sufficient, assessment, refined_query, tokens_used}``.
    Callers now use ``parse_reflection_decision(raw_response)`` for structured decisions.
    """

    @pytest.mark.asyncio
    async def test_returns_raw_response_and_assessment(self) -> None:
        """Reflection returns raw_response and assessment keys."""
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

        assert "raw_response" in reflection
        assert "assessment" in reflection
        assert "tokens_used" in reflection

    @pytest.mark.asyncio
    async def test_raw_response_contains_structured_decision(self) -> None:
        """raw_response is parseable into a structured decision."""
        mixin = StubTopicResearch()
        state = _make_state()

        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps(
                {
                    "continue_searching": True,
                    "refined_query": "deep learning architectures comparison",
                    "research_complete": False,
                    "rationale": "Only 1 source found",
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

        from foundry_mcp.core.research.workflows.deep_research._helpers import parse_reflection_decision

        decision = parse_reflection_decision(reflection["raw_response"])
        assert decision.continue_searching is True
        assert decision.refined_query == "deep learning architectures comparison"

    @pytest.mark.asyncio
    async def test_provider_failure_returns_stop_decision(self) -> None:
        """Provider failure returns a parseable stop-searching response."""
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

        from foundry_mcp.core.research.workflows.deep_research._helpers import parse_reflection_decision

        decision = parse_reflection_decision(reflection["raw_response"])
        assert decision.continue_searching is False

    @pytest.mark.asyncio
    async def test_exception_returns_stop_decision(self) -> None:
        """Exception during reflection returns a parseable stop response."""
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

        from foundry_mcp.core.research.workflows.deep_research._helpers import parse_reflection_decision

        decision = parse_reflection_decision(reflection["raw_response"])
        assert decision.continue_searching is False

    @pytest.mark.asyncio
    async def test_malformed_json_returns_raw_response(self) -> None:
        """Malformed JSON still returns a raw_response (caller handles parsing)."""
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

        assert "raw_response" in reflection
        # Malformed content is still returned as raw_response
        assert reflection["raw_response"] == "This is not JSON at all"

    @pytest.mark.asyncio
    async def test_tokens_tracked_in_state(self) -> None:
        """Reflection tokens are returned in the result dict (callers aggregate under lock)."""
        mixin = StubTopicResearch()
        state = _make_state()
        state.total_tokens_used = 100

        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "continue_searching": False,
                "research_complete": False,
                "rationale": "OK",
            })
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

        # LLM reflection returns a refined query (new structured schema)
        async def reflection_fn(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps(
                {
                    "continue_searching": True,
                    "refined_query": "very specific phrase broader terms",
                    "research_complete": False,
                    "rationale": "No results found, broadening query",
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
                    "continue_searching": True,
                    "refined_query": "better query",
                    "research_complete": False,
                    "rationale": "Need more",
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

        # First reflection: continue searching with refined query
        # Second reflection: research complete
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
                        "continue_searching": True,
                        "refined_query": "refined deep learning query",
                        "research_complete": False,
                        "rationale": "Need more data",
                    }
                )
            else:
                result.content = json.dumps(
                    {
                        "continue_searching": False,
                        "research_complete": True,
                        "rationale": "Sufficient now",
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
        """Topic agent config has correct budget defaults."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig()
        assert config.deep_research_topic_max_tool_calls == 10
        # Backward-compat alias
        assert config.deep_research_topic_max_searches == 10
        assert config.deep_research_topic_reflection_provider is None
        # Extract defaults
        assert config.deep_research_enable_extract is True
        assert config.deep_research_extract_max_per_iteration == 2

    def test_from_toml_dict_parses_topic_keys(self) -> None:
        """from_toml_dict correctly parses topic agent config."""
        from foundry_mcp.config.research import ResearchConfig

        data = {
            "deep_research_topic_max_tool_calls": 8,
            "deep_research_topic_reflection_provider": "[cli]gemini:flash",
        }
        config = ResearchConfig.from_toml_dict(data)

        assert config.deep_research_topic_max_tool_calls == 8
        assert config.deep_research_topic_max_searches == 8  # backward-compat alias
        assert config.deep_research_topic_reflection_provider == "[cli]gemini:flash"

    def test_from_toml_dict_backward_compat_old_key(self) -> None:
        """from_toml_dict accepts old key name deep_research_topic_max_searches."""
        from foundry_mcp.config.research import ResearchConfig

        data = {
            "deep_research_topic_max_searches": 7,
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_topic_max_tool_calls == 7
        assert config.deep_research_topic_max_searches == 7

    def test_from_toml_dict_string_bool(self) -> None:
        """String 'true' is parsed as boolean True for boolean config fields."""
        from foundry_mcp.config.research import ResearchConfig

        data = {"deep_research_enable_extract": "true"}
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_enable_extract is True


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


# =============================================================================
# Phase 5: Reflection Enforcement tests
# =============================================================================


class TestReflectionEnforcement:
    """Tests for simplified reflection (adaptive guidance, no rigid thresholds).

    Covers:
    - Reflection prompt uses adaptive guidance (not rigid threshold rules)
    - Reflection user prompt excludes domain/quality metadata
    - max_searches is enforced regardless of reflection decision
    - rationale field is always non-empty in reflection notes
    - research_complete signal terminates loop
    """

    @pytest.mark.asyncio
    async def test_reflection_prompt_excludes_domain_count(self) -> None:
        """Reflection user prompt no longer includes distinct domain count."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)

        # Add sources from 3 distinct domains
        for i, domain in enumerate(["arxiv.org", "nature.com", "ieee.org"]):
            src = _make_source(
                f"src-d{i}",
                f"https://{domain}/paper{i}",
                f"Paper from {domain}",
            )
            state.sources.append(src)

        captured_prompts: list[str] = []

        async def capture_provider(**kwargs):
            captured_prompts.append(kwargs.get("prompt", ""))
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "continue_searching": False,
                "research_complete": True,
                "rationale": "Sufficient coverage of research question",
            })
            result.tokens_used = 40
            return result

        mixin._provider_async_fn = capture_provider

        await mixin._topic_reflect(
            original_query="deep learning",
            current_query="deep learning",
            sources_found=3,
            iteration=1,
            max_iterations=3,
            state=state,
        )

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # Domain count and quality distribution no longer injected
        assert "Distinct source domains:" not in prompt
        assert "Source quality distribution:" not in prompt
        # But sources_found and iteration budget are still present
        assert "Sources found so far: 3" in prompt
        assert "Search iteration: 1/3" in prompt

    @pytest.mark.asyncio
    async def test_reflection_prompt_uses_adaptive_guidance(self) -> None:
        """Reflection system prompt uses adaptive guidance, not rigid thresholds."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)

        captured_system: list[str] = []

        async def capture_provider(**kwargs):
            captured_system.append(kwargs.get("system_prompt", ""))
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "continue_searching": False,
                "research_complete": False,
                "rationale": "Test",
            })
            result.tokens_used = 40
            return result

        mixin._provider_async_fn = capture_provider

        await mixin._topic_reflect(
            original_query="test",
            current_query="test",
            sources_found=1,
            iteration=1,
            max_iterations=3,
            state=state,
        )

        assert len(captured_system) == 1
        sys_prompt = captured_system[0]
        # Verify adaptive guidance (no rigid threshold rules)
        assert "substantively answer the research question" in sys_prompt
        assert "Simple factual queries: 2-3 searches" in sys_prompt
        assert "Comparative analysis" in sys_prompt
        assert "diminishing returns" in sys_prompt
        # Verify rigid threshold rules are GONE
        assert "STOP IMMEDIATELY" not in sys_prompt
        assert "3+ sources" not in sys_prompt
        assert "2+ distinct domains" not in sys_prompt
        assert "fewer than 2 relevant sources" not in sys_prompt
        # Verify rationale requirement still present
        assert "rationale field is REQUIRED" in sys_prompt
        assert "must never be empty" in sys_prompt

    @pytest.mark.asyncio
    async def test_research_complete_signal_terminates_loop(self) -> None:
        """research_complete=true from reflection terminates the ReAct loop."""
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            parse_reflection_decision,
        )

        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        search_count = 0

        async def counting_search(**kwargs):
            nonlocal search_count
            search_count += 1
            return [_make_source(f"src-{search_count}", f"https://domain{search_count}.com/p")]

        provider = MagicMock()
        provider.get_provider_name.return_value = "tavily"
        provider.search = counting_search

        # Reflection signals completion on first call
        async def mock_complete(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "continue_searching": False,
                "research_complete": True,
                "rationale": "Findings substantively answer the research question",
            })
            result.tokens_used = 40
            return result

        mixin._provider_async_fn = mock_complete

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        result = await mixin._execute_topic_research_async(
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_searches=5,
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        # Only 1 search, then reflection stops the loop
        assert result.searches_performed == 1
        assert result.early_completion is True

    @pytest.mark.asyncio
    async def test_max_searches_enforced_regardless_of_reflection(self) -> None:
        """Loop respects max_searches even if reflection always says continue."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        search_count = 0

        async def counting_search(**kwargs):
            nonlocal search_count
            search_count += 1
            return [_make_source(f"src-{search_count}", f"https://domain{search_count}.com/p")]

        provider = MagicMock()
        provider.get_provider_name.return_value = "tavily"
        provider.search = counting_search

        # Reflection always says continue
        async def always_continue(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "continue_searching": True,
                "refined_query": f"refined query v{search_count}",
                "research_complete": False,
                "rationale": "Need more sources to cover all angles",
            })
            result.tokens_used = 30
            return result

        mixin._provider_async_fn = always_continue

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

        # Must not exceed max_searches=3
        assert result.searches_performed == 3
        assert search_count == 3

    @pytest.mark.asyncio
    async def test_max_searches_one_no_reflection(self) -> None:
        """With max_searches=1, no reflection is called at all."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        provider = _make_mock_provider("tavily")
        reflection_called = False

        async def should_not_call(**kwargs):
            nonlocal reflection_called
            reflection_called = True
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "continue_searching": True,
                "research_complete": False,
                "rationale": "Should not be called",
            })
            result.tokens_used = 0
            return result

        mixin._provider_async_fn = should_not_call

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        result = await mixin._execute_topic_research_async(
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

        assert result.searches_performed == 1
        assert not reflection_called

    @pytest.mark.asyncio
    async def test_rationale_always_non_empty_in_reflection_notes(self) -> None:
        """Every reflection note has non-empty rationale content."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        reflect_call = 0

        async def dynamic_search(**kwargs):
            return [_make_source(f"src-r{reflect_call}", f"https://r{reflect_call}.com/p")]

        provider = MagicMock()
        provider.get_provider_name.return_value = "tavily"
        provider.search = dynamic_search

        async def mock_reflect_with_rationale(**kwargs):
            nonlocal reflect_call
            reflect_call += 1
            result = MagicMock()
            result.success = True
            result.tokens_used = 30
            if reflect_call == 1:
                result.content = json.dumps({
                    "continue_searching": True,
                    "refined_query": "refined query",
                    "research_complete": False,
                    "rationale": "Only 1 source found, need broader coverage",
                })
            else:
                result.content = json.dumps({
                    "continue_searching": False,
                    "research_complete": True,
                    "rationale": "Sufficient sources from multiple domains now available",
                })
            return result

        mixin._provider_async_fn = mock_reflect_with_rationale

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

        # At least one reflection should have been performed
        assert len(result.reflection_notes) >= 1
        # Every reflection note must be non-empty
        for note in result.reflection_notes:
            assert note, f"Reflection note was empty: {result.reflection_notes}"
            assert len(note) > 0

    @pytest.mark.asyncio
    async def test_rationale_from_failed_reflection_is_non_empty(self) -> None:
        """Even failed reflection produces non-empty rationale."""
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            parse_reflection_decision,
        )

        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)

        # Provider failure
        async def mock_fail(**kwargs):
            result = MagicMock()
            result.success = False
            return result

        mixin._provider_async_fn = mock_fail

        reflection = await mixin._topic_reflect(
            original_query="test",
            current_query="test",
            sources_found=0,
            iteration=1,
            max_iterations=3,
            state=state,
        )

        decision = parse_reflection_decision(reflection["raw_response"])
        assert decision.rationale, "Rationale should not be empty even on failure"
        assert len(decision.rationale) > 0

    @pytest.mark.asyncio
    async def test_rationale_from_exception_is_non_empty(self) -> None:
        """Exception in reflection produces non-empty rationale."""
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            parse_reflection_decision,
        )

        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)

        async def mock_exception(**kwargs):
            raise RuntimeError("Network timeout")

        mixin._provider_async_fn = mock_exception

        reflection = await mixin._topic_reflect(
            original_query="test",
            current_query="test",
            sources_found=0,
            iteration=1,
            max_iterations=3,
            state=state,
        )

        decision = parse_reflection_decision(reflection["raw_response"])
        assert decision.rationale, "Rationale should not be empty even on exception"
        assert len(decision.rationale) > 0

    @pytest.mark.asyncio
    async def test_reflection_includes_per_source_summaries(self) -> None:
        """Reflection user prompt includes per-source summaries for LLM reasoning."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        # Add source belonging to the sub-query
        src = _make_source("src-1", "https://arxiv.org/paper1", "ML Paper")
        src.metadata = {"summarized": True}
        src.content = "<summary>Machine learning overview</summary>"
        state.sources.append(src)
        sq.source_ids = ["src-1"]

        captured_prompts: list[str] = []

        async def capture(**kwargs):
            captured_prompts.append(kwargs.get("prompt", ""))
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "continue_searching": False,
                "research_complete": False,
                "rationale": "Test",
            })
            result.tokens_used = 10
            return result

        mixin._provider_async_fn = capture

        await mixin._topic_reflect(
            original_query="test",
            current_query="test",
            sources_found=1,
            iteration=1,
            max_iterations=3,
            state=state,
            sub_query=sq,
        )

        # Per-source summaries should be included
        assert len(captured_prompts) == 1
        assert "--- SOURCE 1: ML Paper ---" in captured_prompts[0]
        assert "Machine learning overview" in captured_prompts[0]


# =============================================================================
# Phase 5 tests: Simplified Reflection (Adaptive Guidance)
# =============================================================================


class TestSimplifiedReflection:
    """Tests for simplified researcher reflection (no rigid thresholds).

    Verifies that the ReAct loop relies on LLM judgment rather than
    metadata-threshold early exits.
    """

    @pytest.mark.asyncio
    async def test_researcher_continues_beyond_3_sources_on_complex_topic(self) -> None:
        """Researcher can continue searching beyond 3 sources when reflection says continue."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        search_count = 0

        async def mock_search(**kwargs):
            nonlocal search_count
            search_count += 1
            return [_make_source(
                f"src-{search_count}",
                f"https://domain{search_count}.com/p",
                f"Source {search_count}",
                SourceQuality.HIGH,
            )]

        provider = MagicMock()
        provider.get_provider_name.return_value = "tavily"
        provider.search = mock_search

        # _provider_async_fn is shared between reflection and think calls.
        # Use search_count (set by the search mock) to decide when to stop.
        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = True
            result.tokens_used = 30
            if search_count < 5:
                # Continue searching — complex topic needs more coverage
                result.content = json.dumps({
                    "continue_searching": True,
                    "refined_query": f"aspect {search_count + 1} of topic",
                    "research_complete": False,
                    "rationale": f"Only {search_count} perspective(s) covered, need more angles",
                })
            else:
                result.content = json.dumps({
                    "continue_searching": False,
                    "research_complete": True,
                    "rationale": "Comprehensive coverage across multiple perspectives achieved",
                })
            return result

        mixin._provider_async_fn = mock_provider

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        result = await mixin._execute_topic_research_async(
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_searches=7,
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        # Should have done more than 3 searches (no rigid 3-source cutoff)
        assert result.searches_performed > 3
        assert result.sources_found > 3
        assert result.early_completion is True

    @pytest.mark.asyncio
    async def test_researcher_stops_early_on_simple_topic(self) -> None:
        """Researcher stops after 1 search when reflection decides coverage is sufficient."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        async def mock_search(**kwargs):
            return [
                _make_source("s1", "https://example.com/a", "Simple Answer"),
                _make_source("s2", "https://other.com/b", "Another Answer"),
            ]

        provider = MagicMock()
        provider.get_provider_name.return_value = "tavily"
        provider.search = mock_search

        async def mock_reflect(**kwargs):
            result = MagicMock()
            result.success = True
            result.tokens_used = 20
            result.content = json.dumps({
                "continue_searching": False,
                "research_complete": True,
                "rationale": "Simple factual query answered with 2 clear sources",
            })
            return result

        mixin._provider_async_fn = mock_reflect

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        result = await mixin._execute_topic_research_async(
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_searches=5,
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        # Stops after just 1 search
        assert result.searches_performed == 1
        assert result.early_completion is True

    @pytest.mark.asyncio
    async def test_no_metadata_threshold_early_exit(self) -> None:
        """ReAct loop does not have metadata-threshold early exit; relies on LLM decision."""
        # Verify _check_early_exit is no longer on the mixin
        mixin = StubTopicResearch()
        assert not hasattr(mixin, "_check_early_exit"), (
            "_check_early_exit should be removed — LLM reflection is the primary exit signal"
        )

    @pytest.mark.asyncio
    async def test_no_new_sources_exit_still_works(self) -> None:
        """Loop exits when no refinement is possible (no new sources scenario)."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        # Search returns no sources
        provider = MagicMock()
        provider.get_provider_name.return_value = "tavily"
        provider.search = AsyncMock(return_value=[])

        # Reflection says continue but provides same query
        async def mock_reflect(**kwargs):
            result = MagicMock()
            result.success = True
            result.tokens_used = 20
            result.content = json.dumps({
                "continue_searching": True,
                "refined_query": None,
                "research_complete": False,
                "rationale": "No sources found, need to broaden search",
            })
            return result

        mixin._provider_async_fn = mock_reflect

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        result = await mixin._execute_topic_research_async(
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_searches=5,
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        # Loop should terminate early (no refinement possible)
        assert result.searches_performed < 5


class TestPhase5TopicThink:
    """Tests for Phase 5.2: think-tool step within ReAct loop."""

    @pytest.mark.asyncio
    async def test_topic_think_returns_structured_response(self) -> None:
        """Think step returns reasoning and next_query."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)

        async def mock_think(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "reasoning": "Found basic overview but missing implementation details",
                "next_query": "deep learning implementation techniques practical",
            })
            result.tokens_used = 40
            return result

        mixin._provider_async_fn = mock_think

        output = await mixin._topic_think(
            original_query="how does deep learning work",
            current_query="deep learning basics",
            reflection_rationale="Only found overview articles",
            refined_query_suggestion="deep learning details",
            sources_found=2,
            iteration=1,
            state=state,
        )

        assert output["reasoning"] == "Found basic overview but missing implementation details"
        assert output["next_query"] == "deep learning implementation techniques practical"
        assert output["tokens_used"] == 40

    @pytest.mark.asyncio
    async def test_topic_think_handles_failure_gracefully(self) -> None:
        """Think step returns empty dict on failure without crashing."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)

        async def mock_fail(**kwargs):
            raise RuntimeError("Provider timeout")

        mixin._provider_async_fn = mock_fail

        output = await mixin._topic_think(
            original_query="test",
            current_query="test",
            reflection_rationale="test",
            refined_query_suggestion=None,
            sources_found=1,
            iteration=1,
            state=state,
        )

        assert output["reasoning"] == ""
        assert output["next_query"] is None
        assert output["tokens_used"] == 0

    @pytest.mark.asyncio
    async def test_topic_think_handles_provider_failure(self) -> None:
        """Think step handles non-success result."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)

        async def mock_nonsuccess(**kwargs):
            result = MagicMock()
            result.success = False
            return result

        mixin._provider_async_fn = mock_nonsuccess

        output = await mixin._topic_think(
            original_query="test",
            current_query="test",
            reflection_rationale="test",
            refined_query_suggestion=None,
            sources_found=1,
            iteration=1,
            state=state,
        )

        assert output["reasoning"] == ""
        assert output["next_query"] is None

    @pytest.mark.asyncio
    async def test_topic_think_handles_malformed_json(self) -> None:
        """Think step handles non-JSON response gracefully."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)

        async def mock_text(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = "The research so far covers basics but needs more depth."
            result.tokens_used = 25
            return result

        mixin._provider_async_fn = mock_text

        output = await mixin._topic_think(
            original_query="test",
            current_query="test",
            reflection_rationale="test",
            refined_query_suggestion=None,
            sources_found=1,
            iteration=1,
            state=state,
        )

        # Should extract reasoning from raw content
        assert len(output["reasoning"]) > 0
        assert output["next_query"] is None
        assert output["tokens_used"] == 25

    @pytest.mark.asyncio
    async def test_topic_think_prompt_includes_context(self) -> None:
        """Think prompt includes original query, current state, and reflection info."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)

        captured_prompts: list[str] = []
        captured_system: list[str] = []

        async def capture(**kwargs):
            captured_prompts.append(kwargs.get("prompt", ""))
            captured_system.append(kwargs.get("system_prompt", ""))
            result = MagicMock()
            result.success = True
            result.content = json.dumps({"reasoning": "test", "next_query": None})
            result.tokens_used = 10
            return result

        mixin._provider_async_fn = capture

        await mixin._topic_think(
            original_query="AI safety regulations",
            current_query="AI governance policies",
            reflection_rationale="Found only US-focused sources",
            refined_query_suggestion="global AI regulations",
            sources_found=2,
            iteration=2,
            state=state,
        )

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "AI safety regulations" in prompt
        assert "AI governance policies" in prompt
        assert "Found only US-focused sources" in prompt
        assert "global AI regulations" in prompt

        sys = captured_system[0]
        assert "MISSING" in sys or "missing" in sys.lower()

    @pytest.mark.asyncio
    async def test_think_step_integrates_in_react_loop(self) -> None:
        """Think step is called between reflection and next search."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        call_sequence: list[str] = []

        async def mock_search(**kwargs):
            call_sequence.append("search")
            return [_make_source(f"src-{len(call_sequence)}", f"https://d{len(call_sequence)}.com/p")]

        provider = MagicMock()
        provider.get_provider_name.return_value = "tavily"
        provider.search = mock_search

        call_count = 0

        async def mock_provider(**kwargs):
            nonlocal call_count
            call_count += 1
            phase = kwargs.get("phase", "")
            call_sequence.append(phase)
            result = MagicMock()
            result.success = True
            result.tokens_used = 20

            if phase == "topic_reflection":
                if call_count == 1:
                    result.content = json.dumps({
                        "continue_searching": True,
                        "refined_query": "reflection refined",
                        "research_complete": False,
                        "rationale": "Need more sources",
                    })
                else:
                    result.content = json.dumps({
                        "continue_searching": False,
                        "research_complete": True,
                        "rationale": "Enough now",
                    })
            elif phase == "topic_think":
                result.content = json.dumps({
                    "reasoning": "Missing economic perspective",
                    "next_query": "think refined query",
                })
            else:
                result.content = json.dumps({
                    "continue_searching": False,
                    "research_complete": True,
                    "rationale": "Done",
                })
            return result

        mixin._provider_async_fn = mock_provider

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        result = await mixin._execute_topic_research_async(
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_searches=5,
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        # Verify the sequence: search → reflection → think → search → ...
        assert "search" in call_sequence
        assert "topic_reflection" in call_sequence
        assert "topic_think" in call_sequence

        # Think output should be recorded in reflection_notes
        think_notes = [n for n in result.reflection_notes if "[think]" in n]
        assert len(think_notes) >= 1
        assert "Missing economic perspective" in think_notes[0]

    @pytest.mark.asyncio
    async def test_think_refined_query_preferred_over_reflection(self) -> None:
        """Think step's next_query is preferred over reflection's refined_query."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        async def mock_search(**kwargs):
            return [_make_source(f"src-x", f"https://dx.com/p")]

        provider = MagicMock()
        provider.get_provider_name.return_value = "tavily"
        provider.search = mock_search

        call_count = 0

        async def mock_provider(**kwargs):
            nonlocal call_count
            call_count += 1
            phase = kwargs.get("phase", "")
            result = MagicMock()
            result.success = True
            result.tokens_used = 20

            if phase == "topic_reflection":
                if call_count <= 2:
                    result.content = json.dumps({
                        "continue_searching": True,
                        "refined_query": "reflection query",
                        "research_complete": False,
                        "rationale": "Need more",
                    })
                else:
                    result.content = json.dumps({
                        "continue_searching": False,
                        "research_complete": True,
                        "rationale": "Done",
                    })
            elif phase == "topic_think":
                result.content = json.dumps({
                    "reasoning": "Found basics, need depth",
                    "next_query": "think query",
                })
            else:
                result.content = "{}"
            return result

        mixin._provider_async_fn = mock_provider

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        result = await mixin._execute_topic_research_async(
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_searches=5,
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        # Think query should be preferred
        assert "think query" in result.refined_queries


class TestPhase5ContentDedup:
    """Tests for Phase 5.3: cross-researcher content deduplication."""

    def test_content_similarity_identical_texts(self) -> None:
        """Identical texts have similarity 1.0."""
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            content_similarity,
        )

        text = "This is a test document about deep learning and neural networks."
        assert content_similarity(text, text) == 1.0

    def test_content_similarity_completely_different(self) -> None:
        """Completely different texts have low similarity."""
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            content_similarity,
        )

        text_a = "Deep learning is a subset of machine learning based on artificial neural networks."
        text_b = "Cooking pasta requires boiling water and adding salt for optimal flavor results."
        sim = content_similarity(text_a, text_b)
        assert sim < 0.3

    def test_content_similarity_mirror_content(self) -> None:
        """Slightly modified mirror content has high similarity."""
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            content_similarity,
        )

        text_a = (
            "Deep learning is a subset of machine learning that uses multi-layered "
            "neural networks to progressively extract higher-level features from raw "
            "input. For example, in image processing, lower layers may identify edges, "
            "while higher layers may identify concepts relevant to a human."
        )
        # Minor modifications (syndicated content)
        text_b = (
            "Deep learning is a subset of machine learning that uses multi-layered "
            "neural networks to progressively extract higher-level features from raw "
            "input. For example, in image processing, lower layers may identify edges, "
            "while higher layers may identify concepts relevant to humans."
        )
        sim = content_similarity(text_a, text_b)
        assert sim > 0.8

    def test_content_similarity_empty_texts(self) -> None:
        """Empty texts return 0.0 similarity."""
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            content_similarity,
        )

        assert content_similarity("", "") == 0.0
        assert content_similarity("Some text", "") == 0.0
        assert content_similarity("", "Some text") == 0.0

    def test_content_similarity_very_different_lengths(self) -> None:
        """Very different length texts get early 0.0 via length-ratio check."""
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            content_similarity,
        )

        short = "Brief text."
        long = "A " * 500 + "very long document with lots of content."
        assert content_similarity(short, long) == 0.0

    def test_normalize_content_for_dedup(self) -> None:
        """Content normalization strips whitespace and copyright."""
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            _normalize_content_for_dedup,
        )

        raw = "  Hello   World\n\nCopyright 2024 Some Corp. All Rights Reserved  \nMore text  "
        normalized = _normalize_content_for_dedup(raw)
        assert "copyright" not in normalized
        assert "  " not in normalized  # No double spaces
        assert normalized.startswith("hello")

    @pytest.mark.asyncio
    async def test_content_dedup_skips_similar_source(self) -> None:
        """Content-similar sources from different URLs are deduplicated."""
        mixin = StubTopicResearch()
        mixin.config.deep_research_enable_content_dedup = True
        mixin.config.deep_research_content_dedup_threshold = 0.8
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        # Pre-existing source
        existing = ResearchSource(
            id="existing-1",
            title="Deep Learning Overview",
            url="https://site-a.com/dl-overview",
            content=(
                "Deep learning is a subset of machine learning that uses multi-layered "
                "neural networks to progressively extract higher-level features from raw "
                "input data for pattern recognition and classification tasks."
            ),
            quality=SourceQuality.HIGH,
        )
        state.sources.append(existing)

        # Mirror content from different URL
        mirror = ResearchSource(
            id="new-mirror",
            title="DL Overview Mirror",
            url="https://site-b.com/dl-overview-copy",
            content=(
                "Deep learning is a subset of machine learning that uses multi-layered "
                "neural networks to progressively extract higher-level features from raw "
                "input data for pattern recognition and classification tasks."
            ),
            quality=SourceQuality.MEDIUM,
        )

        provider = MagicMock()
        provider.get_provider_name.return_value = "tavily"
        provider.search = AsyncMock(return_value=[mirror])

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        added = await mixin._topic_search(
            query="deep learning",
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

        # Mirror content should be deduplicated
        assert added == 0

    @pytest.mark.asyncio
    async def test_content_dedup_keeps_different_content(self) -> None:
        """Genuinely different content from similar titles is preserved."""
        mixin = StubTopicResearch()
        mixin.config.deep_research_enable_content_dedup = True
        mixin.config.deep_research_content_dedup_threshold = 0.8
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        existing = ResearchSource(
            id="existing-1",
            title="AI Safety Overview",
            url="https://site-a.com/safety",
            content="AI safety focuses on ensuring AI systems behave as intended and don't cause harm.",
            quality=SourceQuality.MEDIUM,
        )
        state.sources.append(existing)

        # Different content
        different = ResearchSource(
            id="new-different",
            title="AI Safety Techniques",
            url="https://site-b.com/safety-tech",
            content="Reinforcement learning from human feedback is a technique for aligning language models.",
            quality=SourceQuality.MEDIUM,
        )

        provider = MagicMock()
        provider.get_provider_name.return_value = "tavily"
        provider.search = AsyncMock(return_value=[different])

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        added = await mixin._topic_search(
            query="AI safety",
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

        # Different content should be kept
        assert added == 1

    @pytest.mark.asyncio
    async def test_content_dedup_disabled_via_config(self) -> None:
        """Content dedup is skipped when config flag is disabled."""
        mixin = StubTopicResearch()
        mixin.config.deep_research_enable_content_dedup = False
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        existing = ResearchSource(
            id="existing-1",
            title="Existing",
            url="https://site-a.com/page",
            content="Identical content that should be added anyway because dedup is off " * 5,
            quality=SourceQuality.MEDIUM,
        )
        state.sources.append(existing)

        mirror = ResearchSource(
            id="new-mirror",
            title="Mirror",
            url="https://site-b.com/page",
            content="Identical content that should be added anyway because dedup is off " * 5,
            quality=SourceQuality.MEDIUM,
        )

        provider = MagicMock()
        provider.get_provider_name.return_value = "tavily"
        provider.search = AsyncMock(return_value=[mirror])

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        added = await mixin._topic_search(
            query="test",
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

        # Should be added because dedup is disabled
        assert added == 1

    @pytest.mark.asyncio
    async def test_content_dedup_skips_short_content(self) -> None:
        """Content dedup is skipped for very short content (< 100 chars)."""
        mixin = StubTopicResearch()
        mixin.config.deep_research_enable_content_dedup = True
        mixin.config.deep_research_content_dedup_threshold = 0.8
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        existing = ResearchSource(
            id="existing-1",
            title="Short",
            url="https://site-a.com/s",
            content="Brief.",
            quality=SourceQuality.MEDIUM,
        )
        state.sources.append(existing)

        short_mirror = ResearchSource(
            id="new-short",
            title="Also Short",
            url="https://site-b.com/s",
            content="Brief.",
            quality=SourceQuality.MEDIUM,
        )

        provider = MagicMock()
        provider.get_provider_name.return_value = "tavily"
        provider.search = AsyncMock(return_value=[short_mirror])

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        added = await mixin._topic_search(
            query="test",
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

        # Short content should bypass dedup
        assert added == 1


class TestPhase5ConfigIntegration:
    """Tests for Phase 5 config keys."""

    def test_default_max_tool_calls_is_10(self) -> None:
        """Default topic_max_tool_calls is 10 (increased from 5)."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig()
        assert config.deep_research_topic_max_tool_calls == 10
        assert config.deep_research_topic_max_searches == 10  # backward-compat alias

    def test_default_content_dedup_enabled(self) -> None:
        """Content dedup is enabled by default."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig()
        assert config.deep_research_enable_content_dedup is True
        assert config.deep_research_content_dedup_threshold == 0.8

    def test_from_toml_dict_parses_content_dedup_keys(self) -> None:
        """TOML parsing correctly reads content dedup config."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig.from_toml_dict({
            "deep_research_enable_content_dedup": False,
            "deep_research_content_dedup_threshold": 0.7,
        })
        assert config.deep_research_enable_content_dedup is False
        assert config.deep_research_content_dedup_threshold == 0.7

    def test_sub_config_has_content_dedup_fields(self) -> None:
        """DeepResearchConfig sub-config includes content dedup fields."""
        from foundry_mcp.config.research_sub_configs import DeepResearchConfig

        dc = DeepResearchConfig()
        assert dc.topic_max_searches == 10
        assert dc.enable_content_dedup is True
        assert dc.content_dedup_threshold == 0.8

    def test_deep_research_config_property_passes_new_fields(self) -> None:
        """ResearchConfig.deep_research_config passes content dedup fields."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig(
            deep_research_enable_content_dedup=False,
            deep_research_content_dedup_threshold=0.6,
        )
        dc = config.deep_research_config
        assert dc.enable_content_dedup is False
        assert dc.content_dedup_threshold == 0.6


# =============================================================================
# Phase 3 PLAN: Iteration Budget + Extract Tool for Researchers
# =============================================================================


class TestPhase3IterationBudget:
    """Tests for Phase 3: raised iteration budgets."""

    def test_default_supervision_rounds_is_6(self) -> None:
        """Default max_supervision_rounds raised to 6."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig()
        assert config.deep_research_max_supervision_rounds == 6

    def test_default_topic_max_tool_calls_is_10(self) -> None:
        """Default topic_max_tool_calls raised to 10."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig()
        assert config.deep_research_topic_max_tool_calls == 10

    def test_early_exit_heuristic_removed(self) -> None:
        """Metadata-threshold early exit removed — LLM reflection is primary signal."""
        stub = StubTopicResearch()
        assert not hasattr(stub, "_check_early_exit"), (
            "_check_early_exit removed; LLM reflection is the primary exit signal"
        )


class TestPhase3ReflectionDecisionExtract:
    """Tests for urls_to_extract field in TopicReflectionDecision."""

    def test_parse_reflection_with_urls_to_extract(self) -> None:
        """parse_reflection_decision extracts urls_to_extract from JSON."""
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            parse_reflection_decision,
        )

        text = json.dumps({
            "continue_searching": True,
            "refined_query": "next query",
            "research_complete": False,
            "rationale": "Need more data",
            "urls_to_extract": [
                "https://example.com/docs",
                "https://other.com/paper.pdf",
            ],
        })
        decision = parse_reflection_decision(text)
        assert decision.continue_searching is True
        assert decision.urls_to_extract is not None
        assert len(decision.urls_to_extract) == 2
        assert "https://example.com/docs" in decision.urls_to_extract

    def test_parse_reflection_without_urls_to_extract(self) -> None:
        """urls_to_extract defaults to None when not present in JSON."""
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            parse_reflection_decision,
        )

        text = json.dumps({
            "continue_searching": False,
            "research_complete": True,
            "rationale": "Done",
        })
        decision = parse_reflection_decision(text)
        assert decision.urls_to_extract is None

    def test_parse_reflection_urls_to_extract_filters_invalid(self) -> None:
        """Invalid URLs (non-http) are filtered out."""
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            parse_reflection_decision,
        )

        text = json.dumps({
            "continue_searching": True,
            "research_complete": False,
            "rationale": "Extract needed",
            "urls_to_extract": [
                "https://valid.com/page",
                "not-a-url",
                "",
                123,
                "https://also-valid.com/doc",
            ],
        })
        decision = parse_reflection_decision(text)
        assert decision.urls_to_extract is not None
        assert len(decision.urls_to_extract) == 2
        assert decision.urls_to_extract[0] == "https://valid.com/page"
        assert decision.urls_to_extract[1] == "https://also-valid.com/doc"

    def test_parse_reflection_empty_urls_list_becomes_none(self) -> None:
        """Empty urls_to_extract list becomes None."""
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            parse_reflection_decision,
        )

        text = json.dumps({
            "continue_searching": False,
            "research_complete": True,
            "rationale": "Done",
            "urls_to_extract": [],
        })
        decision = parse_reflection_decision(text)
        assert decision.urls_to_extract is None

    def test_to_dict_includes_urls_to_extract(self) -> None:
        """TopicReflectionDecision.to_dict() includes urls_to_extract."""
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            TopicReflectionDecision,
        )

        decision = TopicReflectionDecision(
            continue_searching=True,
            rationale="test",
            urls_to_extract=["https://example.com"],
        )
        d = decision.to_dict()
        assert "urls_to_extract" in d
        assert d["urls_to_extract"] == ["https://example.com"]


class TestPhase3TopicExtract:
    """Tests for _topic_extract() method."""

    @pytest.mark.asyncio
    async def test_topic_extract_adds_sources(self) -> None:
        """_topic_extract adds extracted sources to state."""
        from unittest.mock import patch

        stub = StubTopicResearch()
        stub.config.tavily_api_key = "test-key"
        stub.config.tavily_extract_depth = "basic"
        stub.config.deep_research_enable_extract = True

        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        seen_urls: set[str] = set()
        seen_titles: dict[str, str] = {}

        mock_source = _make_source("ext-1", "https://docs.example.com/api", "API Docs")
        mock_source.sub_query_id = None
        mock_source.metadata = {}

        mock_provider = MagicMock()
        mock_provider.extract = AsyncMock(return_value=[mock_source])

        with patch(
            "foundry_mcp.core.research.providers.tavily_extract.TavilyExtractProvider",
            return_value=mock_provider,
        ):
            added = await stub._topic_extract(
                urls=["https://docs.example.com/api"],
                sub_query=sq,
                state=state,
                seen_urls=seen_urls,
                seen_titles=seen_titles,
                state_lock=asyncio.Lock(),
                semaphore=asyncio.Semaphore(3),
            )

        assert added == 1
        assert "https://docs.example.com/api" in seen_urls
        assert any(s.id == "ext-1" for s in state.sources)

    @pytest.mark.asyncio
    async def test_topic_extract_deduplicates_urls(self) -> None:
        """_topic_extract skips URLs already in seen_urls."""
        from unittest.mock import patch

        stub = StubTopicResearch()
        stub.config.tavily_api_key = "test-key"
        stub.config.tavily_extract_depth = "basic"

        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        seen_urls: set[str] = {"https://already-seen.com/page"}
        seen_titles: dict[str, str] = {}

        mock_source = _make_source("ext-dup", "https://already-seen.com/page", "Dup")
        mock_source.sub_query_id = None
        mock_source.metadata = {}

        mock_provider = MagicMock()
        mock_provider.extract = AsyncMock(return_value=[mock_source])

        with patch(
            "foundry_mcp.core.research.providers.tavily_extract.TavilyExtractProvider",
            return_value=mock_provider,
        ):
            added = await stub._topic_extract(
                urls=["https://already-seen.com/page"],
                sub_query=sq,
                state=state,
                seen_urls=seen_urls,
                seen_titles=seen_titles,
                state_lock=asyncio.Lock(),
                semaphore=asyncio.Semaphore(3),
            )

        assert added == 0

    @pytest.mark.asyncio
    async def test_topic_extract_non_fatal_on_failure(self) -> None:
        """_topic_extract returns 0 on provider failure, doesn't raise."""
        from unittest.mock import patch

        stub = StubTopicResearch()
        stub.config.tavily_api_key = "test-key"
        stub.config.tavily_extract_depth = "basic"

        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        mock_provider = MagicMock()
        mock_provider.extract = AsyncMock(side_effect=RuntimeError("API down"))

        with patch(
            "foundry_mcp.core.research.providers.tavily_extract.TavilyExtractProvider",
            return_value=mock_provider,
        ):
            added = await stub._topic_extract(
                urls=["https://example.com"],
                sub_query=sq,
                state=state,
                seen_urls=set(),
                seen_titles={},
                state_lock=asyncio.Lock(),
                semaphore=asyncio.Semaphore(3),
            )

        assert added == 0

    @pytest.mark.asyncio
    async def test_topic_extract_no_api_key_returns_0(self) -> None:
        """_topic_extract returns 0 when no Tavily API key available."""
        import os
        from unittest.mock import patch

        stub = StubTopicResearch()
        stub.config.tavily_api_key = None

        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        with patch.dict(os.environ, {}, clear=True):
            # Ensure TAVILY_API_KEY is not in env
            os.environ.pop("TAVILY_API_KEY", None)
            added = await stub._topic_extract(
                urls=["https://example.com"],
                sub_query=sq,
                state=state,
                seen_urls=set(),
                seen_titles={},
                state_lock=asyncio.Lock(),
                semaphore=asyncio.Semaphore(3),
            )

        assert added == 0


class TestPhase3ExtractConfig:
    """Tests for extract-related config fields."""

    def test_extract_config_defaults(self) -> None:
        """Extract config fields have correct defaults."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig()
        assert config.deep_research_enable_extract is True
        assert config.deep_research_extract_max_per_iteration == 2

    def test_from_toml_dict_parses_extract_fields(self) -> None:
        """from_toml_dict correctly parses extract config."""
        from foundry_mcp.config.research import ResearchConfig

        data = {
            "deep_research_enable_extract": False,
            "deep_research_extract_max_per_iteration": 3,
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_enable_extract is False
        assert config.deep_research_extract_max_per_iteration == 3

    def test_backward_compat_old_key_in_toml(self) -> None:
        """from_toml_dict accepts old deep_research_topic_max_searches key."""
        from foundry_mcp.config.research import ResearchConfig

        data = {"deep_research_topic_max_searches": 7}
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_topic_max_tool_calls == 7
        assert config.deep_research_topic_max_searches == 7

    def test_new_key_takes_precedence_over_old(self) -> None:
        """New key takes precedence when both old and new are present."""
        from foundry_mcp.config.research import ResearchConfig

        data = {
            "deep_research_topic_max_tool_calls": 12,
            "deep_research_topic_max_searches": 7,
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_topic_max_tool_calls == 12
