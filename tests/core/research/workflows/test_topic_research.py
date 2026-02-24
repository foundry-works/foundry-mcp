"""Unit and integration tests for Phase 2: Tool-Calling Researchers (ReAct Agent).

Tests cover:
1. TopicResearchResult model — fields, defaults, serialization
2. _execute_topic_research_async() — ReAct tool-calling loop
3. _topic_search() — search with deduplication and budget splitting
4. Tool dispatch — web_search, extract_content, think, research_complete
5. Budget tracking — budget-exempt tools, budget enforcement
6. Message history — accumulates across turns
7. Concurrent researchers — independent histories, semaphore
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
    ResearcherResponse,
    ResearcherToolCall,
    TopicResearchResult,
    parse_researcher_response,
)
from foundry_mcp.core.research.models.sources import (
    ResearchSource,
    SourceQuality,
    SubQuery,
)
from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
    TopicResearchMixin,
    _build_react_user_prompt,
    _build_researcher_system_prompt,
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


def _react_response(*tool_calls: dict) -> str:
    """Build a JSON ReAct response string from tool call dicts."""
    return json.dumps({"tool_calls": list(tool_calls)})


def _web_search_call(query: str = "test query", max_results: int = 5) -> dict:
    return {"tool": "web_search", "arguments": {"query": query, "max_results": max_results}}


def _think_call(reasoning: str = "Assessing findings...") -> dict:
    return {"tool": "think", "arguments": {"reasoning": reasoning}}


def _complete_call(summary: str = "Research is complete.") -> dict:
    return {"tool": "research_complete", "arguments": {"summary": summary}}


def _extract_call(urls: list[str] | None = None) -> dict:
    return {"tool": "extract_content", "arguments": {"urls": urls or ["https://example.com/doc"]}}


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
        self.config.deep_research_enable_extract = False
        self.config.deep_research_extract_max_per_iteration = 2
        self.config.deep_research_inline_compression = False
        self.config.deep_research_enable_content_dedup = True
        self.config.deep_research_content_dedup_threshold = 0.8
        self.config.resolve_model_for_role = MagicMock(return_value=(None, None))
        self.config.get_phase_fallback_providers = MagicMock(return_value=[])
        self.config.deep_research_max_retries = 1
        self.config.deep_research_retry_delay = 1.0
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

    def _get_tavily_search_kwargs(self, state: Any) -> dict[str, Any]:
        return {"search_depth": "basic"}

    def _get_perplexity_search_kwargs(self, state: Any) -> dict[str, Any]:
        return {}

    def _get_semantic_scholar_search_kwargs(self, state: Any) -> dict[str, Any]:
        return {}

    async def _execute_provider_async(self, **kwargs: Any) -> MagicMock:
        """Mock provider async execution for researcher LLM calls."""
        if self._provider_async_fn:
            return await self._provider_async_fn(**kwargs)
        # Default: signal research complete immediately
        result = MagicMock()
        result.success = True
        result.content = _react_response(_complete_call("Default completion"))
        result.tokens_used = 50
        result.error = None
        return result

    async def _compress_single_topic_async(self, **kwargs: Any) -> tuple[int, int, bool]:
        return (0, 0, True)


# =============================================================================
# Unit tests: TopicResearchResult model
# =============================================================================


class TestTopicResearchResult:
    """Tests for TopicResearchResult model."""

    def test_default_values(self) -> None:
        result = TopicResearchResult(sub_query_id="sq-1")
        assert result.sub_query_id == "sq-1"
        assert result.searches_performed == 0
        assert result.sources_found == 0
        assert result.per_topic_summary is None
        assert result.reflection_notes == []
        assert result.refined_queries == []
        assert result.source_ids == []

    def test_field_updates(self) -> None:
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
# Unit tests: parse_researcher_response
# =============================================================================


class TestParseResearcherResponse:
    """Tests for parse_researcher_response()."""

    def test_valid_tool_calls(self) -> None:
        content = json.dumps({
            "tool_calls": [
                {"tool": "web_search", "arguments": {"query": "test"}},
                {"tool": "think", "arguments": {"reasoning": "hmm"}},
            ]
        })
        resp = parse_researcher_response(content)
        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].tool == "web_search"
        assert resp.tool_calls[1].tool == "think"

    def test_empty_content_returns_empty(self) -> None:
        resp = parse_researcher_response("")
        assert resp.tool_calls == []

    def test_malformed_json_returns_empty(self) -> None:
        resp = parse_researcher_response("This is not JSON")
        assert resp.tool_calls == []

    def test_json_in_code_block(self) -> None:
        content = '```json\n{"tool_calls": [{"tool": "research_complete", "arguments": {"summary": "done"}}]}\n```'
        resp = parse_researcher_response(content)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].tool == "research_complete"

    def test_research_complete_parsed(self) -> None:
        content = _react_response(_complete_call("All done"))
        resp = parse_researcher_response(content)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].tool == "research_complete"
        assert resp.tool_calls[0].arguments["summary"] == "All done"


# =============================================================================
# Unit tests: _build_researcher_system_prompt
# =============================================================================


class TestBuildResearcherSystemPrompt:
    """Tests for system prompt construction."""

    def test_budget_in_prompt(self) -> None:
        prompt = _build_researcher_system_prompt(
            budget_total=10, budget_remaining=7, extract_enabled=True, date_str="2026-02-24"
        )
        assert "7 of 10" in prompt
        assert "2026-02-24" in prompt

    def test_extract_disabled_removes_tool(self) -> None:
        prompt = _build_researcher_system_prompt(
            budget_total=5, budget_remaining=5, extract_enabled=False
        )
        assert "### extract_content" not in prompt

    def test_extract_enabled_includes_tool(self) -> None:
        prompt = _build_researcher_system_prompt(
            budget_total=5, budget_remaining=5, extract_enabled=True
        )
        assert "extract_content" in prompt


# =============================================================================
# Unit tests: _build_react_user_prompt
# =============================================================================


class TestBuildReactUserPrompt:
    """Tests for user prompt construction with message history."""

    def test_initial_prompt_has_topic(self) -> None:
        prompt = _build_react_user_prompt(
            topic="deep learning", message_history=[], budget_remaining=5, budget_total=5,
        )
        assert "deep learning" in prompt
        assert "5 of 5" in prompt

    def test_history_included(self) -> None:
        history = [
            {"role": "assistant", "content": '{"tool_calls": []}'},
            {"role": "tool", "tool": "web_search", "content": "Found 3 sources"},
        ]
        prompt = _build_react_user_prompt(
            topic="test", message_history=history, budget_remaining=4, budget_total=5,
        )
        assert "conversation_history" in prompt
        assert "Found 3 sources" in prompt
        assert "4 of 5" in prompt


# =============================================================================
# Unit tests: _topic_search
# =============================================================================


class TestTopicSearch:
    """Tests for TopicResearchMixin._topic_search()."""

    @pytest.mark.asyncio
    async def test_adds_sources_to_state(self) -> None:
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily")

        added = await mixin._topic_search(
            query=sq.query, sub_query=sq, state=state,
            available_providers=[provider], max_sources_per_provider=5,
            timeout=30.0, seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )
        assert added == 1
        assert len(state.sources) == 1

    @pytest.mark.asyncio
    async def test_url_deduplication(self) -> None:
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        source = _make_source("src-dup", "https://example.com/dup")
        provider = _make_mock_provider("tavily", [source])

        added = await mixin._topic_search(
            query=sq.query, sub_query=sq, state=state,
            available_providers=[provider], max_sources_per_provider=5,
            timeout=30.0, seen_urls={"https://example.com/dup"}, seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )
        assert added == 0

    @pytest.mark.asyncio
    async def test_budget_split_max_results(self) -> None:
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1, max_sources_per_query=10)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily", [])

        await mixin._topic_search(
            query=sq.query, sub_query=sq, state=state,
            available_providers=[provider], max_sources_per_provider=2,
            timeout=30.0, seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )
        call_kwargs = provider.search.call_args
        assert call_kwargs.kwargs.get("max_results") == 2

    @pytest.mark.asyncio
    async def test_none_budget_falls_back_to_state(self) -> None:
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1, max_sources_per_query=7)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily", [])

        await mixin._topic_search(
            query=sq.query, sub_query=sq, state=state,
            available_providers=[provider], max_sources_per_provider=None,
            timeout=30.0, seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )
        call_kwargs = provider.search.call_args
        assert call_kwargs.kwargs.get("max_results") == 7

    @pytest.mark.asyncio
    async def test_provider_error_handled(self) -> None:
        from foundry_mcp.core.research.providers import SearchProviderError

        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        bad_provider = MagicMock()
        bad_provider.get_provider_name.return_value = "bad"
        bad_provider.search = AsyncMock(side_effect=SearchProviderError("bad", "API error"))
        good_provider = _make_mock_provider("good")

        added = await mixin._topic_search(
            query=sq.query, sub_query=sq, state=state,
            available_providers=[bad_provider, good_provider],
            max_sources_per_provider=5, timeout=30.0,
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )
        assert added == 1

    @pytest.mark.asyncio
    async def test_timeout_handled(self) -> None:
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        slow_provider = MagicMock()
        slow_provider.get_provider_name.return_value = "slow"
        slow_provider.search = AsyncMock(side_effect=asyncio.TimeoutError())

        added = await mixin._topic_search(
            query=sq.query, sub_query=sq, state=state,
            available_providers=[slow_provider], max_sources_per_provider=5,
            timeout=30.0, seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )
        assert added == 0

    @pytest.mark.asyncio
    async def test_citation_numbers_assigned(self) -> None:
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        sources = [
            _make_source("src-a", "https://a.com", "Source A"),
            _make_source("src-b", "https://b.com", "Source B"),
        ]
        provider = _make_mock_provider("tavily", sources)

        await mixin._topic_search(
            query=sq.query, sub_query=sq, state=state,
            available_providers=[provider], max_sources_per_provider=5,
            timeout=30.0, seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )
        assert len(state.sources) == 2
        assert state.sources[0].citation_number == 1
        assert state.sources[1].citation_number == 2

    @pytest.mark.asyncio
    async def test_search_provider_stats_tracked(self) -> None:
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily")

        await mixin._topic_search(
            query=sq.query, sub_query=sq, state=state,
            available_providers=[provider], max_sources_per_provider=5,
            timeout=30.0, seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )
        assert state.search_provider_stats.get("tavily") == 1


# =============================================================================
# Unit tests: ReAct loop (_execute_topic_research_async)
# =============================================================================


class TestReActLoop:
    """Tests for the full tool-calling ReAct loop."""

    @pytest.mark.asyncio
    async def test_research_complete_terminates_loop(self) -> None:
        """ResearchComplete tool call terminates the loop and records summary."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily")

        call_count = 0

        async def mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.success = True
            result.tokens_used = 50
            result.error = None
            if call_count == 1:
                result.content = _react_response(
                    _web_search_call("deep learning basics")
                )
            else:
                result.content = _react_response(
                    _complete_call("Found comprehensive information on deep learning")
                )
            return result

        mixin._provider_async_fn = mock_llm

        topic_result = await mixin._execute_topic_research_async(
            sub_query=sq, state=state, available_providers=[provider],
            max_searches=5, timeout=30.0,
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )

        assert topic_result.early_completion is True
        assert "comprehensive information" in topic_result.completion_rationale
        assert topic_result.searches_performed == 1
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_web_search_dispatches_to_provider(self) -> None:
        """WebSearch tool call dispatches to configured search providers."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily")

        call_count = 0

        async def mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.success = True
            result.tokens_used = 30
            result.error = None
            if call_count == 1:
                result.content = _react_response(_web_search_call("query A"))
            else:
                result.content = _react_response(_complete_call("Done"))
            return result

        mixin._provider_async_fn = mock_llm

        topic_result = await mixin._execute_topic_research_async(
            sub_query=sq, state=state, available_providers=[provider],
            max_searches=5, timeout=30.0,
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )

        assert topic_result.searches_performed == 1
        assert topic_result.sources_found >= 1
        provider.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_think_tool_logs_and_does_not_count_budget(self) -> None:
        """Think tool records reasoning but does not count against budget."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily")

        call_count = 0

        async def mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.success = True
            result.tokens_used = 30
            result.error = None
            if call_count == 1:
                # Search + think in same turn
                result.content = _react_response(
                    _web_search_call("test"),
                    _think_call("Analyzing initial results..."),
                )
            else:
                result.content = _react_response(_complete_call("Done"))
            return result

        mixin._provider_async_fn = mock_llm

        topic_result = await mixin._execute_topic_research_async(
            sub_query=sq, state=state, available_providers=[provider],
            max_searches=5, timeout=30.0,
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )

        # Think note was recorded
        think_notes = [n for n in topic_result.reflection_notes if "[think]" in n]
        assert len(think_notes) == 1
        assert "Analyzing initial results" in think_notes[0]
        # Only 1 search counted (think is budget-exempt)
        assert topic_result.searches_performed == 1

    @pytest.mark.asyncio
    async def test_budget_exhaustion_stops_loop(self) -> None:
        """Loop stops when tool call budget is exhausted."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily")

        call_count = 0

        async def mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.success = True
            result.tokens_used = 30
            result.error = None
            # Alternate search → think → search → think to follow reflection protocol
            if call_count % 2 == 1:
                result.content = _react_response(
                    _web_search_call(f"query iteration {call_count}")
                )
            else:
                result.content = _react_response(
                    _think_call(f"Reflecting on iteration {call_count}...")
                )
            return result

        mixin._provider_async_fn = mock_llm

        topic_result = await mixin._execute_topic_research_async(
            sub_query=sq, state=state, available_providers=[provider],
            max_searches=2, timeout=30.0,  # Budget of 2
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )

        # Exactly 2 searches should have been performed (budget cap)
        assert topic_result.searches_performed == 2

    @pytest.mark.asyncio
    async def test_no_tool_calls_terminates_gracefully(self) -> None:
        """When LLM returns no tool calls, loop terminates."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily")

        async def mock_llm(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({"tool_calls": []})
            result.tokens_used = 20
            result.error = None
            return result

        mixin._provider_async_fn = mock_llm

        topic_result = await mixin._execute_topic_research_async(
            sub_query=sq, state=state, available_providers=[provider],
            max_searches=5, timeout=30.0,
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )

        assert topic_result.searches_performed == 0
        assert topic_result.early_completion is False

    @pytest.mark.asyncio
    async def test_llm_failure_terminates_loop(self) -> None:
        """LLM call failure terminates the loop."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily")

        async def mock_llm(**kwargs):
            result = MagicMock()
            result.success = False
            result.error = "Provider unavailable"
            return result

        mixin._provider_async_fn = mock_llm

        topic_result = await mixin._execute_topic_research_async(
            sub_query=sq, state=state, available_providers=[provider],
            max_searches=5, timeout=30.0,
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )

        assert topic_result.searches_performed == 0

    @pytest.mark.asyncio
    async def test_think_executed_before_action_tools(self) -> None:
        """When Think and WebSearch are in the same turn, Think runs first."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily")

        execution_order: list[str] = []
        original_handle_think = mixin._handle_think_tool

        def tracking_think(tool_call, sub_query, result):
            execution_order.append("think")
            return original_handle_think(tool_call, sub_query, result)

        mixin._handle_think_tool = tracking_think

        original_search = mixin._topic_search

        async def tracking_search(*args, **kwargs):
            execution_order.append("search")
            return await original_search(*args, **kwargs)

        mixin._topic_search = tracking_search

        call_count = 0

        async def mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.success = True
            result.tokens_used = 30
            result.error = None
            if call_count == 1:
                # Both think and search in same turn (search first in list, but Think should execute first)
                result.content = _react_response(
                    _web_search_call("test"),
                    _think_call("Let me reflect..."),
                )
            else:
                result.content = _react_response(_complete_call("Done"))
            return result

        mixin._provider_async_fn = mock_llm

        await mixin._execute_topic_research_async(
            sub_query=sq, state=state, available_providers=[provider],
            max_searches=5, timeout=30.0,
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )

        # Think should appear before search in execution order
        assert execution_order[0] == "think"
        assert execution_order[1] == "search"

    @pytest.mark.asyncio
    async def test_message_history_accumulates(self) -> None:
        """Message history accumulates across turns."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily")

        captured_prompts: list[str] = []
        call_count = 0

        async def mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            captured_prompts.append(kwargs.get("prompt", ""))
            result = MagicMock()
            result.success = True
            result.tokens_used = 30
            result.error = None
            if call_count == 1:
                result.content = _react_response(_web_search_call("first query"))
            elif call_count == 2:
                result.content = _react_response(_web_search_call("second query"))
            else:
                result.content = _react_response(_complete_call("Done"))
            return result

        mixin._provider_async_fn = mock_llm

        await mixin._execute_topic_research_async(
            sub_query=sq, state=state, available_providers=[provider],
            max_searches=5, timeout=30.0,
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )

        # Second prompt should contain conversation history from first turn
        assert len(captured_prompts) >= 2
        assert "conversation_history" in captured_prompts[1]

    @pytest.mark.asyncio
    async def test_refined_queries_tracked(self) -> None:
        """Queries different from the original are tracked as refined."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily")

        call_count = 0

        async def mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.success = True
            result.tokens_used = 30
            result.error = None
            if call_count == 1:
                result.content = _react_response(
                    _web_search_call("completely different refined query")
                )
            else:
                result.content = _react_response(_complete_call("Done"))
            return result

        mixin._provider_async_fn = mock_llm

        topic_result = await mixin._execute_topic_research_async(
            sub_query=sq, state=state, available_providers=[provider],
            max_searches=5, timeout=30.0,
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )

        assert "completely different refined query" in topic_result.refined_queries

    @pytest.mark.asyncio
    async def test_sub_query_marked_completed_on_sources(self) -> None:
        """SubQuery is marked completed when sources are found."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily")

        call_count = 0

        async def mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.success = True
            result.tokens_used = 30
            result.error = None
            if call_count == 1:
                result.content = _react_response(_web_search_call("test"))
            else:
                result.content = _react_response(_complete_call("Done"))
            return result

        mixin._provider_async_fn = mock_llm

        await mixin._execute_topic_research_async(
            sub_query=sq, state=state, available_providers=[provider],
            max_searches=5, timeout=30.0,
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )

        assert sq.status == "completed"

    @pytest.mark.asyncio
    async def test_sub_query_marked_failed_no_sources(self) -> None:
        """SubQuery is marked failed when no sources found."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily", [])  # Returns no sources

        async def mock_llm(**kwargs):
            result = MagicMock()
            result.success = True
            result.tokens_used = 30
            result.error = None
            result.content = _react_response(_web_search_call("test"))
            return result

        mixin._provider_async_fn = mock_llm

        await mixin._execute_topic_research_async(
            sub_query=sq, state=state, available_providers=[provider],
            max_searches=1, timeout=30.0,
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )

        assert sq.status == "failed"

    @pytest.mark.asyncio
    async def test_audit_event_emitted(self) -> None:
        """Audit event emitted on topic research completion."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily")

        topic_result = await mixin._execute_topic_research_async(
            sub_query=sq, state=state, available_providers=[provider],
            max_searches=5, timeout=30.0,
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )

        assert len(mixin._audit_events) >= 1
        event_name, event_data = mixin._audit_events[-1]
        assert event_name == "topic_research_complete"
        assert event_data["data"]["sub_query_id"] == sq.id

    @pytest.mark.asyncio
    async def test_tokens_merged_to_state(self) -> None:
        """Tokens used by researcher LLM calls are merged to state."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        state.total_tokens_used = 100
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily")

        async def mock_llm(**kwargs):
            result = MagicMock()
            result.success = True
            result.tokens_used = 75
            result.error = None
            result.content = _react_response(_complete_call("Done"))
            return result

        mixin._provider_async_fn = mock_llm

        await mixin._execute_topic_research_async(
            sub_query=sq, state=state, available_providers=[provider],
            max_searches=5, timeout=30.0,
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )

        assert state.total_tokens_used == 175  # 100 + 75


# =============================================================================
# Unit tests: concurrent researchers
# =============================================================================


class TestConcurrentResearchers:
    """Tests for parallel topic researchers with independent state."""

    @pytest.mark.asyncio
    async def test_parallel_researchers_independent(self) -> None:
        """Multiple concurrent researchers have independent results."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=3)

        # Each sub_query gets its own provider returning unique sources
        providers = []
        for i in range(3):
            src = _make_source(f"src-topic-{i}", f"https://topic{i}.com/1", f"Topic {i} result")
            providers.append(_make_mock_provider(f"provider-{i}", [src]))

        call_counts: dict[str, int] = {}

        async def mock_llm(**kwargs):
            prompt = kwargs.get("prompt", "")
            # Determine which sub-query this is for
            for i in range(3):
                if f"sq-{i}" in prompt or f"Sub-query {i}" in prompt:
                    key = f"sq-{i}"
                    call_counts[key] = call_counts.get(key, 0) + 1
                    break

            result = MagicMock()
            result.success = True
            result.tokens_used = 30
            result.error = None
            result.content = _react_response(_complete_call("Done"))
            return result

        mixin._provider_async_fn = mock_llm

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()
        seen_urls: set[str] = set()
        seen_titles: dict[str, str] = {}

        tasks = [
            mixin._execute_topic_research_async(
                sub_query=state.sub_queries[i],
                state=state,
                available_providers=[providers[i]],
                max_searches=3,
                timeout=30.0,
                seen_urls=seen_urls,
                seen_titles=seen_titles,
                state_lock=state_lock,
                semaphore=semaphore,
            )
            for i in range(3)
        ]

        results = await asyncio.gather(*tasks)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, TopicResearchResult)


# =============================================================================
# Unit tests: tool schemas
# =============================================================================


class TestToolSchemas:
    """Tests for researcher tool Pydantic schemas."""

    def test_web_search_tool_validates(self) -> None:
        from foundry_mcp.core.research.models.deep_research import WebSearchTool
        tool = WebSearchTool(query="test query", max_results=3)
        assert tool.query == "test query"
        assert tool.max_results == 3

    def test_extract_content_tool_caps_urls(self) -> None:
        from foundry_mcp.core.research.models.deep_research import ExtractContentTool
        tool = ExtractContentTool(urls=["https://a.com", "https://b.com", "https://c.com"])
        assert len(tool.urls) == 2  # Capped at 2

    def test_think_tool_validates(self) -> None:
        from foundry_mcp.core.research.models.deep_research import ThinkTool
        tool = ThinkTool(reasoning="Analyzing coverage gaps")
        assert tool.reasoning == "Analyzing coverage gaps"

    def test_research_complete_tool_validates(self) -> None:
        from foundry_mcp.core.research.models.deep_research import ResearchCompleteTool
        tool = ResearchCompleteTool(summary="All findings address the question")
        assert tool.summary == "All findings address the question"

    def test_researcher_response_model(self) -> None:
        resp = ResearcherResponse(
            tool_calls=[
                ResearcherToolCall(tool="web_search", arguments={"query": "test"}),
            ],
            reasoning="Starting research",
        )
        assert len(resp.tool_calls) == 1
        assert resp.reasoning == "Starting research"

    def test_budget_exempt_tools(self) -> None:
        from foundry_mcp.core.research.models.deep_research import BUDGET_EXEMPT_TOOLS
        assert "think" in BUDGET_EXEMPT_TOOLS
        assert "research_complete" in BUDGET_EXEMPT_TOOLS
        assert "web_search" not in BUDGET_EXEMPT_TOOLS

    def test_researcher_tool_registry(self) -> None:
        from foundry_mcp.core.research.models.deep_research import RESEARCHER_TOOL_SCHEMAS
        assert "web_search" in RESEARCHER_TOOL_SCHEMAS
        assert "extract_content" in RESEARCHER_TOOL_SCHEMAS
        assert "think" in RESEARCHER_TOOL_SCHEMAS
        assert "research_complete" in RESEARCHER_TOOL_SCHEMAS


# =============================================================================
# Unit tests: forced reflection (Phase 2)
# =============================================================================


class TestForcedReflection:
    """Tests for Phase 2 forced reflection enforcement."""

    def test_prompt_contains_reflection_protocol(self) -> None:
        """System prompt includes the CRITICAL Reflection Protocol section."""
        prompt = _build_researcher_system_prompt(
            budget_total=5, budget_remaining=5, extract_enabled=True
        )
        assert "CRITICAL: Reflection Protocol" in prompt
        assert "search → think → search → think" in prompt
        assert "MUST call think after every web_search" in prompt

    @pytest.mark.asyncio
    async def test_search_think_search_pattern(self) -> None:
        """Researcher alternates search → think → search correctly."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily")

        call_count = 0

        async def mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.success = True
            result.tokens_used = 30
            result.error = None
            if call_count == 1:
                # First turn: search (first search turn, exempt from reflection)
                result.content = _react_response(_web_search_call("query 1"))
            elif call_count == 2:
                # Second turn: think (correct — reflecting after search)
                result.content = _react_response(
                    _think_call("Analyzing first search results...")
                )
            elif call_count == 3:
                # Third turn: search (allowed — think was done)
                result.content = _react_response(_web_search_call("query 2"))
            elif call_count == 4:
                # Fourth turn: think (correct — reflecting after search)
                result.content = _react_response(
                    _think_call("Analyzing second search results...")
                )
            else:
                result.content = _react_response(_complete_call("Done"))
            return result

        mixin._provider_async_fn = mock_llm

        topic_result = await mixin._execute_topic_research_async(
            sub_query=sq, state=state, available_providers=[provider],
            max_searches=5, timeout=30.0,
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )

        assert topic_result.searches_performed == 2
        assert topic_result.early_completion is True
        # No reflections should have been injected — pattern was correct
        audit_data = mixin._audit_events[-1][1]["data"]
        assert audit_data["reflection_injections"] == 0

    @pytest.mark.asyncio
    async def test_first_turn_parallel_searches_allowed(self) -> None:
        """Multiple web_search calls on first turn are allowed (initial broadening)."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        # Provider returning different sources for each call
        call_idx = 0
        original_search = AsyncMock()

        async def multi_source_search(**kwargs):
            nonlocal call_idx
            call_idx += 1
            return [
                _make_source(
                    f"src-broad-{call_idx}",
                    f"https://broad{call_idx}.com",
                    f"Broad result {call_idx}",
                )
            ]

        provider = MagicMock()
        provider.get_provider_name.return_value = "tavily"
        provider.search = multi_source_search

        llm_call_count = 0

        async def mock_llm(**kwargs):
            nonlocal llm_call_count
            llm_call_count += 1
            result = MagicMock()
            result.success = True
            result.tokens_used = 30
            result.error = None
            if llm_call_count == 1:
                # First turn: two parallel searches (initial broadening)
                result.content = _react_response(
                    _web_search_call("broad query A"),
                    _web_search_call("broad query B"),
                )
            elif llm_call_count == 2:
                # Second turn: think (reflecting on broadening)
                result.content = _react_response(
                    _think_call("Got broad coverage from initial searches...")
                )
            else:
                result.content = _react_response(_complete_call("Done"))
            return result

        mixin._provider_async_fn = mock_llm

        topic_result = await mixin._execute_topic_research_async(
            sub_query=sq, state=state, available_providers=[provider],
            max_searches=5, timeout=30.0,
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )

        # Both searches on the first turn should have executed
        assert topic_result.searches_performed == 2
        # No reflection injections — first turn is exempt
        audit_data = mixin._audit_events[-1][1]["data"]
        assert audit_data["reflection_injections"] == 0

    @pytest.mark.asyncio
    async def test_synthetic_reflection_injection(self) -> None:
        """When researcher skips think after search, synthetic reflection is injected."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily")

        llm_call_count = 0

        async def mock_llm(**kwargs):
            nonlocal llm_call_count
            llm_call_count += 1
            result = MagicMock()
            result.success = True
            result.tokens_used = 30
            result.error = None
            if llm_call_count == 1:
                # Turn 1: search (first search turn, exempt)
                result.content = _react_response(_web_search_call("query 1"))
            elif llm_call_count == 2:
                # Turn 2: search without think (violation! should trigger injection)
                result.content = _react_response(_web_search_call("query 2"))
            elif llm_call_count == 3:
                # Turn 3: after injection, model should think
                result.content = _react_response(
                    _think_call("Reflecting after being prompted...")
                )
            elif llm_call_count == 4:
                # Turn 4: now can search
                result.content = _react_response(_web_search_call("query 3"))
            else:
                result.content = _react_response(_complete_call("Done"))
            return result

        mixin._provider_async_fn = mock_llm

        topic_result = await mixin._execute_topic_research_async(
            sub_query=sq, state=state, available_providers=[provider],
            max_searches=5, timeout=30.0,
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )

        # Reflection was injected once (turn 2 skipped think)
        audit_data = mixin._audit_events[-1][1]["data"]
        assert audit_data["reflection_injections"] == 1

        # The injected turn's search was NOT executed (only turns 1, 4 executed)
        # Turn 1: search (executed, 1 search)
        # Turn 2: search (rejected, injection, 0 searches)
        # Turn 3: think (no search)
        # Turn 4: search (executed, 1 search) -- previous turn was think, allowed
        # Turn 5: complete
        assert topic_result.searches_performed == 2

        # Message history should contain the injection
        injection_msgs = [
            m for m in topic_result.message_history
            if m.get("tool") == "system" and "REFLECTION REQUIRED" in m.get("content", "")
        ]
        assert len(injection_msgs) == 1

    @pytest.mark.asyncio
    async def test_reflection_not_needed_when_think_present(self) -> None:
        """When researcher includes think with search, no injection needed."""
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]
        provider = _make_mock_provider("tavily")

        llm_call_count = 0

        async def mock_llm(**kwargs):
            nonlocal llm_call_count
            llm_call_count += 1
            result = MagicMock()
            result.success = True
            result.tokens_used = 30
            result.error = None
            if llm_call_count == 1:
                # Turn 1: search (first turn)
                result.content = _react_response(_web_search_call("query 1"))
            elif llm_call_count == 2:
                # Turn 2: search + think together (think present, no injection needed)
                result.content = _react_response(
                    _web_search_call("query 2"),
                    _think_call("Reflecting on results..."),
                )
            else:
                result.content = _react_response(_complete_call("Done"))
            return result

        mixin._provider_async_fn = mock_llm

        topic_result = await mixin._execute_topic_research_async(
            sub_query=sq, state=state, available_providers=[provider],
            max_searches=5, timeout=30.0,
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )

        # No reflection injections — think was present
        audit_data = mixin._audit_events[-1][1]["data"]
        assert audit_data["reflection_injections"] == 0
        assert topic_result.searches_performed == 2


# =============================================================================
# Phase 5: Researcher Stop Heuristics
# =============================================================================


class TestResearcherStopHeuristics:
    """Tests for explicit stop heuristics in the researcher prompt and think tool."""

    def test_system_prompt_contains_stop_heuristics(self) -> None:
        """5.1: System prompt includes 'Stop Immediately When' block with all three rules."""
        prompt = _build_researcher_system_prompt(
            budget_total=5, budget_remaining=5, extract_enabled=True, date_str="2026-02-24",
        )
        assert "Stop Immediately When" in prompt
        assert "3 or more high-quality" in prompt or "3+" in prompt
        assert "last 2 searches" in prompt
        assert "Comprehensive answer" in prompt or "comprehensively" in prompt

    def test_stop_heuristics_present_with_extract_disabled(self) -> None:
        """Stop heuristics remain when extract_content tool is removed."""
        prompt = _build_researcher_system_prompt(
            budget_total=5, budget_remaining=5, extract_enabled=False, date_str="2026-02-24",
        )
        assert "Stop Immediately When" in prompt
        assert "research_complete" in prompt

    def test_think_tool_response_includes_stop_checklist(self) -> None:
        """5.2: Think tool acknowledgment includes stop-criteria checklist."""
        mixin = StubTopicResearch()
        result = TopicResearchResult(sub_query_id="sq-test")
        sq = SubQuery(id="sq-test", query="test query", rationale="test", priority=1)

        tool_call = ResearcherToolCall(
            tool="think",
            arguments={"reasoning": "Found several good sources on the topic."},
        )
        response_text = mixin._handle_think_tool(
            tool_call=tool_call, sub_query=sq, result=result,
        )

        # The response should include the stop-criteria checklist
        assert "3+ high-quality relevant sources" in response_text
        assert "last 2 searches" in response_text
        assert "research_complete" in response_text

    @pytest.mark.asyncio
    async def test_researcher_completes_early_with_sufficient_sources(self) -> None:
        """5.3: Researcher calls research_complete when 3+ sources found.

        Simulates a researcher that finds 3+ sources from its first search,
        reflects via think, and then correctly calls research_complete.
        """
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        # Provider returns 3 high-quality sources per search
        sources = [
            _make_source(f"src-{i}", f"https://example.com/{i}", f"Source {i}", SourceQuality.HIGH)
            for i in range(3)
        ]
        provider = _make_mock_provider("tavily", sources=sources)

        llm_call_count = 0

        async def mock_llm(**kwargs):
            nonlocal llm_call_count
            llm_call_count += 1
            result = MagicMock()
            result.success = True
            result.tokens_used = 50
            result.error = None
            if llm_call_count == 1:
                # Turn 1: initial search
                result.content = _react_response(_web_search_call("deep learning basics"))
            elif llm_call_count == 2:
                # Turn 2: think — researcher reflects and recognizes 3+ sources
                result.content = _react_response(
                    _think_call("I've found 3 high-quality sources. Stopping criteria met.")
                )
            elif llm_call_count == 3:
                # Turn 3: research_complete (the correct stop behavior)
                result.content = _react_response(
                    _complete_call("Found 3 high-quality sources addressing the question.")
                )
            else:
                # Should not reach here
                result.content = _react_response(_complete_call("Fallback"))
            return result

        mixin._provider_async_fn = mock_llm

        topic_result = await mixin._execute_topic_research_async(
            sub_query=sq, state=state, available_providers=[provider],
            max_searches=10, timeout=30.0,
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )

        # Researcher stopped early after only 1 search (budget of 10)
        assert topic_result.early_completion is True
        assert topic_result.searches_performed == 1
        assert topic_result.sources_found == 3
        assert llm_call_count == 3  # search → think → complete

    @pytest.mark.asyncio
    async def test_researcher_stops_after_overlapping_searches(self) -> None:
        """5.4: Researcher stops after 2 searches returning similar results.

        Simulates a researcher whose second search returns overlapping content.
        After reflecting, the researcher correctly calls research_complete
        due to diminishing returns.
        """
        mixin = StubTopicResearch()
        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        search_count = 0

        def _make_overlapping_provider():
            """Provider that returns similar sources on subsequent calls."""
            prov = MagicMock()
            prov.get_provider_name.return_value = "tavily"

            async def search_fn(**kwargs):
                nonlocal search_count
                search_count += 1
                # Both searches return similar-looking sources (different URLs but
                # the researcher should note the overlap in its think step)
                return [
                    _make_source(
                        f"src-search{search_count}-1",
                        f"https://example.com/search{search_count}/1",
                        f"Deep Learning Overview (Search {search_count})",
                        SourceQuality.MEDIUM,
                    ),
                ]

            prov.search = AsyncMock(side_effect=search_fn)
            return prov

        provider = _make_overlapping_provider()

        llm_call_count = 0

        async def mock_llm(**kwargs):
            nonlocal llm_call_count
            llm_call_count += 1
            result = MagicMock()
            result.success = True
            result.tokens_used = 50
            result.error = None
            if llm_call_count == 1:
                # Turn 1: first search
                result.content = _react_response(_web_search_call("deep learning"))
            elif llm_call_count == 2:
                # Turn 2: think after first search
                result.content = _react_response(
                    _think_call("Found 1 source. Need more information.")
                )
            elif llm_call_count == 3:
                # Turn 3: second search (different query, similar results)
                result.content = _react_response(_web_search_call("deep learning overview"))
            elif llm_call_count == 4:
                # Turn 4: think — researcher notices overlap from last 2 searches
                result.content = _react_response(
                    _think_call(
                        "Last 2 searches returned substantially similar information. "
                        "Diminishing returns detected. Stopping."
                    )
                )
            elif llm_call_count == 5:
                # Turn 5: research_complete — correct stop on diminishing returns
                result.content = _react_response(
                    _complete_call("Stopping due to diminishing returns from searches.")
                )
            else:
                result.content = _react_response(_complete_call("Fallback"))
            return result

        mixin._provider_async_fn = mock_llm

        topic_result = await mixin._execute_topic_research_async(
            sub_query=sq, state=state, available_providers=[provider],
            max_searches=10, timeout=30.0,
            seen_urls=set(), seen_titles={},
            state_lock=asyncio.Lock(), semaphore=asyncio.Semaphore(3),
        )

        # Researcher stopped after 2 searches (budget of 10) due to overlap
        assert topic_result.early_completion is True
        assert topic_result.searches_performed == 2
        assert llm_call_count == 5  # search → think → search → think → complete

        # Reflection notes should contain the diminishing-returns reasoning
        think_notes = [n for n in topic_result.reflection_notes if n.startswith("[think]")]
        assert len(think_notes) == 2
        assert "diminishing returns" in think_notes[1].lower() or "similar" in think_notes[1].lower()


# =============================================================================
# Phase 1: Per-Result Summarization at Search Time
# =============================================================================


class TestPerResultSummarization:
    """Tests for Phase 1: per-result summarization in _handle_web_search_tool."""

    @pytest.mark.asyncio
    async def test_summarized_sources_show_summary_block(self) -> None:
        """After search, sources with long content are summarized and message
        history contains SUMMARY: blocks instead of raw content."""
        from unittest.mock import patch

        from foundry_mcp.core.research.providers.shared import SourceSummarizationResult

        mixin = StubTopicResearch()
        # Set config fields for summarization
        mixin.config.deep_research_summarization_min_content_length = 300
        mixin.config.deep_research_summarization_timeout = 30
        mixin.config.deep_research_max_content_length = 50000

        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        # Create a source with long content that should trigger summarization
        long_content = "A" * 500
        source = ResearchSource(
            id="src-long-1",
            title="Long Source",
            url="https://example.com/long",
            content=long_content,
            quality=SourceQuality.MEDIUM,
        )
        provider = _make_mock_provider("tavily", [source])

        # Mock the SourceSummarizer to return a known summary
        mock_summary_result = SourceSummarizationResult(
            executive_summary="This is the executive summary.",
            key_excerpts=["Key excerpt one", "Key excerpt two"],
            input_tokens=100,
            output_tokens=50,
        )

        with patch(
            "foundry_mcp.core.research.providers.shared.SourceSummarizer"
        ) as MockSummarizer:
            mock_instance = AsyncMock()
            mock_instance.summarize_sources = AsyncMock(
                return_value={"src-long-1": mock_summary_result}
            )
            MockSummarizer.return_value = mock_instance
            MockSummarizer.format_summarized_content = (
                lambda summary, excerpts: f"<summary>{summary}</summary>"
            )

            tool_call = ResearcherToolCall(
                tool="web_search",
                arguments={"query": sq.query, "max_results": 5},
            )
            result_text = await mixin._handle_web_search_tool(
                tool_call=tool_call,
                sub_query=sq,
                state=state,
                result=TopicResearchResult(sub_query_id=sq.id),
                available_providers=[provider],
                max_sources_per_provider=5,
                timeout=30.0,
                seen_urls=set(),
                seen_titles={},
                state_lock=asyncio.Lock(),
                semaphore=asyncio.Semaphore(3),
            )

        # The formatted output should contain SUMMARY: block
        assert "SUMMARY:" in result_text
        assert "executive summary" in result_text

        # Source in state should be marked as summarized
        summarized_src = next(s for s in state.sources if s.id == "src-long-1")
        assert summarized_src.metadata.get("summarized") is True
        assert summarized_src.raw_content == long_content

    @pytest.mark.asyncio
    async def test_fallback_to_raw_on_summarization_failure(self) -> None:
        """When summarization fails entirely, raw content is used as fallback."""
        from unittest.mock import patch

        mixin = StubTopicResearch()
        mixin.config.deep_research_summarization_min_content_length = 300
        mixin.config.deep_research_summarization_timeout = 30
        mixin.config.deep_research_max_content_length = 50000

        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        long_content = "B" * 600
        source = ResearchSource(
            id="src-fail-1",
            title="Fail Source",
            url="https://example.com/fail",
            content=long_content,
            quality=SourceQuality.MEDIUM,
        )
        provider = _make_mock_provider("tavily", [source])

        with patch(
            "foundry_mcp.core.research.providers.shared.SourceSummarizer"
        ) as MockSummarizer:
            # Summarizer raises an exception
            mock_instance = AsyncMock()
            mock_instance.summarize_sources = AsyncMock(
                side_effect=RuntimeError("Summarization provider unavailable")
            )
            MockSummarizer.return_value = mock_instance

            tool_call = ResearcherToolCall(
                tool="web_search",
                arguments={"query": sq.query, "max_results": 5},
            )
            result_text = await mixin._handle_web_search_tool(
                tool_call=tool_call,
                sub_query=sq,
                state=state,
                result=TopicResearchResult(sub_query_id=sq.id),
                available_providers=[provider],
                max_sources_per_provider=5,
                timeout=30.0,
                seen_urls=set(),
                seen_titles={},
                state_lock=asyncio.Lock(),
                semaphore=asyncio.Semaphore(3),
            )

        # Raw content should be used (truncated to 500 chars)
        assert "CONTENT:" in result_text
        assert "SUMMARY:" not in result_text
        # Source should NOT be marked as summarized
        src = next(s for s in state.sources if s.id == "src-fail-1")
        assert not src.metadata.get("summarized")

    @pytest.mark.asyncio
    async def test_short_content_not_summarized(self) -> None:
        """Sources with content below the threshold are not summarized."""
        from unittest.mock import patch

        mixin = StubTopicResearch()
        mixin.config.deep_research_summarization_min_content_length = 300
        mixin.config.deep_research_summarization_timeout = 30
        mixin.config.deep_research_max_content_length = 50000

        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        # Short content below 300 chars
        short_content = "Short result content"
        source = ResearchSource(
            id="src-short-1",
            title="Short Source",
            url="https://example.com/short",
            content=short_content,
            quality=SourceQuality.MEDIUM,
        )
        provider = _make_mock_provider("tavily", [source])

        with patch(
            "foundry_mcp.core.research.providers.shared.SourceSummarizer"
        ) as MockSummarizer:
            mock_instance = AsyncMock()
            mock_instance.summarize_sources = AsyncMock(return_value={})
            MockSummarizer.return_value = mock_instance

            tool_call = ResearcherToolCall(
                tool="web_search",
                arguments={"query": sq.query, "max_results": 5},
            )
            result_text = await mixin._handle_web_search_tool(
                tool_call=tool_call,
                sub_query=sq,
                state=state,
                result=TopicResearchResult(sub_query_id=sq.id),
                available_providers=[provider],
                max_sources_per_provider=5,
                timeout=30.0,
                seen_urls=set(),
                seen_titles={},
                state_lock=asyncio.Lock(),
                semaphore=asyncio.Semaphore(3),
            )

        # Short content should NOT be summarized — shown as CONTENT:
        assert "SUMMARY:" not in result_text
        # Source not marked as summarized
        src = next(s for s in state.sources if s.id == "src-short-1")
        assert not src.metadata.get("summarized")

    @pytest.mark.asyncio
    async def test_already_summarized_sources_skipped(self) -> None:
        """Sources already summarized by the provider layer are not re-summarized."""
        from unittest.mock import patch

        mixin = StubTopicResearch()
        mixin.config.deep_research_summarization_min_content_length = 300
        mixin.config.deep_research_summarization_timeout = 30
        mixin.config.deep_research_max_content_length = 50000

        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        # Source already summarized by provider
        source = ResearchSource(
            id="src-presumm-1",
            title="Pre-Summarized Source",
            url="https://example.com/presumm",
            content="<summary>Already summarized</summary>",
            quality=SourceQuality.MEDIUM,
            metadata={"summarized": True},
        )
        provider = _make_mock_provider("tavily", [source])

        with patch(
            "foundry_mcp.core.research.providers.shared.SourceSummarizer"
        ) as MockSummarizer:
            mock_instance = AsyncMock()
            # Should return empty because the source is already summarized
            mock_instance.summarize_sources = AsyncMock(return_value={})
            MockSummarizer.return_value = mock_instance

            tool_call = ResearcherToolCall(
                tool="web_search",
                arguments={"query": sq.query, "max_results": 5},
            )
            result_text = await mixin._handle_web_search_tool(
                tool_call=tool_call,
                sub_query=sq,
                state=state,
                result=TopicResearchResult(sub_query_id=sq.id),
                available_providers=[provider],
                max_sources_per_provider=5,
                timeout=30.0,
                seen_urls=set(),
                seen_titles={},
                state_lock=asyncio.Lock(),
                semaphore=asyncio.Semaphore(3),
            )

        # SUMMARY: block shown because metadata["summarized"] is True
        assert "SUMMARY:" in result_text
        assert "Already summarized" in result_text

    @pytest.mark.asyncio
    async def test_raw_content_preserved_in_metadata(self) -> None:
        """Summarized sources preserve original content in raw_content field."""
        from unittest.mock import patch

        from foundry_mcp.core.research.providers.shared import SourceSummarizationResult

        mixin = StubTopicResearch()
        mixin.config.deep_research_summarization_min_content_length = 100
        mixin.config.deep_research_summarization_timeout = 30
        mixin.config.deep_research_max_content_length = 50000

        state = _make_state(num_sub_queries=1)
        sq = state.sub_queries[0]

        original_content = "Original detailed content " * 20  # > 100 chars
        source = ResearchSource(
            id="src-raw-1",
            title="Raw Preserved",
            url="https://example.com/raw",
            content=original_content,
            quality=SourceQuality.MEDIUM,
        )
        provider = _make_mock_provider("tavily", [source])

        mock_summary = SourceSummarizationResult(
            executive_summary="Compressed summary.",
            key_excerpts=["Quote A"],
            input_tokens=80,
            output_tokens=30,
        )

        with patch(
            "foundry_mcp.core.research.providers.shared.SourceSummarizer"
        ) as MockSummarizer:
            mock_instance = AsyncMock()
            mock_instance.summarize_sources = AsyncMock(
                return_value={"src-raw-1": mock_summary}
            )
            MockSummarizer.return_value = mock_instance
            MockSummarizer.format_summarized_content = (
                lambda summary, excerpts: f"<summary>{summary}</summary>"
            )

            tool_call = ResearcherToolCall(
                tool="web_search",
                arguments={"query": sq.query, "max_results": 5},
            )
            await mixin._handle_web_search_tool(
                tool_call=tool_call,
                sub_query=sq,
                state=state,
                result=TopicResearchResult(sub_query_id=sq.id),
                available_providers=[provider],
                max_sources_per_provider=5,
                timeout=30.0,
                seen_urls=set(),
                seen_titles={},
                state_lock=asyncio.Lock(),
                semaphore=asyncio.Semaphore(3),
            )

        # Verify raw content preserved
        src = next(s for s in state.sources if s.id == "src-raw-1")
        assert src.raw_content == original_content
        assert src.content == "<summary>Compressed summary.</summary>"
        assert src.metadata["summarized"] is True
        assert src.metadata["excerpts"] == ["Quote A"]
        assert src.metadata["summarization_input_tokens"] == 80
        assert src.metadata["summarization_output_tokens"] == 30

    @pytest.mark.asyncio
    async def test_summarization_tokens_tracked(self) -> None:
        """Token usage from summarization is added to state.total_tokens_used."""
        from unittest.mock import patch

        from foundry_mcp.core.research.providers.shared import SourceSummarizationResult

        mixin = StubTopicResearch()
        mixin.config.deep_research_summarization_min_content_length = 100
        mixin.config.deep_research_summarization_timeout = 30
        mixin.config.deep_research_max_content_length = 50000

        state = _make_state(num_sub_queries=1)
        state.total_tokens_used = 0
        sq = state.sub_queries[0]

        source = ResearchSource(
            id="src-tok-1",
            title="Token Source",
            url="https://example.com/tok",
            content="C" * 500,
            quality=SourceQuality.MEDIUM,
        )
        provider = _make_mock_provider("tavily", [source])

        mock_summary = SourceSummarizationResult(
            executive_summary="Summary.",
            key_excerpts=[],
            input_tokens=200,
            output_tokens=100,
        )

        with patch(
            "foundry_mcp.core.research.providers.shared.SourceSummarizer"
        ) as MockSummarizer:
            mock_instance = AsyncMock()
            mock_instance.summarize_sources = AsyncMock(
                return_value={"src-tok-1": mock_summary}
            )
            MockSummarizer.return_value = mock_instance
            MockSummarizer.format_summarized_content = (
                lambda summary, excerpts: f"<summary>{summary}</summary>"
            )

            tool_call = ResearcherToolCall(
                tool="web_search",
                arguments={"query": sq.query, "max_results": 5},
            )
            await mixin._handle_web_search_tool(
                tool_call=tool_call,
                sub_query=sq,
                state=state,
                result=TopicResearchResult(sub_query_id=sq.id),
                available_providers=[provider],
                max_sources_per_provider=5,
                timeout=30.0,
                seen_urls=set(),
                seen_titles={},
                state_lock=asyncio.Lock(),
                semaphore=asyncio.Semaphore(3),
            )

        # 200 input + 100 output = 300 tokens tracked
        assert state.total_tokens_used == 300
