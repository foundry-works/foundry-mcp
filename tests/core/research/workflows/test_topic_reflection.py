"""Unit tests for Phase 2: Forced Reflection in Topic Research.

Tests cover:
1. TopicReflectionDecision dataclass — fields, defaults, serialization
2. parse_reflection_decision() — valid JSON, fallback regex, edge cases
3. Updated _topic_reflect() — enhanced prompt with source quality, structured response
4. Mandatory reflection in _execute_topic_research_async — always reflects after search
5. Early exit on research_complete=True
6. Exit on continue_searching=False
7. Hard cap max_searches overrides reflection decisions
8. TopicResearchResult new fields (early_completion, completion_rationale)
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
from foundry_mcp.core.research.workflows.deep_research._helpers import (
    TopicReflectionDecision,
    parse_reflection_decision,
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
    num_sub_queries: int = 1,
) -> DeepResearchState:
    """Create a DeepResearchState with pending sub-queries for testing."""
    state = DeepResearchState(
        id="deepres-test-reflection",
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
    """Concrete class inheriting TopicResearchMixin for testing."""

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
        """Mock provider async execution for reflection calls."""
        if self._provider_async_fn:
            return await self._provider_async_fn(**kwargs)
        # Default: return a structured reflection decision (research complete)
        result = MagicMock()
        result.success = True
        result.content = json.dumps({
            "continue_searching": False,
            "refined_query": None,
            "research_complete": False,
            "rationale": "Adequate sources found",
        })
        result.tokens_used = 50
        return result


# =============================================================================
# Unit tests: TopicReflectionDecision
# =============================================================================


class TestTopicReflectionDecision:
    """Tests for TopicReflectionDecision dataclass."""

    def test_default_values(self) -> None:
        decision = TopicReflectionDecision()
        assert decision.continue_searching is False
        assert decision.refined_query is None
        assert decision.research_complete is False
        assert decision.rationale == ""

    def test_custom_values(self) -> None:
        decision = TopicReflectionDecision(
            continue_searching=True,
            refined_query="better query",
            research_complete=False,
            rationale="Need more data",
        )
        assert decision.continue_searching is True
        assert decision.refined_query == "better query"
        assert decision.rationale == "Need more data"

    def test_to_dict(self) -> None:
        decision = TopicReflectionDecision(
            continue_searching=True,
            refined_query="q",
            research_complete=True,
            rationale="Done",
        )
        d = decision.to_dict()
        assert d["continue_searching"] is True
        assert d["refined_query"] == "q"
        assert d["research_complete"] is True
        assert d["rationale"] == "Done"


# =============================================================================
# Unit tests: parse_reflection_decision
# =============================================================================


class TestParseReflectionDecision:
    """Tests for parse_reflection_decision()."""

    def test_valid_json_continue_searching(self) -> None:
        """Valid JSON with continue_searching=true is parsed correctly."""
        text = json.dumps({
            "continue_searching": True,
            "refined_query": "deep learning architectures",
            "research_complete": False,
            "rationale": "Only 1 source found, need more",
        })
        decision = parse_reflection_decision(text)
        assert decision.continue_searching is True
        assert decision.refined_query == "deep learning architectures"
        assert decision.research_complete is False
        assert "Only 1 source" in decision.rationale

    def test_valid_json_research_complete(self) -> None:
        """Valid JSON with research_complete=true is parsed correctly."""
        text = json.dumps({
            "continue_searching": False,
            "refined_query": None,
            "research_complete": True,
            "rationale": "Found 5 relevant sources from diverse domains",
        })
        decision = parse_reflection_decision(text)
        assert decision.research_complete is True
        assert decision.continue_searching is False

    def test_valid_json_stop_searching(self) -> None:
        """Valid JSON with continue_searching=false, not complete."""
        text = json.dumps({
            "continue_searching": False,
            "research_complete": False,
            "rationale": "Sources are adequate",
        })
        decision = parse_reflection_decision(text)
        assert decision.continue_searching is False
        assert decision.research_complete is False

    def test_json_in_code_block(self) -> None:
        """JSON wrapped in markdown code block is extracted."""
        text = """```json
{"continue_searching": true, "refined_query": "ML basics", "research_complete": false, "rationale": "Too few sources"}
```"""
        decision = parse_reflection_decision(text)
        assert decision.continue_searching is True
        assert decision.refined_query == "ML basics"

    def test_json_with_surrounding_text(self) -> None:
        """JSON embedded in prose is extracted."""
        text = """Here is my assessment:
{"continue_searching": false, "research_complete": true, "rationale": "Sufficient coverage"}
Based on this analysis..."""
        decision = parse_reflection_decision(text)
        assert decision.research_complete is True

    def test_malformed_json_fallback_regex(self) -> None:
        """Malformed JSON falls back to regex extraction."""
        text = '"continue_searching": true, "refined_query": "better terms", "research_complete": false'
        decision = parse_reflection_decision(text)
        assert decision.continue_searching is True
        assert decision.refined_query == "better terms"

    def test_research_complete_regex_fallback(self) -> None:
        """research_complete=true detected via regex when JSON fails."""
        text = 'The analysis shows "research_complete": true because we have enough.'
        decision = parse_reflection_decision(text)
        assert decision.research_complete is True

    def test_empty_string_returns_default(self) -> None:
        """Empty string returns conservative default."""
        decision = parse_reflection_decision("")
        assert decision.continue_searching is False
        assert decision.research_complete is False

    def test_no_json_no_patterns_returns_default(self) -> None:
        """Plain text without recognizable patterns returns default."""
        decision = parse_reflection_decision("The sources look great, we should stop.")
        assert decision.continue_searching is False
        assert decision.research_complete is False

    def test_missing_fields_have_defaults(self) -> None:
        """JSON with missing fields uses defaults for missing keys."""
        text = json.dumps({"continue_searching": True})
        decision = parse_reflection_decision(text)
        assert decision.continue_searching is True
        assert decision.refined_query is None
        assert decision.research_complete is False
        assert decision.rationale == ""


# =============================================================================
# Unit tests: Updated _topic_reflect with enhanced prompt
# =============================================================================


class TestUpdatedTopicReflect:
    """Tests for the updated _topic_reflect() method."""

    @pytest.mark.asyncio
    async def test_returns_raw_response(self) -> None:
        """Reflection returns raw_response for structured parsing."""
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
        assert "tokens_used" in reflection

    @pytest.mark.asyncio
    async def test_prompt_excludes_quality_metadata(self) -> None:
        """Reflection prompt no longer includes quality distribution (simplified reflection)."""
        mixin = StubTopicResearch()
        state = _make_state()
        # Add sources with different quality levels
        state.sources.append(_make_source("src-1", quality=SourceQuality.HIGH))
        state.sources.append(_make_source("src-2", quality=SourceQuality.MEDIUM))

        captured_prompt = None

        async def capture_prompt(**kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs.get("prompt", "")
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "continue_searching": False,
                "research_complete": False,
                "rationale": "OK",
            })
            result.tokens_used = 40
            return result

        mixin._provider_async_fn = capture_prompt

        await mixin._topic_reflect(
            original_query="test",
            current_query="test",
            sources_found=2,
            iteration=1,
            max_iterations=3,
            state=state,
        )

        assert captured_prompt is not None
        # Quality and domain metadata no longer injected (simplified reflection)
        assert "Source quality distribution:" not in captured_prompt
        assert "Distinct source domains:" not in captured_prompt
        # Basic context is still present
        assert "Sources found so far: 2" in captured_prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_source_count(self) -> None:
        """Reflection prompt includes source count info."""
        mixin = StubTopicResearch()
        state = _make_state()

        captured_prompt = None

        async def capture_prompt(**kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs.get("prompt", "")
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "continue_searching": False,
                "research_complete": False,
                "rationale": "OK",
            })
            result.tokens_used = 40
            return result

        mixin._provider_async_fn = capture_prompt

        await mixin._topic_reflect(
            original_query="test",
            current_query="test",
            sources_found=5,
            iteration=2,
            max_iterations=3,
            state=state,
        )

        assert captured_prompt is not None
        assert "Sources found so far: 5" in captured_prompt
        assert "Search iteration: 2/3" in captured_prompt

    @pytest.mark.asyncio
    async def test_prompt_uses_adaptive_guidance(self) -> None:
        """Reflection system prompt uses adaptive guidance, not rigid thresholds."""
        mixin = StubTopicResearch()
        state = _make_state()

        captured_system = None

        async def capture_system(**kwargs):
            nonlocal captured_system
            captured_system = kwargs.get("system_prompt", "")
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "continue_searching": False,
                "research_complete": False,
                "rationale": "OK",
            })
            result.tokens_used = 40
            return result

        mixin._provider_async_fn = capture_system

        await mixin._topic_reflect(
            original_query="test",
            current_query="test",
            sources_found=4,
            iteration=1,
            max_iterations=3,
            state=state,
        )

        assert captured_system is not None
        # Adaptive guidance present
        assert "substantively answer the research question" in captured_system
        assert "Simple factual queries" in captured_system
        assert "diminishing returns" in captured_system
        # Rigid threshold rules removed
        assert "STOP IMMEDIATELY" not in captured_system
        assert "3+ sources" not in captured_system

    @pytest.mark.asyncio
    async def test_prompt_includes_original_and_current_query(self) -> None:
        """Reflection prompt includes both original and current query text."""
        mixin = StubTopicResearch()
        state = _make_state()

        captured_prompt = None

        async def capture_prompt(**kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs.get("prompt", "")
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "continue_searching": False,
                "research_complete": False,
                "rationale": "OK",
            })
            result.tokens_used = 40
            return result

        mixin._provider_async_fn = capture_prompt

        await mixin._topic_reflect(
            original_query="How does deep learning work?",
            current_query="deep learning architectures and transformers",
            sources_found=3,
            iteration=1,
            max_iterations=3,
            state=state,
        )

        assert captured_prompt is not None
        assert "How does deep learning work?" in captured_prompt
        assert "deep learning architectures and transformers" in captured_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_requests_structured_json(self) -> None:
        """System prompt requests the new structured JSON schema."""
        mixin = StubTopicResearch()
        state = _make_state()

        captured_system = None

        async def capture_system(**kwargs):
            nonlocal captured_system
            captured_system = kwargs.get("system_prompt", "")
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "continue_searching": False,
                "research_complete": False,
                "rationale": "OK",
            })
            result.tokens_used = 40
            return result

        mixin._provider_async_fn = capture_system

        await mixin._topic_reflect(
            original_query="test",
            current_query="test",
            sources_found=3,
            iteration=1,
            max_iterations=3,
            state=state,
        )

        assert captured_system is not None
        assert "continue_searching" in captured_system
        assert "research_complete" in captured_system
        assert "refined_query" in captured_system

    @pytest.mark.asyncio
    async def test_provider_failure_returns_stop(self) -> None:
        """Provider failure returns a parseable stop-searching response."""
        mixin = StubTopicResearch()
        state = _make_state()

        async def fail_provider(**kwargs):
            result = MagicMock()
            result.success = False
            return result

        mixin._provider_async_fn = fail_provider

        reflection = await mixin._topic_reflect(
            original_query="test",
            current_query="test",
            sources_found=2,
            iteration=1,
            max_iterations=3,
            state=state,
        )

        # Should be parseable as a stop decision
        decision = parse_reflection_decision(reflection["raw_response"])
        assert decision.continue_searching is False

    @pytest.mark.asyncio
    async def test_exception_returns_stop(self) -> None:
        """Exception during reflection returns a parseable stop response."""
        mixin = StubTopicResearch()
        state = _make_state()

        async def raise_error(**kwargs):
            raise RuntimeError("Network error")

        mixin._provider_async_fn = raise_error

        reflection = await mixin._topic_reflect(
            original_query="test",
            current_query="test",
            sources_found=0,
            iteration=1,
            max_iterations=3,
            state=state,
        )

        decision = parse_reflection_decision(reflection["raw_response"])
        assert decision.continue_searching is False


# =============================================================================
# Unit tests: Mandatory reflection in ReAct loop
# =============================================================================


class TestMandatoryReflectionLoop:
    """Tests for forced reflection in _execute_topic_research_async."""

    @pytest.mark.asyncio
    async def test_reflection_called_after_every_search(self) -> None:
        """Reflection is called after each search iteration, not conditionally."""
        mixin = StubTopicResearch()
        state = _make_state()
        sq = state.sub_queries[0]

        reflect_call_count = 0
        think_call_count = 0

        async def counting_calls(**kwargs):
            nonlocal reflect_call_count, think_call_count
            phase = kwargs.get("phase", "")
            result = MagicMock()
            result.success = True
            result.tokens_used = 30

            if phase == "topic_reflection":
                reflect_call_count += 1
                result.content = json.dumps({
                    "continue_searching": True,
                    "refined_query": f"refined-{reflect_call_count}",
                    "research_complete": False,
                    "rationale": f"Iteration {reflect_call_count}",
                })
            elif phase == "topic_think":
                think_call_count += 1
                result.content = json.dumps({
                    "reasoning": "Think step",
                    "next_query": None,
                })
            else:
                result.content = "{}"
            return result

        mixin._provider_async_fn = counting_calls

        # Provider always returns 1 source
        provider = _make_mock_provider("tavily")

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        await mixin._execute_topic_research_async(
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

        # With max_searches=3, reflection is called after iterations 0 and 1
        # (not after the last iteration). Since reflection always says continue
        # with a refined query, we get exactly 2 reflection calls.
        # Phase 5 adds a think step after each reflection-continue, so we also
        # get 2 think calls.
        assert reflect_call_count == 2
        assert think_call_count == 2

    @pytest.mark.asyncio
    async def test_reflection_called_even_with_zero_sources(self) -> None:
        """Reflection is mandatory even when no sources are found."""
        mixin = StubTopicResearch()
        state = _make_state()
        sq = state.sub_queries[0]

        reflect_call_count = 0

        async def counting_reflect(**kwargs):
            nonlocal reflect_call_count
            reflect_call_count += 1
            result = MagicMock()
            result.success = True
            # Stop after first reflection
            result.content = json.dumps({
                "continue_searching": False,
                "research_complete": False,
                "rationale": "No sources, stopping",
            })
            result.tokens_used = 30
            return result

        mixin._provider_async_fn = counting_reflect
        provider = _make_mock_provider("tavily", [])  # No sources

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        await mixin._execute_topic_research_async(
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

        # Reflection was called even with 0 sources
        assert reflect_call_count == 1

    @pytest.mark.asyncio
    async def test_early_exit_on_research_complete(self) -> None:
        """Loop exits early when reflection signals research_complete=True."""
        mixin = StubTopicResearch()
        state = _make_state()
        sq = state.sub_queries[0]

        sources = [_make_source(f"src-{i}", f"https://ex.com/{i}") for i in range(3)]
        provider = _make_mock_provider("tavily", sources)

        async def complete_reflect(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "continue_searching": False,
                "research_complete": True,
                "rationale": "Found 3 relevant sources from distinct domains",
            })
            result.tokens_used = 40
            return result

        mixin._provider_async_fn = complete_reflect

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        topic_result = await mixin._execute_topic_research_async(
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_searches=5,  # High budget
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        # Only 1 search performed, then early exit
        assert topic_result.searches_performed == 1
        assert topic_result.early_completion is True
        assert "distinct domains" in topic_result.completion_rationale

    @pytest.mark.asyncio
    async def test_exit_on_continue_searching_false(self) -> None:
        """Loop exits when reflection says continue_searching=False."""
        mixin = StubTopicResearch()
        state = _make_state()
        sq = state.sub_queries[0]

        provider = _make_mock_provider("tavily")

        async def stop_reflect(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "continue_searching": False,
                "research_complete": False,
                "rationale": "Sources are adequate for this topic",
            })
            result.tokens_used = 30
            return result

        mixin._provider_async_fn = stop_reflect

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        topic_result = await mixin._execute_topic_research_async(
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

        assert topic_result.searches_performed == 1
        assert topic_result.early_completion is False
        assert "adequate" in topic_result.completion_rationale

    @pytest.mark.asyncio
    async def test_max_searches_hard_cap(self) -> None:
        """max_searches is the hard cap regardless of reflection decisions."""
        mixin = StubTopicResearch()
        state = _make_state()
        sq = state.sub_queries[0]

        search_count = 0

        async def always_continue(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "continue_searching": True,
                "refined_query": f"refined-{search_count}",
                "research_complete": False,
                "rationale": "Need more data",
            })
            result.tokens_used = 30
            return result

        mixin._provider_async_fn = always_continue

        async def counting_search(**kwargs):
            nonlocal search_count
            search_count += 1
            return [_make_source(f"src-{search_count}", f"https://ex.com/{search_count}")]

        provider = MagicMock()
        provider.get_provider_name.return_value = "tavily"
        provider.search = counting_search

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        topic_result = await mixin._execute_topic_research_async(
            sub_query=sq,
            state=state,
            available_providers=[provider],
            max_searches=2,  # Hard cap at 2
            timeout=30.0,
            seen_urls=set(),
            seen_titles={},
            state_lock=state_lock,
            semaphore=semaphore,
        )

        # Even though reflection says continue, max_searches=2 is the hard cap
        assert topic_result.searches_performed <= 2

    @pytest.mark.asyncio
    async def test_refine_loop_with_structured_decision(self) -> None:
        """Search → reflect (continue with refined query) → search again."""
        mixin = StubTopicResearch()
        state = _make_state()
        sq = state.sub_queries[0]

        search_call_count = 0

        async def dynamic_search(**kwargs):
            nonlocal search_call_count
            search_call_count += 1
            return [_make_source(f"src-{search_call_count}", f"https://ex.com/{search_call_count}")]

        provider = MagicMock()
        provider.get_provider_name.return_value = "tavily"
        provider.search = dynamic_search

        reflect_count = 0

        async def dynamic_reflect(**kwargs):
            nonlocal reflect_count
            reflect_count += 1
            result = MagicMock()
            result.success = True
            result.tokens_used = 40
            if reflect_count == 1:
                result.content = json.dumps({
                    "continue_searching": True,
                    "refined_query": "refined deep learning query",
                    "research_complete": False,
                    "rationale": "Need more diverse sources",
                })
            else:
                result.content = json.dumps({
                    "continue_searching": False,
                    "research_complete": True,
                    "rationale": "Now have sufficient coverage",
                })
            return result

        mixin._provider_async_fn = dynamic_reflect

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        topic_result = await mixin._execute_topic_research_async(
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

        assert topic_result.searches_performed == 2
        assert "refined deep learning query" in topic_result.refined_queries
        assert topic_result.early_completion is True

    @pytest.mark.asyncio
    async def test_audit_event_includes_new_fields(self) -> None:
        """Audit event includes early_completion and completion_rationale."""
        mixin = StubTopicResearch()
        state = _make_state()
        sq = state.sub_queries[0]

        provider = _make_mock_provider("tavily")

        async def complete_reflect(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "continue_searching": False,
                "research_complete": True,
                "rationale": "Well covered",
            })
            result.tokens_used = 30
            return result

        mixin._provider_async_fn = complete_reflect

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        await mixin._execute_topic_research_async(
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

        assert len(mixin._audit_events) >= 1
        event_name, event_data = mixin._audit_events[-1]
        assert event_name == "topic_research_complete"
        assert event_data["data"]["early_completion"] is True
        assert "Well covered" in event_data["data"]["completion_rationale"]

    @pytest.mark.asyncio
    async def test_tokens_tracked_from_reflection(self) -> None:
        """Reflection tokens are accumulated and merged into state."""
        mixin = StubTopicResearch()
        state = _make_state()
        state.total_tokens_used = 100
        sq = state.sub_queries[0]

        async def token_reflect(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "continue_searching": False,
                "research_complete": False,
                "rationale": "OK",
            })
            result.tokens_used = 75
            return result

        mixin._provider_async_fn = token_reflect
        provider = _make_mock_provider("tavily")

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        await mixin._execute_topic_research_async(
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

        # Tokens should have been merged into state
        assert state.total_tokens_used == 175  # 100 + 75


# =============================================================================
# Unit tests: TopicResearchResult new fields
# =============================================================================


class TestTopicResearchResultNewFields:
    """Tests for early_completion and completion_rationale fields."""

    def test_default_values(self) -> None:
        result = TopicResearchResult(sub_query_id="sq-1")
        assert result.early_completion is False
        assert result.completion_rationale == ""

    def test_set_early_completion(self) -> None:
        result = TopicResearchResult(sub_query_id="sq-1")
        result.early_completion = True
        result.completion_rationale = "Found sufficient sources"
        assert result.early_completion is True
        assert result.completion_rationale == "Found sufficient sources"

    def test_serialization_includes_new_fields(self) -> None:
        result = TopicResearchResult(
            sub_query_id="sq-test",
            early_completion=True,
            completion_rationale="Research complete signal",
        )
        d = result.model_dump()
        assert d["early_completion"] is True
        assert d["completion_rationale"] == "Research complete signal"

    def test_backward_compat_deserialization(self) -> None:
        """Old serialized data without new fields deserializes correctly."""
        old_data = {
            "sub_query_id": "sq-old",
            "searches_performed": 2,
            "sources_found": 3,
        }
        result = TopicResearchResult(**old_data)
        assert result.early_completion is False
        assert result.completion_rationale == ""


# =============================================================================
# Integration tests
# =============================================================================


class TestForcedReflectionIntegration:
    """Integration tests for the forced reflection behavior."""

    @pytest.mark.asyncio
    async def test_single_search_max_1_no_reflection(self) -> None:
        """With max_searches=1, no reflection occurs (last iteration skips it)."""
        mixin = StubTopicResearch()
        state = _make_state()
        sq = state.sub_queries[0]

        reflect_count = 0

        async def counting_reflect(**kwargs):
            nonlocal reflect_count
            reflect_count += 1
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "continue_searching": False,
                "research_complete": False,
                "rationale": "OK",
            })
            result.tokens_used = 30
            return result

        mixin._provider_async_fn = counting_reflect
        provider = _make_mock_provider("tavily")

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        topic_result = await mixin._execute_topic_research_async(
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

        # max_searches=1 means the loop body runs once and breaks before reflection
        assert reflect_count == 0
        assert topic_result.searches_performed == 1

    @pytest.mark.asyncio
    async def test_fallback_query_broadening_on_zero_sources(self) -> None:
        """With zero sources and no refined query, falls back to broadening."""
        mixin = StubTopicResearch()
        state = _make_state()
        sq = state.sub_queries[0]
        sq.query = '"very specific phrase"'

        search_count = 0

        async def dynamic_search(**kwargs):
            nonlocal search_count
            search_count += 1
            if search_count == 1:
                return []
            return [_make_source(f"src-{search_count}", f"https://retry.com/{search_count}")]

        provider = MagicMock()
        provider.get_provider_name.return_value = "tavily"
        provider.search = dynamic_search

        # Reflection says continue but doesn't provide a refined query
        async def no_refined_query(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "continue_searching": True,
                "refined_query": None,
                "research_complete": False,
                "rationale": "No sources found, try broader",
            })
            result.tokens_used = 30
            return result

        mixin._provider_async_fn = no_refined_query

        semaphore = asyncio.Semaphore(3)
        state_lock = asyncio.Lock()

        topic_result = await mixin._execute_topic_research_async(
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

        # Should have broadened the query (removed quotes)
        assert any("very specific phrase" in q for q in topic_result.refined_queries)
