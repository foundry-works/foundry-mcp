"""Tests for PLAN Phase 5: Compression Token-Limit Retry Strategy.

Covers:
- 5a: Message-boundary-aware truncation for compression retries
  - _group_message_pairs groups messages correctly
  - _classify_message_pair identifies think/search/research_complete/other
  - truncate_message_history_for_retry drops oldest pairs progressively
  - Protected messages (last 2 thinks, last search, research_complete) preserved
  - Integration: retry loop uses message-boundary truncation with message_history
  - Integration: retry loop falls back to percentage truncation without message_history
- 5b: Truncation metadata recorded on TopicResearchResult
  - compression_messages_dropped, compression_retry_count, compression_original_message_count
  - Metadata recorded on success after retry
  - Metadata recorded on failure after retry exhaustion
  - Backward compat: fields default to 0
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
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research.phases.compression import (
    CompressionMixin,
    _classify_message_pair,
    _group_message_pairs,
    truncate_message_history_for_retry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    query: str = "What are the benefits of renewable energy?",
    phase: DeepResearchPhase = DeepResearchPhase.GATHERING,
    num_sub_queries: int = 1,
    sources_per_query: int = 3,
) -> DeepResearchState:
    """Create a DeepResearchState with sub-queries and sources for testing."""
    state = DeepResearchState(
        id="deepres-test-retry",
        original_query=query,
        research_brief="Investigation of renewable energy benefits",
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


def _make_topic_result_with_history(
    state: DeepResearchState,
    sub_query_id: str = "sq-0",
    num_search_pairs: int = 3,
    num_think_pairs: int = 2,
    include_research_complete: bool = True,
) -> TopicResearchResult:
    """Create a TopicResearchResult with a realistic message_history."""
    source_ids = [s.id for s in state.sources if s.sub_query_id == sub_query_id]
    tr = TopicResearchResult(
        sub_query_id=sub_query_id,
        searches_performed=num_search_pairs,
        sources_found=len(source_ids),
        source_ids=source_ids,
    )

    history: list[dict[str, str]] = []

    # Interleave search and think pairs
    for i in range(num_search_pairs):
        history.append({
            "role": "assistant",
            "content": f'{{"tool_calls": [{{"tool": "web_search", "arguments": {{"query": "search {i}"}}}}]}}',
        })
        history.append({
            "role": "tool",
            "tool": "web_search",
            "content": f"Found results for search {i}",
        })
        if i < num_think_pairs:
            history.append({
                "role": "assistant",
                "content": f'{{"tool_calls": [{{"tool": "think", "arguments": {{"reasoning": "thinking about {i}"}}}}]}}',
            })
            history.append({
                "role": "tool",
                "tool": "think",
                "content": f"Reflection {i} recorded.",
            })

    if include_research_complete:
        history.append({
            "role": "assistant",
            "content": '{"tool_calls": [{"tool": "research_complete", "arguments": {"summary": "Done"}}]}',
        })
        history.append({
            "role": "tool",
            "tool": "research_complete",
            "content": "Research complete.",
        })

    tr.message_history = history
    state.topic_research_results.append(tr)
    return tr


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


# =============================================================================
# 5a: _classify_message_pair
# =============================================================================


class TestClassifyMessagePair:
    """Tests for _classify_message_pair."""

    def test_think_from_tool_name(self):
        assistant = {"role": "assistant", "content": "..."}
        tool = {"role": "tool", "tool": "think", "content": "Reflection"}
        assert _classify_message_pair(assistant, tool) == "think"

    def test_search_from_tool_name(self):
        assistant = {"role": "assistant", "content": "..."}
        tool = {"role": "tool", "tool": "web_search", "content": "Results"}
        assert _classify_message_pair(assistant, tool) == "search"

    def test_research_complete_from_tool_name(self):
        assistant = {"role": "assistant", "content": "..."}
        tool = {"role": "tool", "tool": "research_complete", "content": "Done"}
        assert _classify_message_pair(assistant, tool) == "research_complete"

    def test_other_tool(self):
        assistant = {"role": "assistant", "content": "..."}
        tool = {"role": "tool", "tool": "extract_url", "content": "Page content"}
        assert _classify_message_pair(assistant, tool) == "other"

    def test_infer_from_assistant_content_when_no_tool(self):
        assistant = {"role": "assistant", "content": '"research_complete"'}
        assert _classify_message_pair(assistant, None) == "research_complete"

    def test_infer_think_from_content(self):
        assistant = {"role": "assistant", "content": '"think"'}
        assert _classify_message_pair(assistant, None) == "think"

    def test_infer_search_from_content(self):
        assistant = {"role": "assistant", "content": '"web_search"'}
        assert _classify_message_pair(assistant, None) == "search"


# =============================================================================
# 5a: _group_message_pairs
# =============================================================================


class TestGroupMessagePairs:
    """Tests for _group_message_pairs."""

    def test_pairs_assistant_tool_messages(self):
        history = [
            {"role": "assistant", "content": "search call"},
            {"role": "tool", "tool": "web_search", "content": "results"},
            {"role": "assistant", "content": "think call"},
            {"role": "tool", "tool": "think", "content": "reflection"},
        ]
        pairs = _group_message_pairs(history)
        assert len(pairs) == 2
        assert pairs[0][0] == "search"
        assert len(pairs[0][1]) == 2
        assert pairs[1][0] == "think"
        assert len(pairs[1][1]) == 2

    def test_handles_single_message_at_end(self):
        history = [
            {"role": "assistant", "content": "search call"},
            {"role": "tool", "tool": "web_search", "content": "results"},
            {"role": "assistant", "content": "orphan assistant message"},
        ]
        pairs = _group_message_pairs(history)
        assert len(pairs) == 2
        # Last message is a single (no tool response)
        assert len(pairs[1][1]) == 1

    def test_empty_history(self):
        assert _group_message_pairs([]) == []

    def test_all_types_present(self):
        history = [
            {"role": "assistant", "content": "s1"},
            {"role": "tool", "tool": "web_search", "content": "r1"},
            {"role": "assistant", "content": "t1"},
            {"role": "tool", "tool": "think", "content": "r2"},
            {"role": "assistant", "content": "e1"},
            {"role": "tool", "tool": "extract_url", "content": "r3"},
            {"role": "assistant", "content": "rc"},
            {"role": "tool", "tool": "research_complete", "content": "done"},
        ]
        pairs = _group_message_pairs(history)
        types = [t for t, _ in pairs]
        assert types == ["search", "think", "other", "research_complete"]


# =============================================================================
# 5a: truncate_message_history_for_retry
# =============================================================================


class TestTruncateMessageHistoryForRetry:
    """Tests for truncate_message_history_for_retry."""

    def _build_history(
        self,
        n_search: int = 4,
        n_think: int = 3,
        research_complete: bool = True,
    ) -> list[dict[str, str]]:
        """Build a message history with n_search, n_think, and optional research_complete."""
        history: list[dict[str, str]] = []
        for i in range(n_search):
            history.append({"role": "assistant", "content": f"search {i}"})
            history.append({"role": "tool", "tool": "web_search", "content": f"results {i}"})
        for i in range(n_think):
            history.append({"role": "assistant", "content": f"think {i}"})
            history.append({"role": "tool", "tool": "think", "content": f"reflection {i}"})
        if research_complete:
            history.append({"role": "assistant", "content": "complete"})
            history.append({"role": "tool", "tool": "research_complete", "content": "done"})
        return history

    def test_attempt_0_returns_unchanged(self):
        history = self._build_history()
        result, dropped = truncate_message_history_for_retry(history, 0)
        assert result == history
        assert dropped == 0

    def test_empty_history_returns_unchanged(self):
        result, dropped = truncate_message_history_for_retry([], 1)
        assert result == []
        assert dropped == 0

    def test_single_pair_returns_unchanged(self):
        history = [
            {"role": "assistant", "content": "only"},
            {"role": "tool", "tool": "web_search", "content": "result"},
        ]
        result, dropped = truncate_message_history_for_retry(history, 1)
        assert result == history
        assert dropped == 0

    def test_preserves_research_complete(self):
        """research_complete pair is never dropped."""
        history = self._build_history(n_search=4, n_think=0)
        result, dropped = truncate_message_history_for_retry(history, 3, max_attempts=3)
        # research_complete should still be present
        tool_names = [m.get("tool") for m in result if m.get("role") == "tool"]
        assert "research_complete" in tool_names

    def test_preserves_last_2_thinks(self):
        """The most recent 2 think pairs are never dropped."""
        history = self._build_history(n_search=2, n_think=3)
        result, dropped = truncate_message_history_for_retry(history, 3, max_attempts=3)
        # Count think tool messages in result
        think_tools = [m for m in result if m.get("tool") == "think"]
        assert len(think_tools) >= 2

    def test_preserves_last_search(self):
        """The most recent search pair is never dropped."""
        history = self._build_history(n_search=4, n_think=0, research_complete=False)
        result, dropped = truncate_message_history_for_retry(history, 3, max_attempts=3)
        # At least one search should remain
        search_tools = [m for m in result if m.get("tool") == "web_search"]
        assert len(search_tools) >= 1
        # It should be the last one
        assert "results 3" in search_tools[-1]["content"]

    def test_progressive_removal_attempt_1(self):
        """Attempt 1 drops ~1/3 of droppable pairs."""
        # 4 search + 0 think + research_complete = 5 pairs
        # Protected: last search (idx 3), research_complete (idx 4)
        # Droppable: 3 pairs (search 0, 1, 2)
        # Attempt 1: drop 1/3 of 3 = 1 pair
        history = self._build_history(n_search=4, n_think=0)
        result, dropped = truncate_message_history_for_retry(history, 1, max_attempts=3)
        assert dropped == 2  # 1 pair = 2 messages

    def test_progressive_removal_attempt_3_drops_all_droppable(self):
        """Attempt 3 (max) drops all droppable pairs."""
        # 4 search + 0 think + research_complete = 5 pairs
        # Protected: last search (idx 3), research_complete (idx 4)
        # Droppable: 3 pairs (search 0, 1, 2)
        # Attempt 3: drop all 3
        history = self._build_history(n_search=4, n_think=0)
        result, dropped = truncate_message_history_for_retry(history, 3, max_attempts=3)
        assert dropped == 6  # 3 pairs = 6 messages
        # Only 2 pairs remain (last search + research_complete)
        assert len(result) == 4

    def test_drops_oldest_first(self):
        """Oldest pairs are dropped before newer ones."""
        history = self._build_history(n_search=4, n_think=0, research_complete=False)
        result, dropped = truncate_message_history_for_retry(history, 1, max_attempts=3)
        # search 0 should be dropped (oldest droppable)
        contents = [m.get("content", "") for m in result]
        assert "results 0" not in contents
        # search 3 should remain (protected as last search)
        assert "results 3" in contents


# =============================================================================
# 5b: Truncation metadata fields on TopicResearchResult
# =============================================================================


class TestTruncationMetadataFields:
    """Tests for compression truncation metadata on TopicResearchResult."""

    def test_default_values(self):
        """New fields default to 0 (backward compat)."""
        tr = TopicResearchResult(sub_query_id="sq-test")
        assert tr.compression_messages_dropped == 0
        assert tr.compression_retry_count == 0
        assert tr.compression_original_message_count == 0

    def test_backward_compat_deserialization(self):
        """Old data without new fields deserializes cleanly."""
        data = {
            "sub_query_id": "sq-old",
            "searches_performed": 2,
            "sources_found": 3,
        }
        tr = TopicResearchResult.model_validate(data)
        assert tr.compression_messages_dropped == 0
        assert tr.compression_retry_count == 0
        assert tr.compression_original_message_count == 0

    def test_roundtrip_serialization(self):
        """Metadata survives model_dump -> model_validate roundtrip."""
        original = TopicResearchResult(
            sub_query_id="sq-rt",
            compression_messages_dropped=4,
            compression_retry_count=2,
            compression_original_message_count=12,
        )
        data = original.model_dump()
        restored = TopicResearchResult.model_validate(data)
        assert restored.compression_messages_dropped == 4
        assert restored.compression_retry_count == 2
        assert restored.compression_original_message_count == 12

    def test_fields_in_dump(self):
        """Fields appear in model_dump output."""
        tr = TopicResearchResult(
            sub_query_id="sq-dump",
            compression_messages_dropped=2,
            compression_retry_count=1,
            compression_original_message_count=8,
        )
        data = tr.model_dump()
        assert "compression_messages_dropped" in data
        assert "compression_retry_count" in data
        assert "compression_original_message_count" in data


# =============================================================================
# 5a+5b: Integration — retry loop uses message-boundary truncation
# =============================================================================


class TestCompressionRetryWithMessageBoundary:
    """Integration tests for the retry loop in _compress_single_topic_async."""

    @pytest.mark.asyncio
    async def test_no_retry_records_zero_metadata(self):
        """When compression succeeds on first try, metadata stays at 0."""
        state = _make_state()
        tr = _make_topic_result_with_history(state)
        stub = StubCompression()

        mock_result = MagicMock()
        mock_result.result.success = True
        mock_result.result.content = "Compressed output"
        mock_result.result.input_tokens = 100
        mock_result.result.output_tokens = 50

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            inp, out, success = await stub._compress_single_topic_async(
                tr, state, 120.0
            )

        assert success is True
        assert tr.compression_retry_count == 0
        assert tr.compression_messages_dropped == 0
        assert tr.compression_original_message_count == len(tr.message_history)

    @pytest.mark.asyncio
    async def test_retry_uses_message_boundary_truncation(self):
        """On context-window error with message_history, uses message-boundary truncation."""
        state = _make_state()
        tr = _make_topic_result_with_history(
            state, num_search_pairs=5, num_think_pairs=3,
        )
        stub = StubCompression()

        call_count = 0
        captured_prompts: list[str] = []

        async def mock_execute_llm_call(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            captured_prompts.append(kwargs["user_prompt"])

            if call_count == 1:
                # First call: context window exceeded
                return WorkflowResult(
                    success=False,
                    content="",
                    error="Context window exceeded",
                    metadata={"error_type": "context_window_exceeded"},
                )
            # Second call: success
            result = MagicMock()
            result.result.success = True
            result.result.content = "Compressed after retry"
            result.result.input_tokens = 80
            result.result.output_tokens = 40
            return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ):
            inp, out, success = await stub._compress_single_topic_async(
                tr, state, 120.0
            )

        assert success is True
        assert tr.compression_retry_count == 1
        assert tr.compression_messages_dropped > 0
        assert tr.compression_original_message_count == len(tr.message_history)
        # The second prompt should be shorter (messages dropped)
        assert len(captured_prompts[1]) < len(captured_prompts[0])

    @pytest.mark.asyncio
    async def test_retry_preserves_recent_messages_in_prompt(self):
        """After retry truncation, the prompt still contains the most recent messages."""
        state = _make_state()
        tr = _make_topic_result_with_history(
            state, num_search_pairs=5, num_think_pairs=3,
        )
        stub = StubCompression()

        call_count = 0
        captured_prompts: list[str] = []

        async def mock_execute(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            captured_prompts.append(kwargs["user_prompt"])

            if call_count <= 2:
                return WorkflowResult(
                    success=False, content="", error="overflow",
                    metadata={"error_type": "context_window_exceeded"},
                )
            result = MagicMock()
            result.result.success = True
            result.result.content = "Compressed"
            result.result.input_tokens = 0
            result.result.output_tokens = 0
            return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=mock_execute,
        ):
            await stub._compress_single_topic_async(tr, state, 120.0)

        # The last prompt should still contain the research_complete content
        last_prompt = captured_prompts[-1]
        assert "Research complete" in last_prompt or "research_complete" in last_prompt.lower()

    @pytest.mark.asyncio
    async def test_metadata_on_retry_exhaustion(self):
        """When all retries fail, metadata is still recorded."""
        state = _make_state()
        tr = _make_topic_result_with_history(state)
        stub = StubCompression()

        async def always_fail(**kwargs: Any) -> Any:
            return WorkflowResult(
                success=False, content="", error="overflow",
                metadata={"error_type": "context_window_exceeded"},
            )

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=always_fail,
        ):
            inp, out, success = await stub._compress_single_topic_async(
                tr, state, 120.0
            )

        assert success is False
        assert tr.compression_retry_count == 3  # MAX_PHASE_TOKEN_RETRIES
        assert tr.compression_messages_dropped > 0
        assert tr.compression_original_message_count == len(tr.message_history)

    @pytest.mark.asyncio
    async def test_fallback_to_percentage_truncation_without_history(self):
        """Without message_history, falls back to percentage-based truncation."""
        state = _make_state()
        source_ids = [s.id for s in state.sources if s.sub_query_id == "sq-0"]
        tr = TopicResearchResult(
            sub_query_id="sq-0",
            searches_performed=2,
            sources_found=len(source_ids),
            source_ids=source_ids,
            reflection_notes=["note"],
            refined_queries=["refined"],
            # No message_history — uses structured metadata fallback
        )
        state.topic_research_results.append(tr)
        stub = StubCompression()

        call_count = 0

        async def mock_execute(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return WorkflowResult(
                    success=False, content="", error="overflow",
                    metadata={"error_type": "context_window_exceeded"},
                )
            result = MagicMock()
            result.result.success = True
            result.result.content = "Compressed"
            result.result.input_tokens = 0
            result.result.output_tokens = 0
            return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=mock_execute,
        ):
            inp, out, success = await stub._compress_single_topic_async(
                tr, state, 120.0
            )

        assert success is True
        # Without message_history, messages_dropped stays 0
        assert tr.compression_messages_dropped == 0
        assert tr.compression_retry_count == 1
        # original_message_count is 0 since there was no message_history
        assert tr.compression_original_message_count == 0

    @pytest.mark.asyncio
    async def test_audit_event_includes_truncation_data(self):
        """Audit events on retry success include message truncation metadata."""
        state = _make_state()
        tr = _make_topic_result_with_history(state, num_search_pairs=4)
        stub = StubCompression()

        call_count = 0

        async def mock_execute(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return WorkflowResult(
                    success=False, content="", error="overflow",
                    metadata={"error_type": "context_window_exceeded"},
                )
            result = MagicMock()
            result.result.success = True
            result.result.content = "Compressed"
            result.result.input_tokens = 10
            result.result.output_tokens = 5
            return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=mock_execute,
        ):
            await stub._compress_single_topic_async(tr, state, 120.0)

        # Check audit events
        retry_events = [
            (evt, data) for evt, data in stub._audit_events
            if evt == "compression_retry_succeeded"
        ]
        assert len(retry_events) == 1
        event_data = retry_events[0][1]["data"]
        assert "messages_dropped" in event_data
        assert "original_message_count" in event_data
        assert event_data["outer_retries"] == 1
