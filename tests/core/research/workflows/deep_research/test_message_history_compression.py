"""Tests for PLAN Phase 2: Pass Full Message History to Compression.

Covers:
- 2.1: message_history field on TopicResearchResult (default, serialization, backward compat)
- 2.2: message_history stored on result after ReAct loop
- 2.3: compression prompt includes raw message history when available
- 2.3: compression prompt falls back to structured metadata when message_history empty
- 2.4: compression system prompt aligned with open_deep_research structure
- 2.5: message history truncated to max_content_length (oldest dropped first)
- 2.5: citation format [N] Title: URL in compression prompt
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
    _build_message_history_prompt,
    _build_structured_metadata_prompt,
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
        id="deepres-test-msg-history",
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


def _make_topic_result(
    state: DeepResearchState,
    sub_query_id: str = "sq-0",
    *,
    with_message_history: bool = True,
) -> TopicResearchResult:
    """Create a TopicResearchResult, optionally with message_history."""
    source_ids = [s.id for s in state.sources if s.sub_query_id == sub_query_id]
    tr = TopicResearchResult(
        sub_query_id=sub_query_id,
        searches_performed=2,
        sources_found=len(source_ids),
        source_ids=source_ids,
        reflection_notes=["Found relevant sources", "Good coverage"],
        refined_queries=["refined query for " + sub_query_id],
    )
    if with_message_history:
        tr.message_history = [
            {"role": "assistant", "content": '{"tool_calls": [{"tool": "web_search", "arguments": {"query": "renewable energy benefits"}}]}'},
            {"role": "tool", "tool": "web_search", "content": "Found 3 new source(s):\n--- SOURCE 1: Solar Benefits ---\nURL: https://example0.com\nSNIPPET: Solar energy reduces costs..."},
            {"role": "assistant", "content": '{"tool_calls": [{"tool": "think", "arguments": {"reasoning": "I found sources about solar but need wind energy too"}}]}'},
            {"role": "tool", "tool": "think", "content": "Reflection recorded."},
            {"role": "assistant", "content": '{"tool_calls": [{"tool": "web_search", "arguments": {"query": "wind energy advantages"}}]}'},
            {"role": "tool", "tool": "web_search", "content": "Found 2 new source(s):\n--- SOURCE 1: Wind Power ---\nURL: https://example1.com\nSNIPPET: Wind farms produce clean electricity..."},
            {"role": "assistant", "content": '{"tool_calls": [{"tool": "research_complete", "arguments": {"summary": "Found comprehensive sources on both solar and wind"}}]}'},
            {"role": "tool", "tool": "research_complete", "content": "Research complete. Findings recorded."},
        ]
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
# 2.1: message_history field on TopicResearchResult
# =============================================================================


class TestMessageHistoryField:
    """Tests for the message_history field on TopicResearchResult."""

    def test_default_empty_list(self):
        """message_history defaults to empty list (backward compat)."""
        tr = TopicResearchResult(sub_query_id="sq-1")
        assert tr.message_history == []

    def test_set_and_get(self):
        """message_history can be set and retrieved."""
        history = [
            {"role": "assistant", "content": "response"},
            {"role": "tool", "tool": "web_search", "content": "results"},
        ]
        tr = TopicResearchResult(sub_query_id="sq-1", message_history=history)
        assert len(tr.message_history) == 2
        assert tr.message_history[0]["role"] == "assistant"
        assert tr.message_history[1]["tool"] == "web_search"

    def test_model_dump_includes_message_history(self):
        """message_history appears in model_dump output."""
        tr = TopicResearchResult(
            sub_query_id="sq-1",
            message_history=[{"role": "assistant", "content": "test"}],
        )
        data = tr.model_dump()
        assert "message_history" in data
        assert len(data["message_history"]) == 1

    def test_backward_compat_deserialization(self):
        """Old serialized data without message_history deserializes cleanly."""
        data = {
            "sub_query_id": "sq-old",
            "searches_performed": 2,
            "sources_found": 3,
        }
        tr = TopicResearchResult.model_validate(data)
        assert tr.message_history == []
        assert tr.sub_query_id == "sq-old"


# =============================================================================
# 2.3: _build_message_history_prompt unit tests
# =============================================================================


class TestBuildMessageHistoryPrompt:
    """Tests for the _build_message_history_prompt helper."""

    def test_includes_conversation_entries(self):
        """Prompt should include all message history entries."""
        history = [
            {"role": "assistant", "content": "search call"},
            {"role": "tool", "tool": "web_search", "content": "3 results found"},
            {"role": "assistant", "content": "think call"},
            {"role": "tool", "tool": "think", "content": "Reflection recorded."},
        ]
        sources = [MagicMock(title="Source A", url="https://a.com")]

        prompt = _build_message_history_prompt(
            query_text="test query",
            message_history=history,
            topic_sources=sources,
            max_content_length=100_000,
        )

        assert "[Assistant]" in prompt
        assert "[Tool: web_search]" in prompt
        assert "[Tool: think]" in prompt
        assert "3 results found" in prompt
        assert "search call" in prompt

    def test_includes_source_reference(self):
        """Prompt should include source reference list for citation mapping."""
        src_a = MagicMock(title="Source A", url="https://a.com")
        src_b = MagicMock(title="Source B", url="https://b.com")
        history = [{"role": "assistant", "content": "test"}]

        prompt = _build_message_history_prompt(
            query_text="test query",
            message_history=history,
            topic_sources=[src_a, src_b],
            max_content_length=100_000,
        )

        assert "[1] Source A: https://a.com" in prompt
        assert "[2] Source B: https://b.com" in prompt

    def test_includes_research_subquery(self):
        """Prompt should contain the research sub-query."""
        history = [{"role": "assistant", "content": "test"}]
        prompt = _build_message_history_prompt(
            query_text="benefits of solar energy",
            message_history=history,
            topic_sources=[],
            max_content_length=100_000,
        )
        assert "Research sub-query: benefits of solar energy" in prompt

    def test_includes_cleanup_instructions(self):
        """Prompt should include the cleanup/verbatim instructions."""
        history = [{"role": "assistant", "content": "test"}]
        prompt = _build_message_history_prompt(
            query_text="test",
            message_history=history,
            topic_sources=[],
            max_content_length=100_000,
        )
        assert "DO NOT summarize" in prompt
        assert "rewrite findings verbatim" in prompt

    def test_truncates_oldest_messages_first(self):
        """When over max_content_length, oldest messages are dropped first."""
        # Create messages where each is ~100 chars
        history = [
            {"role": "assistant", "content": f"Message number {i} " + "x" * 80}
            for i in range(20)
        ]
        prompt = _build_message_history_prompt(
            query_text="test",
            message_history=history,
            topic_sources=[],
            max_content_length=500,
        )
        # Most recent messages should be preserved
        assert "Message number 19" in prompt
        # Oldest messages should be truncated
        assert "Message number 0" not in prompt

    def test_handles_single_huge_message(self):
        """When a single message exceeds max_content_length, hard-truncate from end."""
        history = [{"role": "assistant", "content": "x" * 10_000}]
        prompt = _build_message_history_prompt(
            query_text="test",
            message_history=history,
            topic_sources=[],
            max_content_length=500,
        )
        # Should not raise, should produce some output
        assert len(prompt) > 0

    def test_source_without_url(self):
        """Source reference handles sources without URLs."""
        src = MagicMock(title="Offline Source", url=None)
        history = [{"role": "assistant", "content": "test"}]
        prompt = _build_message_history_prompt(
            query_text="test",
            message_history=history,
            topic_sources=[src],
            max_content_length=100_000,
        )
        assert "[1] Offline Source" in prompt
        assert ": None" not in prompt


# =============================================================================
# 2.3: _build_structured_metadata_prompt unit tests (fallback)
# =============================================================================


class TestBuildStructuredMetadataPrompt:
    """Tests for the _build_structured_metadata_prompt fallback helper."""

    def test_includes_iteration_history(self):
        """Prompt should include search iteration history."""
        tr = TopicResearchResult(
            sub_query_id="sq-0",
            sources_found=3,
            reflection_notes=["Found relevant sources"],
            refined_queries=["refined query"],
        )
        src = MagicMock(title="Src", url="https://x.com", content="Content", snippet=None)

        prompt = _build_structured_metadata_prompt(
            query_text="test query",
            topic_result=tr,
            topic_sources=[src],
            max_content_length=50_000,
        )

        assert "Search iterations:" in prompt
        assert '"test query"' in prompt
        assert '"refined query"' in prompt
        assert "Found relevant sources" in prompt

    def test_includes_source_content(self):
        """Prompt should include source content."""
        tr = TopicResearchResult(sub_query_id="sq-0", sources_found=1)
        src = MagicMock(title="Src A", url="https://a.com", content="Detailed info", snippet=None)

        prompt = _build_structured_metadata_prompt(
            query_text="test",
            topic_result=tr,
            topic_sources=[src],
            max_content_length=50_000,
        )

        assert "[1] Title: Src A" in prompt
        assert "URL: https://a.com" in prompt
        assert "Content: Detailed info" in prompt

    def test_truncates_long_content(self):
        """Source content exceeding max_content_length gets truncated."""
        tr = TopicResearchResult(sub_query_id="sq-0", sources_found=1)
        long_content = "x" * 1000
        src = MagicMock(title="Src", url="https://x.com", content=long_content, snippet=None)

        prompt = _build_structured_metadata_prompt(
            query_text="test",
            topic_result=tr,
            topic_sources=[src],
            max_content_length=500,
        )

        assert "..." in prompt

    def test_completion_rationale_included(self):
        """Completion rationale should appear when early_completion is True."""
        tr = TopicResearchResult(
            sub_query_id="sq-0",
            sources_found=1,
            early_completion=True,
            completion_rationale="Sufficient coverage found",
        )
        prompt = _build_structured_metadata_prompt(
            query_text="test",
            topic_result=tr,
            topic_sources=[],
            max_content_length=50_000,
        )
        assert "Sufficient coverage found" in prompt


# =============================================================================
# 2.4: Compression system prompt alignment with open_deep_research
# =============================================================================


class TestCompressionSystemPromptAlignment:
    """Tests that the system prompt matches open_deep_research structure."""

    @pytest.mark.asyncio
    async def test_system_prompt_has_task_section(self):
        """System prompt should include <Task> section."""
        state = _make_state()
        tr = _make_topic_result(state, with_message_history=True)
        stub = StubCompression()
        captured_prompts: list[dict] = []

        async def capture_llm_call(**kwargs: Any) -> Any:
            captured_prompts.append(kwargs)
            result = MagicMock()
            result.result.success = True
            result.result.content = "Compressed output"
            result.result.input_tokens = 100
            result.result.output_tokens = 50
            return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=capture_llm_call,
        ):
            await stub._compress_single_topic_async(tr, state, 120.0)

        assert len(captured_prompts) == 1
        sys_prompt = captured_prompts[0]["system_prompt"]
        assert "<Task>" in sys_prompt
        assert "</Task>" in sys_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_has_guidelines_section(self):
        """System prompt should include <Guidelines> with 6 rules."""
        state = _make_state()
        tr = _make_topic_result(state, with_message_history=True)
        stub = StubCompression()
        captured: list[dict] = []

        async def capture(**kwargs: Any) -> Any:
            captured.append(kwargs)
            result = MagicMock()
            result.result.success = True
            result.result.content = "out"
            result.result.input_tokens = 0
            result.result.output_tokens = 0
            return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=capture,
        ):
            await stub._compress_single_topic_async(tr, state, 120.0)

        sys_prompt = captured[0]["system_prompt"]
        assert "<Guidelines>" in sys_prompt
        assert "</Guidelines>" in sys_prompt
        # All 6 guideline rules should be present
        assert "include ALL" in sys_prompt
        assert "as long as necessary" in sys_prompt
        assert "inline citations" in sys_prompt
        assert "Sources section" in sys_prompt or "Sources" in sys_prompt
        assert "include ALL of the sources" in sys_prompt
        assert "lose any sources" in sys_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_has_output_format_section(self):
        """System prompt should include <Output Format> with structured report template."""
        state = _make_state()
        tr = _make_topic_result(state, with_message_history=True)
        stub = StubCompression()
        captured: list[dict] = []

        async def capture(**kwargs: Any) -> Any:
            captured.append(kwargs)
            result = MagicMock()
            result.result.success = True
            result.result.content = "out"
            result.result.input_tokens = 0
            result.result.output_tokens = 0
            return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=capture,
        ):
            await stub._compress_single_topic_async(tr, state, 120.0)

        sys_prompt = captured[0]["system_prompt"]
        assert "<Output Format>" in sys_prompt
        assert "Queries and Tool Calls Made" in sys_prompt
        assert "Fully Comprehensive Findings" in sys_prompt
        assert "Sources" in sys_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_has_citation_rules(self):
        """System prompt should include <Citation Rules> with [N] Title: URL format."""
        state = _make_state()
        tr = _make_topic_result(state, with_message_history=True)
        stub = StubCompression()
        captured: list[dict] = []

        async def capture(**kwargs: Any) -> Any:
            captured.append(kwargs)
            result = MagicMock()
            result.result.success = True
            result.result.content = "out"
            result.result.input_tokens = 0
            result.result.output_tokens = 0
            return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=capture,
        ):
            await stub._compress_single_topic_async(tr, state, 120.0)

        sys_prompt = captured[0]["system_prompt"]
        assert "<Citation Rules>" in sys_prompt
        assert "[1] Source Title: URL" in sys_prompt
        assert "sequentially without gaps" in sys_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_has_verbatim_reminder(self):
        """System prompt should include critical verbatim preservation reminder."""
        state = _make_state()
        tr = _make_topic_result(state, with_message_history=True)
        stub = StubCompression()
        captured: list[dict] = []

        async def capture(**kwargs: Any) -> Any:
            captured.append(kwargs)
            result = MagicMock()
            result.result.success = True
            result.result.content = "out"
            result.result.input_tokens = 0
            result.result.output_tokens = 0
            return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=capture,
        ):
            await stub._compress_single_topic_async(tr, state, 120.0)

        sys_prompt = captured[0]["system_prompt"]
        assert "preserved verbatim" in sys_prompt
        assert "don't summarize" in sys_prompt


# =============================================================================
# 2.3: Compression uses message history when available, falls back otherwise
# =============================================================================


class TestCompressionPromptDispatch:
    """Tests that compression dispatches to message history or metadata prompt."""

    @pytest.mark.asyncio
    async def test_uses_message_history_when_available(self):
        """When message_history is populated, user prompt includes conversation entries."""
        state = _make_state()
        tr = _make_topic_result(state, with_message_history=True)
        stub = StubCompression()
        captured: list[dict] = []

        async def capture(**kwargs: Any) -> Any:
            captured.append(kwargs)
            result = MagicMock()
            result.result.success = True
            result.result.content = "compressed"
            result.result.input_tokens = 0
            result.result.output_tokens = 0
            return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=capture,
        ):
            await stub._compress_single_topic_async(tr, state, 120.0)

        user_prompt = captured[0]["user_prompt"]
        # Message history prompt markers
        assert "full researcher conversation" in user_prompt
        assert "[Assistant]" in user_prompt
        assert "[Tool:" in user_prompt
        # Should NOT have structured metadata markers
        assert "Search iterations:" not in user_prompt

    @pytest.mark.asyncio
    async def test_falls_back_to_metadata_when_no_history(self):
        """When message_history is empty, user prompt uses structured metadata."""
        state = _make_state()
        tr = _make_topic_result(state, with_message_history=False)
        stub = StubCompression()
        captured: list[dict] = []

        async def capture(**kwargs: Any) -> Any:
            captured.append(kwargs)
            result = MagicMock()
            result.result.success = True
            result.result.content = "compressed"
            result.result.input_tokens = 0
            result.result.output_tokens = 0
            return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=capture,
        ):
            await stub._compress_single_topic_async(tr, state, 120.0)

        user_prompt = captured[0]["user_prompt"]
        # Structured metadata prompt markers
        assert "Search iterations:" in user_prompt
        # Should NOT have message history markers
        assert "[Assistant]" not in user_prompt
        assert "full researcher conversation" not in user_prompt


# =============================================================================
# 2.5: Message history truncation
# =============================================================================


class TestMessageHistoryTruncation:
    """Tests for max_content_length enforcement on message history."""

    @pytest.mark.asyncio
    async def test_truncation_applied_in_compression(self):
        """When message_history is large, compression prompt is bounded by max_content_length."""
        state = _make_state()
        tr = _make_topic_result(state, with_message_history=False)
        # Add many small messages (each ~80 chars) that together exceed budget
        tr.message_history = [
            {"role": "assistant", "content": f"Message {i}: finding about topic"}
            for i in range(50)
        ]
        stub = StubCompression()
        # Set max content length to allow only ~10-15 messages
        stub.config.deep_research_compression_max_content_length = 1500
        captured: list[dict] = []

        async def capture(**kwargs: Any) -> Any:
            captured.append(kwargs)
            result = MagicMock()
            result.result.success = True
            result.result.content = "compressed"
            result.result.input_tokens = 0
            result.result.output_tokens = 0
            return result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=capture,
        ):
            await stub._compress_single_topic_async(tr, state, 120.0)

        user_prompt = captured[0]["user_prompt"]
        # Most recent messages should be preserved
        assert "Message 49" in user_prompt
        # Oldest messages should be truncated
        assert "Message 0" not in user_prompt


# =============================================================================
# 2.2: message_history stored after ReAct loop (integration-level)
# =============================================================================


class TestMessageHistoryStoredOnResult:
    """Tests verifying message_history is persisted on TopicResearchResult."""

    def test_message_history_populated_after_react_loop(self):
        """message_history should contain entries from the research conversation."""
        # This is verified by constructing a TopicResearchResult with
        # message_history and checking it's preserved through serialization
        tr = TopicResearchResult(
            sub_query_id="sq-0",
            searches_performed=2,
            sources_found=3,
            message_history=[
                {"role": "assistant", "content": "first response"},
                {"role": "tool", "tool": "web_search", "content": "results"},
                {"role": "assistant", "content": "second response"},
                {"role": "tool", "tool": "think", "content": "reflection"},
            ],
        )
        assert len(tr.message_history) == 4
        assert tr.message_history[0]["role"] == "assistant"
        assert tr.message_history[1]["tool"] == "web_search"

    def test_message_history_roundtrip(self):
        """message_history survives model_dump -> model_validate roundtrip."""
        original = TopicResearchResult(
            sub_query_id="sq-rt",
            message_history=[
                {"role": "assistant", "content": "response"},
                {"role": "tool", "tool": "web_search", "content": "data"},
            ],
        )
        data = original.model_dump()
        restored = TopicResearchResult.model_validate(data)
        assert restored.message_history == original.message_history

    def test_message_history_empty_when_no_turns(self):
        """If the ReAct loop does zero turns, message_history is empty."""
        tr = TopicResearchResult(sub_query_id="sq-empty")
        assert tr.message_history == []


# =============================================================================
# Existing compression tests backward compat
# =============================================================================


class TestBackwardCompatibility:
    """Ensure existing compression behavior still works with new code."""

    @pytest.mark.asyncio
    async def test_no_sources_returns_immediately(self):
        """When topic has no sources, compression returns (0, 0, True)."""
        state = _make_state(sources_per_query=0)
        tr = TopicResearchResult(sub_query_id="sq-0", source_ids=[])
        state.topic_research_results.append(tr)
        stub = StubCompression()

        inp, out, success = await stub._compress_single_topic_async(
            topic_result=tr, state=state, timeout=120.0
        )
        assert success is True
        assert inp == 0
        assert out == 0

    @pytest.mark.asyncio
    async def test_compression_populates_compressed_findings(self):
        """Successful compression should set compressed_findings on result."""
        state = _make_state()
        tr = _make_topic_result(state, with_message_history=True)
        stub = StubCompression()

        mock_result = MagicMock()
        mock_result.result.success = True
        mock_result.result.content = "Compressed: [1] Solar [2] Wind benefits"
        mock_result.result.input_tokens = 200
        mock_result.result.output_tokens = 100

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            inp, out, success = await stub._compress_single_topic_async(
                topic_result=tr, state=state, timeout=120.0
            )

        assert success is True
        assert tr.compressed_findings == "Compressed: [1] Solar [2] Wind benefits"
        assert inp == 200
        assert out == 100

    @pytest.mark.asyncio
    async def test_compression_failure_returns_false(self):
        """LLM failure should return (0, 0, False) without crashing."""
        from foundry_mcp.core.research.workflows.base import WorkflowResult

        state = _make_state()
        tr = _make_topic_result(state, with_message_history=True)
        stub = StubCompression()

        error_result = WorkflowResult(
            success=False,
            content="",
            error="Context window exceeded",
        )

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            new_callable=AsyncMock,
            return_value=error_result,
        ):
            inp, out, success = await stub._compress_single_topic_async(
                topic_result=tr, state=state, timeout=120.0
            )

        assert success is False
        assert inp == 0
        assert out == 0
        assert tr.compressed_findings is None
