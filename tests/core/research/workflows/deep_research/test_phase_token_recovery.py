"""Tests for PLAN Phase 4: Message-Aware Token Limit Recovery.

Covers:
- 4.1: truncate_prompt_for_retry() helper (boundary cases, min threshold,
       progressive truncation 20% → 30% → 40%)
- 4.2: Compression outer retry loop on simulated token-limit errors
- 4.3: Synthesis outer retry loop on simulated token-limit errors
- 4.4: Provider-specific error detection for OpenAI, Anthropic, Google
- 4.5: System prompt never truncated, most recent content preserved,
       max 3 retries then fallback, non-token-limit errors NOT retried,
       retry metadata recorded in audit events
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.core.errors.provider import ContextWindowError
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
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    MAX_PHASE_TOKEN_RETRIES,
    LLMCallResult,
    _is_context_window_error,
    _is_context_window_exceeded,
    truncate_prompt_for_retry,
)
from foundry_mcp.core.research.workflows.deep_research.phases.compression import (
    CompressionMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases.synthesis import (
    SynthesisPhaseMixin,
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
        id="deepres-token-recovery",
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
        # Generate enough message history to exceed 1000 chars (truncation minimum)
        tr.message_history = [
            {"role": "assistant", "content": "Search for renewable energy benefits " + "context " * 30},
            {"role": "tool", "tool": "web_search", "content": "Found 3 sources about solar energy and its benefits to the environment and economy. " * 5},
            {"role": "assistant", "content": "Good results on solar, now searching for wind energy advantages and applications " + "reasoning " * 20},
            {"role": "tool", "tool": "web_search", "content": "Found 2 sources about wind power including offshore and onshore installations. " * 5},
            {"role": "assistant", "content": "I need to verify some contradicting data about energy costs " + "analysis " * 20},
            {"role": "tool", "tool": "think", "content": "Reflection: Sources agree on environmental benefits but disagree on cost effectiveness. " * 3},
            {"role": "assistant", "content": "Comprehensive coverage achieved across solar, wind, and cost analysis " + "summary " * 15},
            {"role": "tool", "tool": "research_complete", "content": "Research complete. Findings recorded with high confidence. " * 3},
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
        self.config.audit_verbosity = "standard"
        self.memory = MagicMock()
        self._audit_events: list[tuple[str, dict]] = []
        self._cancelled = False

    def _write_audit_event(self, state: Any, event: str, **kwargs: Any) -> None:
        self._audit_events.append((event, kwargs))

    def _check_cancellation(self, state: Any) -> None:
        if self._cancelled:
            raise asyncio.CancelledError()


class StubSynthesis(SynthesisPhaseMixin):
    """Concrete class for testing SynthesisPhaseMixin in isolation."""

    def __init__(self) -> None:
        self.config = MagicMock()
        self.config.default_provider = "test-provider"
        self.config.resolve_model_for_role = MagicMock(
            return_value=("test-provider", None)
        )
        self.config.get_phase_fallback_providers = MagicMock(return_value=[])
        self.config.deep_research_max_retries = 1
        self.config.deep_research_retry_delay = 0.1
        self.config.audit_verbosity = "standard"
        self.memory = MagicMock()
        self._audit_events: list[tuple[str, dict]] = []
        self._cancelled = False

    def _write_audit_event(self, state: Any, event: str, **kwargs: Any) -> None:
        self._audit_events.append((event, kwargs))

    def _check_cancellation(self, state: Any) -> None:
        if self._cancelled:
            raise asyncio.CancelledError()


def _make_context_window_error_result(**extra_metadata: Any) -> WorkflowResult:
    """Build a WorkflowResult that simulates context window exhaustion."""
    metadata = {
        "error_type": "context_window_exceeded",
        "phase": "compression",
        "prompt_tokens": 200_000,
        "max_tokens": 128_000,
        "token_limit_retries": 3,
    }
    metadata.update(extra_metadata)
    return WorkflowResult(
        success=False,
        content="",
        error="Context window exceeded",
        metadata=metadata,
    )


def _make_success_llm_result(content: str = "Compressed output") -> LLMCallResult:
    """Build a successful LLMCallResult."""
    result = WorkflowResult(
        success=True,
        content=content,
        provider_id="test-provider",
        model_used="test-model",
        tokens_used=150,
        input_tokens=100,
        output_tokens=50,
        duration_ms=200.0,
    )
    return LLMCallResult(result=result, llm_call_duration_ms=200.0)


def _make_non_token_error_result() -> WorkflowResult:
    """Build a WorkflowResult for a non-token-limit error (e.g., timeout)."""
    return WorkflowResult(
        success=False,
        content="",
        error="Request timed out",
        metadata={"error_type": "timeout", "timeout": True},
    )


# =============================================================================
# 4.1: truncate_prompt_for_retry() unit tests
# =============================================================================


class TestTruncatePromptForRetry:
    """Tests for the truncate_prompt_for_retry helper function."""

    def test_attempt_1_removes_20_percent(self):
        """Attempt 1 should remove the first 20% of content."""
        prompt = "A" * 10_000
        result = truncate_prompt_for_retry(prompt, attempt=1)
        expected_len = 8_000  # 80% of 10_000
        assert len(result) == expected_len
        # Should be the tail of the original
        assert result == prompt[2_000:]

    def test_attempt_2_removes_30_percent(self):
        """Attempt 2 should remove the first 30% of content."""
        prompt = "A" * 10_000
        result = truncate_prompt_for_retry(prompt, attempt=2)
        expected_len = 7_000  # 70% of 10_000
        assert len(result) == expected_len
        assert result == prompt[3_000:]

    def test_attempt_3_removes_40_percent(self):
        """Attempt 3 should remove the first 40% of content."""
        prompt = "A" * 10_000
        result = truncate_prompt_for_retry(prompt, attempt=3)
        expected_len = 6_000  # 60% of 10_000
        assert len(result) == expected_len
        assert result == prompt[4_000:]

    def test_preserves_most_recent_content(self):
        """The tail (most recent content) should always be preserved."""
        prompt = "OLD_CONTENT_" * 500 + "RECENT_IMPORTANT_DATA"
        result = truncate_prompt_for_retry(prompt, attempt=1)
        assert result.endswith("RECENT_IMPORTANT_DATA")

    def test_never_truncates_below_minimum(self):
        """Should never produce output below 1000 characters."""
        prompt = "A" * 1_500
        result = truncate_prompt_for_retry(prompt, attempt=3)
        # 40% removal of 1500 = 900 chars → below minimum → clamp to 1000
        assert len(result) == 1_000
        # Should be the last 1000 chars
        assert result == prompt[-1_000:]

    def test_short_prompt_unchanged(self):
        """Prompts at or below 1000 chars are never truncated."""
        prompt = "A" * 1_000
        result = truncate_prompt_for_retry(prompt, attempt=3)
        assert result == prompt

    def test_very_short_prompt_unchanged(self):
        """Very short prompts are returned as-is."""
        prompt = "Short prompt"
        result = truncate_prompt_for_retry(prompt, attempt=1)
        assert result == prompt

    def test_attempt_0_returns_unchanged(self):
        """Attempt 0 (invalid) returns the original prompt."""
        prompt = "A" * 5_000
        result = truncate_prompt_for_retry(prompt, attempt=0)
        assert result == prompt

    def test_attempt_above_max_returns_unchanged(self):
        """Attempts beyond max_attempts return the original prompt."""
        prompt = "A" * 5_000
        result = truncate_prompt_for_retry(prompt, attempt=4, max_attempts=3)
        assert result == prompt

    def test_negative_attempt_returns_unchanged(self):
        """Negative attempt values return the original prompt."""
        prompt = "A" * 5_000
        result = truncate_prompt_for_retry(prompt, attempt=-1)
        assert result == prompt

    def test_custom_max_attempts(self):
        """Custom max_attempts should be respected."""
        prompt = "A" * 10_000
        # With max_attempts=5, attempt 5 → removal_pct = 0.1 + 5*0.1 = 0.6
        result = truncate_prompt_for_retry(prompt, attempt=5, max_attempts=5)
        expected_len = 4_000  # 40% of 10_000
        assert len(result) == expected_len

    def test_empty_prompt_returns_empty(self):
        """Empty prompt is returned as-is."""
        result = truncate_prompt_for_retry("", attempt=1)
        assert result == ""

    def test_progressive_truncation_is_monotonic(self):
        """Each successive attempt should produce a shorter result."""
        prompt = "A" * 10_000
        lengths = [
            len(truncate_prompt_for_retry(prompt, attempt=i))
            for i in range(1, 4)
        ]
        assert lengths[0] > lengths[1] > lengths[2]


# =============================================================================
# 4.1: _is_context_window_exceeded helper
# =============================================================================


class TestIsContextWindowExceeded:
    """Tests for the _is_context_window_exceeded helper."""

    def test_detects_context_window_error_type(self):
        """Should return True when metadata has error_type=context_window_exceeded."""
        result = _make_context_window_error_result()
        assert _is_context_window_exceeded(result) is True

    def test_returns_false_for_other_errors(self):
        """Should return False for non-context-window errors."""
        result = _make_non_token_error_result()
        assert _is_context_window_exceeded(result) is False

    def test_returns_false_for_success_result(self):
        """Should return False for successful results."""
        result = WorkflowResult(success=True, content="ok")
        assert _is_context_window_exceeded(result) is False

    def test_returns_false_for_no_metadata(self):
        """Should return False when metadata is None."""
        result = WorkflowResult(success=False, content="", error="err")
        assert _is_context_window_exceeded(result) is False


# =============================================================================
# 4.2: Compression retry loop
# =============================================================================


class TestCompressionRetry:
    """Tests for the outer retry loop in _compress_single_topic_async."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """Compression succeeds on first attempt — no retry needed."""
        stub = StubCompression()
        state = _make_state()
        tr = _make_topic_result(state)

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=[_make_success_llm_result()],
        ):
            inp, out, success = await stub._compress_single_topic_async(tr, state, 120.0)

        assert success is True
        assert inp == 100
        assert out == 50
        assert tr.compressed_findings == "Compressed output"
        # No retry audit events
        retry_events = [e for e, _ in stub._audit_events if "retry" in e]
        assert len(retry_events) == 0

    @pytest.mark.asyncio
    async def test_retries_on_context_window_error(self):
        """Compression should retry with truncated prompt on context window error."""
        stub = StubCompression()
        state = _make_state()
        tr = _make_topic_result(state)

        call_results = [
            _make_context_window_error_result(),  # First attempt fails
            _make_success_llm_result("Compressed after retry"),  # Retry succeeds
        ]

        captured_prompts: list[str] = []

        async def capture_llm_call(**kwargs: Any) -> Any:
            captured_prompts.append(kwargs.get("user_prompt", ""))
            return call_results.pop(0)

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=capture_llm_call,
        ):
            inp, out, success = await stub._compress_single_topic_async(tr, state, 120.0)

        assert success is True
        assert tr.compressed_findings == "Compressed after retry"
        # Second prompt should be shorter (truncated)
        assert len(captured_prompts) == 2
        assert len(captured_prompts[1]) < len(captured_prompts[0])

    @pytest.mark.asyncio
    async def test_records_retry_audit_event_on_success(self):
        """Should record compression_retry_succeeded audit event."""
        stub = StubCompression()
        state = _make_state()
        tr = _make_topic_result(state)

        call_results = [
            _make_context_window_error_result(),
            _make_success_llm_result(),
        ]

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=call_results,
        ):
            await stub._compress_single_topic_async(tr, state, 120.0)

        retry_events = [e for e, _ in stub._audit_events if e == "compression_retry_succeeded"]
        assert len(retry_events) == 1

    @pytest.mark.asyncio
    async def test_exhausts_retries_returns_failure(self):
        """After max retries, should return (0, 0, False)."""
        stub = StubCompression()
        state = _make_state()
        tr = _make_topic_result(state)

        # All attempts fail with context window error
        call_results = [_make_context_window_error_result()] * (MAX_PHASE_TOKEN_RETRIES + 1)

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=call_results,
        ):
            inp, out, success = await stub._compress_single_topic_async(tr, state, 120.0)

        assert success is False
        assert inp == 0
        assert out == 0
        assert tr.compressed_findings is None

    @pytest.mark.asyncio
    async def test_records_exhausted_audit_event(self):
        """Should record compression_retry_exhausted on final failure."""
        stub = StubCompression()
        state = _make_state()
        tr = _make_topic_result(state)

        call_results = [_make_context_window_error_result()] * (MAX_PHASE_TOKEN_RETRIES + 1)

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=call_results,
        ):
            await stub._compress_single_topic_async(tr, state, 120.0)

        exhausted_events = [e for e, _ in stub._audit_events if e == "compression_retry_exhausted"]
        assert len(exhausted_events) == 1

    @pytest.mark.asyncio
    async def test_non_token_error_not_retried(self):
        """Non-context-window errors should NOT trigger outer retry."""
        stub = StubCompression()
        state = _make_state()
        tr = _make_topic_result(state)

        call_count = 0

        async def count_calls(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return _make_non_token_error_result()

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=count_calls,
        ):
            inp, out, success = await stub._compress_single_topic_async(tr, state, 120.0)

        assert success is False
        # Only called once — no retry for non-token errors
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_progressive_truncation_applied(self):
        """Each retry should apply progressively more truncation."""
        stub = StubCompression()
        state = _make_state()
        tr = _make_topic_result(state)

        captured_prompts: list[str] = []

        async def capture_and_fail(**kwargs: Any) -> Any:
            captured_prompts.append(kwargs.get("user_prompt", ""))
            if len(captured_prompts) <= MAX_PHASE_TOKEN_RETRIES:
                return _make_context_window_error_result()
            return _make_success_llm_result()

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=capture_and_fail,
        ):
            await stub._compress_single_topic_async(tr, state, 120.0)

        # Should have original + N retries with progressively shorter prompts
        assert len(captured_prompts) >= 2
        for i in range(1, len(captured_prompts)):
            assert len(captured_prompts[i]) <= len(captured_prompts[i - 1])


# =============================================================================
# 4.3: Synthesis retry loop
# =============================================================================


_SYNTHESIS_LLM_PATCH = "foundry_mcp.core.research.workflows.deep_research.phases.synthesis.execute_llm_call"
_SYNTHESIS_BUDGET_PATCH = "foundry_mcp.core.research.workflows.deep_research.phases.synthesis.allocate_synthesis_budget"
_SYNTHESIS_FIT_PATCH = "foundry_mcp.core.research.workflows.deep_research.phases.synthesis.final_fit_validate"
_SYNTHESIS_CITE_PATCH = "foundry_mcp.core.research.workflows.deep_research.phases.synthesis.postprocess_citations"


class TestSynthesisRetry:
    """Tests for the outer retry loop in _execute_synthesis_async."""

    def _make_synthesis_state(self) -> DeepResearchState:
        """Create a state suitable for synthesis testing."""
        state = _make_state(phase=DeepResearchPhase.SYNTHESIS)
        tr = _make_topic_result(state)
        tr.compressed_findings = "Compressed findings for testing"
        return state

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """Synthesis succeeds on first attempt — no retry."""
        stub = StubSynthesis()
        state = self._make_synthesis_state()

        with patch(
            _SYNTHESIS_LLM_PATCH,
            side_effect=[_make_success_llm_result("# Report\n\nFindings here")],
        ), patch(
            _SYNTHESIS_BUDGET_PATCH,
        ) as mock_budget, patch(
            _SYNTHESIS_FIT_PATCH,
        ) as mock_fit, patch(
            _SYNTHESIS_CITE_PATCH,
            return_value=("# Report\n\nFindings here", {"citations_processed": 0}),
        ):
            mock_budget.return_value = MagicMock(
                items=[], dropped_ids=[], fidelity=1.0, to_dict=lambda: {"fidelity": 1.0}
            )
            mock_fit.return_value = (True, None, "sys prompt", "user prompt")

            result = await stub._execute_synthesis_async(state, "test-provider", 120.0)

        assert result.success is True
        assert state.report is not None
        retry_events = [e for e, _ in stub._audit_events if "retry" in e]
        assert len(retry_events) == 0

    @pytest.mark.asyncio
    async def test_retries_on_context_window_error(self):
        """Synthesis should retry with truncated prompt on context window error."""
        stub = StubSynthesis()
        state = self._make_synthesis_state()

        call_results = [
            _make_context_window_error_result(phase="synthesis"),
            _make_success_llm_result("# Report\n\nRetried findings"),
        ]

        captured_prompts: list[str] = []

        async def capture_llm_call(**kwargs: Any) -> Any:
            captured_prompts.append(kwargs.get("user_prompt", ""))
            return call_results.pop(0)

        with patch(
            _SYNTHESIS_LLM_PATCH,
            side_effect=capture_llm_call,
        ), patch(
            _SYNTHESIS_BUDGET_PATCH,
        ) as mock_budget, patch(
            _SYNTHESIS_FIT_PATCH,
        ) as mock_fit, patch(
            _SYNTHESIS_CITE_PATCH,
            return_value=("# Report\n\nRetried findings", {"citations_processed": 0}),
        ):
            mock_budget.return_value = MagicMock(
                items=[], dropped_ids=[], fidelity=1.0, to_dict=lambda: {"fidelity": 1.0}
            )
            mock_fit.return_value = (True, None, "sys prompt", "user prompt " + "x" * 2000)

            result = await stub._execute_synthesis_async(state, "test-provider", 120.0)

        assert result.success is True
        assert len(captured_prompts) == 2
        # Second prompt should be truncated (shorter)
        assert len(captured_prompts[1]) < len(captured_prompts[0])

    @pytest.mark.asyncio
    async def test_records_retry_succeeded_audit(self):
        """Should record synthesis_retry_succeeded audit event."""
        stub = StubSynthesis()
        state = self._make_synthesis_state()

        call_results = [
            _make_context_window_error_result(phase="synthesis"),
            _make_success_llm_result("# Report"),
        ]

        with patch(
            _SYNTHESIS_LLM_PATCH,
            side_effect=call_results,
        ), patch(
            _SYNTHESIS_BUDGET_PATCH,
        ) as mock_budget, patch(
            _SYNTHESIS_FIT_PATCH,
        ) as mock_fit, patch(
            _SYNTHESIS_CITE_PATCH,
            return_value=("# Report", {}),
        ):
            mock_budget.return_value = MagicMock(
                items=[], dropped_ids=[], fidelity=1.0, to_dict=lambda: {"fidelity": 1.0}
            )
            mock_fit.return_value = (True, None, "sys", "usr " + "x" * 2000)

            await stub._execute_synthesis_async(state, "test-provider", 120.0)

        retry_events = [e for e, _ in stub._audit_events if e == "synthesis_retry_succeeded"]
        assert len(retry_events) == 1

    @pytest.mark.asyncio
    async def test_exhausts_retries_returns_error(self):
        """After max retries, should return the error WorkflowResult."""
        stub = StubSynthesis()
        state = self._make_synthesis_state()

        call_results = [
            _make_context_window_error_result(phase="synthesis")
        ] * (MAX_PHASE_TOKEN_RETRIES + 1)

        with patch(
            _SYNTHESIS_LLM_PATCH,
            side_effect=call_results,
        ), patch(
            _SYNTHESIS_BUDGET_PATCH,
        ) as mock_budget, patch(
            _SYNTHESIS_FIT_PATCH,
        ) as mock_fit:
            mock_budget.return_value = MagicMock(
                items=[], dropped_ids=[], fidelity=1.0, to_dict=lambda: {"fidelity": 1.0}
            )
            mock_fit.return_value = (True, None, "sys", "usr " + "x" * 2000)

            result = await stub._execute_synthesis_async(state, "test-provider", 120.0)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_records_exhausted_audit_event(self):
        """Should record synthesis_retry_exhausted on final failure."""
        stub = StubSynthesis()
        state = self._make_synthesis_state()

        call_results = [
            _make_context_window_error_result(phase="synthesis")
        ] * (MAX_PHASE_TOKEN_RETRIES + 1)

        with patch(
            _SYNTHESIS_LLM_PATCH,
            side_effect=call_results,
        ), patch(
            _SYNTHESIS_BUDGET_PATCH,
        ) as mock_budget, patch(
            _SYNTHESIS_FIT_PATCH,
        ) as mock_fit:
            mock_budget.return_value = MagicMock(
                items=[], dropped_ids=[], fidelity=1.0, to_dict=lambda: {"fidelity": 1.0}
            )
            mock_fit.return_value = (True, None, "sys", "usr " + "x" * 2000)

            await stub._execute_synthesis_async(state, "test-provider", 120.0)

        exhausted_events = [e for e, _ in stub._audit_events if e == "synthesis_retry_exhausted"]
        assert len(exhausted_events) == 1

    @pytest.mark.asyncio
    async def test_non_token_error_not_retried(self):
        """Non-context-window errors should NOT trigger outer retry."""
        stub = StubSynthesis()
        state = self._make_synthesis_state()

        call_count = 0

        async def count_calls(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return _make_non_token_error_result()

        with patch(
            _SYNTHESIS_LLM_PATCH,
            side_effect=count_calls,
        ), patch(
            _SYNTHESIS_BUDGET_PATCH,
        ) as mock_budget, patch(
            _SYNTHESIS_FIT_PATCH,
        ) as mock_fit:
            mock_budget.return_value = MagicMock(
                items=[], dropped_ids=[], fidelity=1.0, to_dict=lambda: {"fidelity": 1.0}
            )
            mock_fit.return_value = (True, None, "sys", "usr")

            result = await stub._execute_synthesis_async(state, "test-provider", 120.0)

        assert result.success is False
        assert call_count == 1  # Only called once — no retry

    @pytest.mark.asyncio
    async def test_system_prompt_never_truncated(self):
        """The system prompt should remain unchanged across retries."""
        stub = StubSynthesis()
        state = self._make_synthesis_state()

        captured_system_prompts: list[str] = []

        call_results = [
            _make_context_window_error_result(phase="synthesis"),
            _make_success_llm_result("# Report"),
        ]

        async def capture_system(**kwargs: Any) -> Any:
            captured_system_prompts.append(kwargs.get("system_prompt", ""))
            return call_results.pop(0)

        with patch(
            _SYNTHESIS_LLM_PATCH,
            side_effect=capture_system,
        ), patch(
            _SYNTHESIS_BUDGET_PATCH,
        ) as mock_budget, patch(
            _SYNTHESIS_FIT_PATCH,
        ) as mock_fit, patch(
            _SYNTHESIS_CITE_PATCH,
            return_value=("# Report", {}),
        ):
            mock_budget.return_value = MagicMock(
                items=[], dropped_ids=[], fidelity=1.0, to_dict=lambda: {"fidelity": 1.0}
            )
            mock_fit.return_value = (True, None, "sys prompt", "user prompt " + "x" * 2000)

            await stub._execute_synthesis_async(state, "test-provider", 120.0)

        # System prompt should be identical across all attempts
        assert len(captured_system_prompts) == 2
        assert captured_system_prompts[0] == captured_system_prompts[1]


# =============================================================================
# 4.4: Provider-specific error detection
# =============================================================================


class TestProviderErrorDetection:
    """Tests for _is_context_window_error provider-specific detection."""

    def test_openai_maximum_context_length(self):
        """OpenAI: 'maximum context length' detected."""
        exc = Exception("This model's maximum context length is 128000 tokens")
        assert _is_context_window_error(exc) is True

    def test_openai_too_many_tokens(self):
        """OpenAI: 'too many tokens' detected."""
        exc = Exception("Request has too many tokens: 200000")
        assert _is_context_window_error(exc) is True

    def test_anthropic_prompt_too_long(self):
        """Anthropic: 'prompt is too long' detected."""
        exc = Exception("prompt is too long: 250000 tokens > 200000 maximum")
        assert _is_context_window_error(exc) is True

    def test_google_resource_exhausted_class(self):
        """Google: ResourceExhausted class name detected."""

        class ResourceExhausted(Exception):
            pass

        exc = ResourceExhausted("Quota exceeded")
        assert _is_context_window_error(exc) is True

    def test_google_invalid_argument_class(self):
        """Google: InvalidArgument class name detected."""

        class InvalidArgument(Exception):
            pass

        exc = InvalidArgument("Token limit exceeded")
        assert _is_context_window_error(exc) is True

    def test_google_token_limit_message(self):
        """Google: 'token limit' in message detected."""
        exc = Exception("Request exceeds token limit for this model")
        assert _is_context_window_error(exc) is True

    def test_generic_token_keyword(self):
        """Generic: message containing 'token' keyword detected."""
        exc = Exception("Error: token count exceeded for request")
        assert _is_context_window_error(exc) is True

    def test_generic_context_keyword(self):
        """Generic: message containing 'context' keyword detected."""
        exc = Exception("context length exceeded")
        assert _is_context_window_error(exc) is True

    def test_unrelated_error_not_detected(self):
        """Unrelated errors should NOT be classified as context-window."""
        exc = Exception("Authentication failed: invalid API key")
        assert _is_context_window_error(exc) is False

    def test_rate_limit_not_detected(self):
        """Rate limit errors should NOT be classified as context-window."""
        exc = Exception("Rate limit exceeded. Please retry after 60 seconds.")
        assert _is_context_window_error(exc) is False

    def test_network_error_not_detected(self):
        """Network errors should NOT be classified as context-window."""
        exc = Exception("Connection refused: could not reach API endpoint")
        assert _is_context_window_error(exc) is False

    def test_too_many_tokens_cross_provider(self):
        """Cross-provider: 'too many tokens' pattern matched."""
        exc = Exception("BadRequestError: Too many tokens in the input")
        assert _is_context_window_error(exc) is True


# =============================================================================
# 4.5: MAX_PHASE_TOKEN_RETRIES constant
# =============================================================================


class TestMaxPhaseTokenRetries:
    """Tests for the MAX_PHASE_TOKEN_RETRIES constant."""

    def test_value_is_3(self):
        """MAX_PHASE_TOKEN_RETRIES should be 3."""
        assert MAX_PHASE_TOKEN_RETRIES == 3
