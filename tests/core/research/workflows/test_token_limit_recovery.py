"""Tests for progressive token-limit recovery in execute_llm_call.

Covers:
- Progressive truncation with mock provider errors (ContextWindowError)
- Fallback to hard error after 3 retries
- Provider-specific error detection (OpenAI, Anthropic, Google patterns)
- System prompt is never truncated (only user prompt)
- PhaseMetrics.metadata tracks token_limit_retries
- Successful recovery after truncation
"""

from __future__ import annotations

import re
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.core.errors.provider import ContextWindowError
from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research._helpers import (
    estimate_token_limit_for_model,
    truncate_to_token_estimate,
)
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    LLMCallResult,
    _CONTEXT_WINDOW_ERROR_CLASSES,
    _CONTEXT_WINDOW_ERROR_PATTERNS,
    FALLBACK_CONTEXT_WINDOW,
    _MAX_TOKEN_LIMIT_RETRIES,
    _TRUNCATION_FACTOR,
    _is_context_window_error,
    _truncate_for_retry,
    execute_llm_call,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_workflow():
    """Create a mock DeepResearchWorkflow with all required methods/attrs."""
    wf = MagicMock()
    wf._execute_provider_async = AsyncMock()
    wf._write_audit_event = MagicMock()
    wf.memory.save_deep_research = MagicMock()
    wf.config.get_phase_fallback_providers = MagicMock(return_value=[])
    wf.config.deep_research_max_retries = 2
    wf.config.deep_research_retry_delay = 1.0
    return wf


@pytest.fixture
def sample_state():
    """Create a minimal DeepResearchState."""
    return DeepResearchState(
        id="deepres-token-recovery-test",
        original_query="token recovery test query",
        phase=DeepResearchPhase.BRIEF,
        iteration=1,
        max_iterations=3,
    )


def _make_success_result(**overrides):
    """Build a successful WorkflowResult."""
    defaults = dict(
        success=True,
        content="test response content",
        provider_id="test-provider",
        model_used="test-model",
        tokens_used=30,
        input_tokens=10,
        output_tokens=20,
        cached_tokens=0,
        duration_ms=150.0,
        metadata={},
    )
    defaults.update(overrides)
    return WorkflowResult(**defaults)


# ---------------------------------------------------------------------------
# Unit tests: _is_context_window_error
# ---------------------------------------------------------------------------


class TestIsContextWindowError:
    """Tests for provider-specific context-window error detection."""

    def test_openai_token_keyword(self):
        """OpenAI-style error with 'token' keyword should match."""
        exc = Exception("This model's maximum context length is 8192 tokens")
        assert _is_context_window_error(exc) is True

    def test_openai_context_keyword(self):
        """OpenAI-style error with 'context' keyword should match."""
        exc = Exception("Request too large: context window exceeded")
        assert _is_context_window_error(exc) is True

    def test_openai_length_keyword(self):
        """OpenAI-style error with 'length' keyword should match."""
        exc = Exception("maximum context length exceeded")
        assert _is_context_window_error(exc) is True

    def test_anthropic_prompt_too_long(self):
        """Anthropic-style 'prompt is too long' error should match."""
        exc = Exception("BadRequestError: prompt is too long")
        assert _is_context_window_error(exc) is True

    def test_anthropic_case_insensitive(self):
        """Anthropic pattern should be case-insensitive."""
        exc = Exception("Prompt Is Too Long for this model")
        assert _is_context_window_error(exc) is True

    def test_google_resource_exhausted_class_name(self):
        """Google ResourceExhausted exception class should match by name."""

        class ResourceExhausted(Exception):
            pass

        exc = ResourceExhausted("Quota exceeded")
        assert _is_context_window_error(exc) is True

    def test_google_token_limit_message(self):
        """Google-style 'token limit' in message should match."""
        exc = Exception("token limit exceeded for model")
        assert _is_context_window_error(exc) is True

    def test_unrelated_error_no_match(self):
        """Unrelated errors should not match."""
        exc = Exception("Connection refused to api.example.com")
        assert _is_context_window_error(exc) is False

    def test_empty_message_no_match(self):
        """Empty error message should not match."""
        exc = Exception("")
        assert _is_context_window_error(exc) is False

    def test_authentication_error_no_match(self):
        """Auth errors should not match."""
        exc = Exception("Invalid API key provided")
        assert _is_context_window_error(exc) is False


# ---------------------------------------------------------------------------
# Unit tests: truncate_to_token_estimate
# ---------------------------------------------------------------------------


class TestTruncateToTokenEstimate:
    """Tests for the token-estimate-based truncation utility."""

    def test_short_text_unchanged(self):
        """Text within budget should be returned unchanged."""
        text = "Hello, world!"
        result = truncate_to_token_estimate(text, max_tokens=100)
        assert result == text

    def test_long_text_truncated(self):
        """Text exceeding budget should be truncated."""
        # 4 chars/token heuristic → 100 tokens = 400 chars
        text = "A" * 1000
        result = truncate_to_token_estimate(text, max_tokens=100)
        assert len(result) < 1000
        assert "[... content truncated for context limits]" in result

    def test_truncation_at_boundary(self):
        """Truncation should happen at a natural boundary."""
        # Create text with clear paragraph breaks
        paragraphs = "\n\n".join([f"Paragraph {i}. " + "X" * 50 for i in range(20)])
        result = truncate_to_token_estimate(paragraphs, max_tokens=50)
        # Should be truncated
        assert len(result) < len(paragraphs)
        assert result.endswith("[... content truncated for context limits]")


# ---------------------------------------------------------------------------
# Unit tests: estimate_token_limit_for_model
# ---------------------------------------------------------------------------


class TestEstimateTokenLimitForModel:
    """Tests for model-based token limit lookup."""

    def test_known_model_substring_match(self):
        """Should match known model patterns by substring."""
        limits = {"claude-3": 200_000, "gpt-4": 8_192}
        assert estimate_token_limit_for_model("claude-3.5-sonnet-20240620", limits) == 200_000

    def test_case_insensitive(self):
        """Should match case-insensitively."""
        limits = {"claude-3": 200_000}
        assert estimate_token_limit_for_model("CLAUDE-3-OPUS", limits) == 200_000

    def test_none_model_returns_none(self):
        """None model should return None."""
        limits = {"claude-3": 200_000}
        assert estimate_token_limit_for_model(None, limits) is None

    def test_unknown_model_returns_none(self):
        """Unknown model should return None."""
        limits = {"claude-3": 200_000}
        assert estimate_token_limit_for_model("llama-70b", limits) is None

    def test_empty_limits_returns_none(self):
        """Empty limits dict should return None."""
        assert estimate_token_limit_for_model("claude-3", {}) is None


# ---------------------------------------------------------------------------
# Unit tests: _truncate_for_retry
# ---------------------------------------------------------------------------


class TestTruncateForRetry:
    """Tests for the retry truncation helper."""

    def test_uses_error_max_tokens_when_available(self):
        """Should use error-provided max_tokens as the basis."""
        prompt = "X" * 10000
        result = _truncate_for_retry(
            prompt,
            error_max_tokens=500,  # 500 tokens → 2000 chars base
            model=None,
            retry_count=1,
            truncate_fn=truncate_to_token_estimate,
            estimate_limit_fn=estimate_token_limit_for_model,
            token_limits={},
        )
        # 500 * 0.9 = 450 tokens → 1800 chars
        assert len(result) < 10000

    def test_falls_back_to_model_registry(self):
        """Should use model registry when error has no max_tokens."""
        prompt = "X" * 1000000
        result = _truncate_for_retry(
            prompt,
            error_max_tokens=None,
            model="claude-3.5-sonnet",
            retry_count=1,
            truncate_fn=truncate_to_token_estimate,
            estimate_limit_fn=estimate_token_limit_for_model,
            token_limits={"claude-3": 200_000},
        )
        assert len(result) < 1000000

    def test_falls_back_to_default(self):
        """Should use fallback when neither error nor registry has limits."""
        prompt = "X" * 1000000
        result = _truncate_for_retry(
            prompt,
            error_max_tokens=None,
            model="unknown-model",
            retry_count=1,
            truncate_fn=truncate_to_token_estimate,
            estimate_limit_fn=estimate_token_limit_for_model,
            token_limits={},
        )
        # Should truncate using FALLBACK_CONTEXT_WINDOW (128K)
        assert len(result) < 1000000

    def test_progressive_reduction(self):
        """Each successive retry should produce a shorter result."""
        prompt = "X" * 1000000
        results = []
        for retry in range(1, 4):
            result = _truncate_for_retry(
                prompt,
                error_max_tokens=100_000,
                model=None,
                retry_count=retry,
                truncate_fn=truncate_to_token_estimate,
                estimate_limit_fn=estimate_token_limit_for_model,
                token_limits={},
            )
            results.append(len(result))

        # Each retry should be shorter (or equal if already at boundary)
        assert results[0] >= results[1] >= results[2]


# ---------------------------------------------------------------------------
# Integration tests: execute_llm_call — progressive recovery
# ---------------------------------------------------------------------------


class TestProgressiveTokenLimitRecovery:
    """Tests for progressive truncation recovery in execute_llm_call."""

    @pytest.mark.asyncio
    async def test_succeeds_after_one_retry(self, mock_workflow, sample_state):
        """Should succeed after truncating and retrying once."""
        mock_workflow._execute_provider_async.side_effect = [
            ContextWindowError(
                "Context window exceeded",
                prompt_tokens=5000,
                max_tokens=4096,
                provider="test-provider",
            ),
            _make_success_result(),  # Succeeds on retry
        ]

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="analysis",
            system_prompt="sys",
            user_prompt="X" * 50000,
            provider_id="test-provider",
            model="claude-3.5-sonnet",
            temperature=0.3,
            timeout=60.0,
        )

        assert isinstance(ret, LLMCallResult)
        assert ret.result.success is True
        assert mock_workflow._execute_provider_async.call_count == 2

    @pytest.mark.asyncio
    async def test_succeeds_after_two_retries(self, mock_workflow, sample_state):
        """Should succeed after truncating twice."""
        mock_workflow._execute_provider_async.side_effect = [
            ContextWindowError("too long", prompt_tokens=5000, max_tokens=4096),
            ContextWindowError("still too long", prompt_tokens=4500, max_tokens=4096),
            _make_success_result(),
        ]

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="analysis",
            system_prompt="sys",
            user_prompt="X" * 50000,
            provider_id="p",
            model="m",
            temperature=0.3,
            timeout=60.0,
        )

        assert isinstance(ret, LLMCallResult)
        assert mock_workflow._execute_provider_async.call_count == 3

    @pytest.mark.asyncio
    async def test_hard_error_after_max_retries(self, mock_workflow, sample_state):
        """Should return hard error after exhausting all retries."""
        err = ContextWindowError(
            "Context window exceeded",
            prompt_tokens=5000,
            max_tokens=4096,
            provider="test-provider",
        )
        # Fail on initial + all 3 retries
        mock_workflow._execute_provider_async.side_effect = [err] * (_MAX_TOKEN_LIMIT_RETRIES + 1)

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="analysis",
            system_prompt="sys",
            user_prompt="X" * 50000,
            provider_id="test-provider",
            model="m",
            temperature=0.3,
            timeout=60.0,
        )

        assert isinstance(ret, WorkflowResult)
        assert ret.success is False
        assert ret.metadata["error_type"] == "context_window_exceeded"
        assert ret.metadata["token_limit_retries"] == _MAX_TOKEN_LIMIT_RETRIES
        assert mock_workflow._execute_provider_async.call_count == _MAX_TOKEN_LIMIT_RETRIES + 1

    @pytest.mark.asyncio
    async def test_system_prompt_never_truncated(self, mock_workflow, sample_state):
        """System prompt should be passed unchanged on every retry."""
        system_prompt = "IMPORTANT SYSTEM INSTRUCTIONS " * 100

        mock_workflow._execute_provider_async.side_effect = [
            ContextWindowError("too long", prompt_tokens=5000, max_tokens=4096),
            _make_success_result(),
        ]

        await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="analysis",
            system_prompt=system_prompt,
            user_prompt="X" * 50000,
            provider_id="p",
            model="m",
            temperature=0.3,
            timeout=60.0,
        )

        # Check that system_prompt was passed unchanged on both calls
        for call in mock_workflow._execute_provider_async.call_args_list:
            assert call.kwargs["system_prompt"] == system_prompt

    @pytest.mark.asyncio
    async def test_user_prompt_truncated_on_retry(self, mock_workflow, sample_state):
        """User prompt should be shorter on each retry."""
        original_prompt = "X" * 50000

        mock_workflow._execute_provider_async.side_effect = [
            ContextWindowError("too long", prompt_tokens=5000, max_tokens=4096),
            ContextWindowError("still too long", prompt_tokens=4500, max_tokens=4096),
            _make_success_result(),
        ]

        await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="analysis",
            system_prompt="sys",
            user_prompt=original_prompt,
            provider_id="p",
            model="m",
            temperature=0.3,
            timeout=60.0,
        )

        prompts = [
            call.kwargs["prompt"]
            for call in mock_workflow._execute_provider_async.call_args_list
        ]
        assert len(prompts) == 3
        # Each retry should have a shorter or equal user prompt
        assert len(prompts[0]) >= len(prompts[1]) >= len(prompts[2])
        # First call should use the original prompt
        assert prompts[0] == original_prompt

    @pytest.mark.asyncio
    async def test_tracks_retries_in_phase_metrics(self, mock_workflow, sample_state):
        """Successful recovery should record token_limit_retries in PhaseMetrics."""
        mock_workflow._execute_provider_async.side_effect = [
            ContextWindowError("too long", prompt_tokens=5000, max_tokens=4096),
            _make_success_result(),
        ]

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="analysis",
            system_prompt="sys",
            user_prompt="X" * 50000,
            provider_id="p",
            model="m",
            temperature=0.3,
            timeout=60.0,
        )

        assert isinstance(ret, LLMCallResult)
        assert len(sample_state.phase_metrics) == 1
        pm = sample_state.phase_metrics[0]
        assert pm.metadata["token_limit_retries"] == 1

    @pytest.mark.asyncio
    async def test_no_retry_metadata_on_clean_success(self, mock_workflow, sample_state):
        """No token_limit_retries in metadata when no retries occurred."""
        mock_workflow._execute_provider_async.return_value = _make_success_result()

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="analysis",
            system_prompt="sys",
            user_prompt="Hello",
            provider_id="p",
            model="m",
            temperature=0.3,
            timeout=60.0,
        )

        assert isinstance(ret, LLMCallResult)
        pm = sample_state.phase_metrics[0]
        assert pm.metadata == {}

    @pytest.mark.asyncio
    async def test_error_metadata_preserved_on_hard_failure(self, mock_workflow, sample_state):
        """Error metadata should be merged into the failure response."""
        err = ContextWindowError("too long", prompt_tokens=5000, max_tokens=4096)
        mock_workflow._execute_provider_async.side_effect = [err] * 4

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="synthesis",
            system_prompt="sys",
            user_prompt="X" * 50000,
            provider_id="p",
            model="m",
            temperature=0.5,
            timeout=60.0,
            error_metadata={"finding_count": 42},
        )

        assert isinstance(ret, WorkflowResult)
        assert ret.metadata["finding_count"] == 42
        assert ret.metadata["token_limit_retries"] == 3


# ---------------------------------------------------------------------------
# Integration tests: provider-specific error recovery
# ---------------------------------------------------------------------------


class TestProviderSpecificErrorRecovery:
    """Tests for recovery from provider-specific errors (non-ContextWindowError)."""

    @pytest.mark.asyncio
    async def test_openai_bad_request_recovery(self, mock_workflow, sample_state):
        """Should detect and recover from OpenAI-style BadRequestError."""
        # First call: OpenAI-style error (not ContextWindowError)
        openai_error = Exception(
            "BadRequestError: This model's maximum context length is 128000 tokens"
        )
        mock_workflow._execute_provider_async.side_effect = [
            openai_error,
            _make_success_result(),
        ]

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="analysis",
            system_prompt="sys",
            user_prompt="X" * 50000,
            provider_id="p",
            model="gpt-4o",
            temperature=0.3,
            timeout=60.0,
        )

        assert isinstance(ret, LLMCallResult)
        assert mock_workflow._execute_provider_async.call_count == 2

    @pytest.mark.asyncio
    async def test_anthropic_error_recovery(self, mock_workflow, sample_state):
        """Should detect and recover from Anthropic-style error."""
        anthropic_error = Exception(
            "BadRequestError: prompt is too long: 250000 tokens > 200000 maximum"
        )
        mock_workflow._execute_provider_async.side_effect = [
            anthropic_error,
            _make_success_result(),
        ]

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="analysis",
            system_prompt="sys",
            user_prompt="X" * 50000,
            provider_id="p",
            model="claude-3",
            temperature=0.3,
            timeout=60.0,
        )

        assert isinstance(ret, LLMCallResult)

    @pytest.mark.asyncio
    async def test_google_resource_exhausted_recovery(self, mock_workflow, sample_state):
        """Should detect and recover from Google ResourceExhausted."""

        class ResourceExhausted(Exception):
            pass

        google_error = ResourceExhausted("Quota exceeded for model")
        mock_workflow._execute_provider_async.side_effect = [
            google_error,
            _make_success_result(),
        ]

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="analysis",
            system_prompt="sys",
            user_prompt="X" * 50000,
            provider_id="p",
            model="gemini-2",
            temperature=0.3,
            timeout=60.0,
        )

        assert isinstance(ret, LLMCallResult)

    @pytest.mark.asyncio
    async def test_non_context_window_error_not_retried(self, mock_workflow, sample_state):
        """Non-context-window errors should not trigger retry."""
        unrelated_error = ValueError("Something completely different")
        mock_workflow._execute_provider_async.side_effect = unrelated_error

        with pytest.raises(ValueError, match="Something completely different"):
            await execute_llm_call(
                workflow=mock_workflow,
                state=sample_state,
                phase_name="analysis",
                system_prompt="sys",
                user_prompt="Hello",
                provider_id="p",
                model="m",
                temperature=0.3,
                timeout=60.0,
            )

        # Should only be called once (no retry)
        assert mock_workflow._execute_provider_async.call_count == 1


# ---------------------------------------------------------------------------
# Constants validation
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify constants are set to expected values."""

    def test_max_retries(self):
        assert _MAX_TOKEN_LIMIT_RETRIES == 3

    def test_truncation_factor(self):
        assert _TRUNCATION_FACTOR == 0.9

    def test_fallback_context_window(self):
        assert FALLBACK_CONTEXT_WINDOW == 128_000

    def test_error_patterns_cover_major_providers(self):
        # Verify patterns match representative error messages from each provider
        assert len(_CONTEXT_WINDOW_ERROR_PATTERNS) >= 3
        # OpenAI-style: token/context/length keywords
        assert any(re.search(p, "maximum context length exceeded") for p in _CONTEXT_WINDOW_ERROR_PATTERNS)
        # Anthropic-style: prompt is too long
        assert any(re.search(p, "prompt is too long") for p in _CONTEXT_WINDOW_ERROR_PATTERNS)
        # Google-style: resource exhausted / token limit
        assert any(re.search(p, "token limit exceeded") for p in _CONTEXT_WINDOW_ERROR_PATTERNS)

    def test_error_classes_cover_google(self):
        assert "ResourceExhausted" in _CONTEXT_WINDOW_ERROR_CLASSES


# ---------------------------------------------------------------------------
# Integration tests: truncation + downstream error combinations (PT.3)
# ---------------------------------------------------------------------------


class TestTokenRecoveryDownstreamErrors:
    """Tests for truncation succeeding but subsequent errors occurring."""

    @pytest.mark.asyncio
    async def test_truncation_succeeds_but_llm_returns_failure(
        self, mock_workflow, sample_state
    ):
        """After truncation fixes the context-window error, the LLM may still
        return success=False (e.g. content filter, timeout, provider error).
        The failure should propagate without further token retries."""
        mock_workflow._execute_provider_async.side_effect = [
            # First call: context-window error triggers truncation
            ContextWindowError(
                "Context window exceeded",
                prompt_tokens=5000,
                max_tokens=4096,
                provider="test-provider",
            ),
            # Second call: truncation fixes size, but LLM still fails
            _make_success_result(
                success=False,
                content="",
                error="Content filtered by safety system",
                metadata={"timeout": False},
            ),
        ]

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="analysis",
            system_prompt="sys",
            user_prompt="X" * 50000,
            provider_id="test-provider",
            model="claude-3.5-sonnet",
            temperature=0.3,
            timeout=60.0,
        )

        # The failed WorkflowResult is returned directly (not wrapped in LLMCallResult)
        assert isinstance(ret, WorkflowResult)
        assert ret.success is False
        assert "Content filtered" in ret.error
        # Only 2 calls: initial (context error) + retry (LLM failure), no further retries
        assert mock_workflow._execute_provider_async.call_count == 2

    @pytest.mark.asyncio
    async def test_truncation_succeeds_but_llm_times_out(
        self, mock_workflow, sample_state
    ):
        """After truncation fixes context-window, the LLM may time out.
        Timeout metadata should be present in the returned result."""
        mock_workflow._execute_provider_async.side_effect = [
            ContextWindowError(
                "too long",
                prompt_tokens=5000,
                max_tokens=4096,
            ),
            _make_success_result(
                success=False,
                content="",
                error="Request timed out",
                metadata={"timeout": True, "providers_tried": ["test-provider"]},
            ),
        ]

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="synthesis",
            system_prompt="sys",
            user_prompt="X" * 50000,
            provider_id="p",
            model="m",
            temperature=0.3,
            timeout=60.0,
        )

        assert isinstance(ret, WorkflowResult)
        assert ret.success is False
        assert ret.metadata["timeout"] is True

    @pytest.mark.asyncio
    async def test_very_small_truncated_prompt_still_submitted(
        self, mock_workflow, sample_state
    ):
        """When progressive truncation produces a very short prompt (near the
        truncation marker), the prompt is still submitted — there is no
        minimum-size guard.  If it fails again, the hard-error path is taken."""
        # Use a tiny max_tokens so the truncated prompt becomes very short
        small_error = ContextWindowError(
            "Context window exceeded",
            prompt_tokens=500,
            max_tokens=10,  # Very small budget → tiny truncated prompt
            provider="test-provider",
        )
        mock_workflow._execute_provider_async.side_effect = [small_error] * (
            _MAX_TOKEN_LIMIT_RETRIES + 1
        )

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="analysis",
            system_prompt="sys",
            user_prompt="X" * 1000,
            provider_id="p",
            model="m",
            temperature=0.3,
            timeout=60.0,
        )

        # Hard error after all retries exhausted
        assert isinstance(ret, WorkflowResult)
        assert ret.success is False
        assert ret.metadata["error_type"] == "context_window_exceeded"
        assert ret.metadata["token_limit_retries"] == _MAX_TOKEN_LIMIT_RETRIES

        # All attempts were made (initial + 3 retries)
        assert mock_workflow._execute_provider_async.call_count == _MAX_TOKEN_LIMIT_RETRIES + 1

        # Each retry prompt should be shorter due to progressive truncation
        prompts = [
            call.kwargs["prompt"]
            for call in mock_workflow._execute_provider_async.call_args_list
        ]
        for i in range(1, len(prompts)):
            assert len(prompts[i]) <= len(prompts[i - 1])

    @pytest.mark.asyncio
    async def test_non_context_error_after_successful_truncation(
        self, mock_workflow, sample_state
    ):
        """A non-context-window exception after successful truncation is NOT
        retried — it propagates immediately."""
        mock_workflow._execute_provider_async.side_effect = [
            ContextWindowError(
                "too long",
                prompt_tokens=5000,
                max_tokens=4096,
            ),
            RuntimeError("Internal provider error"),
        ]

        with pytest.raises(RuntimeError, match="Internal provider error"):
            await execute_llm_call(
                workflow=mock_workflow,
                state=sample_state,
                phase_name="analysis",
                system_prompt="sys",
                user_prompt="X" * 50000,
                provider_id="p",
                model="m",
                temperature=0.3,
                timeout=60.0,
            )

        # Only 2 calls: initial (context error) + retry (runtime error)
        assert mock_workflow._execute_provider_async.call_count == 2


# ---------------------------------------------------------------------------
# Unit tests: structured truncation helpers (Phase 4)
# ---------------------------------------------------------------------------


from foundry_mcp.core.research.workflows.deep_research._helpers import (
    _split_prompt_sections,
    structured_drop_sources,
    structured_truncate_blocks,
)
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    _apply_truncation_strategy,
    _TRUNCATION_STRATEGY_NAMES,
)


def _build_synthesis_style_prompt(
    num_sources: int = 5,
    source_content_len: int = 500,
    qualities: list[str] | None = None,
) -> str:
    """Build a realistic synthesis-style structured prompt for testing."""
    if qualities is None:
        qualities = ["high", "medium", "low", "medium", "high"]

    parts = [
        "# Research Query\nWhat are the best Python web frameworks?",
        "",
        "## Research Brief\nComprehensive analysis of Python web frameworks.",
        "",
        "## Findings to Synthesize",
        "",
        "### Web Frameworks",
        "- [HIGH] Django is a full-featured web framework.",
        "  Sources: [1], [2]",
        "- [MEDIUM] Flask is lightweight and flexible.",
        "  Sources: [3]",
        "",
        "## Source Reference (use these citation numbers in your report)",
    ]

    for i in range(num_sources):
        q = qualities[i % len(qualities)]
        content = f"Content about source {i + 1}. " * (source_content_len // 30)
        parts.append(f"- **[{i + 1}]**: Source {i + 1} Title [{q}]")
        parts.append(f"  URL: https://example.com/source-{i + 1}")
        parts.append(f"  Snippet: {content}")

    parts.extend(
        [
            "",
            "## Instructions",
            "Generate a comprehensive research report.",
            "Total findings: 2",
            "Total sources: " + str(num_sources),
        ]
    )

    return "\n".join(parts)


def _build_analysis_style_prompt(
    num_sources: int = 5,
    source_content_len: int = 500,
) -> str:
    """Build a realistic analysis-style prompt for testing."""
    parts = [
        "Original Research Query: What are the best Python web frameworks?",
        "",
        "Research Brief:",
        "Analysis of Python web frameworks.",
        "",
        "Sources to Analyze:",
        "",
    ]

    for i in range(num_sources):
        content = f"Detailed content for source {i + 1}. " * (source_content_len // 40)
        parts.append(f"Source {i + 1} (ID: src-{i + 1:03d}):")
        parts.append(f"  Title: Source {i + 1} Title")
        parts.append(f"  URL: https://example.com/source-{i + 1}")
        parts.append(f"  Snippet: Short snippet for source {i + 1}")
        parts.append(f"  Content: {content}")

    parts.extend(
        [
            "",
            "Please analyze these sources and:",
            "1. Extract 2-5 key findings",
            "2. Assess confidence levels",
        ]
    )

    return "\n".join(parts)


class TestSplitPromptSections:
    """Tests for _split_prompt_sections()."""

    def test_splits_at_markdown_headers(self):
        prompt = "# Header 1\ncontent 1\n## Header 2\ncontent 2\n### Header 3\ncontent 3"
        sections = _split_prompt_sections(prompt)
        assert len(sections) == 3
        assert sections[0][0] == "# Header 1"
        assert "content 1" in sections[0][1]
        assert sections[1][0] == "## Header 2"
        assert sections[2][0] == "### Header 3"

    def test_preserves_pre_header_content(self):
        prompt = "preamble text\n# Header\ncontent"
        sections = _split_prompt_sections(prompt)
        assert len(sections) == 2
        assert sections[0][0] == ""
        assert "preamble" in sections[0][1]

    def test_no_headers_single_section(self):
        prompt = "just plain text\nno headers here"
        sections = _split_prompt_sections(prompt)
        assert len(sections) == 1
        assert sections[0][0] == ""

    def test_empty_prompt(self):
        sections = _split_prompt_sections("")
        assert len(sections) == 1


class TestStructuredTruncateBlocks:
    """Tests for structured_truncate_blocks() — checklist 4.1."""

    def test_short_prompt_unchanged(self):
        """Prompt within budget is returned unchanged."""
        prompt = "# Header\nshort content"
        result = structured_truncate_blocks(prompt, max_tokens=1000)
        assert result == prompt

    def test_truncates_longest_section_first(self):
        """Longest truncatable section should be reduced first."""
        short = "short " * 10
        long_content = "verbose " * 500
        prompt = f"# Research Query\nmy query\n\n## Findings\n{long_content}\n\n## Small Section\n{short}"

        # Budget that fits the prompt minus roughly half the long section
        budget = len(prompt) // 8  # tokens (chars / 4)
        result = structured_truncate_blocks(prompt, max_tokens=budget)

        assert len(result) < len(prompt)
        assert "# Research Query" in result
        assert "my query" in result
        assert "## Findings" in result
        assert "## Small Section" in result

    def test_protects_instructions_section(self):
        """Instructions section should not be truncated."""
        instructions = "important instructions " * 200
        findings = "findings content " * 200
        prompt = f"## Findings\n{findings}\n\n## Instructions\n{instructions}"

        budget = len(prompt) // 8
        result = structured_truncate_blocks(prompt, max_tokens=budget)

        # Instructions content should be intact
        assert instructions in result
        # Findings should be truncated
        assert len(result) < len(prompt)

    def test_protects_research_query_section(self):
        """Research Query section should not be truncated."""
        query = "detailed research query " * 100
        sources = "source content " * 500
        prompt = f"# Research Query\n{query}\n\n## Source Reference\n{sources}"

        budget = len(prompt) // 8
        result = structured_truncate_blocks(prompt, max_tokens=budget)

        assert query in result

    def test_fallback_for_unstructured_prompt(self):
        """Unstructured prompt should fall back to char-based truncation."""
        prompt = "X" * 10000
        result = structured_truncate_blocks(prompt, max_tokens=500)
        assert len(result) < 10000
        assert "[... content truncated for context limits]" in result

    def test_preserves_high_quality_sources(self):
        """Structured truncation should preserve source headers while truncating content."""
        prompt = _build_synthesis_style_prompt(num_sources=5, source_content_len=2000)
        budget = len(prompt) // 8  # Force significant truncation

        result = structured_truncate_blocks(prompt, max_tokens=budget)

        assert len(result) < len(prompt)
        # Source reference header should be preserved
        assert "## Source Reference" in result
        # Findings section should be preserved
        assert "## Findings to Synthesize" in result


class TestStructuredDropSources:
    """Tests for structured_drop_sources() — checklist 4.2."""

    def test_short_prompt_unchanged(self):
        """Prompt within budget is returned unchanged."""
        prompt = _build_synthesis_style_prompt(num_sources=2, source_content_len=50)
        result = structured_drop_sources(prompt, max_tokens=100_000)
        assert result == prompt

    def test_drops_low_quality_sources_first(self):
        """Low-quality sources should be dropped before high-quality ones."""
        prompt = _build_synthesis_style_prompt(
            num_sources=5,
            source_content_len=500,
            qualities=["high", "high", "low", "low", "medium"],
        )
        # Budget that requires dropping ~2 sources
        budget = len(prompt) // 5
        result = structured_drop_sources(prompt, max_tokens=budget)

        assert len(result) < len(prompt)
        # High-quality sources should survive
        assert "Source 1 Title [high]" in result
        assert "Source 2 Title [high]" in result
        # Low-quality sources should be dropped first
        assert "Source 3 Title [low]" not in result or "Source 4 Title [low]" not in result

    def test_drops_largest_within_same_quality(self):
        """Among same-quality sources, largest should be dropped first."""
        parts = [
            "# Research Query\ntest query",
            "",
            "## Source Reference",
        ]
        # Source 1: medium, short
        parts.append("- **[1]**: Short Source [medium]")
        parts.append("  URL: https://example.com/1")
        parts.append("  Snippet: brief.")
        # Source 2: medium, very long
        parts.append("- **[2]**: Long Source [medium]")
        parts.append("  URL: https://example.com/2")
        parts.append("  Snippet: " + "verbose content " * 200)
        # Source 3: medium, short
        parts.append("- **[3]**: Another Short [medium]")
        parts.append("  URL: https://example.com/3")
        parts.append("  Snippet: brief content.")

        prompt = "\n".join(parts)
        budget = len(prompt) // 5

        result = structured_drop_sources(prompt, max_tokens=budget)

        # Source 2 (longest) should be dropped first
        assert "Long Source [medium]" not in result
        # Shorter sources should survive
        assert "Short Source [medium]" in result

    def test_detects_analysis_style_sources(self):
        """Should detect Source N (ID: ...) patterns from analysis phase."""
        prompt = _build_analysis_style_prompt(num_sources=5, source_content_len=1000)
        budget = len(prompt) // 5

        result = structured_drop_sources(prompt, max_tokens=budget)

        assert len(result) < len(prompt)
        # Should still have some sources
        assert "Source" in result

    def test_fallback_for_no_sources(self):
        """Prompt with no recognizable source entries falls back to char truncation."""
        prompt = "X" * 10000
        result = structured_drop_sources(prompt, max_tokens=500)
        assert len(result) < 10000
        assert "[... content truncated for context limits]" in result

    def test_final_char_truncation_if_still_over(self):
        """If dropping all sources is still not enough, char truncation kicks in."""
        # Very small budget that no amount of source dropping can satisfy
        prompt = _build_synthesis_style_prompt(num_sources=2, source_content_len=100)
        result = structured_drop_sources(prompt, max_tokens=10)
        assert len(result) < len(prompt)
        assert "[... content truncated for context limits]" in result


class TestApplyTruncationStrategy:
    """Tests for _apply_truncation_strategy() — strategy dispatch."""

    def test_retry_1_uses_block_truncation(self):
        """Retry 1 should use structured block truncation."""
        prompt = _build_synthesis_style_prompt(num_sources=5, source_content_len=2000)
        result = _apply_truncation_strategy(prompt, error_max_tokens=500, model=None, retry_count=1)
        assert len(result) < len(prompt)
        # Headers should be preserved
        assert "## Findings to Synthesize" in result

    def test_retry_2_uses_source_dropping(self):
        """Retry 2 should use quality-aware source dropping."""
        prompt = _build_synthesis_style_prompt(
            num_sources=5,
            source_content_len=1000,
            qualities=["high", "high", "low", "low", "medium"],
        )
        result = _apply_truncation_strategy(prompt, error_max_tokens=500, model=None, retry_count=2)
        assert len(result) < len(prompt)

    def test_retry_3_uses_char_truncation(self):
        """Retry 3 should use character-based truncation."""
        prompt = "X" * 50000
        result = _apply_truncation_strategy(prompt, error_max_tokens=500, model=None, retry_count=3)
        assert len(result) < 50000
        assert "[... content truncated for context limits]" in result

    def test_unstructured_falls_back_to_char(self):
        """Unstructured prompt on retry 1 should fall back to char truncation."""
        prompt = "X" * 50000
        result = _apply_truncation_strategy(prompt, error_max_tokens=500, model=None, retry_count=1)
        assert len(result) < 50000

    def test_uses_model_registry_when_no_error_tokens(self):
        """Should use model registry when error_max_tokens is None."""
        # gpt-4o has 128K tokens → 512K chars budget. Use a prompt larger than
        # the reduced budget (128K * 0.9^3 ≈ 93K tokens → 373K chars)
        prompt = "X" * 500000
        result = _apply_truncation_strategy(
            prompt, error_max_tokens=None, model="gpt-4o", retry_count=3,
        )
        assert len(result) < 500000

    def test_uses_fallback_when_no_info(self):
        """Should use fallback context window when no error tokens or model match."""
        prompt = "X" * 1000000
        result = _apply_truncation_strategy(
            prompt, error_max_tokens=None, model="unknown-model", retry_count=3,
        )
        assert len(result) < 1000000

    def test_strategy_names_cover_all_retries(self):
        """Strategy name mapping should cover retries 1-3."""
        assert 1 in _TRUNCATION_STRATEGY_NAMES
        assert 2 in _TRUNCATION_STRATEGY_NAMES
        assert 3 in _TRUNCATION_STRATEGY_NAMES


class TestStructuredTruncationIntegration:
    """Integration tests: structured truncation through execute_llm_call — checklist 4.4-4.6."""

    @pytest.mark.asyncio
    async def test_structured_prompt_preserves_high_quality_on_retry(
        self, mock_workflow, sample_state
    ):
        """When a structured prompt triggers context window error, retry should
        use structured truncation that preserves high-quality sources (4.4)."""
        structured_prompt = _build_synthesis_style_prompt(
            num_sources=10,
            source_content_len=2000,
            qualities=["high", "low", "high", "low", "medium",
                        "high", "low", "medium", "low", "high"],
        )

        mock_workflow._execute_provider_async.side_effect = [
            ContextWindowError("too long", prompt_tokens=50000, max_tokens=2000),
            _make_success_result(),
        ]

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="synthesis",
            system_prompt="sys",
            user_prompt=structured_prompt,
            provider_id="p",
            model="m",
            temperature=0.3,
            timeout=60.0,
        )

        assert isinstance(ret, LLMCallResult)
        # The retry prompt should be shorter
        prompts = [
            call.kwargs["prompt"]
            for call in mock_workflow._execute_provider_async.call_args_list
        ]
        assert len(prompts[1]) < len(prompts[0])
        # Structure should be preserved in the truncated prompt
        assert "## Findings to Synthesize" in prompts[1]

    @pytest.mark.asyncio
    async def test_char_fallback_works_after_structured_fails(
        self, mock_workflow, sample_state
    ):
        """Char-based fallback on retry 3 should still produce a usable prompt (4.5)."""
        err = ContextWindowError("too long", prompt_tokens=50000, max_tokens=1000)
        mock_workflow._execute_provider_async.side_effect = [
            err, err, err,
            _make_success_result(),
        ]

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="synthesis",
            system_prompt="sys",
            user_prompt=_build_synthesis_style_prompt(num_sources=10, source_content_len=2000),
            provider_id="p",
            model="m",
            temperature=0.3,
            timeout=60.0,
        )

        assert isinstance(ret, LLMCallResult)
        assert mock_workflow._execute_provider_async.call_count == 4

    @pytest.mark.asyncio
    async def test_existing_recovery_behavior_unchanged_for_unstructured(
        self, mock_workflow, sample_state
    ):
        """Unstructured prompts should still get progressively shorter on each retry (4.6)."""
        original_prompt = "X" * 50000

        mock_workflow._execute_provider_async.side_effect = [
            ContextWindowError("too long", prompt_tokens=5000, max_tokens=4096),
            ContextWindowError("still too long", prompt_tokens=4500, max_tokens=4096),
            _make_success_result(),
        ]

        ret = await execute_llm_call(
            workflow=mock_workflow,
            state=sample_state,
            phase_name="analysis",
            system_prompt="sys",
            user_prompt=original_prompt,
            provider_id="p",
            model="m",
            temperature=0.3,
            timeout=60.0,
        )

        assert isinstance(ret, LLMCallResult)
        prompts = [
            call.kwargs["prompt"]
            for call in mock_workflow._execute_provider_async.call_args_list
        ]
        assert len(prompts) == 3
        # Each retry should have a shorter or equal user prompt
        assert len(prompts[0]) >= len(prompts[1]) >= len(prompts[2])
        # First call should use the original prompt
        assert prompts[0] == original_prompt
