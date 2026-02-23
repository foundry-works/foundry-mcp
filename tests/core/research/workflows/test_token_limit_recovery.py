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
    _FALLBACK_CONTEXT_WINDOW,
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
        phase=DeepResearchPhase.PLANNING,
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
        # Should truncate using _FALLBACK_CONTEXT_WINDOW (128K)
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
        assert _FALLBACK_CONTEXT_WINDOW == 128_000

    def test_error_patterns_cover_major_providers(self):
        providers = {p for p, _ in _CONTEXT_WINDOW_ERROR_PATTERNS}
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers

    def test_error_classes_cover_google(self):
        assert "ResourceExhausted" in _CONTEXT_WINDOW_ERROR_CLASSES
