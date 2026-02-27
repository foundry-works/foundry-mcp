"""Tests for context-window error detection in _lifecycle.py.

Regression tests for false-positive patterns that previously caused
incorrect truncation-retry loops (e.g., "invalid authentication token"
was matched by the overly broad token/context regex).
"""

from __future__ import annotations

from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    _is_context_window_error,
)

# ---------------------------------------------------------------------------
# Helper: create an exception with a specific class name
# ---------------------------------------------------------------------------


def _make_exception(cls_name: str, message: str) -> Exception:
    """Create an exception with a dynamic class name for testing."""
    exc_cls = type(cls_name, (Exception,), {})
    return exc_cls(message)


# ---------------------------------------------------------------------------
# True positives: SHOULD be classified as context-window errors
# ---------------------------------------------------------------------------


class TestTruePositives:
    """Messages and class names that should trigger context-window detection."""

    def test_maximum_context_length_exceeded(self):
        exc = Exception("maximum context length exceeded")
        assert _is_context_window_error(exc) is True

    def test_token_limit_exceeded(self):
        exc = Exception("token limit exceeded for model gpt-4")
        assert _is_context_window_error(exc) is True

    def test_context_window_overflow(self):
        exc = Exception("context window overflow: 200000 tokens")
        assert _is_context_window_error(exc) is True

    def test_prompt_is_too_long(self):
        exc = Exception("prompt is too long: 150000 tokens")
        assert _is_context_window_error(exc) is True

    def test_too_many_tokens(self):
        exc = Exception("too many tokens in request")
        assert _is_context_window_error(exc) is True

    def test_resource_exhausted_class(self):
        exc = _make_exception("ResourceExhausted", "some message")
        assert _is_context_window_error(exc) is True

    def test_exceeds_token_limit(self):
        exc = Exception("Request exceeds the token limit")
        assert _is_context_window_error(exc) is True

    def test_over_context_limit(self):
        exc = Exception("Input is over the context limit")
        assert _is_context_window_error(exc) is True

    def test_context_length_exceeded(self):
        exc = Exception("context length exceeded")
        assert _is_context_window_error(exc) is True

    def test_token_limit_in_message(self):
        exc = Exception("token limit reached")
        assert _is_context_window_error(exc) is True

    def test_invalid_argument_with_token_limit(self):
        """InvalidArgument with token-related message IS a context-window error."""
        exc = _make_exception("InvalidArgument", "token limit exceeded for this model")
        assert _is_context_window_error(exc) is True

    def test_invalid_argument_with_context_window(self):
        exc = _make_exception("InvalidArgument", "context window exceeded")
        assert _is_context_window_error(exc) is True

    def test_maximum_token(self):
        exc = Exception("maximum token count exceeded")
        assert _is_context_window_error(exc) is True


# ---------------------------------------------------------------------------
# False positives: should NOT be classified as context-window errors
# ---------------------------------------------------------------------------


class TestFalsePositives:
    """Messages that previously false-positived and should NOT trigger."""

    def test_invalid_authentication_token(self):
        exc = Exception("invalid authentication token")
        assert _is_context_window_error(exc) is False

    def test_context_parameter_is_required(self):
        exc = Exception("context parameter is required")
        assert _is_context_window_error(exc) is False

    def test_length_must_be_positive(self):
        exc = Exception("length must be positive")
        assert _is_context_window_error(exc) is False

    def test_invalid_argument_auth_token(self):
        """InvalidArgument with non-token message is NOT a context-window error."""
        exc = _make_exception("InvalidArgument", "invalid authentication token")
        assert _is_context_window_error(exc) is False

    def test_invalid_argument_missing_field(self):
        exc = _make_exception("InvalidArgument", "required field 'model' is missing")
        assert _is_context_window_error(exc) is False

    def test_invalid_argument_context_param(self):
        exc = _make_exception("InvalidArgument", "context parameter is required")
        assert _is_context_window_error(exc) is False

    def test_token_in_url(self):
        exc = Exception("failed to fetch https://api.example.com/token/refresh")
        assert _is_context_window_error(exc) is False

    def test_generic_bad_request(self):
        exc = Exception("Bad request: invalid model name")
        assert _is_context_window_error(exc) is False

    def test_context_in_unrelated_message(self):
        exc = Exception("missing context in request headers")
        assert _is_context_window_error(exc) is False
