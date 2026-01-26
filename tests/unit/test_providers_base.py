"""
Unit tests for foundry_mcp.core.providers.base module.

Tests cover:
- ProviderStatus.is_retryable() - verifies retry semantics for each status
- ProviderStatus.to_error_type() - verifies mapping to ErrorType enum
- Dataclass creation and immutability for all provider dataclasses
- ProviderHooks lifecycle callback execution
- ProviderCapability enum values
"""

import pytest

from foundry_mcp.core.providers.base import (
    # Enums
    ProviderCapability,
    ProviderStatus,
    # Request/Response dataclasses
    ProviderRequest,
    ProviderResult,
    TokenUsage,
    StreamChunk,
    # Metadata dataclasses
    ModelDescriptor,
    ProviderMetadata,
    # Hooks
    ProviderHooks,
    # Errors
    ProviderError,
    ProviderUnavailableError,
    ProviderExecutionError,
    ProviderTimeoutError,
)
from foundry_mcp.core.responses import ErrorType


# =============================================================================
# ProviderStatus Tests
# =============================================================================


class TestProviderStatusIsRetryable:
    """Tests for ProviderStatus.is_retryable() method."""

    def test_timeout_is_retryable(self):
        """TIMEOUT status should be retryable."""
        assert ProviderStatus.TIMEOUT.is_retryable() is True

    def test_error_is_retryable(self):
        """ERROR status should be retryable."""
        assert ProviderStatus.ERROR.is_retryable() is True

    def test_success_is_not_retryable(self):
        """SUCCESS status should not be retryable (already succeeded)."""
        assert ProviderStatus.SUCCESS.is_retryable() is False

    def test_not_found_is_not_retryable(self):
        """NOT_FOUND status should not be retryable (provider unavailable)."""
        assert ProviderStatus.NOT_FOUND.is_retryable() is False

    def test_invalid_output_is_not_retryable(self):
        """INVALID_OUTPUT status should not be retryable (malformed response)."""
        assert ProviderStatus.INVALID_OUTPUT.is_retryable() is False

    def test_canceled_is_not_retryable(self):
        """CANCELED status should not be retryable (explicitly canceled)."""
        assert ProviderStatus.CANCELED.is_retryable() is False

    def test_all_statuses_covered(self):
        """Verify all ProviderStatus values have defined retry behavior."""
        retryable = {ProviderStatus.TIMEOUT, ProviderStatus.ERROR}
        non_retryable = {
            ProviderStatus.SUCCESS,
            ProviderStatus.NOT_FOUND,
            ProviderStatus.INVALID_OUTPUT,
            ProviderStatus.CANCELED,
        }
        all_statuses = set(ProviderStatus)

        assert retryable | non_retryable == all_statuses
        for status in retryable:
            assert status.is_retryable() is True
        for status in non_retryable:
            assert status.is_retryable() is False


class TestProviderStatusToErrorType:
    """Tests for ProviderStatus.to_error_type() method."""

    def test_timeout_maps_to_unavailable(self):
        """TIMEOUT status should map to UNAVAILABLE error type."""
        assert ProviderStatus.TIMEOUT.to_error_type() == ErrorType.UNAVAILABLE

    def test_not_found_maps_to_not_found(self):
        """NOT_FOUND status should map to NOT_FOUND error type."""
        assert ProviderStatus.NOT_FOUND.to_error_type() == ErrorType.NOT_FOUND

    def test_invalid_output_maps_to_validation(self):
        """INVALID_OUTPUT status should map to VALIDATION error type."""
        assert ProviderStatus.INVALID_OUTPUT.to_error_type() == ErrorType.VALIDATION

    def test_error_maps_to_internal(self):
        """ERROR status should map to INTERNAL error type."""
        assert ProviderStatus.ERROR.to_error_type() == ErrorType.INTERNAL

    def test_canceled_maps_to_internal(self):
        """CANCELED status should map to INTERNAL error type."""
        assert ProviderStatus.CANCELED.to_error_type() == ErrorType.INTERNAL

    def test_success_raises_value_error(self):
        """SUCCESS status should raise ValueError (not an error state)."""
        with pytest.raises(ValueError, match="SUCCESS.*cannot be mapped"):
            ProviderStatus.SUCCESS.to_error_type()

    def test_all_error_statuses_mapped(self):
        """Verify all non-SUCCESS statuses have error type mappings."""
        error_statuses = [s for s in ProviderStatus if s != ProviderStatus.SUCCESS]
        for status in error_statuses:
            error_type = status.to_error_type()
            assert isinstance(error_type, ErrorType)


# =============================================================================
# ProviderCapability Tests
# =============================================================================


class TestProviderCapability:
    """Tests for ProviderCapability enum."""

    def test_text_capability_value(self):
        """TEXT capability should have expected value."""
        assert ProviderCapability.TEXT.value == "text_generation"

    def test_vision_capability_value(self):
        """VISION capability should have expected value."""
        assert ProviderCapability.VISION.value == "vision"

    def test_function_calling_capability_value(self):
        """FUNCTION_CALLING capability should have expected value."""
        assert ProviderCapability.FUNCTION_CALLING.value == "function_calling"

    def test_streaming_capability_value(self):
        """STREAMING capability should have expected value."""
        assert ProviderCapability.STREAMING.value == "streaming"

    def test_thinking_capability_value(self):
        """THINKING capability should have expected value."""
        assert ProviderCapability.THINKING.value == "thinking"

    def test_all_capabilities_unique(self):
        """All capability values should be unique."""
        values = [cap.value for cap in ProviderCapability]
        assert len(values) == len(set(values))


# =============================================================================
# Dataclass Creation Tests
# =============================================================================


class TestTokenUsageDataclass:
    """Tests for TokenUsage dataclass."""

    def test_default_values(self):
        """TokenUsage should have sensible defaults."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cached_input_tokens == 0
        assert usage.total_tokens == 0

    def test_custom_values(self):
        """TokenUsage should accept custom values."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cached_input_tokens=25,
            total_tokens=150,
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cached_input_tokens == 25
        assert usage.total_tokens == 150

    def test_immutability(self):
        """TokenUsage should be immutable (frozen)."""
        usage = TokenUsage()
        with pytest.raises(Exception):  # FrozenInstanceError
            usage.input_tokens = 100


class TestProviderRequestDataclass:
    """Tests for ProviderRequest dataclass."""

    def test_minimal_creation(self):
        """ProviderRequest should work with just prompt."""
        request = ProviderRequest(prompt="Hello")
        assert request.prompt == "Hello"
        assert request.system_prompt is None
        assert request.model is None
        assert request.timeout is None
        assert request.temperature is None
        assert request.max_tokens is None
        assert request.metadata == {}
        assert request.stream is False
        assert list(request.attachments) == []

    def test_full_creation(self):
        """ProviderRequest should accept all parameters."""
        request = ProviderRequest(
            prompt="Hello",
            system_prompt="You are helpful.",
            model="pro",
            timeout=30.0,
            temperature=0.7,
            max_tokens=1000,
            metadata={"trace_id": "abc123"},
            stream=True,
            attachments=["image.png"],
        )
        assert request.prompt == "Hello"
        assert request.system_prompt == "You are helpful."
        assert request.model == "pro"
        assert request.timeout == 30.0
        assert request.temperature == 0.7
        assert request.max_tokens == 1000
        assert request.metadata == {"trace_id": "abc123"}
        assert request.stream is True
        assert list(request.attachments) == ["image.png"]

    def test_immutability(self):
        """ProviderRequest should be immutable (frozen)."""
        request = ProviderRequest(prompt="Hello")
        with pytest.raises(Exception):  # FrozenInstanceError
            request.prompt = "Changed"


class TestProviderResultDataclass:
    """Tests for ProviderResult dataclass."""

    def test_minimal_creation(self):
        """ProviderResult should work with required fields only."""
        result = ProviderResult(
            content="Hello, world!",
            provider_id="gemini",
            model_used="gemini:pro",
            status=ProviderStatus.SUCCESS,
        )
        assert result.content == "Hello, world!"
        assert result.provider_id == "gemini"
        assert result.model_used == "gemini:pro"
        assert result.status == ProviderStatus.SUCCESS
        assert result.tokens == TokenUsage()
        assert result.duration_ms is None
        assert result.stderr is None
        assert result.raw_payload == {}

    def test_full_creation(self):
        """ProviderResult should accept all parameters."""
        tokens = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        result = ProviderResult(
            content="Response text",
            provider_id="codex",
            model_used="codex:latest",
            status=ProviderStatus.SUCCESS,
            tokens=tokens,
            duration_ms=1234.5,
            stderr="Some debug output",
            raw_payload={"trace": "data"},
        )
        assert result.content == "Response text"
        assert result.tokens == tokens
        assert result.duration_ms == 1234.5
        assert result.stderr == "Some debug output"
        assert result.raw_payload == {"trace": "data"}

    def test_immutability(self):
        """ProviderResult should be immutable (frozen)."""
        result = ProviderResult(
            content="Test",
            provider_id="test",
            model_used="test:model",
            status=ProviderStatus.SUCCESS,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            result.content = "Changed"


class TestModelDescriptorDataclass:
    """Tests for ModelDescriptor dataclass."""

    def test_minimal_creation(self):
        """ModelDescriptor should work with just id."""
        model = ModelDescriptor(id="pro")
        assert model.id == "pro"
        assert model.display_name is None
        assert model.capabilities == set()
        assert model.routing_hints == {}

    def test_full_creation(self):
        """ModelDescriptor should accept all parameters."""
        caps = {ProviderCapability.TEXT, ProviderCapability.VISION}
        model = ModelDescriptor(
            id="pro",
            display_name="Pro Model",
            capabilities=caps,
            routing_hints={"cost": "high"},
        )
        assert model.id == "pro"
        assert model.display_name == "Pro Model"
        assert model.capabilities == caps
        assert model.routing_hints == {"cost": "high"}


class TestProviderMetadataDataclass:
    """Tests for ProviderMetadata dataclass."""

    def test_minimal_creation(self):
        """ProviderMetadata should work with just provider_id."""
        metadata = ProviderMetadata(provider_id="test-provider")
        assert metadata.provider_id == "test-provider"
        assert metadata.display_name is None
        assert list(metadata.models) == []
        assert metadata.default_model is None
        assert metadata.capabilities == set()
        assert metadata.security_flags == {}
        assert metadata.extra == {}

    def test_full_creation(self):
        """ProviderMetadata should accept all parameters."""
        model = ModelDescriptor(id="pro")
        caps = {ProviderCapability.TEXT}
        metadata = ProviderMetadata(
            provider_id="gemini",
            display_name="Gemini AI",
            models=[model],
            default_model="pro",
            capabilities=caps,
            security_flags={"sandbox": True},
            extra={"version": "1.0"},
        )
        assert metadata.provider_id == "gemini"
        assert metadata.display_name == "Gemini AI"
        assert list(metadata.models) == [model]
        assert metadata.default_model == "pro"
        assert metadata.capabilities == caps
        assert metadata.security_flags == {"sandbox": True}
        assert metadata.extra == {"version": "1.0"}


class TestStreamChunkDataclass:
    """Tests for StreamChunk dataclass."""

    def test_minimal_creation(self):
        """StreamChunk should work with required fields."""
        chunk = StreamChunk(content="Hello", index=0)
        assert chunk.content == "Hello"
        assert chunk.index == 0
        assert chunk.metadata == {}

    def test_with_metadata(self):
        """StreamChunk should accept metadata."""
        chunk = StreamChunk(
            content="World",
            index=1,
            metadata={"partial": True},
        )
        assert chunk.content == "World"
        assert chunk.index == 1
        assert chunk.metadata == {"partial": True}


# =============================================================================
# ProviderHooks Tests
# =============================================================================


class TestProviderHooks:
    """Tests for ProviderHooks lifecycle callbacks."""

    def test_default_hooks_are_none(self):
        """ProviderHooks should default to None callbacks."""
        hooks = ProviderHooks()
        assert hooks.before_execute is None
        assert hooks.on_stream_chunk is None
        assert hooks.after_result is None

    def test_emit_before_with_no_hook(self):
        """emit_before should be no-op when hook not registered."""
        hooks = ProviderHooks()
        metadata = ProviderMetadata(provider_id="test")
        request = ProviderRequest(prompt="test")
        # Should not raise
        hooks.emit_before(request, metadata)

    def test_emit_before_calls_hook(self):
        """emit_before should call registered hook."""
        calls = []

        def before_hook(req, meta):
            calls.append((req, meta))

        hooks = ProviderHooks(before_execute=before_hook)
        metadata = ProviderMetadata(provider_id="test")
        request = ProviderRequest(prompt="test")

        hooks.emit_before(request, metadata)

        assert len(calls) == 1
        assert calls[0] == (request, metadata)

    def test_emit_stream_with_no_hook(self):
        """emit_stream should be no-op when hook not registered."""
        hooks = ProviderHooks()
        metadata = ProviderMetadata(provider_id="test")
        chunk = StreamChunk(content="data", index=0)
        # Should not raise
        hooks.emit_stream(chunk, metadata)

    def test_emit_stream_calls_hook(self):
        """emit_stream should call registered hook."""
        calls = []

        def stream_hook(chunk, meta):
            calls.append((chunk, meta))

        hooks = ProviderHooks(on_stream_chunk=stream_hook)
        metadata = ProviderMetadata(provider_id="test")
        chunk = StreamChunk(content="data", index=0)

        hooks.emit_stream(chunk, metadata)

        assert len(calls) == 1
        assert calls[0] == (chunk, metadata)

    def test_emit_after_with_no_hook(self):
        """emit_after should be no-op when hook not registered."""
        hooks = ProviderHooks()
        metadata = ProviderMetadata(provider_id="test")
        result = ProviderResult(
            content="result",
            provider_id="test",
            model_used="test:model",
            status=ProviderStatus.SUCCESS,
        )
        # Should not raise
        hooks.emit_after(result, metadata)

    def test_emit_after_calls_hook(self):
        """emit_after should call registered hook."""
        calls = []

        def after_hook(res, meta):
            calls.append((res, meta))

        hooks = ProviderHooks(after_result=after_hook)
        metadata = ProviderMetadata(provider_id="test")
        result = ProviderResult(
            content="result",
            provider_id="test",
            model_used="test:model",
            status=ProviderStatus.SUCCESS,
        )

        hooks.emit_after(result, metadata)

        assert len(calls) == 1
        assert calls[0] == (result, metadata)


# =============================================================================
# Error Hierarchy Tests
# =============================================================================


class TestProviderErrors:
    """Tests for provider error hierarchy."""

    def test_provider_error_base(self):
        """ProviderError should be base exception."""
        err = ProviderError("Something failed", provider="test")
        assert str(err) == "Something failed"
        assert err.provider == "test"
        assert isinstance(err, RuntimeError)

    def test_provider_error_without_provider(self):
        """ProviderError should work without provider kwarg."""
        err = ProviderError("Error occurred")
        assert str(err) == "Error occurred"
        assert err.provider is None

    def test_provider_unavailable_error(self):
        """ProviderUnavailableError should inherit from ProviderError."""
        err = ProviderUnavailableError("Binary not found", provider="codex")
        assert isinstance(err, ProviderError)
        assert isinstance(err, RuntimeError)
        assert err.provider == "codex"

    def test_provider_execution_error(self):
        """ProviderExecutionError should inherit from ProviderError."""
        err = ProviderExecutionError("Non-zero exit", provider="gemini")
        assert isinstance(err, ProviderError)
        assert isinstance(err, RuntimeError)
        assert err.provider == "gemini"

    def test_provider_timeout_error(self):
        """ProviderTimeoutError should inherit from ProviderError."""
        err = ProviderTimeoutError("Exceeded 30s", provider="cursor-agent")
        assert isinstance(err, ProviderError)
        assert isinstance(err, RuntimeError)
        assert err.provider == "cursor-agent"

    def test_provider_timeout_error_with_metadata(self):
        """ProviderTimeoutError should include elapsed and timeout fields."""
        err = ProviderTimeoutError(
            "Command timed out after 30 seconds",
            provider="gemini",
            elapsed=30.5,
            timeout=30.0,
        )
        assert err.provider == "gemini"
        assert err.elapsed == 30.5
        assert err.timeout == 30.0
        assert "30 seconds" in str(err)

    def test_error_hierarchy_catch_all(self):
        """All provider errors should be catchable via ProviderError."""
        errors = [
            ProviderUnavailableError("test"),
            ProviderExecutionError("test"),
            ProviderTimeoutError("test"),
        ]
        for err in errors:
            try:
                raise err
            except ProviderError as caught:
                assert caught is err
