"""
Provider abstractions for foundry-mcp.

This package provides pluggable LLM provider backends for CLI operations,
with support for capability negotiation, request/response normalization,
and lifecycle hooks.

Example usage:
    from foundry_mcp.core.providers import (
        ProviderCapability,
        ProviderRequest,
        ProviderResult,
        ProviderContext,
    )

    # Check if provider supports streaming
    if provider.supports(ProviderCapability.STREAMING):
        request = ProviderRequest(prompt="Hello", stream=True)
        result = provider.generate(request)
"""

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
    StreamChunkCallback,
    BeforeExecuteHook,
    AfterResultHook,
    # Errors
    ProviderError,
    ProviderUnavailableError,
    ProviderExecutionError,
    ProviderTimeoutError,
    # ABC
    ProviderContext,
)

__all__ = [
    # Enums
    "ProviderCapability",
    "ProviderStatus",
    # Request/Response dataclasses
    "ProviderRequest",
    "ProviderResult",
    "TokenUsage",
    "StreamChunk",
    # Metadata dataclasses
    "ModelDescriptor",
    "ProviderMetadata",
    # Hooks
    "ProviderHooks",
    "StreamChunkCallback",
    "BeforeExecuteHook",
    "AfterResultHook",
    # Errors
    "ProviderError",
    "ProviderUnavailableError",
    "ProviderExecutionError",
    "ProviderTimeoutError",
    # ABC
    "ProviderContext",
]
