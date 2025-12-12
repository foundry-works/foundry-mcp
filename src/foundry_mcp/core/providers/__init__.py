"""
Provider abstractions for foundry-mcp.

This package provides pluggable LLM provider backends for CLI operations,
with support for capability negotiation, request/response normalization,
lifecycle hooks, availability detection, and registry management.

Example usage:
    from foundry_mcp.core.providers import (
        # Core types
        ProviderCapability,
        ProviderRequest,
        ProviderResult,
        ProviderContext,
        ProviderHooks,
        # Detection
        detect_provider_availability,
        get_provider_statuses,
        # Registry
        register_provider,
        resolve_provider,
        available_providers,
    )

    # Check provider availability
    if detect_provider_availability("gemini"):
        # Register and resolve a provider
        hooks = ProviderHooks()
        provider = resolve_provider("gemini", hooks=hooks)

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

from foundry_mcp.core.providers.detectors import (
    ProviderDetector,
    register_detector,
    get_detector,
    detect_provider_availability,
    get_provider_statuses,
    list_detectors,
    reset_detectors,
)

from foundry_mcp.core.providers.registry import (
    # Types
    ProviderFactory,
    ProviderRegistration,
    AvailabilityCheck,
    MetadataResolver,
    LazyFactoryLoader,
    DependencyResolver,
    # Registration
    register_provider,
    register_lazy_provider,
    # Resolution
    available_providers,
    check_provider_available,
    resolve_provider,
    get_provider_metadata,
    describe_providers,
    # Dependency Injection
    set_dependency_resolver,
    # Testing
    reset_registry,
    get_registration,
)

from foundry_mcp.core.providers.validation import (
    # Validation
    ValidationError,
    strip_ansi,
    ensure_utf8,
    sanitize_prompt,
    validate_request,
    # Command allowlists
    COMMON_SAFE_COMMANDS,
    BLOCKED_COMMANDS,
    is_command_allowed,
    # Observability
    ExecutionSpan,
    create_execution_span,
    log_span,
    # Retry
    RETRYABLE_STATUSES,
    is_retryable,
    is_retryable_error,
    # Circuit breaker
    CircuitState,
    CircuitBreaker,
    get_circuit_breaker,
    reset_circuit_breakers,
    # Rate limiting
    RateLimiter,
    get_rate_limiter,
    reset_rate_limiters,
    # Execution wrapper
    with_validation_and_resilience,
)

# ---------------------------------------------------------------------------
# Import provider modules to trigger auto-registration with the registry.
# Each provider module calls register_provider() at import time.
# ---------------------------------------------------------------------------
from foundry_mcp.core.providers import gemini as _gemini_provider  # noqa: F401
from foundry_mcp.core.providers import codex as _codex_provider  # noqa: F401
from foundry_mcp.core.providers import cursor_agent as _cursor_agent_provider  # noqa: F401
from foundry_mcp.core.providers import claude as _claude_provider  # noqa: F401
from foundry_mcp.core.providers import opencode as _opencode_provider  # noqa: F401
from foundry_mcp.core.providers import test_provider as _test_provider  # noqa: F401

__all__ = [
    # === Base Types (base.py) ===
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
    # === Detection (detectors.py) ===
    "ProviderDetector",
    "register_detector",
    "get_detector",
    "detect_provider_availability",
    "get_provider_statuses",
    "list_detectors",
    "reset_detectors",
    # === Registry (registry.py) ===
    # Types
    "ProviderFactory",
    "ProviderRegistration",
    "AvailabilityCheck",
    "MetadataResolver",
    "LazyFactoryLoader",
    "DependencyResolver",
    # Registration
    "register_provider",
    "register_lazy_provider",
    # Resolution
    "available_providers",
    "check_provider_available",
    "resolve_provider",
    "get_provider_metadata",
    "describe_providers",
    # Dependency Injection
    "set_dependency_resolver",
    # Testing
    "reset_registry",
    "get_registration",
    # === Validation & Resilience (validation.py) ===
    # Validation
    "ValidationError",
    "strip_ansi",
    "ensure_utf8",
    "sanitize_prompt",
    "validate_request",
    # Command allowlists
    "COMMON_SAFE_COMMANDS",
    "BLOCKED_COMMANDS",
    "is_command_allowed",
    # Observability
    "ExecutionSpan",
    "create_execution_span",
    "log_span",
    # Retry
    "RETRYABLE_STATUSES",
    "is_retryable",
    "is_retryable_error",
    # Circuit breaker
    "CircuitState",
    "CircuitBreaker",
    "get_circuit_breaker",
    "reset_circuit_breakers",
    # Rate limiting
    "RateLimiter",
    "get_rate_limiter",
    "reset_rate_limiters",
    # Execution wrapper
    "with_validation_and_resilience",
]
