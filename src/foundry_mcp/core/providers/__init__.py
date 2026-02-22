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

from foundry_mcp.core.errors.provider import (
    ContextWindowError,
    ProviderError,
    ProviderExecutionError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from foundry_mcp.core.providers import claude as _claude_provider  # noqa: F401
from foundry_mcp.core.providers import claude_zai as _claude_zai_provider  # noqa: F401
from foundry_mcp.core.providers import codex as _codex_provider  # noqa: F401
from foundry_mcp.core.providers import cursor_agent as _cursor_agent_provider  # noqa: F401

# ---------------------------------------------------------------------------
# Import provider modules to trigger auto-registration with the registry.
# Each provider module calls register_provider() at import time.
# ---------------------------------------------------------------------------
from foundry_mcp.core.providers import gemini as _gemini_provider  # noqa: F401
from foundry_mcp.core.providers import opencode as _opencode_provider  # noqa: F401
from foundry_mcp.core.providers import test_provider as _test_provider  # noqa: F401
from foundry_mcp.core.providers.base import (
    AfterResultHook,
    BeforeExecuteHook,
    # Metadata dataclasses
    ModelDescriptor,
    # Enums
    ProviderCapability,
    # ABC
    ProviderContext,
    # Hooks
    ProviderHooks,
    ProviderMetadata,
    # Request/Response dataclasses
    ProviderRequest,
    ProviderResult,
    ProviderStatus,
    StreamChunk,
    StreamChunkCallback,
    TokenUsage,
)
from foundry_mcp.core.providers.detectors import (
    ProviderDetector,
    detect_provider_availability,
    get_detector,
    get_provider_statuses,
    list_detectors,
    register_detector,
    reset_detectors,
)
from foundry_mcp.core.providers.registry import (
    AvailabilityCheck,
    DependencyResolver,
    LazyFactoryLoader,
    MetadataResolver,
    # Types
    ProviderFactory,
    ProviderRegistration,
    # Resolution
    available_providers,
    check_provider_available,
    describe_providers,
    get_provider_metadata,
    get_registration,
    register_lazy_provider,
    # Registration
    register_provider,
    # Testing
    reset_registry,
    resolve_provider,
    # Dependency Injection
    set_dependency_resolver,
)
from foundry_mcp.core.providers.validation import (
    BLOCKED_COMMANDS,
    # Command allowlists
    COMMON_SAFE_COMMANDS,
    # Context window detection
    CONTEXT_WINDOW_ERROR_PATTERNS,
    # Retry
    RETRYABLE_STATUSES,
    CircuitBreaker,
    # Circuit breaker
    CircuitState,
    # Observability
    ExecutionSpan,
    # Rate limiting
    RateLimiter,
    # Validation
    ValidationError,
    create_context_window_guidance,
    create_execution_span,
    ensure_utf8,
    extract_token_counts,
    get_circuit_breaker,
    get_rate_limiter,
    is_command_allowed,
    is_context_window_error,
    is_retryable,
    is_retryable_error,
    log_span,
    reset_circuit_breakers,
    reset_rate_limiters,
    sanitize_prompt,
    strip_ansi,
    validate_request,
    # Execution wrapper
    with_validation_and_resilience,
)

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
    "ContextWindowError",
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
    # Context window detection
    "CONTEXT_WINDOW_ERROR_PATTERNS",
    "is_context_window_error",
    "extract_token_counts",
    "create_context_window_guidance",
]
