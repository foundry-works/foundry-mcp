"""Error-to-ErrorCode mapping registry.

Provides a centralized mapping from exception types to (ErrorCode, ErrorType) tuples,
enabling consistent error response generation across the codebase.

Usage:
    from foundry_mcp.core.errors.base import error_to_response

    try:
        do_something()
    except Exception as e:
        result = error_to_response(e)
        if result is not None:
            return result
        raise  # Unknown error, re-raise
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Type

from foundry_mcp.core.errors.authorization import PathValidationError
from foundry_mcp.core.errors.execution import ActionRouterError, ExecutorExhaustedError
from foundry_mcp.core.errors.llm import (
    AuthenticationError as LLMAuthenticationError,
)
from foundry_mcp.core.errors.llm import (
    ContentFilterError,
    InvalidRequestError,
    ModelNotFoundError,
)
from foundry_mcp.core.errors.llm import (
    RateLimitError as LLMRateLimitError,
)
from foundry_mcp.core.errors.provider import (
    ContextWindowError,
    ProviderExecutionError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from foundry_mcp.core.errors.resilience import (
    CircuitBreakerError,
    RateLimitWaitError,
    TimeBudgetExceededError,
    TimeoutException,
)
from foundry_mcp.core.errors.search import (
    AuthenticationError as SearchAuthenticationError,
)
from foundry_mcp.core.errors.search import (
    RateLimitError as SearchRateLimitError,
)
from foundry_mcp.core.errors.search import (
    SearchProviderError,
)
from foundry_mcp.core.errors.storage import (
    CursorError,
    LockAcquisitionError,
    MigrationError,
    SessionCorrupted,
    VersionConflictError,
)
from foundry_mcp.core.responses.types import (
    ErrorCode,
    ErrorType,
)

ERROR_MAPPINGS: Dict[Type[Exception], Tuple[ErrorCode, ErrorType]] = {
    # --- Provider errors ---
    ProviderUnavailableError: (ErrorCode.UNAVAILABLE, ErrorType.UNAVAILABLE),
    ProviderTimeoutError: (ErrorCode.AI_PROVIDER_TIMEOUT, ErrorType.UNAVAILABLE),
    ProviderExecutionError: (ErrorCode.AI_PROVIDER_ERROR, ErrorType.AI_PROVIDER),
    ContextWindowError: (ErrorCode.AI_CONTEXT_TOO_LARGE, ErrorType.AI_PROVIDER),
    # --- LLM errors ---
    LLMRateLimitError: (ErrorCode.RATE_LIMIT_EXCEEDED, ErrorType.RATE_LIMIT),
    LLMAuthenticationError: (ErrorCode.UNAUTHORIZED, ErrorType.AUTHENTICATION),
    InvalidRequestError: (ErrorCode.VALIDATION_ERROR, ErrorType.VALIDATION),
    ModelNotFoundError: (ErrorCode.AI_NO_PROVIDER, ErrorType.NOT_FOUND),
    ContentFilterError: (ErrorCode.FORBIDDEN, ErrorType.AI_PROVIDER),
    # --- Search provider errors ---
    SearchProviderError: (ErrorCode.AI_PROVIDER_ERROR, ErrorType.AI_PROVIDER),
    SearchRateLimitError: (ErrorCode.RATE_LIMIT_EXCEEDED, ErrorType.RATE_LIMIT),
    SearchAuthenticationError: (ErrorCode.UNAUTHORIZED, ErrorType.AUTHENTICATION),
    # --- Storage / concurrency errors ---
    CursorError: (ErrorCode.INVALID_FORMAT, ErrorType.VALIDATION),
    LockAcquisitionError: (ErrorCode.UNAVAILABLE, ErrorType.UNAVAILABLE),
    VersionConflictError: (ErrorCode.VERSION_CONFLICT, ErrorType.CONFLICT),
    MigrationError: (ErrorCode.INTERNAL_ERROR, ErrorType.INTERNAL),
    SessionCorrupted: (ErrorCode.INTERNAL_ERROR, ErrorType.INTERNAL),
    # --- Authorization errors ---
    PathValidationError: (ErrorCode.FORBIDDEN, ErrorType.AUTHORIZATION),
    # --- Execution errors ---
    ActionRouterError: (ErrorCode.VALIDATION_ERROR, ErrorType.VALIDATION),
    ExecutorExhaustedError: (ErrorCode.UNAVAILABLE, ErrorType.UNAVAILABLE),
    # --- Resilience errors ---
    TimeoutException: (ErrorCode.AI_PROVIDER_TIMEOUT, ErrorType.UNAVAILABLE),
    CircuitBreakerError: (ErrorCode.UNAVAILABLE, ErrorType.UNAVAILABLE),
    TimeBudgetExceededError: (ErrorCode.AI_PROVIDER_TIMEOUT, ErrorType.UNAVAILABLE),
    RateLimitWaitError: (ErrorCode.RATE_LIMIT_EXCEEDED, ErrorType.RATE_LIMIT),
}


def error_to_response(exc: Exception) -> Optional[dict]:
    """Convert a known exception to a standard error_response dict, or None if unknown.

    Looks up the exception's *exact* type in ERROR_MAPPINGS and, if found,
    generates a standardized error response using the mapped ErrorCode and ErrorType.

    Args:
        exc: The exception to convert.

    Returns:
        A dict suitable for MCP tool response, or None if the exception type
        is not registered in ERROR_MAPPINGS.
    """
    mapping = ERROR_MAPPINGS.get(type(exc))
    if mapping is None:
        return None

    from dataclasses import asdict

    from foundry_mcp.core.responses.builders import error_response

    code, error_type = mapping
    return asdict(error_response(str(exc), error_code=code, error_type=error_type))
