"""Unified error hierarchy for foundry-mcp.

All custom exception classes are defined in domain-specific modules within
this package. This __init__.py re-exports everything for convenient access.

Usage:
    # Import from domain modules for specificity
    from foundry_mcp.core.errors.llm import LLMError, RateLimitError

    # Or import from the package with qualified aliases for disambiguation
    from foundry_mcp.core.errors import LLMRateLimitError, SearchRateLimitError

    # Registry helper
    from foundry_mcp.core.errors import error_to_response
"""

# --- Base / Registry ---
# --- Authorization errors ---
from foundry_mcp.core.errors.authorization import (
    PathValidationError,
    StdinTimeoutError,
)
from foundry_mcp.core.errors.base import ERROR_MAPPINGS, error_to_response

# --- Execution errors ---
from foundry_mcp.core.errors.execution import (
    ActionRouterError,
    ExecutorExhaustedError,
    FoundryImplementV2Error,
)

# --- LLM errors ---
from foundry_mcp.core.errors.llm import (
    AuthenticationError as LLMAuthenticationError,
)
from foundry_mcp.core.errors.llm import (
    ContentFilterError,
    InvalidRequestError,
    LLMError,
    ModelNotFoundError,
)
from foundry_mcp.core.errors.llm import (
    RateLimitError as LLMRateLimitError,
)

# --- Provider errors ---
from foundry_mcp.core.errors.provider import (
    ContextWindowError,
    ProviderError,
    ProviderExecutionError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ValidationError,
)

# --- Research errors ---
from foundry_mcp.core.errors.research import (
    InvalidPDFError,
    PDFSecurityError,
    PDFSizeError,
    ProtectedContentOverflowError,
    ProviderExhaustedError,
    SSRFError,
    SummarizationError,
    SummarizationValidationError,
    UrlValidationError,
)

# --- Resilience errors ---
from foundry_mcp.core.errors.resilience import (
    CircuitBreakerError,
    RateLimitWaitError,
    TimeBudgetExceededError,
    TimeoutException,
)

# --- Search provider errors ---
from foundry_mcp.core.errors.search import (
    AuthenticationError as SearchAuthenticationError,
)
from foundry_mcp.core.errors.search import (
    RateLimitError as SearchRateLimitError,
)
from foundry_mcp.core.errors.search import (
    SearchProviderError,
)

# --- Storage errors ---
from foundry_mcp.core.errors.storage import (
    CursorError,
    LockAcquisitionError,
    MigrationError,
    SessionCorrupted,
    VersionConflictError,
)

__all__ = [
    # Base / Registry
    "ERROR_MAPPINGS",
    "error_to_response",
    # LLM errors (qualified aliases to disambiguate from search)
    "LLMError",
    "LLMRateLimitError",
    "LLMAuthenticationError",
    "ContentFilterError",
    "InvalidRequestError",
    "ModelNotFoundError",
    # Provider errors
    "ProviderError",
    "ProviderUnavailableError",
    "ProviderExecutionError",
    "ProviderTimeoutError",
    "ContextWindowError",
    "ValidationError",
    # Search provider errors (qualified aliases)
    "SearchProviderError",
    "SearchRateLimitError",
    "SearchAuthenticationError",
    # Research errors
    "PDFSecurityError",
    "SSRFError",
    "InvalidPDFError",
    "PDFSizeError",
    "SummarizationError",
    "ProviderExhaustedError",
    "SummarizationValidationError",
    "ProtectedContentOverflowError",
    "UrlValidationError",
    # Resilience errors
    "TimeoutException",
    "CircuitBreakerError",
    "TimeBudgetExceededError",
    "RateLimitWaitError",
    # Storage errors
    "CursorError",
    "LockAcquisitionError",
    "MigrationError",
    "VersionConflictError",
    "SessionCorrupted",
    # Authorization errors
    "PathValidationError",
    "StdinTimeoutError",
    # Execution errors
    "ExecutorExhaustedError",
    "ActionRouterError",
    "FoundryImplementV2Error",
]
