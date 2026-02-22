"""Search provider error classes.

Moved from foundry_mcp.core.research.providers.base for centralized error management.
"""

from typing import Optional


class SearchProviderError(Exception):
    """Base exception for search provider errors.

    Attributes:
        provider: Name of the provider that raised the error
        message: Human-readable error description
        retryable: Whether the error is potentially transient
        original_error: The underlying exception if available
    """

    def __init__(
        self,
        provider: str,
        message: str,
        retryable: bool = False,
        original_error: Optional[Exception] = None,
    ):
        self.provider = provider
        self.message = message
        self.retryable = retryable
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")


class RateLimitError(SearchProviderError):
    """Raised when a provider's rate limit is exceeded.

    This error is always retryable. The retry_after field indicates
    how long to wait before retrying (if provided by the API).
    The optional reason can distinguish quota exhaustion from generic
    throttling for providers that expose that nuance.
    """

    def __init__(
        self,
        provider: str,
        retry_after: Optional[float] = None,
        reason: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.retry_after = retry_after
        self.reason = reason
        message = "Rate limit exceeded"
        if reason:
            message += f" ({reason})"
        if retry_after:
            message += f" (retry after {retry_after}s)"
        super().__init__(
            provider=provider,
            message=message,
            retryable=True,
            original_error=original_error,
        )


class AuthenticationError(SearchProviderError):
    """Raised when API authentication fails.

    This error is NOT retryable - the API key or credentials
    need to be fixed before retrying.
    """

    def __init__(
        self,
        provider: str,
        message: str = "Authentication failed",
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            provider=provider,
            message=message,
            retryable=False,
            original_error=original_error,
        )
