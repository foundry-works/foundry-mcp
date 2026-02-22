"""LLM provider error classes.

Moved from foundry_mcp.core.llm_provider for centralized error management.
"""

from typing import Optional


class LLMError(Exception):
    """Base exception for LLM operations.

    Attributes:
        message: Human-readable error description
        provider: Name of the provider that raised the error
        retryable: Whether the operation can be retried
        status_code: HTTP status code if applicable
    """

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        retryable: bool = False,
        status_code: Optional[int] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code


class RateLimitError(LLMError):
    """Rate limit exceeded error.

    Attributes:
        retry_after: Seconds to wait before retrying
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        provider: Optional[str] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message, provider=provider, retryable=True, status_code=429)
        self.retry_after = retry_after


class AuthenticationError(LLMError):
    """Authentication failed error."""

    def __init__(
        self,
        message: str = "Authentication failed",
        *,
        provider: Optional[str] = None,
    ):
        super().__init__(message, provider=provider, retryable=False, status_code=401)


class InvalidRequestError(LLMError):
    """Invalid request error (bad parameters, etc.)."""

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        param: Optional[str] = None,
    ):
        super().__init__(message, provider=provider, retryable=False, status_code=400)
        self.param = param


class ModelNotFoundError(LLMError):
    """Requested model not found or not accessible."""

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(message, provider=provider, retryable=False, status_code=404)
        self.model = model


class ContentFilterError(LLMError):
    """Content was filtered due to policy violation."""

    def __init__(
        self,
        message: str = "Content filtered",
        *,
        provider: Optional[str] = None,
    ):
        super().__init__(message, provider=provider, retryable=False, status_code=400)
