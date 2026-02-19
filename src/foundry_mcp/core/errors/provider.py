"""Provider orchestration error classes.

Moved from foundry_mcp.core.providers.base and foundry_mcp.core.providers.validation
for centralized error management.
"""

from typing import Any, Optional


class ProviderError(RuntimeError):
    """Base exception for provider orchestration errors."""

    def __init__(self, message: str, *, provider: Optional[str] = None):
        self.provider = provider
        super().__init__(message)


class ProviderUnavailableError(ProviderError):
    """Raised when a provider cannot be instantiated (binary missing, auth issues)."""


class ProviderExecutionError(ProviderError):
    """Raised when a provider command returns a non-retryable error."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider exceeds its allotted execution time.

    This error indicates the provider did not respond within the configured
    timeout period. It includes timing information to help with debugging
    and timeout configuration tuning.

    Attributes:
        provider: Provider that timed out
        elapsed: Actual elapsed time in seconds before timeout
        timeout: Configured timeout value in seconds
    """

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        elapsed: Optional[float] = None,
        timeout: Optional[float] = None,
    ):
        """Initialize provider timeout error.

        Args:
            message: Error message describing the timeout
            provider: Provider that timed out
            elapsed: Actual elapsed time in seconds
            timeout: Configured timeout in seconds
        """
        super().__init__(message, provider=provider)
        self.elapsed = elapsed
        self.timeout = timeout


class ContextWindowError(ProviderExecutionError):
    """Raised when prompt exceeds the model's context window limit.

    This error indicates the prompt/context size exceeded what the model
    can process. It includes token counts to help with debugging and
    provides actionable guidance for resolution.

    Attributes:
        prompt_tokens: Estimated tokens in the prompt (if known)
        max_tokens: Maximum context window size (if known)
        provider: Provider that raised the error
        truncation_needed: How many tokens need to be removed
    """

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        """Initialize context window error.

        Args:
            message: Error message describing the issue
            provider: Provider ID that raised the error
            prompt_tokens: Number of tokens in the prompt (if known)
            max_tokens: Maximum tokens allowed (if known)
        """
        super().__init__(message, provider=provider)
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.truncation_needed = (prompt_tokens - max_tokens) if prompt_tokens and max_tokens else None


class ValidationError(ProviderExecutionError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs: Any) -> None:
        self.field = field
        super().__init__(message, **kwargs)
