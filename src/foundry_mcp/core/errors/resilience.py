"""Resilience error classes.

Moved from foundry_mcp.core.resilience and
foundry_mcp.core.research.providers.resilience for centralized error management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from foundry_mcp.core.resilience import CircuitState


class TimeoutException(Exception):
    """Operation timed out.

    Attributes:
        timeout_seconds: The timeout duration that was exceeded.
        operation: Name of the operation that timed out.
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
    ):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class CircuitBreakerError(Exception):
    """Circuit breaker is open and rejecting requests.

    Attributes:
        breaker_name: Name of the circuit breaker.
        state: Current state of the breaker.
        retry_after: Seconds until recovery timeout.
    """

    def __init__(
        self,
        message: str,
        breaker_name: Optional[str] = None,
        state: Optional[CircuitState] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.breaker_name = breaker_name
        self.state = state
        self.retry_after = retry_after


class TimeBudgetExceededError(Exception):
    """Time budget for operation has been exhausted.

    Attributes:
        budget_seconds: The original time budget.
        elapsed_seconds: Time elapsed before budget exceeded.
        operation: Name of the operation.
    """

    def __init__(
        self,
        message: str,
        budget_seconds: Optional[float] = None,
        elapsed_seconds: Optional[float] = None,
        operation: Optional[str] = None,
    ):
        super().__init__(message)
        self.budget_seconds = budget_seconds
        self.elapsed_seconds = elapsed_seconds
        self.operation = operation


class RateLimitWaitError(Exception):
    """Rate limiter wait time would exceed max_wait_seconds.

    Attributes:
        wait_needed: Seconds needed to wait for token.
        max_wait: Maximum allowed wait time.
        provider: Provider name.
    """

    def __init__(
        self,
        message: str,
        wait_needed: Optional[float] = None,
        max_wait: Optional[float] = None,
        provider: Optional[str] = None,
    ):
        super().__init__(message)
        self.wait_needed = wait_needed
        self.max_wait = max_wait
        self.provider = provider
