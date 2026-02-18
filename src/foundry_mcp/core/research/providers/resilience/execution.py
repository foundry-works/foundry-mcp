"""Full resilience stack execution.

Combines circuit breaker, rate limiting, timeout, and retry with
proper error classification and state recording.
"""

import asyncio
import random
import time
from typing import Awaitable, Callable, Optional, TypeVar

from foundry_mcp.core.context import get_correlation_id
from foundry_mcp.core.observability import audit_log
from foundry_mcp.core.errors.resilience import (
    CircuitBreakerError,
    RateLimitWaitError,
    TimeBudgetExceededError,
)
from foundry_mcp.core.resilience import CircuitState

from foundry_mcp.core.research.providers.resilience.config import get_provider_config
from foundry_mcp.core.research.providers.resilience.manager import (
    ProviderResilienceManager,
    get_resilience_manager,
)
from foundry_mcp.core.research.providers.resilience.models import (
    ErrorClassification,
    ErrorType,
    ProviderResilienceConfig,
)

T = TypeVar("T")


async def execute_with_resilience(
    func: Callable[[], Awaitable[T]],
    provider_name: str,
    *,
    time_budget: Optional[float] = None,
    max_wait_seconds: float = 5.0,
    classify_error: Optional[Callable[[Exception], ErrorClassification]] = None,
    manager: Optional[ProviderResilienceManager] = None,
    resilience_config: Optional[ProviderResilienceConfig] = None,
) -> T:
    """Execute an async function with full resilience stack.

    Combines circuit breaker, rate limiting, timeout, and retry with
    proper error classification and state recording.

    Execution order:
    1. Check circuit breaker (fail fast if OPEN)
    2. Acquire rate limiter token (wait up to max_wait_seconds)
    3. Execute with time budget (remaining budget minus overhead)
    4. Retry transient failures with jitter
    5. Record success/failure to circuit breaker

    Args:
        func: Async function to execute (no arguments; use lambda for args).
        provider_name: Name of the provider for resilience lookup.
        time_budget: Total time budget in seconds (None = no limit).
        max_wait_seconds: Max time to wait for rate limit token (default 5.0).
        classify_error: Custom error classifier (default uses HTTP status patterns).
        manager: Optional resilience manager (uses singleton if None).
        resilience_config: Optional per-provider config override.

    Returns:
        Result from the function on success.

    Raises:
        CircuitBreakerError: If circuit is open and rejecting requests.
        RateLimitWaitError: If rate limit wait would exceed max_wait_seconds.
        TimeBudgetExceededError: If time budget exhausted during execution.
        Exception: Original exception if all retries exhausted.

    Example:
        >>> result = await execute_with_resilience(
        ...     lambda: http_client.get(url),
        ...     provider_name="tavily",
        ...     time_budget=30.0,
        ... )
    """
    _manager = manager or get_resilience_manager()
    config = resilience_config or get_provider_config(provider_name)
    start_time = time.monotonic()
    correlation_id = get_correlation_id()

    def remaining_budget() -> Optional[float]:
        if time_budget is None:
            return None
        elapsed = time.monotonic() - start_time
        return max(0.0, time_budget - elapsed)

    def _audit(event_type: str, **details: object) -> None:
        """Log audit event with correlation_id if available."""
        if correlation_id:
            details["correlation_id"] = correlation_id
        details["provider"] = provider_name
        audit_log(event_type, **details)

    # Step 1: Initialize circuit breaker
    breaker = _manager._get_or_create_circuit_breaker(provider_name, config=config)

    limiter = _manager._get_or_create_rate_limiter(provider_name, config=config)

    async def _acquire_rate_limit_token(attempt: int) -> None:
        """Acquire a rate limit token, waiting if needed."""
        while True:
            rate_result = limiter.check()
            if rate_result.allowed:
                acquire_result = limiter.acquire()
                if acquire_result.allowed:
                    return
                rate_result = acquire_result

            wait_needed = rate_result.reset_in
            if wait_needed > max_wait_seconds:
                raise RateLimitWaitError(
                    f"Rate limit wait {wait_needed:.1f}s exceeds max {max_wait_seconds:.1f}s",
                    wait_needed=wait_needed,
                    max_wait=max_wait_seconds,
                    provider=provider_name,
                )

            budget = remaining_budget()
            if budget is not None and wait_needed > budget:
                _audit(
                    "budget_exceeded",
                    elapsed_ms=int((time.monotonic() - start_time) * 1000),
                    budget_ms=int((time_budget or 0) * 1000),
                    attempts=attempt,
                    phase="rate_limit_wait",
                )
                raise TimeBudgetExceededError(
                    f"Rate limit wait {wait_needed:.1f}s exceeds remaining budget {budget:.1f}s",
                    budget_seconds=time_budget,
                    elapsed_seconds=time.monotonic() - start_time,
                    operation=provider_name,
                )

            _audit(
                "rate_limit_wait",
                wait_ms=int(wait_needed * 1000),
                attempt=attempt + 1,
            )
            await asyncio.sleep(wait_needed)

    # Step 4: Execute with retry
    last_exception: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            # Check circuit breaker at the start of each attempt
            old_state = breaker.state
            if not breaker.can_execute():
                _audit(
                    "circuit_state_change",
                    old_state=old_state.value,
                    new_state=breaker.state.value,
                    action="rejected",
                )
                raise CircuitBreakerError(
                    f"Circuit breaker open for {provider_name}",
                    breaker_name=provider_name,
                    state=breaker.state,
                    retry_after=config.circuit_recovery_timeout,
                )

            # Check remaining budget before acquiring token
            budget = remaining_budget()
            if budget is not None and budget <= 0:
                _audit(
                    "budget_exceeded",
                    elapsed_ms=int((time.monotonic() - start_time) * 1000),
                    budget_ms=int((time_budget or 0) * 1000),
                    attempts=attempt,
                    phase="pre_execution",
                )
                raise TimeBudgetExceededError(
                    f"Time budget exhausted before execution for {provider_name}",
                    budget_seconds=time_budget,
                    elapsed_seconds=time.monotonic() - start_time,
                    operation=provider_name,
                )

            await _acquire_rate_limit_token(attempt)

            # Apply timeout if we have a budget
            budget = remaining_budget()
            if budget is not None:
                if budget <= 0:
                    _audit(
                        "budget_exceeded",
                        elapsed_ms=int((time.monotonic() - start_time) * 1000),
                        budget_ms=int((time_budget or 0) * 1000),
                        attempts=attempt,
                        phase="during_retry",
                    )
                    raise TimeBudgetExceededError(
                        f"Time budget exhausted during retry for {provider_name}",
                        budget_seconds=time_budget,
                        elapsed_seconds=time.monotonic() - start_time,
                        operation=provider_name,
                    )
                result = await asyncio.wait_for(func(), timeout=budget)
            else:
                result = await func()

            # Step 5: Record success and log circuit state if it changed
            pre_success_state = breaker.state
            breaker.record_success()
            if breaker.state != pre_success_state:
                _audit(
                    "circuit_state_change",
                    old_state=pre_success_state.value,
                    new_state=breaker.state.value,
                    action="recovery",
                )
            return result

        except asyncio.TimeoutError:
            _audit(
                "budget_exceeded",
                elapsed_ms=int((time.monotonic() - start_time) * 1000),
                budget_ms=int((time_budget or 0) * 1000),
                attempts=attempt + 1,
                phase="timeout",
            )
            last_exception = TimeBudgetExceededError(
                f"Operation timed out for {provider_name}",
                budget_seconds=time_budget,
                elapsed_seconds=time.monotonic() - start_time,
                operation=provider_name,
            )
            pre_failure_state = breaker.state
            breaker.record_failure()
            if breaker.state != pre_failure_state:
                _audit(
                    "circuit_state_change",
                    old_state=pre_failure_state.value,
                    new_state=breaker.state.value,
                    action="tripped",
                )
            break  # No retry on timeout

        except RateLimitWaitError:
            # Avoid consuming HALF_OPEN probe slots on local rate-limit waits.
            if breaker.state == CircuitState.HALF_OPEN:
                with breaker._lock:
                    if breaker.half_open_calls > 0:
                        breaker.half_open_calls -= 1
            raise

        except Exception as e:
            last_exception = e

            # Classify the error
            if classify_error:
                classification = classify_error(e)
            else:
                classification = _default_classify_error(e)

            # Record failure if it trips breaker
            if classification.trips_breaker:
                pre_failure_state = breaker.state
                breaker.record_failure()
                if breaker.state != pre_failure_state:
                    _audit(
                        "circuit_state_change",
                        old_state=pre_failure_state.value,
                        new_state=breaker.state.value,
                        action="tripped",
                    )
                if breaker.state == CircuitState.OPEN:
                    break

            # Check if retryable
            if not classification.retryable or attempt == config.max_retries:
                break

            # Log retry attempt
            _audit(
                "retry_attempt",
                attempt=attempt + 1,
                max_attempts=config.max_retries + 1,
                error_type=classification.error_type.value,
                error_message=str(e)[:200],
            )

            # Calculate retry delay
            delay = min(
                config.base_delay * (2.0**attempt),
                config.max_delay,
            )
            if classification.backoff_seconds is not None:
                delay = max(delay, classification.backoff_seconds)

            # Apply jitter within +/- jitter fraction unless backoff_seconds is explicit
            if config.jitter > 0 and classification.backoff_seconds is None:
                jitter = min(max(config.jitter, 0.0), 1.0)
                jitter_factor = (1.0 - jitter) + (2.0 * jitter * random.random())
                delay = delay * jitter_factor

            # Check budget before sleeping
            budget = remaining_budget()
            if budget is not None and delay > budget:
                _audit(
                    "budget_exceeded",
                    elapsed_ms=int((time.monotonic() - start_time) * 1000),
                    budget_ms=int((time_budget or 0) * 1000),
                    attempts=attempt + 1,
                    phase="retry_delay",
                )
                raise TimeBudgetExceededError(
                    f"Retry delay {delay:.1f}s exceeds remaining budget {budget:.1f}s",
                    budget_seconds=time_budget,
                    elapsed_seconds=time.monotonic() - start_time,
                    operation=provider_name,
                )

            await asyncio.sleep(delay)

    # All retries exhausted
    if last_exception:
        raise last_exception
    raise RuntimeError("execute_with_resilience: unexpected state")


def _default_classify_error(error: Exception) -> ErrorClassification:
    """Default error classifier based on common HTTP patterns.

    Classifies errors for retry and circuit breaker decisions.
    """
    error_str = str(error).lower()
    error_type_name = type(error).__name__.lower()

    # Rate limit errors (retryable, don't trip breaker)
    if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
        return ErrorClassification(
            retryable=True,
            trips_breaker=False,
            error_type=ErrorType.RATE_LIMIT,
        )

    # Authentication errors (not retryable, trip breaker)
    if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
        return ErrorClassification(
            retryable=False,
            trips_breaker=True,
            error_type=ErrorType.AUTHENTICATION,
        )

    # Server errors (retryable, trip breaker)
    if any(code in error_str for code in ["500", "502", "503", "504"]):
        return ErrorClassification(
            retryable=True,
            trips_breaker=True,
            error_type=ErrorType.SERVER_ERROR,
        )

    # Timeout errors (retryable, trip breaker)
    if "timeout" in error_str or "timed out" in error_str or "timeouterror" in error_type_name:
        return ErrorClassification(
            retryable=True,
            trips_breaker=True,
            error_type=ErrorType.TIMEOUT,
        )

    # Network errors (retryable, trip breaker)
    if any(
        term in error_str
        for term in ["connection", "network", "dns", "refused", "reset"]
    ):
        return ErrorClassification(
            retryable=True,
            trips_breaker=True,
            error_type=ErrorType.NETWORK,
        )

    # Default: not retryable, trips breaker
    return ErrorClassification(
        retryable=False,
        trips_breaker=True,
        error_type=ErrorType.UNKNOWN,
    )
