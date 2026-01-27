"""Provider resilience configuration and error classification.

Centralized resilience utilities for search providers including:
- Per-provider configuration for rate limiting, retries, and circuit breakers
- Error classification for unified retry/circuit-breaker decisions
- ProviderResilienceManager singleton for state management
"""

import asyncio
import random
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import (
    Awaitable,
    Callable,
    Optional,
    Protocol,
    Type,
    TypeVar,
)

T = TypeVar("T")

from foundry_mcp.core.context import get_correlation_id
from foundry_mcp.core.observability import audit_log
from foundry_mcp.core.rate_limit import RateLimitConfig, TokenBucketLimiter
from foundry_mcp.core.resilience import CircuitBreaker, CircuitBreakerError, CircuitState


class ErrorType(str, Enum):
    """Classification of error types for resilience decisions."""

    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    QUOTA_EXCEEDED = "quota_exceeded"
    NETWORK = "network"
    INVALID_REQUEST = "invalid_request"
    UNKNOWN = "unknown"


@dataclass
class ProviderResilienceConfig:
    """Per-provider resilience configuration.

    Configures rate limiting, retry behavior, and circuit breaker thresholds.
    """

    # Rate limiting
    requests_per_second: float = 1.0
    burst_limit: int = 3

    # Retry behavior
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: float = 0.5  # Fractional jitter range around base delay (0.0-1.0, 0.5 => 50-150%)

    # Circuit breaker
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: float = 30.0


@dataclass
class ErrorClassification:
    """Classification result for an error.

    Determines how the resilience layer should handle a specific error.
    """

    retryable: bool
    trips_breaker: bool
    backoff_seconds: Optional[float] = None
    error_type: ErrorType = ErrorType.UNKNOWN


# Provider-specific configurations with tuned defaults
PROVIDER_CONFIGS: dict[str, ProviderResilienceConfig] = {
    "tavily": ProviderResilienceConfig(
        requests_per_second=1.0,
        burst_limit=3,
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        jitter=0.5,
        circuit_failure_threshold=5,
        circuit_recovery_timeout=30.0,
    ),
    "google": ProviderResilienceConfig(
        requests_per_second=1.0,
        burst_limit=3,
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        jitter=0.5,
        circuit_failure_threshold=5,
        circuit_recovery_timeout=30.0,
    ),
    "perplexity": ProviderResilienceConfig(
        requests_per_second=1.0,
        burst_limit=3,
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        jitter=0.5,
        circuit_failure_threshold=5,
        circuit_recovery_timeout=30.0,
    ),
    "semantic_scholar": ProviderResilienceConfig(
        # Slightly under 1 RPS to stay within Semantic Scholar's rate limits
        requests_per_second=0.9,
        burst_limit=2,
        max_retries=3,
        base_delay=1.5,  # Slightly higher base delay for academic API
        max_delay=60.0,
        jitter=0.5,
        circuit_failure_threshold=5,
        circuit_recovery_timeout=30.0,
    ),
    "tavily_extract": ProviderResilienceConfig(
        requests_per_second=1.0,
        burst_limit=3,
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        jitter=0.5,
        circuit_failure_threshold=5,
        circuit_recovery_timeout=30.0,
    ),
}


def get_provider_config(provider_name: str) -> ProviderResilienceConfig:
    """Get resilience configuration for a provider.

    Args:
        provider_name: Name of the provider (e.g., 'tavily', 'google')

    Returns:
        Provider-specific config or default config if provider not found
    """
    return PROVIDER_CONFIGS.get(provider_name, ProviderResilienceConfig())


@dataclass
class ProviderStatus:
    """Status of a provider's resilience components."""

    provider_name: str
    is_available: bool
    circuit_state: str
    circuit_failure_count: int
    rate_limit_remaining: int
    rate_limit_reset_in: float


class ProviderResilienceManager:
    """Singleton manager for provider resilience state.

    Manages per-provider rate limiters and circuit breakers with lazy
    initialization. Provides state inspection API for observability.

    Thread-safe via threading.Lock for sync operations and asyncio.Lock for async.
    """

    _instance: Optional["ProviderResilienceManager"] = None
    _async_lock: asyncio.Lock
    _sync_lock: threading.Lock

    def __init__(self) -> None:
        """Initialize manager with empty state.

        Use get_resilience_manager() to get the singleton instance.
        """
        self._rate_limiters: dict[str, TokenBucketLimiter] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._async_lock = asyncio.Lock()
        self._sync_lock = threading.Lock()

    def _get_or_create_rate_limiter(
        self,
        provider_name: str,
        config: Optional[ProviderResilienceConfig] = None,
    ) -> TokenBucketLimiter:
        """Get or lazily create a rate limiter for a provider."""
        if provider_name not in self._rate_limiters:
            config = config or get_provider_config(provider_name)
            # Convert RPS to RPM for TokenBucketLimiter
            rpm = int(config.requests_per_second * 60)
            rate_config = RateLimitConfig(
                requests_per_minute=rpm,
                burst_limit=config.burst_limit,
                enabled=True,
                reason=f"{provider_name} rate limit",
            )
            self._rate_limiters[provider_name] = TokenBucketLimiter(rate_config)
        elif config is not None:
            # Update existing limiter config to honor overrides
            limiter = self._rate_limiters[provider_name]
            rpm = int(config.requests_per_second * 60)
            limiter.config.requests_per_minute = rpm
            limiter.config.burst_limit = config.burst_limit
            limiter.config.enabled = True
            limiter.config.reason = f"{provider_name} rate limit"
            limiter.state.tokens = min(
                limiter.state.tokens,
                float(limiter.config.burst_limit),
            )
        return self._rate_limiters[provider_name]

    def _get_or_create_circuit_breaker(
        self,
        provider_name: str,
        config: Optional[ProviderResilienceConfig] = None,
    ) -> CircuitBreaker:
        """Get or lazily create a circuit breaker for a provider."""
        if provider_name not in self._circuit_breakers:
            config = config or get_provider_config(provider_name)
            self._circuit_breakers[provider_name] = CircuitBreaker(
                name=provider_name,
                failure_threshold=config.circuit_failure_threshold,
                recovery_timeout=config.circuit_recovery_timeout,
            )
        elif config is not None:
            breaker = self._circuit_breakers[provider_name]
            breaker.failure_threshold = config.circuit_failure_threshold
            breaker.recovery_timeout = config.circuit_recovery_timeout
        return self._circuit_breakers[provider_name]

    def _is_breaker_available(self, breaker: CircuitBreaker) -> bool:
        """Check breaker availability without mutating probe counters."""
        with breaker._lock:
            if breaker.state == CircuitState.CLOSED:
                return True
            if breaker.state == CircuitState.OPEN:
                elapsed = time.time() - breaker.last_failure_time
                return elapsed >= breaker.recovery_timeout
            if breaker.state == CircuitState.HALF_OPEN:
                return breaker.half_open_calls < breaker.half_open_max_calls
            return False

    async def get_rate_limiter(
        self,
        provider_name: str,
        config: Optional[ProviderResilienceConfig] = None,
    ) -> TokenBucketLimiter:
        """Get rate limiter for a provider (thread-safe).

        Args:
            provider_name: Name of the provider

        Returns:
            TokenBucketLimiter for the provider
        """
        async with self._async_lock:
            return self._get_or_create_rate_limiter(provider_name, config=config)

    async def get_circuit_breaker(
        self,
        provider_name: str,
        config: Optional[ProviderResilienceConfig] = None,
    ) -> CircuitBreaker:
        """Get circuit breaker for a provider (thread-safe).

        Args:
            provider_name: Name of the provider

        Returns:
            CircuitBreaker for the provider
        """
        async with self._async_lock:
            return self._get_or_create_circuit_breaker(provider_name, config=config)

    def get_breaker_state(self, provider_name: str) -> CircuitState:
        """Get current circuit breaker state for a provider (thread-safe).

        Args:
            provider_name: Name of the provider

        Returns:
            CircuitState (CLOSED, OPEN, or HALF_OPEN)
        """
        with self._sync_lock:
            breaker = self._get_or_create_circuit_breaker(provider_name)
            return breaker.state

    def is_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is available (circuit not open, thread-safe).

        Args:
            provider_name: Name of the provider

        Returns:
            True if provider can accept requests
        """
        with self._sync_lock:
            breaker = self._get_or_create_circuit_breaker(provider_name)
            return self._is_breaker_available(breaker)

    def get_provider_status(self, provider_name: str) -> ProviderStatus:
        """Get comprehensive status for a provider (thread-safe).

        Args:
            provider_name: Name of the provider

        Returns:
            ProviderStatus with circuit and rate limit state
        """
        with self._sync_lock:
            breaker = self._get_or_create_circuit_breaker(provider_name)
            limiter = self._get_or_create_rate_limiter(provider_name)
            rate_check = limiter.check()
            is_available = self._is_breaker_available(breaker)

            return ProviderStatus(
                provider_name=provider_name,
                is_available=is_available,
                circuit_state=breaker.state.value,
                circuit_failure_count=breaker.failure_count,
                rate_limit_remaining=rate_check.remaining,
                rate_limit_reset_in=rate_check.reset_in,
            )

    def get_all_provider_statuses(self) -> dict[str, ProviderStatus]:
        """Get status for all known providers.

        Returns:
            Dict mapping provider names to their status
        """
        statuses = {}
        for provider_name in PROVIDER_CONFIGS:
            statuses[provider_name] = self.get_provider_status(provider_name)
        return statuses

    def reset(self) -> None:
        """Reset all resilience state (for testing)."""
        self._rate_limiters.clear()
        self._circuit_breakers.clear()


# Module-level singleton
_resilience_manager: Optional[ProviderResilienceManager] = None


def get_resilience_manager() -> ProviderResilienceManager:
    """Get the singleton ProviderResilienceManager instance.

    Returns:
        The global ProviderResilienceManager instance
    """
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = ProviderResilienceManager()
    return _resilience_manager


def reset_resilience_manager_for_testing() -> None:
    """Reset the singleton manager for test isolation.

    Creates a fresh manager instance with no state.
    """
    global _resilience_manager
    _resilience_manager = ProviderResilienceManager()


class SleepFunc(Protocol):
    """Protocol for injectable sleep function."""

    async def __call__(self, seconds: float) -> None: ...


async def async_retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[list[Type[Exception]]] = None,
    rng: Optional[random.Random] = None,
    sleep_func: Optional[SleepFunc] = None,
) -> T:
    """Async retry with exponential backoff and jitter.

    Retries an async function on failure with increasing delays.
    Jitter adds 50-150% randomness to delay to prevent thundering herd.

    Args:
        func: Async function to retry (no arguments; use lambda for args).
        max_retries: Maximum retry attempts (default 3).
        base_delay: Initial delay in seconds (default 1.0).
        max_delay: Maximum delay cap in seconds (default 60.0).
        exponential_base: Multiplier per retry (default 2.0).
        jitter: Add randomness to delay (default True, 50-150% of base).
        retryable_exceptions: Exceptions to retry on (default: all).
        rng: Injectable Random instance for deterministic testing.
        sleep_func: Injectable sleep function for time control in tests.

    Returns:
        Result from the function on success.

    Raises:
        Exception: The last exception if all retries exhausted.

    Example:
        >>> result = await async_retry_with_backoff(
        ...     lambda: http_client.get(url),
        ...     max_retries=3,
        ...     retryable_exceptions=[ConnectionError, TimeoutError],
        ... )

    Testing example:
        >>> seeded_rng = random.Random(42)
        >>> sleep_times = []
        >>> async def fake_sleep(s): sleep_times.append(s)
        >>> await async_retry_with_backoff(
        ...     func, rng=seeded_rng, sleep_func=fake_sleep
        ... )
    """
    retryable = tuple(retryable_exceptions or [Exception])
    last_exception: Optional[Exception] = None
    _rng = rng or random.Random()
    _sleep = sleep_func or asyncio.sleep

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retryable as e:
            last_exception = e

            if attempt == max_retries:
                break

            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base**attempt), max_delay)

            # Add jitter to prevent thundering herd (50-150% of delay)
            if jitter:
                jitter_factor = 0.5 + _rng.random()  # Range: 0.5 to 1.5
                delay = delay * jitter_factor

            await _sleep(delay)

    # All retries exhausted
    if last_exception:
        raise last_exception
    raise RuntimeError("async_retry_with_backoff: unexpected state")


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
