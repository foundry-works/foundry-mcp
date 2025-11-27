"""
Resilience primitives for MCP tool operations.

Provides timeout budgets, retry patterns, circuit breakers, and health checks
for building robust MCP tools that handle failures gracefully.

Timeout Budget Categories
=========================

Use the appropriate timeout category based on operation type:

    FAST_TIMEOUT (5s)       - Cache lookups, simple queries
    MEDIUM_TIMEOUT (30s)    - Database operations, API calls
    SLOW_TIMEOUT (120s)     - File processing, complex operations
    BACKGROUND_TIMEOUT (600s) - Batch jobs, large transfers

Example usage:

    from foundry_mcp.core.resilience import (
        MEDIUM_TIMEOUT,
        with_timeout,
        retry_with_backoff,
        CircuitBreaker,
    )

    @mcp.tool()
    @with_timeout(MEDIUM_TIMEOUT, "Database query timed out")
    async def query_database(query: str) -> dict:
        result = await db.execute(query)
        return asdict(success_response(data={"result": result}))
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar
import asyncio
import random
import time


# ---------------------------------------------------------------------------
# Timeout Budget Constants
# ---------------------------------------------------------------------------

#: Fast operations: cache lookups, simple queries (default 5s, max 10s)
FAST_TIMEOUT: float = 5.0
FAST_TIMEOUT_MAX: float = 10.0

#: Medium operations: database ops, API calls (default 30s, max 60s)
MEDIUM_TIMEOUT: float = 30.0
MEDIUM_TIMEOUT_MAX: float = 60.0

#: Slow operations: file processing, complex operations (default 120s, max 300s)
SLOW_TIMEOUT: float = 120.0
SLOW_TIMEOUT_MAX: float = 300.0

#: Background operations: batch jobs, large transfers (default 600s, max 3600s)
BACKGROUND_TIMEOUT: float = 600.0
BACKGROUND_TIMEOUT_MAX: float = 3600.0


T = TypeVar("T")


# ---------------------------------------------------------------------------
# Timeout Error
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Timeout Decorator
# ---------------------------------------------------------------------------


def with_timeout(
    seconds: float,
    error_message: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add timeout to async functions.

    Uses asyncio.wait_for to enforce timeout on async operations.
    On timeout, raises TimeoutException with details.

    Args:
        seconds: Timeout duration in seconds.
        error_message: Custom error message (defaults to function name).

    Returns:
        Decorated async function with timeout enforcement.

    Example:
        >>> @with_timeout(30, "Database query timed out")
        ... async def query_database(query: str):
        ...     return await db.execute(query)

    Raises:
        TimeoutException: If the operation exceeds the timeout.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds,
                )
            except asyncio.TimeoutError:
                msg = error_message or f"{func.__name__} timed out after {seconds}s"
                raise TimeoutException(
                    msg,
                    timeout_seconds=seconds,
                    operation=func.__name__,
                )

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Retry with Backoff
# ---------------------------------------------------------------------------


def retry_with_backoff(
    func: Callable[..., T],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[List[Type[Exception]]] = None,
) -> T:
    """Retry a function with exponential backoff.

    Retries the function on failure with increasing delays between attempts.
    Supports jitter to prevent thundering herd problems.

    Args:
        func: Function to retry (should take no arguments; use lambda for args).
        max_retries: Maximum number of retry attempts (default 3).
        base_delay: Initial delay in seconds (default 1.0).
        max_delay: Maximum delay cap in seconds (default 60.0).
        exponential_base: Multiplier for each retry (default 2.0).
        jitter: Add randomness to delay (default True).
        retryable_exceptions: List of exceptions to retry on (default: all).

    Returns:
        Result from the function on success.

    Raises:
        Exception: The last exception if all retries exhausted.

    Example:
        >>> result = retry_with_backoff(
        ...     lambda: http_client.get(url),
        ...     max_retries=3,
        ...     retryable_exceptions=[ConnectionError, TimeoutException],
        ... )
    """
    retryable = tuple(retryable_exceptions or [Exception])
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except retryable as e:
            last_exception = e

            if attempt == max_retries:
                break

            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base**attempt), max_delay)

            # Add jitter to prevent thundering herd
            if jitter:
                delay = delay * (0.5 + random.random())

            time.sleep(delay)

    # All retries exhausted
    if last_exception:
        raise last_exception
    raise RuntimeError("retry_with_backoff: unexpected state")


def retryable(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for automatic retries with exponential backoff.

    Args:
        max_retries: Maximum retry attempts (default 3).
        delay: Base delay in seconds (default 1.0).
        exceptions: Tuple of exceptions to retry on.

    Returns:
        Decorated function with retry logic.

    Example:
        >>> @retryable(max_retries=3, exceptions=(ConnectionError,))
        ... def call_api(endpoint: str):
        ...     return http_client.get(endpoint)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return retry_with_backoff(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                base_delay=delay,
                retryable_exceptions=list(exceptions),
            )

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class CircuitState(Enum):
    """Circuit breaker states.

    CLOSED: Normal operation, requests flow through.
    OPEN: Failures exceeded threshold, requests rejected.
    HALF_OPEN: Testing recovery, limited requests allowed.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


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


@dataclass
class CircuitBreaker:
    """Circuit breaker for external dependencies.

    Prevents cascade failures by tracking failures and temporarily
    blocking requests when a dependency is unhealthy.

    States:
        CLOSED: Normal operation, requests pass through.
        OPEN: Too many failures, requests rejected immediately.
        HALF_OPEN: Testing recovery, limited requests allowed.

    Attributes:
        name: Identifier for this circuit breaker.
        failure_threshold: Failures before opening circuit (default 5).
        recovery_timeout: Seconds before testing recovery (default 30).
        half_open_max_calls: Test calls allowed in half-open (default 3).

    Example:
        >>> breaker = CircuitBreaker(name="database")
        >>>
        >>> if breaker.can_execute():
        ...     try:
        ...         result = db.query()
        ...         breaker.record_success()
        ...     except Exception:
        ...         breaker.record_failure()
        ...         raise
        ... else:
        ...     raise CircuitBreakerError("Database circuit open")
    """

    name: str = "default"
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3

    # Internal state (initialized in __post_init__)
    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    failure_count: int = field(default=0, init=False)
    last_failure_time: float = field(default=0.0, init=False)
    half_open_calls: int = field(default=0, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)

    def can_execute(self) -> bool:
        """Check if request should proceed.

        Returns:
            True if request can proceed, False if circuit is open.
        """
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    return True
                return False

            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls < self.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False

            return False

    def record_success(self) -> None:
        """Record successful call.

        In HALF_OPEN state, successful calls contribute to recovery.
        Once enough calls succeed, circuit closes.
        Note: half_open_calls is already incremented in can_execute().
        """
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                # Check if enough successful calls for recovery
                # (counter already incremented in can_execute)
                if self.half_open_calls >= self.half_open_max_calls:
                    # Recovery successful
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            else:
                # Reset failure count on success
                self.failure_count = 0

    def record_failure(self) -> None:
        """Record failed call.

        Increments failure count. If threshold exceeded, opens circuit.
        In HALF_OPEN state, any failure returns to OPEN.
        """
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                # Recovery failed, back to open
                self.state = CircuitState.OPEN

            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
            self.last_failure_time = 0.0

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status.

        Returns:
            Dict with state, failure_count, and other metrics.
        """
        with self._lock:
            retry_after = None
            if self.state == CircuitState.OPEN:
                elapsed = time.time() - self.last_failure_time
                retry_after = max(0.0, self.recovery_timeout - elapsed)

            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "retry_after_seconds": retry_after,
            }


def with_circuit_breaker(
    breaker: CircuitBreaker,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to wrap function with circuit breaker protection.

    Args:
        breaker: CircuitBreaker instance to use.

    Returns:
        Decorated function that checks circuit before execution.

    Example:
        >>> db_breaker = CircuitBreaker(name="database", failure_threshold=3)
        >>>
        >>> @with_circuit_breaker(db_breaker)
        ... def query_database(sql: str):
        ...     return db.execute(sql)

    Raises:
        CircuitBreakerError: If circuit is open and rejecting requests.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not breaker.can_execute():
                status = breaker.get_status()
                raise CircuitBreakerError(
                    f"Circuit breaker '{breaker.name}' is open",
                    breaker_name=breaker.name,
                    state=breaker.state,
                    retry_after=status.get("retry_after_seconds"),
                )

            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception:
                breaker.record_failure()
                raise

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Health Check Utilities
# ---------------------------------------------------------------------------


@dataclass
class HealthStatus:
    """Health status for a dependency.

    Attributes:
        name: Dependency identifier.
        healthy: Whether dependency is healthy.
        latency_ms: Check latency in milliseconds.
        last_check: Timestamp of the check.
        error: Error message if unhealthy.
    """

    name: str
    healthy: bool
    latency_ms: float
    last_check: datetime
    error: Optional[str] = None


async def health_check(
    name: str,
    check_func: Callable[[], Any],
    timeout: float = FAST_TIMEOUT,
) -> HealthStatus:
    """Check health of a dependency with timeout.

    Args:
        name: Identifier for the dependency.
        check_func: Async callable that tests dependency health.
        timeout: Maximum time to wait for check (default FAST_TIMEOUT).

    Returns:
        HealthStatus with check results.

    Example:
        >>> status = await health_check(
        ...     "database",
        ...     lambda: db.execute("SELECT 1"),
        ... )
        >>> if not status.healthy:
        ...     logger.warning(f"DB unhealthy: {status.error}")
    """
    start = time.perf_counter()
    try:
        result = check_func()
        # Handle both sync and async callables
        if asyncio.iscoroutine(result):
            await asyncio.wait_for(result, timeout=timeout)
        latency = (time.perf_counter() - start) * 1000

        return HealthStatus(
            name=name,
            healthy=True,
            latency_ms=latency,
            last_check=datetime.utcnow(),
        )
    except asyncio.TimeoutError:
        latency = (time.perf_counter() - start) * 1000
        return HealthStatus(
            name=name,
            healthy=False,
            latency_ms=latency,
            last_check=datetime.utcnow(),
            error=f"Health check timed out after {timeout}s",
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return HealthStatus(
            name=name,
            healthy=False,
            latency_ms=latency,
            last_check=datetime.utcnow(),
            error=str(e),
        )


async def check_dependencies(
    checks: Dict[str, Callable[[], Any]],
    timeout_per_check: float = FAST_TIMEOUT,
) -> Dict[str, Any]:
    """Check health of multiple dependencies concurrently.

    Args:
        checks: Dict mapping dependency names to check functions.
        timeout_per_check: Timeout per individual check.

    Returns:
        Dict with overall status and per-dependency results.

    Example:
        >>> results = await check_dependencies({
        ...     "database": lambda: db.execute("SELECT 1"),
        ...     "cache": lambda: cache.ping(),
        ...     "api": lambda: http.get(health_url),
        ... })
        >>> if results["status"] == "degraded":
        ...     logger.warning(f"Unhealthy: {results['unhealthy']}")
    """
    results: Dict[str, Dict[str, Any]] = {}

    # Run all checks concurrently
    statuses = await asyncio.gather(
        *[
            health_check(name, check_func, timeout_per_check)
            for name, check_func in checks.items()
        ],
        return_exceptions=False,
    )

    unhealthy: List[str] = []
    for status in statuses:
        results[status.name] = {
            "healthy": status.healthy,
            "latency_ms": round(status.latency_ms, 2),
            "error": status.error,
        }
        if not status.healthy:
            unhealthy.append(status.name)

    return {
        "status": "healthy" if not unhealthy else "degraded",
        "dependencies": results,
        "unhealthy": unhealthy,
        "checked_at": datetime.utcnow().isoformat(),
    }
