"""Async retry with exponential backoff and jitter.

Standalone retry utility that can be used independently of the full
resilience stack (circuit breaker, rate limiter, etc.).
"""

import asyncio
import random
from typing import Awaitable, Callable, Optional, Type, TypeVar

from foundry_mcp.core.research.providers.resilience.models import SleepFunc

T = TypeVar("T")


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
