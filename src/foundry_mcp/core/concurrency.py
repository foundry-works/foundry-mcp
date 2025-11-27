"""
Concurrency utilities for foundry-mcp.

Provides concurrency limiting, cancellation handling, and request context
management for async MCP tool operations.

See docs/mcp_best_practices/15-concurrency-patterns.md for guidance.

Example:
    from foundry_mcp.core.concurrency import (
        ConcurrencyLimiter, with_cancellation, request_context
    )

    # Limit concurrent operations
    limiter = ConcurrencyLimiter(max_concurrent=10)
    results = await limiter.gather([fetch(url) for url in urls])

    # Handle cancellation gracefully
    @with_cancellation
    async def long_task():
        ...

    # Track request context
    async with request_context(request_id="abc", client_id="client1"):
        await process()
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

# Schema version for concurrency module
SCHEMA_VERSION = "1.0.0"

# Context variables for request-scoped state
request_id: ContextVar[str] = ContextVar("request_id", default="")
client_id: ContextVar[str] = ContextVar("client_id", default="anonymous")
start_time: ContextVar[float] = ContextVar("start_time", default=0.0)

# Type variable for async functions
T = TypeVar("T")


@dataclass
class ConcurrencyConfig:
    """Configuration for a concurrency limiter.

    Attributes:
        max_concurrent: Maximum number of concurrent operations
        name: Optional name for logging and identification
        timeout: Optional timeout per operation in seconds
    """

    max_concurrent: int = 10
    name: str = ""
    timeout: Optional[float] = None


@dataclass
class ConcurrencyStats:
    """Statistics from concurrent operation execution.

    Attributes:
        total: Total operations attempted
        succeeded: Operations completed successfully
        failed: Operations that raised exceptions
        cancelled: Operations that were cancelled
        timed_out: Operations that timed out
        elapsed_seconds: Total execution time
    """

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0
    timed_out: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class GatherResult:
    """Result of a gather operation with detailed status.

    Attributes:
        results: List of successful results (None for failed operations)
        errors: List of errors (None for successful operations)
        stats: Execution statistics
    """

    results: List[Any] = field(default_factory=list)
    errors: List[Optional[Exception]] = field(default_factory=list)
    stats: ConcurrencyStats = field(default_factory=ConcurrencyStats)

    @property
    def all_succeeded(self) -> bool:
        """Check if all operations succeeded."""
        return self.stats.failed == 0 and self.stats.cancelled == 0

    def successful_results(self) -> List[Any]:
        """Get only the successful results."""
        return [r for r, e in zip(self.results, self.errors) if e is None]

    def failed_results(self) -> List[tuple[int, Exception]]:
        """Get failed results with their indices."""
        return [(i, e) for i, e in enumerate(self.errors) if e is not None]


class ConcurrencyLimiter:
    """Limit concurrent async operations using a semaphore.

    Provides controlled concurrency for parallel operations like HTTP requests,
    database queries, or file operations to prevent resource exhaustion.

    Example:
        >>> limiter = ConcurrencyLimiter(max_concurrent=5)
        >>> results = await limiter.gather([fetch(url) for url in urls])
        >>> print(f"Completed {results.stats.succeeded}/{results.stats.total}")

        >>> # With timeout per operation
        >>> limiter = ConcurrencyLimiter(max_concurrent=3, timeout=30.0)
        >>> async with limiter.acquire():
        ...     await slow_operation()
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        *,
        name: str = "",
        timeout: Optional[float] = None,
    ):
        """Initialize concurrency limiter.

        Args:
            max_concurrent: Maximum number of concurrent operations (default: 10)
            name: Optional name for logging
            timeout: Optional timeout per operation in seconds
        """
        self.config = ConcurrencyConfig(
            max_concurrent=max_concurrent,
            name=name,
            timeout=timeout,
        )
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0
        self._total_count = 0

    @property
    def max_concurrent(self) -> int:
        """Get maximum concurrent operations."""
        return self.config.max_concurrent

    @property
    def active_count(self) -> int:
        """Get current number of active operations."""
        return self._active_count

    @asynccontextmanager
    async def acquire(self):
        """Acquire a slot for concurrent execution.

        Use as async context manager for single operations:

            async with limiter.acquire():
                await do_something()

        Yields:
            None (the slot is held until context exit)
        """
        async with self._semaphore:
            self._active_count += 1
            self._total_count += 1
            try:
                yield
            finally:
                self._active_count -= 1

    async def run(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        timeout: Optional[float] = None,
    ) -> T:
        """Run a coroutine with concurrency limiting.

        Args:
            coro: The coroutine to run
            timeout: Optional timeout override (uses limiter default if not provided)

        Returns:
            The result of the coroutine

        Raises:
            asyncio.TimeoutError: If operation times out
            asyncio.CancelledError: If operation is cancelled
        """
        effective_timeout = timeout if timeout is not None else self.config.timeout

        async with self.acquire():
            if effective_timeout:
                return await asyncio.wait_for(coro, timeout=effective_timeout)
            return await coro

    async def gather(
        self,
        coros: List[Coroutine[Any, Any, T]],
        *,
        return_exceptions: bool = False,
        timeout: Optional[float] = None,
    ) -> GatherResult:
        """Run multiple coroutines with concurrency limiting.

        Unlike asyncio.gather, this limits how many operations run in parallel.

        Args:
            coros: List of coroutines to execute
            return_exceptions: If True, exceptions are captured in results;
                if False, first exception stops execution
            timeout: Optional timeout per operation

        Returns:
            GatherResult with results, errors, and statistics

        Example:
            >>> limiter = ConcurrencyLimiter(max_concurrent=5)
            >>> result = await limiter.gather([
            ...     fetch(url) for url in urls
            ... ])
            >>> if result.all_succeeded:
            ...     process(result.results)
            ... else:
            ...     handle_errors(result.failed_results())
        """
        start = time.monotonic()
        stats = ConcurrencyStats(total=len(coros))
        results: List[Any] = [None] * len(coros)
        errors: List[Optional[Exception]] = [None] * len(coros)

        async def run_one(index: int, coro: Coroutine[Any, Any, T]) -> None:
            try:
                result = await self.run(coro, timeout=timeout)
                results[index] = result
                stats.succeeded += 1
            except asyncio.TimeoutError as e:
                errors[index] = e
                stats.timed_out += 1
                stats.failed += 1
                if not return_exceptions:
                    raise
            except asyncio.CancelledError as e:
                errors[index] = e
                stats.cancelled += 1
                stats.failed += 1
                if not return_exceptions:
                    raise
            except Exception as e:
                errors[index] = e
                stats.failed += 1
                if not return_exceptions:
                    raise

        try:
            tasks = [
                asyncio.create_task(run_one(i, coro))
                for i, coro in enumerate(coros)
            ]
            await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        except Exception:
            # Cancel remaining tasks on failure
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise
        finally:
            stats.elapsed_seconds = time.monotonic() - start

        return GatherResult(results=results, errors=errors, stats=stats)

    async def map(
        self,
        func: Callable[[T], Coroutine[Any, Any, Any]],
        items: List[T],
        *,
        return_exceptions: bool = False,
        timeout: Optional[float] = None,
    ) -> GatherResult:
        """Apply an async function to items with concurrency limiting.

        Convenience wrapper around gather for mapping operations.

        Args:
            func: Async function to apply to each item
            items: List of items to process
            return_exceptions: If True, capture exceptions in results
            timeout: Optional timeout per operation

        Returns:
            GatherResult with results

        Example:
            >>> async def fetch(url: str) -> dict:
            ...     async with aiohttp.get(url) as resp:
            ...         return await resp.json()
            >>> limiter = ConcurrencyLimiter(max_concurrent=10)
            >>> result = await limiter.map(fetch, urls)
        """
        coros = [func(item) for item in items]
        return await self.gather(
            coros,
            return_exceptions=return_exceptions,
            timeout=timeout,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get current limiter statistics.

        Returns:
            Dictionary with limiter state information
        """
        return {
            "max_concurrent": self.config.max_concurrent,
            "active_count": self._active_count,
            "total_processed": self._total_count,
            "name": self.config.name,
            "timeout": self.config.timeout,
        }


# Registry of per-tool concurrency limiters
_tool_limiters: Dict[str, ConcurrencyLimiter] = {}


def get_tool_limiter(
    tool_name: str,
    default_limit: int = 10,
) -> ConcurrencyLimiter:
    """Get or create a concurrency limiter for a tool.

    Args:
        tool_name: Name of the tool
        default_limit: Default max concurrent if not configured

    Returns:
        ConcurrencyLimiter instance for the tool
    """
    if tool_name not in _tool_limiters:
        _tool_limiters[tool_name] = ConcurrencyLimiter(
            max_concurrent=default_limit,
            name=tool_name,
        )
    return _tool_limiters[tool_name]


def configure_tool_limiter(
    tool_name: str,
    max_concurrent: int,
    *,
    timeout: Optional[float] = None,
) -> ConcurrencyLimiter:
    """Configure a concurrency limiter for a tool.

    Args:
        tool_name: Name of the tool
        max_concurrent: Maximum concurrent operations
        timeout: Optional timeout per operation

    Returns:
        Configured ConcurrencyLimiter instance
    """
    limiter = ConcurrencyLimiter(
        max_concurrent=max_concurrent,
        name=tool_name,
        timeout=timeout,
    )
    _tool_limiters[tool_name] = limiter
    logger.debug(
        f"Configured limiter for {tool_name}: max_concurrent={max_concurrent}"
    )
    return limiter


def get_all_limiter_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all configured tool limiters.

    Returns:
        Dictionary mapping tool names to their limiter stats
    """
    return {name: limiter.get_stats() for name, limiter in _tool_limiters.items()}


# Export all public symbols
__all__ = [
    "SCHEMA_VERSION",
    "ConcurrencyConfig",
    "ConcurrencyStats",
    "GatherResult",
    "ConcurrencyLimiter",
    "get_tool_limiter",
    "configure_tool_limiter",
    "get_all_limiter_stats",
    "request_id",
    "client_id",
    "start_time",
]
