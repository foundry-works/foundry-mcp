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


# -----------------------------------------------------------------------------
# Request Context Management
# -----------------------------------------------------------------------------


@dataclass
class RequestContext:
    """Snapshot of request context for logging and tracking.

    Attributes:
        request_id: Unique identifier for this request
        client_id: Client making the request
        start_time: When the request started (monotonic time)
        start_timestamp: When the request started (wall clock)
    """

    request_id: str
    client_id: str
    start_time: float
    start_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time since request started."""
        return time.monotonic() - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed_seconds * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "request_id": self.request_id,
            "client_id": self.client_id,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "start_timestamp": self.start_timestamp.isoformat(),
        }


@asynccontextmanager
async def request_context(
    req_id: Optional[str] = None,
    cli_id: Optional[str] = None,
):
    """Set up request context for async operations.

    Context automatically propagates through all nested async calls.

    Args:
        req_id: Request ID (auto-generated UUID if not provided)
        cli_id: Client ID (defaults to "anonymous")

    Yields:
        RequestContext object with current context values

    Example:
        >>> async with request_context(cli_id="user123") as ctx:
        ...     print(f"Request {ctx.request_id} started")
        ...     await do_work()
        ...     print(f"Completed in {ctx.elapsed_ms:.2f}ms")
    """
    # Generate request ID if not provided
    actual_req_id = req_id or str(uuid.uuid4())
    actual_cli_id = cli_id or "anonymous"
    actual_start = time.monotonic()

    # Set context variables
    token1 = request_id.set(actual_req_id)
    token2 = client_id.set(actual_cli_id)
    token3 = start_time.set(actual_start)

    ctx = RequestContext(
        request_id=actual_req_id,
        client_id=actual_cli_id,
        start_time=actual_start,
    )

    try:
        logger.debug(
            "Request context started",
            extra={"request_id": actual_req_id, "client_id": actual_cli_id},
        )
        yield ctx
    finally:
        elapsed = ctx.elapsed_ms
        logger.debug(
            "Request context ended",
            extra={
                "request_id": actual_req_id,
                "client_id": actual_cli_id,
                "elapsed_ms": round(elapsed, 2),
            },
        )
        # Reset context variables
        request_id.reset(token1)
        client_id.reset(token2)
        start_time.reset(token3)


def get_current_context() -> RequestContext:
    """Get the current request context.

    Returns:
        RequestContext with current context variable values

    Raises:
        RuntimeError: If called outside of a request context
    """
    req_id = request_id.get()
    if not req_id:
        raise RuntimeError(
            "get_current_context() called outside of request_context"
        )

    return RequestContext(
        request_id=req_id,
        client_id=client_id.get(),
        start_time=start_time.get(),
    )


def get_current_context_or_none() -> Optional[RequestContext]:
    """Get the current request context, or None if not in a context.

    Returns:
        RequestContext if in a request context, None otherwise
    """
    req_id = request_id.get()
    if not req_id:
        return None

    return RequestContext(
        request_id=req_id,
        client_id=client_id.get(),
        start_time=start_time.get(),
    )


def get_request_id() -> str:
    """Get the current request ID.

    Returns:
        Current request ID or empty string if not in context
    """
    return request_id.get()


def get_client_id() -> str:
    """Get the current client ID.

    Returns:
        Current client ID or "anonymous" if not in context
    """
    return client_id.get()


def get_elapsed_time() -> float:
    """Get elapsed time since request started.

    Returns:
        Elapsed time in seconds, or 0.0 if not in context
    """
    start = start_time.get()
    if start == 0.0:
        return 0.0
    return time.monotonic() - start


def log_with_context(
    level: int,
    message: str,
    **extra: Any,
) -> None:
    """Log a message with request context automatically included.

    Args:
        level: Logging level (e.g., logging.INFO)
        message: Log message
        **extra: Additional fields to include in log record
    """
    ctx = get_current_context_or_none()
    if ctx:
        extra.update(ctx.to_dict())
    logger.log(level, message, extra=extra)


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


# -----------------------------------------------------------------------------
# Cancellation Handling
# -----------------------------------------------------------------------------


@dataclass
class CancellationResult:
    """Result of a cancellable operation.

    Attributes:
        completed: Whether the operation completed successfully
        was_cancelled: Whether the operation was cancelled
        partial_results: Any partial results available if cancelled
        processed_count: Number of items processed before cancellation
        total_count: Total items that were to be processed
    """

    completed: bool = False
    was_cancelled: bool = False
    partial_results: List[Any] = field(default_factory=list)
    processed_count: int = 0
    total_count: int = 0


@asynccontextmanager
async def cancellable_scope(
    cleanup_func: Optional[Callable[[], Coroutine[Any, Any, None]]] = None,
):
    """Context manager for operations that may be cancelled.

    Ensures proper cleanup when a cancellation occurs.

    Args:
        cleanup_func: Optional async function to call on cancellation

    Example:
        >>> async def cleanup():
        ...     await close_connections()
        ...
        >>> async with cancellable_scope(cleanup_func=cleanup):
        ...     await long_running_operation()

    Yields:
        None
    """
    try:
        yield
    except asyncio.CancelledError:
        logger.debug("Operation cancelled, performing cleanup")
        if cleanup_func:
            try:
                await cleanup_func()
            except Exception as e:
                logger.warning(f"Cleanup failed during cancellation: {e}")
        raise  # Always re-raise CancelledError


def with_cancellation(
    cleanup_func: Optional[Callable[[], Coroutine[Any, Any, None]]] = None,
):
    """Decorator for async functions that handles cancellation gracefully.

    Ensures cleanup is performed when the function is cancelled.

    Args:
        cleanup_func: Optional async function to call on cancellation

    Example:
        >>> async def close_db():
        ...     await db.close()
        ...
        >>> @with_cancellation(cleanup_func=close_db)
        ... async def query_database():
        ...     return await db.query("SELECT * FROM users")
    """

    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async with cancellable_scope(cleanup_func=cleanup_func):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


async def run_with_cancellation_checkpoints(
    items: List[T],
    process_func: Callable[[T], Coroutine[Any, Any, Any]],
    *,
    checkpoint_interval: int = 10,
    return_partial: bool = True,
) -> CancellationResult:
    """Process items with periodic cancellation checkpoints.

    Allows long-running batch operations to be cancelled cleanly,
    optionally returning partial results.

    Args:
        items: List of items to process
        process_func: Async function to process each item
        checkpoint_interval: Check for cancellation every N items
        return_partial: If True, return partial results on cancellation

    Returns:
        CancellationResult with completion status and any partial results

    Example:
        >>> async def process_item(item: str) -> dict:
        ...     return {"item": item, "processed": True}
        ...
        >>> result = await run_with_cancellation_checkpoints(
        ...     items=["a", "b", "c"],
        ...     process_func=process_item,
        ...     checkpoint_interval=1,
        ... )
        >>> if result.was_cancelled:
        ...     print(f"Processed {result.processed_count}/{result.total_count}")
    """
    results: List[Any] = []
    total = len(items)

    for i, item in enumerate(items):
        # Cancellation checkpoint
        if i % checkpoint_interval == 0:
            await asyncio.sleep(0)  # Yield to allow cancellation

        try:
            result = await process_func(item)
            results.append(result)
        except asyncio.CancelledError:
            logger.info(
                f"Operation cancelled at item {i}/{total}"
            )
            if return_partial:
                return CancellationResult(
                    completed=False,
                    was_cancelled=True,
                    partial_results=results,
                    processed_count=len(results),
                    total_count=total,
                )
            raise

    return CancellationResult(
        completed=True,
        was_cancelled=False,
        partial_results=results,
        processed_count=len(results),
        total_count=total,
    )


async def cancel_tasks_gracefully(
    tasks: List[asyncio.Task],
    *,
    timeout: float = 5.0,
) -> List[Optional[Exception]]:
    """Cancel multiple tasks gracefully with timeout.

    Attempts to cancel all tasks and waits for them to complete,
    with a timeout to prevent hanging.

    Args:
        tasks: List of asyncio.Task objects to cancel
        timeout: Maximum time to wait for tasks to finish cancelling

    Returns:
        List of exceptions from cancelled tasks (None if clean cancellation)

    Example:
        >>> tasks = [asyncio.create_task(op()) for op in operations]
        >>> # Later, need to cancel all
        >>> errors = await cancel_tasks_gracefully(tasks, timeout=10.0)
    """
    if not tasks:
        return []

    # Request cancellation for all tasks
    for task in tasks:
        if not task.done():
            task.cancel()

    # Wait for all tasks to complete with timeout
    errors: List[Optional[Exception]] = []
    try:
        done, pending = await asyncio.wait(
            tasks,
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED,
        )

        # Collect results/exceptions
        for task in tasks:
            if task.done():
                try:
                    task.result()
                    errors.append(None)
                except asyncio.CancelledError:
                    errors.append(None)  # Clean cancellation
                except Exception as e:
                    errors.append(e)
            else:
                # Task didn't finish in time
                errors.append(asyncio.TimeoutError("Task did not finish cancelling"))

        # Force cancel any remaining
        for task in pending:
            task.cancel()

    except Exception as e:
        logger.error(f"Error during graceful cancellation: {e}")
        errors = [e] * len(tasks)

    return errors


class CancellationToken:
    """Token for cooperative cancellation of async operations.

    Allows multiple operations to check for cancellation requests
    without relying solely on asyncio.CancelledError.

    Example:
        >>> token = CancellationToken()
        >>>
        >>> async def worker():
        ...     while not token.is_cancelled:
        ...         await do_work()
        ...         await token.check()  # Raises if cancelled
        ...
        >>> # Later, from another task:
        >>> token.cancel()
    """

    def __init__(self):
        """Initialize cancellation token."""
        self._cancelled = False
        self._cancel_event = asyncio.Event()

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancelled

    def cancel(self) -> None:
        """Request cancellation."""
        self._cancelled = True
        self._cancel_event.set()

    async def check(self) -> None:
        """Check for cancellation and raise if requested.

        Raises:
            asyncio.CancelledError: If cancellation was requested
        """
        if self._cancelled:
            raise asyncio.CancelledError("Cancellation requested via token")

    async def wait_for_cancel(self, timeout: Optional[float] = None) -> bool:
        """Wait for cancellation to be requested.

        Args:
            timeout: Maximum time to wait (None for indefinite)

        Returns:
            True if cancelled, False if timeout reached
        """
        try:
            await asyncio.wait_for(self._cancel_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


# Export all public symbols
__all__ = [
    # Schema
    "SCHEMA_VERSION",
    # Concurrency limiting
    "ConcurrencyConfig",
    "ConcurrencyStats",
    "GatherResult",
    "ConcurrencyLimiter",
    "get_tool_limiter",
    "configure_tool_limiter",
    "get_all_limiter_stats",
    # Cancellation handling
    "CancellationResult",
    "CancellationToken",
    "cancellable_scope",
    "with_cancellation",
    "run_with_cancellation_checkpoints",
    "cancel_tasks_gracefully",
    # Request context management
    "RequestContext",
    "request_context",
    "get_current_context",
    "get_current_context_or_none",
    "get_request_id",
    "get_client_id",
    "get_elapsed_time",
    "log_with_context",
    # Context variables (raw access)
    "request_id",
    "client_id",
    "start_time",
]
