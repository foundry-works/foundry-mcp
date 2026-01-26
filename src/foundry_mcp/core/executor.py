"""Provider Executor for isolating blocking operations.

Provides a dedicated thread pool executor for blocking provider operations
(subprocess calls, sync file I/O) to prevent event loop starvation.

Usage:
    from foundry_mcp.core.executor import provider_executor

    # Run blocking operation in executor
    result = await provider_executor.run_blocking(
        subprocess.run,
        ["claude", "--version"],
        timeout=30.0,
    )
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from threading import Lock
from typing import Any, Callable, Optional, TypeVar

from foundry_mcp.core.observability import get_metrics

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default configuration
DEFAULT_POOL_SIZE = 4
DEFAULT_QUEUE_LIMIT = 100
DEFAULT_SHUTDOWN_TIMEOUT = 30.0


class ExecutorExhaustedError(Exception):
    """Raised when both dedicated and fallback executors are unavailable."""

    def __init__(self, message: str = "Executor pool exhausted"):
        super().__init__(message)
        self.message = message


class ProviderExecutor:
    """Thread pool executor for isolating blocking provider operations.

    Provides a dedicated executor for CLI provider subprocess calls and
    other blocking operations to prevent event loop starvation.

    Features:
    - Configurable pool size and queue limit
    - Graceful shutdown with timeout
    - Fallback to shared executor when dedicated pool is exhausted
    - Metrics for monitoring pool health
    - Feature flag to disable isolation (run inline)

    Example:
        executor = ProviderExecutor(pool_size=4, queue_limit=100)
        await executor.start()

        result = await executor.run_blocking(
            subprocess.run,
            ["claude", "prompt"],
            capture_output=True,
        )

        await executor.shutdown()
    """

    def __init__(
        self,
        pool_size: int = DEFAULT_POOL_SIZE,
        queue_limit: int = DEFAULT_QUEUE_LIMIT,
        shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT,
        enabled: bool = True,
    ):
        """Initialize the provider executor.

        Args:
            pool_size: Maximum number of worker threads
            queue_limit: Maximum pending tasks before rejection
            shutdown_timeout: Seconds to wait for graceful shutdown
            enabled: Feature flag - if False, runs operations inline
        """
        self._pool_size = pool_size
        self._queue_limit = queue_limit
        self._shutdown_timeout = shutdown_timeout
        self._enabled = enabled

        self._executor: Optional[ThreadPoolExecutor] = None
        self._fallback_executor: Optional[ThreadPoolExecutor] = None
        self._lock = Lock()
        self._active_tasks = 0
        self._queued_tasks = 0
        self._fallback_count = 0
        self._started = False

    @property
    def is_started(self) -> bool:
        """Check if executor is started."""
        return self._started

    @property
    def is_enabled(self) -> bool:
        """Check if executor isolation is enabled."""
        return self._enabled

    @property
    def active_workers(self) -> int:
        """Number of currently active worker threads."""
        return self._active_tasks

    @property
    def queued_tasks(self) -> int:
        """Number of tasks waiting in queue."""
        return self._queued_tasks

    @property
    def fallback_count(self) -> int:
        """Number of times fallback executor was used."""
        return self._fallback_count

    def start(self) -> None:
        """Start the executor pool.

        Creates the dedicated thread pool for provider operations.
        Safe to call multiple times (idempotent).
        """
        with self._lock:
            if self._started:
                return

            if self._enabled:
                self._executor = ThreadPoolExecutor(
                    max_workers=self._pool_size,
                    thread_name_prefix="provider-executor-",
                )
                # Smaller fallback pool for overflow
                self._fallback_executor = ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="provider-fallback-",
                )
                logger.info(
                    "Provider executor started: pool_size=%d, queue_limit=%d",
                    self._pool_size,
                    self._queue_limit,
                )
            else:
                logger.info("Provider executor disabled - operations will run inline")

            self._started = True

    async def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor pool gracefully.

        Args:
            wait: If True, wait for pending tasks to complete
        """
        with self._lock:
            if not self._started:
                return

            self._started = False

        if self._executor:
            logger.info("Shutting down provider executor...")
            self._executor.shutdown(wait=wait, cancel_futures=not wait)
            self._executor = None

        if self._fallback_executor:
            self._fallback_executor.shutdown(wait=wait, cancel_futures=not wait)
            self._fallback_executor = None

        logger.info(
            "Provider executor shutdown complete: "
            "fallback_count=%d",
            self._fallback_count,
        )

    async def run_blocking(
        self,
        func: Callable[..., T],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        """Run a blocking function in the executor pool.

        Args:
            func: Blocking function to execute
            *args: Positional arguments for func
            timeout: Optional timeout in seconds
            **kwargs: Keyword arguments for func

        Returns:
            Result of the function call

        Raises:
            ExecutorExhaustedError: When both pools are unavailable
            asyncio.TimeoutError: When timeout is exceeded
            Exception: Any exception raised by func
        """
        # If disabled, run inline (no isolation)
        if not self._enabled:
            return func(*args, **kwargs)

        # Ensure executor is started
        if not self._started:
            self.start()

        # Check queue limit
        with self._lock:
            if self._queued_tasks >= self._queue_limit:
                # Try fallback executor
                return await self._run_with_fallback(func, args, kwargs, timeout)
            self._queued_tasks += 1

        try:
            loop = asyncio.get_running_loop()

            # Track active tasks
            with self._lock:
                self._active_tasks += 1

            # Record metrics
            get_metrics().gauge(
                "provider_executor_active_workers",
                self._active_tasks,
            )
            get_metrics().gauge(
                "provider_executor_queued_tasks",
                self._queued_tasks,
            )

            try:
                # Create partial with kwargs
                if kwargs:
                    func_with_args = partial(func, *args, **kwargs)
                    coro = loop.run_in_executor(self._executor, func_with_args)
                else:
                    coro = loop.run_in_executor(self._executor, func, *args)

                if timeout:
                    return await asyncio.wait_for(coro, timeout=timeout)
                return await coro

            finally:
                with self._lock:
                    self._active_tasks -= 1
                    get_metrics().gauge(
                        "provider_executor_active_workers",
                        self._active_tasks,
                    )
        finally:
            with self._lock:
                self._queued_tasks -= 1
                get_metrics().gauge(
                    "provider_executor_queued_tasks",
                    self._queued_tasks,
                )

    async def _run_with_fallback(
        self,
        func: Callable[..., T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        timeout: Optional[float],
    ) -> T:
        """Run in fallback executor when primary is exhausted."""
        if self._fallback_executor is None:
            raise ExecutorExhaustedError(
                "Primary executor queue full and no fallback available"
            )

        with self._lock:
            self._fallback_count += 1
            get_metrics().counter("provider_executor_fallback_total")

        logger.warning(
            "Primary executor queue full, using fallback: "
            "queued=%d, limit=%d",
            self._queued_tasks,
            self._queue_limit,
        )

        loop = asyncio.get_running_loop()

        if kwargs:
            func_with_args = partial(func, *args, **kwargs)
            coro = loop.run_in_executor(self._fallback_executor, func_with_args)
        else:
            coro = loop.run_in_executor(self._fallback_executor, func, *args)

        if timeout:
            return await asyncio.wait_for(coro, timeout=timeout)
        return await coro

    @contextmanager
    def disabled(self):
        """Context manager to temporarily disable executor isolation.

        Useful for testing or specific code paths that should run inline.
        """
        original = self._enabled
        self._enabled = False
        try:
            yield
        finally:
            self._enabled = original


# Global singleton instance
_provider_executor: Optional[ProviderExecutor] = None
_executor_lock = Lock()


def get_provider_executor() -> ProviderExecutor:
    """Get the global provider executor instance.

    Creates the executor on first access with default configuration.
    Use configure_executor() to customize before first use.
    """
    global _provider_executor
    with _executor_lock:
        if _provider_executor is None:
            _provider_executor = ProviderExecutor()
        return _provider_executor


def configure_executor(
    pool_size: int = DEFAULT_POOL_SIZE,
    queue_limit: int = DEFAULT_QUEUE_LIMIT,
    shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT,
    enabled: bool = True,
) -> ProviderExecutor:
    """Configure and return the global provider executor.

    Must be called before get_provider_executor() for settings to take effect.

    Args:
        pool_size: Maximum number of worker threads
        queue_limit: Maximum pending tasks before rejection
        shutdown_timeout: Seconds to wait for graceful shutdown
        enabled: Feature flag - if False, runs operations inline

    Returns:
        Configured ProviderExecutor instance
    """
    global _provider_executor
    with _executor_lock:
        if _provider_executor is not None and _provider_executor.is_started:
            logger.warning(
                "Reconfiguring already-started executor - "
                "new settings will apply after restart"
            )
        _provider_executor = ProviderExecutor(
            pool_size=pool_size,
            queue_limit=queue_limit,
            shutdown_timeout=shutdown_timeout,
            enabled=enabled,
        )
        return _provider_executor


# Convenience alias
provider_executor = get_provider_executor
