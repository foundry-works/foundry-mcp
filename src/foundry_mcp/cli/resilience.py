"""CLI resilience wrappers for timeout and cancellation.

Provides synchronous timeout and retry decorators for CLI commands,
wrapping the core resilience primitives from foundry_mcp.core.resilience.
"""

import signal
import sys
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from foundry_mcp.core.resilience import (
    FAST_TIMEOUT,
    MEDIUM_TIMEOUT,
    SLOW_TIMEOUT,
    BACKGROUND_TIMEOUT,
    TimeoutException,
    retry_with_backoff,
)

# Re-export constants for CLI usage
__all__ = [
    "FAST_TIMEOUT",
    "MEDIUM_TIMEOUT",
    "SLOW_TIMEOUT",
    "BACKGROUND_TIMEOUT",
    "TimeoutException",
    "with_sync_timeout",
    "cli_retryable",
    "handle_keyboard_interrupt",
]

T = TypeVar("T")


class _TimeoutHandler:
    """Context manager for signal-based timeout on Unix systems."""

    def __init__(self, seconds: float, error_message: str):
        self.seconds = int(seconds)  # signal.alarm requires int
        self.error_message = error_message
        self._old_handler = None

    def _timeout_handler(self, signum: int, frame: Any) -> None:
        raise TimeoutException(
            self.error_message,
            timeout_seconds=float(self.seconds),
            operation="cli_command",
        )

    def __enter__(self) -> "_TimeoutHandler":
        if sys.platform != "win32":
            self._old_handler = signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if sys.platform != "win32":
            signal.alarm(0)
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)


def with_sync_timeout(
    seconds: float = MEDIUM_TIMEOUT,
    error_message: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add timeout to synchronous CLI commands.

    Uses signal.SIGALRM on Unix systems. On Windows, timeout is not
    enforced (function runs without timeout).

    Args:
        seconds: Timeout duration in seconds (default: MEDIUM_TIMEOUT).
        error_message: Custom error message on timeout.

    Returns:
        Decorated function with timeout enforcement.

    Example:
        >>> @with_sync_timeout(FAST_TIMEOUT, "Query timed out")
        ... def fetch_data():
        ...     return expensive_operation()

    Note:
        This uses SIGALRM which only works on Unix. Windows has no
        signal-based timeout mechanism for synchronous code.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            msg = error_message or f"{func.__name__} timed out after {seconds}s"

            if sys.platform == "win32":
                # Windows: run without timeout (signal.alarm not available)
                return func(*args, **kwargs)

            with _TimeoutHandler(seconds, msg):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def cli_retryable(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for automatic retries with exponential backoff.

    Wraps core retry_with_backoff for CLI command usage.

    Args:
        max_retries: Maximum retry attempts (default 3).
        delay: Base delay in seconds (default 1.0).
        exceptions: Tuple of exceptions to retry on.

    Returns:
        Decorated function with retry logic.

    Example:
        >>> @cli_retryable(max_retries=3, exceptions=(IOError,))
        ... def read_spec_file(path):
        ...     return load_json(path)
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


def handle_keyboard_interrupt(
    cleanup: Optional[Callable[[], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to gracefully handle Ctrl+C in CLI commands.

    Catches KeyboardInterrupt, optionally runs cleanup, and exits
    with appropriate code.

    Args:
        cleanup: Optional cleanup function to run on interrupt.

    Returns:
        Decorated function with interrupt handling.

    Example:
        >>> @handle_keyboard_interrupt(cleanup=lambda: print("Cancelled"))
        ... def long_running_task():
        ...     # ... work ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                if cleanup:
                    cleanup()
                # Exit with 130 (128 + SIGINT signal number 2)
                sys.exit(130)

        return wrapper

    return decorator
