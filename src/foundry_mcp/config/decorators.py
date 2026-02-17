"""Metrics and observability decorators.

Provides ``log_call``, ``timed``, and ``require_auth`` decorators that were
historically defined in the monolithic ``config.py`` module.
"""

import functools
import logging
import time
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


def log_call(
    logger_name: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to log function calls with structured data.

    Args:
        logger_name: Optional logger name (defaults to function module)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        log = logging.getLogger(logger_name or func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            log.debug(
                f"Calling {func.__name__}",
                extra={
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                },
            )
            try:
                result = func(*args, **kwargs)
                log.debug(
                    f"Completed {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "success": True,
                    },
                )
                return result
            except Exception as e:
                log.error(
                    f"Error in {func.__name__}: {e}",
                    extra={
                        "function": func.__name__,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                raise

        return wrapper

    return decorator


def timed(
    metric_name: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to measure and log function execution time.

    Args:
        metric_name: Optional metric name (defaults to function name)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = metric_name or func.__name__
        log = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                log.info(
                    f"Timer: {name}",
                    extra={
                        "metric": name,
                        "duration_ms": round(elapsed * 1000, 2),
                        "success": True,
                    },
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                log.info(
                    f"Timer: {name}",
                    extra={
                        "metric": name,
                        "duration_ms": round(elapsed * 1000, 2),
                        "success": False,
                        "error": str(e),
                    },
                )
                raise

        return wrapper

    return decorator


def require_auth(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to require API key authentication for a function.

    The function must accept an 'api_key' keyword argument.
    Raises ValueError if authentication fails.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        from foundry_mcp.config.server import get_config

        config = get_config()
        api_key = kwargs.get("api_key")

        if not config.validate_api_key(api_key):
            raise ValueError("Invalid or missing API key")

        return func(*args, **kwargs)

    return wrapper
