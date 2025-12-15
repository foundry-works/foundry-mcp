"""Structured logging hooks for CLI commands.

Provides context ID generation, metrics emission, and structured
logging for CLI command execution. Wraps core observability primitives
for CLI-specific use cases.
"""

import logging
import time
import uuid
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from foundry_mcp.core.observability import (
    get_metrics,
    redact_sensitive_data,
)

__all__ = [
    "get_request_id",
    "set_request_id",
    "cli_command",
    "get_cli_logger",
    "CLILogContext",
]

T = TypeVar("T")

# Context variable for request/correlation ID
_request_id: ContextVar[str] = ContextVar("request_id", default="")


def generate_request_id() -> str:
    """Generate a unique request ID for CLI command tracking.

    Returns:
        Short UUID suitable for log correlation.
    """
    return f"cli_{uuid.uuid4().hex[:12]}"


def get_request_id() -> str:
    """Get the current request ID for this execution context.

    Returns:
        The request ID, or empty string if not set.
    """
    return _request_id.get()


def set_request_id(request_id: str) -> None:
    """Set the request ID for this execution context.

    Args:
        request_id: The request ID to set.
    """
    _request_id.set(request_id)


class CLILogContext:
    """Context manager for CLI command logging context.

    Automatically generates and sets a request ID for the duration
    of the context, enabling log correlation.

    Example:
        >>> with CLILogContext() as ctx:
        ...     logger.info("Processing", extra={"request_id": ctx.request_id})
    """

    def __init__(self, request_id: Optional[str] = None):
        """Initialize logging context.

        Args:
            request_id: Optional custom request ID (auto-generated if None).
        """
        self.request_id = request_id or generate_request_id()
        self._token = None

    def __enter__(self) -> "CLILogContext":
        self._token = _request_id.set(self.request_id)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._token is not None:
            _request_id.reset(self._token)


class CLILogger:
    """Structured logger for CLI commands.

    Provides JSON-formatted logging with automatic request ID inclusion
    and sensitive data redaction.
    """

    def __init__(self, name: str = "foundry_mcp.cli"):
        self._logger = logging.getLogger(name)

    def _log(
        self,
        level: int,
        message: str,
        **extra: Any,
    ) -> None:
        """Log with structured context."""
        request_id = get_request_id()
        context = {
            "request_id": request_id,
            **redact_sensitive_data(extra),
        }
        self._logger.log(level, message, extra={"cli_context": context})

    def debug(self, message: str, **extra: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **extra)

    def info(self, message: str, **extra: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **extra)

    def warning(self, message: str, **extra: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **extra)

    def error(self, message: str, **extra: Any) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, **extra)


# Global CLI logger
_cli_logger = CLILogger()


def get_cli_logger() -> CLILogger:
    """Get the global CLI logger."""
    return _cli_logger


def cli_command(
    command_name: Optional[str] = None,
    emit_metrics: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for CLI commands with observability.

    Automatically:
    - Generates request ID for correlation
    - Logs command start/end
    - Emits latency and status metrics

    Args:
        command_name: Override command name (defaults to function name).
        emit_metrics: Whether to emit metrics (default: True).

    Returns:
        Decorated function with observability.

    Example:
        >>> @cli_command("fetch-spec")
        ... def fetch_spec(spec_id: str):
        ...     return load_spec(spec_id)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = command_name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            with CLILogContext():
                metrics = get_metrics()
                start = time.perf_counter()
                success = True
                error_msg = None

                _cli_logger.debug(
                    f"CLI command started: {name}",
                    command=name,
                )

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    error_msg = str(e)
                    raise
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000

                    _cli_logger.debug(
                        f"CLI command completed: {name}",
                        command=name,
                        success=success,
                        duration_ms=round(duration_ms, 2),
                        error=error_msg,
                    )

                    if emit_metrics:
                        labels = {
                            "command": name,
                            "status": "success" if success else "error",
                        }
                        metrics.counter("cli.command.invocations", labels=labels)
                        metrics.timer(
                            "cli.command.latency",
                            duration_ms,
                            labels={"command": name},
                        )

        return wrapper

    return decorator
