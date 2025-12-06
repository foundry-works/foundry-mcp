"""Unified context management for request correlation and distributed tracing.

This module provides a single source of truth for request context propagation
across the foundry-mcp codebase, including:

- Correlation ID generation and propagation
- W3C Trace Context (traceparent/tracestate) support
- Thread-safe context variables via contextvars
- Both async and sync context managers

Usage:
    from foundry_mcp.core.context import (
        request_context,
        sync_request_context,
        get_correlation_id,
        get_current_context,
        generate_correlation_id,
    )

    # Async usage
    async with request_context(client_id="user123") as ctx:
        print(ctx.correlation_id)  # e.g., "req_a1b2c3d4e5f6"

    # Sync usage
    with sync_request_context() as ctx:
        print(ctx.correlation_id)

    # Manual ID generation
    corr_id = generate_correlation_id(prefix="task")  # "task_a1b2c3d4e5f6"
"""

from __future__ import annotations

import re
import secrets
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional

__all__ = [
    # Context variables (for advanced use)
    "correlation_id_var",
    "client_id_var",
    "start_time_var",
    "trace_context_var",
    # Dataclasses
    "W3CTraceContext",
    "RequestContext",
    # ID generation
    "generate_correlation_id",
    # Context managers
    "request_context",
    "sync_request_context",
    # Accessors
    "get_correlation_id",
    "get_client_id",
    "get_start_time",
    "get_trace_context",
    "get_current_context",
    # W3C helpers
    "parse_traceparent",
    "format_traceparent",
]

# -----------------------------------------------------------------------------
# Context Variables
# -----------------------------------------------------------------------------

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")
"""Request correlation ID for tracing requests across components."""

client_id_var: ContextVar[str] = ContextVar("client_id", default="anonymous")
"""Identifier for the client making the request."""

start_time_var: ContextVar[float] = ContextVar("start_time", default=0.0)
"""Request start time as Unix timestamp."""

trace_context_var: ContextVar[Optional["W3CTraceContext"]] = ContextVar(
    "trace_context", default=None
)
"""W3C Trace Context for distributed tracing integration."""


# -----------------------------------------------------------------------------
# W3C Trace Context Support
# -----------------------------------------------------------------------------

# W3C traceparent format: version-trace_id-parent_id-flags
# Example: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
_TRACEPARENT_REGEX = re.compile(
    r"^(?P<version>[0-9a-f]{2})-"
    r"(?P<trace_id>[0-9a-f]{32})-"
    r"(?P<parent_id>[0-9a-f]{16})-"
    r"(?P<flags>[0-9a-f]{2})$"
)


@dataclass(frozen=True)
class W3CTraceContext:
    """W3C Trace Context representation for distributed tracing.

    Implements the W3C Trace Context specification:
    https://www.w3.org/TR/trace-context/

    Attributes:
        version: Trace context version (currently "00")
        trace_id: 32-character hex trace identifier
        parent_id: 16-character hex span/parent identifier
        flags: 2-character hex flags (01 = sampled)
        tracestate: Optional vendor-specific trace state
    """

    version: str = "00"
    trace_id: str = ""
    parent_id: str = ""
    flags: str = "00"
    tracestate: Optional[str] = None

    @classmethod
    def parse(
        cls,
        traceparent: Optional[str] = None,
        tracestate: Optional[str] = None,
    ) -> Optional["W3CTraceContext"]:
        """Parse W3C traceparent and tracestate headers.

        Args:
            traceparent: The traceparent header value
            tracestate: Optional tracestate header value

        Returns:
            Parsed W3CTraceContext or None if parsing fails
        """
        if not traceparent:
            return None

        match = _TRACEPARENT_REGEX.match(traceparent.lower().strip())
        if not match:
            return None

        return cls(
            version=match.group("version"),
            trace_id=match.group("trace_id"),
            parent_id=match.group("parent_id"),
            flags=match.group("flags"),
            tracestate=tracestate,
        )

    @classmethod
    def generate(cls, sampled: bool = True) -> "W3CTraceContext":
        """Generate a new W3C Trace Context.

        Args:
            sampled: Whether this trace should be sampled

        Returns:
            New W3CTraceContext with generated IDs
        """
        return cls(
            version="00",
            trace_id=secrets.token_hex(16),  # 32 hex chars
            parent_id=secrets.token_hex(8),  # 16 hex chars
            flags="01" if sampled else "00",
        )

    @property
    def is_sampled(self) -> bool:
        """Check if trace is sampled based on flags."""
        try:
            return (int(self.flags, 16) & 0x01) == 0x01
        except ValueError:
            return False

    @property
    def traceparent(self) -> str:
        """Format as W3C traceparent header value."""
        return f"{self.version}-{self.trace_id}-{self.parent_id}-{self.flags}"

    def with_new_parent(self, sampled: Optional[bool] = None) -> "W3CTraceContext":
        """Create child context with new parent_id, preserving trace_id.

        Args:
            sampled: Override sampling decision (None = inherit)

        Returns:
            New W3CTraceContext with same trace_id but new parent_id
        """
        flags = self.flags
        if sampled is not None:
            flags = "01" if sampled else "00"

        return W3CTraceContext(
            version=self.version,
            trace_id=self.trace_id,
            parent_id=secrets.token_hex(8),
            flags=flags,
            tracestate=self.tracestate,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "version": self.version,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "flags": self.flags,
            "sampled": self.is_sampled,
        }
        if self.tracestate:
            result["tracestate"] = self.tracestate
        return result


def parse_traceparent(header: Optional[str]) -> Optional[W3CTraceContext]:
    """Parse a traceparent header into W3CTraceContext.

    Args:
        header: The traceparent header value

    Returns:
        Parsed context or None
    """
    return W3CTraceContext.parse(header)


def format_traceparent(ctx: W3CTraceContext) -> str:
    """Format W3CTraceContext as traceparent header.

    Args:
        ctx: The trace context

    Returns:
        Formatted traceparent header value
    """
    return ctx.traceparent


# -----------------------------------------------------------------------------
# Correlation ID Generation
# -----------------------------------------------------------------------------


def generate_correlation_id(prefix: str = "req") -> str:
    """Generate a unique correlation ID with optional prefix.

    Format: {prefix}_{12_hex_chars}
    Example: "req_a1b2c3d4e5f6"

    Args:
        prefix: ID prefix (default: "req")

    Returns:
        Unique correlation ID string
    """
    return f"{prefix}_{secrets.token_hex(6)}"


# -----------------------------------------------------------------------------
# Request Context
# -----------------------------------------------------------------------------


@dataclass
class RequestContext:
    """Immutable snapshot of the current request context.

    This dataclass captures all context variables at a point in time,
    providing convenient access to timing information and serialization.

    Attributes:
        correlation_id: Unique request identifier
        client_id: Client/user identifier
        start_time: Request start timestamp
        trace_context: Optional W3C distributed tracing context
    """

    correlation_id: str = ""
    client_id: str = "anonymous"
    start_time: float = field(default_factory=time.time)
    trace_context: Optional[W3CTraceContext] = None

    @property
    def elapsed_seconds(self) -> float:
        """Calculate elapsed time since request start in seconds."""
        if self.start_time <= 0:
            return 0.0
        return time.time() - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Calculate elapsed time since request start in milliseconds."""
        return self.elapsed_seconds * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging/serialization.

        Returns:
            Dictionary with all context fields
        """
        result: Dict[str, Any] = {
            "correlation_id": self.correlation_id,
            "client_id": self.client_id,
            "start_time": self.start_time,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }
        if self.trace_context:
            result["trace"] = self.trace_context.to_dict()
        return result


# -----------------------------------------------------------------------------
# Context Managers
# -----------------------------------------------------------------------------


@contextmanager
def sync_request_context(
    *,
    correlation_id: Optional[str] = None,
    client_id: Optional[str] = None,
    traceparent: Optional[str] = None,
    tracestate: Optional[str] = None,
) -> Generator[RequestContext, None, None]:
    """Synchronous context manager for request context.

    Sets up context variables for the duration of the with block,
    automatically cleaning up on exit.

    Args:
        correlation_id: Request ID (auto-generated if None)
        client_id: Client identifier (default: "anonymous")
        traceparent: W3C traceparent header for distributed tracing
        tracestate: W3C tracestate header

    Yields:
        RequestContext snapshot

    Example:
        with sync_request_context(client_id="user123") as ctx:
            logger.info(f"Processing request {ctx.correlation_id}")
    """
    # Generate or use provided correlation ID
    corr_id = correlation_id or generate_correlation_id()
    client = client_id or "anonymous"
    start = time.time()

    # Parse trace context if provided
    trace_ctx = W3CTraceContext.parse(traceparent, tracestate)

    # Set context variables
    token_corr = correlation_id_var.set(corr_id)
    token_client = client_id_var.set(client)
    token_start = start_time_var.set(start)
    token_trace = trace_context_var.set(trace_ctx)

    try:
        yield RequestContext(
            correlation_id=corr_id,
            client_id=client,
            start_time=start,
            trace_context=trace_ctx,
        )
    finally:
        # Reset context variables
        correlation_id_var.reset(token_corr)
        client_id_var.reset(token_client)
        start_time_var.reset(token_start)
        trace_context_var.reset(token_trace)


async def request_context(
    *,
    correlation_id: Optional[str] = None,
    client_id: Optional[str] = None,
    traceparent: Optional[str] = None,
    tracestate: Optional[str] = None,
) -> RequestContext:
    """Async context manager for request context.

    This is an async generator that can be used with `async with`.
    Sets up context variables for the duration of the async with block.

    Args:
        correlation_id: Request ID (auto-generated if None)
        client_id: Client identifier (default: "anonymous")
        traceparent: W3C traceparent header for distributed tracing
        tracestate: W3C tracestate header

    Returns:
        RequestContext snapshot (use with async context manager)

    Example:
        async with request_context(client_id="user123") as ctx:
            logger.info(f"Processing request {ctx.correlation_id}")
    """
    # This uses the sync implementation since contextvars work across
    # sync/async boundaries in Python 3.7+
    with sync_request_context(
        correlation_id=correlation_id,
        client_id=client_id,
        traceparent=traceparent,
        tracestate=tracestate,
    ) as ctx:
        return ctx


# Make request_context work as both async and sync context manager
class _AsyncContextManager:
    """Wrapper to make request_context work as async context manager."""

    def __init__(
        self,
        correlation_id: Optional[str] = None,
        client_id: Optional[str] = None,
        traceparent: Optional[str] = None,
        tracestate: Optional[str] = None,
    ):
        self.correlation_id = correlation_id
        self.client_id = client_id
        self.traceparent = traceparent
        self.tracestate = tracestate
        self._sync_cm: Optional[Generator[RequestContext, None, None]] = None
        self._ctx: Optional[RequestContext] = None

    async def __aenter__(self) -> RequestContext:
        self._sync_cm = sync_request_context(
            correlation_id=self.correlation_id,
            client_id=self.client_id,
            traceparent=self.traceparent,
            tracestate=self.tracestate,
        )
        self._ctx = self._sync_cm.__enter__()
        return self._ctx

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._sync_cm:
            self._sync_cm.__exit__(exc_type, exc_val, exc_tb)
        return False


def async_request_context(
    *,
    correlation_id: Optional[str] = None,
    client_id: Optional[str] = None,
    traceparent: Optional[str] = None,
    tracestate: Optional[str] = None,
) -> _AsyncContextManager:
    """Create an async context manager for request context.

    Args:
        correlation_id: Request ID (auto-generated if None)
        client_id: Client identifier (default: "anonymous")
        traceparent: W3C traceparent header for distributed tracing
        tracestate: W3C tracestate header

    Returns:
        Async context manager yielding RequestContext

    Example:
        async with async_request_context(client_id="user123") as ctx:
            await some_async_operation()
            logger.info(f"Completed request {ctx.correlation_id}")
    """
    return _AsyncContextManager(
        correlation_id=correlation_id,
        client_id=client_id,
        traceparent=traceparent,
        tracestate=tracestate,
    )


# -----------------------------------------------------------------------------
# Context Accessors
# -----------------------------------------------------------------------------


def get_correlation_id() -> str:
    """Get the current correlation ID from context.

    Returns:
        Current correlation ID or empty string if not set
    """
    return correlation_id_var.get()


def get_client_id() -> str:
    """Get the current client ID from context.

    Returns:
        Current client ID or "anonymous" if not set
    """
    return client_id_var.get()


def get_start_time() -> float:
    """Get the request start time from context.

    Returns:
        Start time as Unix timestamp or 0.0 if not set
    """
    return start_time_var.get()


def get_trace_context() -> Optional[W3CTraceContext]:
    """Get the W3C trace context if set.

    Returns:
        Current W3CTraceContext or None
    """
    return trace_context_var.get()


def get_current_context() -> RequestContext:
    """Get a snapshot of all current context values.

    Returns:
        RequestContext with current values from context variables
    """
    return RequestContext(
        correlation_id=get_correlation_id(),
        client_id=get_client_id(),
        start_time=get_start_time(),
        trace_context=get_trace_context(),
    )


# -----------------------------------------------------------------------------
# Backward Compatibility Aliases
# -----------------------------------------------------------------------------

# For backward compatibility with concurrency.py
request_id = correlation_id_var
"""Alias for correlation_id_var (backward compatibility)."""


def get_request_id() -> str:
    """Alias for get_correlation_id() (backward compatibility)."""
    return get_correlation_id()
