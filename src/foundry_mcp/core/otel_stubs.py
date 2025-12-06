"""No-op stubs for observability when optional dependencies are not installed.

This module provides no-op implementations for tracing and metrics interfaces,
allowing the codebase to use observability features without requiring the
optional dependencies to be installed. All operations are silently ignored.
"""

from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence


# =============================================================================
# Tracing Stubs
# =============================================================================


class NoOpSpan:
    """No-op span that silently ignores all operations."""

    __slots__ = ("_name",)

    def __init__(self, name: str = "") -> None:
        self._name = name

    def __enter__(self) -> "NoOpSpan":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op: ignores attribute setting."""
        pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        """No-op: ignores attributes setting."""
        pass

    def set_status(self, status: Any, description: Optional[str] = None) -> None:
        """No-op: ignores status setting."""
        pass

    def record_exception(
        self,
        exception: BaseException,
        attributes: Optional[dict[str, Any]] = None,
        timestamp: Optional[int] = None,
        escaped: bool = False,
    ) -> None:
        """No-op: ignores exception recording."""
        pass

    def add_event(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        timestamp: Optional[int] = None,
    ) -> None:
        """No-op: ignores event adding."""
        pass

    def is_recording(self) -> bool:
        """No-op spans are never recording."""
        return False

    def end(self, end_time: Optional[int] = None) -> None:
        """No-op: ignores span end."""
        pass


# Singleton instance
_NOOP_SPAN = NoOpSpan()


class NoOpTracer:
    """No-op tracer that returns no-op spans."""

    __slots__ = ("_name",)

    def __init__(self, name: str = "") -> None:
        self._name = name

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        *,
        context: Any = None,
        kind: Any = None,
        attributes: Optional[dict[str, Any]] = None,
        links: Optional[Sequence[Any]] = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Iterator[NoOpSpan]:
        """No-op: yields a no-op span."""
        yield _NOOP_SPAN

    def start_span(
        self,
        name: str,
        *,
        context: Any = None,
        kind: Any = None,
        attributes: Optional[dict[str, Any]] = None,
        links: Optional[Sequence[Any]] = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> NoOpSpan:
        """No-op: returns a no-op span."""
        return _NOOP_SPAN


# Singleton instance
_NOOP_TRACER = NoOpTracer()


def get_noop_tracer(name: str = "") -> NoOpTracer:
    """Get the singleton no-op tracer instance."""
    return _NOOP_TRACER


# =============================================================================
# Metrics Stubs
# =============================================================================


class NoOpCounter:
    """No-op counter that silently ignores all operations."""

    __slots__ = ("_name",)

    def __init__(self, name: str = "") -> None:
        self._name = name

    def add(self, amount: float, attributes: Optional[dict[str, Any]] = None) -> None:
        """No-op: ignores counter increment."""
        pass

    def inc(self, attributes: Optional[dict[str, Any]] = None) -> None:
        """No-op: ignores counter increment by 1."""
        pass


class NoOpGauge:
    """No-op gauge that silently ignores all operations."""

    __slots__ = ("_name",)

    def __init__(self, name: str = "") -> None:
        self._name = name

    def set(self, value: float, attributes: Optional[dict[str, Any]] = None) -> None:
        """No-op: ignores gauge setting."""
        pass

    def inc(self, attributes: Optional[dict[str, Any]] = None) -> None:
        """No-op: ignores gauge increment."""
        pass

    def dec(self, attributes: Optional[dict[str, Any]] = None) -> None:
        """No-op: ignores gauge decrement."""
        pass


class NoOpHistogram:
    """No-op histogram that silently ignores all operations."""

    __slots__ = ("_name",)

    def __init__(self, name: str = "") -> None:
        self._name = name

    def record(
        self, value: float, attributes: Optional[dict[str, Any]] = None
    ) -> None:
        """No-op: ignores value recording."""
        pass

    def observe(
        self, value: float, attributes: Optional[dict[str, Any]] = None
    ) -> None:
        """No-op: ignores value observation (alias for record)."""
        pass


# Singleton instances
_NOOP_COUNTER = NoOpCounter()
_NOOP_GAUGE = NoOpGauge()
_NOOP_HISTOGRAM = NoOpHistogram()


class NoOpMeter:
    """No-op meter that returns no-op metric instruments."""

    __slots__ = ("_name",)

    def __init__(self, name: str = "") -> None:
        self._name = name

    def create_counter(
        self,
        name: str,
        unit: str = "",
        description: str = "",
    ) -> NoOpCounter:
        """No-op: returns a no-op counter."""
        return _NOOP_COUNTER

    def create_up_down_counter(
        self,
        name: str,
        unit: str = "",
        description: str = "",
    ) -> NoOpCounter:
        """No-op: returns a no-op counter."""
        return _NOOP_COUNTER

    def create_gauge(
        self,
        name: str,
        unit: str = "",
        description: str = "",
    ) -> NoOpGauge:
        """No-op: returns a no-op gauge."""
        return _NOOP_GAUGE

    def create_histogram(
        self,
        name: str,
        unit: str = "",
        description: str = "",
    ) -> NoOpHistogram:
        """No-op: returns a no-op histogram."""
        return _NOOP_HISTOGRAM


# Singleton instance
_NOOP_METER = NoOpMeter()


def get_noop_meter(name: str = "") -> NoOpMeter:
    """Get the singleton no-op meter instance."""
    return _NOOP_METER


# =============================================================================
# Convenience exports
# =============================================================================

__all__ = [
    # Tracing
    "NoOpSpan",
    "NoOpTracer",
    "get_noop_tracer",
    # Metrics
    "NoOpCounter",
    "NoOpGauge",
    "NoOpHistogram",
    "NoOpMeter",
    "get_noop_meter",
]
