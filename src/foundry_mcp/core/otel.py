"""OpenTelemetry integration with graceful degradation.

This module provides OpenTelemetry tracing and metrics integration that
gracefully falls back to no-op implementations when the optional
opentelemetry dependencies are not installed.

Usage:
    from foundry_mcp.core.otel import get_tracer, traced

    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("my-operation") as span:
        span.set_attribute("key", "value")
        ...

    @traced("my-function")
    def my_function():
        ...
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union

from foundry_mcp.core.otel_stubs import (
    NoOpMeter,
    NoOpSpan,
    NoOpTracer,
    get_noop_meter,
    get_noop_tracer,
)

# Type checking imports for better IDE support
if TYPE_CHECKING:
    from opentelemetry.metrics import Meter
    from opentelemetry.trace import Span, Tracer

# Try to import OpenTelemetry
try:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry import trace as otel_trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

    _OPENTELEMETRY_AVAILABLE = True
except ImportError:
    _OPENTELEMETRY_AVAILABLE = False

# Type aliases
F = TypeVar("F", bound=Callable[..., Any])
TracerType = Union["Tracer", NoOpTracer]
MeterType = Union["Meter", NoOpMeter]
SpanType = Union["Span", NoOpSpan]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class OTelConfig:
    """Configuration for OpenTelemetry integration.

    Attributes:
        enabled: Whether OpenTelemetry is enabled
        otlp_endpoint: OTLP exporter endpoint (default: localhost:4317)
        service_name: Service name for traces and metrics
        sample_rate: Trace sampling rate (0.0 to 1.0)
        export_interval_ms: Metrics export interval in milliseconds
    """

    enabled: bool = False
    otlp_endpoint: str = "localhost:4317"
    service_name: str = "foundry-mcp"
    sample_rate: float = 1.0
    export_interval_ms: int = 60000
    additional_attributes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env_and_config(
        cls,
        config: Optional[dict[str, Any]] = None,
    ) -> "OTelConfig":
        """Load configuration from environment variables and optional config dict.

        Environment variables take precedence over config dict values.

        Env vars:
            OTEL_ENABLED: "true" or "1" to enable
            OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint
            OTEL_SERVICE_NAME: Service name
            OTEL_TRACE_SAMPLE_RATE: Sample rate (float)
            OTEL_METRIC_EXPORT_INTERVAL: Export interval in ms

        Args:
            config: Optional dict with config values (typically from TOML)

        Returns:
            OTelConfig instance
        """
        config = config or {}

        # Parse enabled from env or config
        env_enabled = os.environ.get("OTEL_ENABLED", "").lower()
        if env_enabled:
            enabled = env_enabled in ("true", "1", "yes")
        else:
            enabled = config.get("enabled", False)

        # Parse endpoint
        otlp_endpoint = os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            config.get("otlp_endpoint", "localhost:4317"),
        )

        # Parse service name
        service_name = os.environ.get(
            "OTEL_SERVICE_NAME",
            config.get("service_name", "foundry-mcp"),
        )

        # Parse sample rate
        sample_rate_str = os.environ.get("OTEL_TRACE_SAMPLE_RATE")
        if sample_rate_str:
            try:
                sample_rate = float(sample_rate_str)
            except ValueError:
                sample_rate = 1.0
        else:
            sample_rate = config.get("sample_rate", 1.0)

        # Parse export interval
        export_interval_str = os.environ.get("OTEL_METRIC_EXPORT_INTERVAL")
        if export_interval_str:
            try:
                export_interval_ms = int(export_interval_str)
            except ValueError:
                export_interval_ms = 60000
        else:
            export_interval_ms = config.get("export_interval_ms", 60000)

        # Additional attributes from config
        additional_attributes = config.get("attributes", {})

        return cls(
            enabled=enabled,
            otlp_endpoint=otlp_endpoint,
            service_name=service_name,
            sample_rate=sample_rate,
            export_interval_ms=export_interval_ms,
            additional_attributes=additional_attributes,
        )


# =============================================================================
# Global State
# =============================================================================

_config: Optional[OTelConfig] = None
_tracer_provider: Any = None
_meter_provider: Any = None
_initialized: bool = False


# =============================================================================
# Initialization
# =============================================================================


def initialize(config: Optional[OTelConfig] = None) -> bool:
    """Initialize OpenTelemetry with the given configuration.

    This function is idempotent - calling it multiple times has no effect
    after the first successful initialization.

    Args:
        config: OTel configuration. If None, loads from env and defaults.

    Returns:
        True if OpenTelemetry was initialized, False if using no-ops
    """
    global _config, _tracer_provider, _meter_provider, _initialized

    if _initialized:
        return _config is not None and _config.enabled and _OPENTELEMETRY_AVAILABLE

    _config = config or OTelConfig.from_env_and_config()

    if not _config.enabled or not _OPENTELEMETRY_AVAILABLE:
        _initialized = True
        return False

    # Create resource with service info
    resource_attrs = {
        "service.name": _config.service_name,
        "service.version": _get_version(),
    }
    resource_attrs.update(_config.additional_attributes)
    resource = Resource.create(resource_attrs)

    # Setup tracing
    sampler = TraceIdRatioBased(_config.sample_rate)
    _tracer_provider = TracerProvider(resource=resource, sampler=sampler)

    # Add OTLP span exporter
    otlp_span_exporter = OTLPSpanExporter(
        endpoint=_config.otlp_endpoint,
        insecure=True,  # Use insecure for local development
    )
    span_processor = BatchSpanProcessor(otlp_span_exporter)
    _tracer_provider.add_span_processor(span_processor)

    # Set as global tracer provider
    otel_trace.set_tracer_provider(_tracer_provider)

    # Setup metrics
    otlp_metric_exporter = OTLPMetricExporter(
        endpoint=_config.otlp_endpoint,
        insecure=True,
    )
    metric_reader = PeriodicExportingMetricReader(
        otlp_metric_exporter,
        export_interval_millis=_config.export_interval_ms,
    )
    _meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])

    # Set as global meter provider
    otel_metrics.set_meter_provider(_meter_provider)

    _initialized = True
    return True


def _ensure_initialized() -> None:
    """Ensure OpenTelemetry is initialized (lazy initialization)."""
    if not _initialized:
        initialize()


def _get_version() -> str:
    """Get the foundry-mcp version."""
    try:
        from importlib.metadata import version

        return version("foundry-mcp")
    except Exception:
        return "unknown"


# =============================================================================
# Public API
# =============================================================================


def is_available() -> bool:
    """Check if OpenTelemetry dependencies are available.

    Returns:
        True if opentelemetry packages are installed
    """
    return _OPENTELEMETRY_AVAILABLE


def is_enabled() -> bool:
    """Check if OpenTelemetry is enabled and initialized.

    Returns:
        True if OTel is enabled, available, and initialized
    """
    _ensure_initialized()
    return (
        _config is not None
        and _config.enabled
        and _OPENTELEMETRY_AVAILABLE
        and _initialized
    )


def get_tracer(name: str = __name__) -> TracerType:
    """Get a tracer instance.

    If OpenTelemetry is not available or not enabled, returns a no-op tracer
    that silently ignores all operations.

    Args:
        name: Tracer name (typically __name__ of the calling module)

    Returns:
        Tracer instance (real or no-op)
    """
    _ensure_initialized()

    if is_enabled():
        return otel_trace.get_tracer(name)
    return get_noop_tracer(name)


def get_meter(name: str = __name__) -> MeterType:
    """Get a meter instance for creating metrics.

    If OpenTelemetry is not available or not enabled, returns a no-op meter
    that silently ignores all operations.

    Args:
        name: Meter name (typically __name__ of the calling module)

    Returns:
        Meter instance (real or no-op)
    """
    _ensure_initialized()

    if is_enabled():
        return otel_metrics.get_meter(name)
    return get_noop_meter(name)


def traced(
    name: Optional[str] = None,
    *,
    attributes: Optional[dict[str, Any]] = None,
    record_exception: bool = True,
    set_status_on_exception: bool = True,
) -> Callable[[F], F]:
    """Decorator to trace a function with a span.

    Usage:
        @traced("my-operation")
        def my_function(arg1, arg2):
            ...

        @traced()  # Uses function name as span name
        async def my_async_function():
            ...

    Args:
        name: Span name. Defaults to function name if not provided.
        attributes: Additional attributes to set on the span.
        record_exception: Whether to record exceptions on the span.
        set_status_on_exception: Whether to set error status on exception.

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__
        tracer = get_tracer(func.__module__)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(
                span_name,
                attributes=attributes,
                record_exception=record_exception,
                set_status_on_exception=set_status_on_exception,
            ) as span:
                # Add function signature info
                if hasattr(span, "set_attribute"):
                    span.set_attribute("code.function", func.__name__)
                    span.set_attribute("code.namespace", func.__module__)
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(
                span_name,
                attributes=attributes,
                record_exception=record_exception,
                set_status_on_exception=set_status_on_exception,
            ) as span:
                # Add function signature info
                if hasattr(span, "set_attribute"):
                    span.set_attribute("code.function", func.__name__)
                    span.set_attribute("code.namespace", func.__module__)
                return await func(*args, **kwargs)

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def shutdown() -> None:
    """Shutdown OpenTelemetry providers and flush pending data.

    Call this during application shutdown to ensure all telemetry
    data is exported.
    """
    global _tracer_provider, _meter_provider, _initialized

    if _tracer_provider is not None and hasattr(_tracer_provider, "shutdown"):
        try:
            _tracer_provider.shutdown()
        except Exception:
            pass
        _tracer_provider = None

    if _meter_provider is not None and hasattr(_meter_provider, "shutdown"):
        try:
            _meter_provider.shutdown()
        except Exception:
            pass
        _meter_provider = None

    _initialized = False


def get_config() -> Optional[OTelConfig]:
    """Get the current OTel configuration.

    Returns:
        Current configuration or None if not initialized
    """
    return _config


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "OTelConfig",
    # Initialization
    "initialize",
    "shutdown",
    # Status
    "is_available",
    "is_enabled",
    "get_config",
    # Tracer/Meter
    "get_tracer",
    "get_meter",
    # Decorator
    "traced",
]
