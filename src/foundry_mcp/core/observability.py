"""
Observability utilities for foundry-mcp.

Provides structured logging, metrics collection, and audit logging
for MCP tools and resources.

FastMCP Middleware Integration:
    The decorators in this module can be applied to FastMCP tool and resource
    handlers to provide consistent observability. Example:

        from fastmcp import FastMCP
        from foundry_mcp.core.observability import mcp_tool, audit_log

        mcp = FastMCP("foundry-mcp")

        @mcp.tool()
        @mcp_tool(tool_name="list_specs")
        async def list_specs(status: str = "all") -> str:
            audit_log("tool_invocation", tool="list_specs", status=status)
            # ... implementation
            return result

    For resources, use the mcp_resource decorator:

        @mcp.resource("specs://{spec_id}")
        @mcp_resource(resource_type="spec")
        async def get_spec(spec_id: str) -> str:
            # ... implementation
            return spec_data
"""

import logging
import functools
import re
import time
import json
from datetime import datetime, timezone
from typing import Final, Optional, Dict, Any, Callable, TypeVar, Union, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from foundry_mcp.core.context import (
    get_correlation_id,
    get_client_id,
    generate_correlation_id,
    sync_request_context,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Optional Dependencies Availability Flags
# =============================================================================

try:
    import opentelemetry  # noqa: F401

    _OPENTELEMETRY_AVAILABLE = True
except ImportError:
    _OPENTELEMETRY_AVAILABLE = False

try:
    import prometheus_client  # noqa: F401

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


def get_observability_status() -> Dict[str, Any]:
    """Get the current observability stack status.

    Returns a dict containing availability information for optional
    observability dependencies and their enabled status.

    Returns:
        Dict with keys:
        - opentelemetry_available: Whether opentelemetry packages are installed
        - prometheus_available: Whether prometheus_client is installed
        - opentelemetry_enabled: Whether OTel is enabled (via otel module)
        - version: foundry-mcp version
    """
    # Check if OTel is actually enabled (requires otel module)
    otel_enabled = False
    if _OPENTELEMETRY_AVAILABLE:
        try:
            from foundry_mcp.core.otel import is_enabled

            otel_enabled = is_enabled()
        except ImportError:
            pass

    # Get version
    try:
        from importlib.metadata import version

        pkg_version = version("foundry-mcp")
    except Exception:
        pkg_version = "unknown"

    return {
        "opentelemetry_available": _OPENTELEMETRY_AVAILABLE,
        "prometheus_available": _PROMETHEUS_AVAILABLE,
        "opentelemetry_enabled": otel_enabled,
        "version": pkg_version,
    }


# =============================================================================
# Observability Manager
# =============================================================================

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foundry_mcp.config import ObservabilityConfig


class ObservabilityManager:
    """Thread-safe singleton manager for observability stack.

    Provides unified access to OpenTelemetry tracing and Prometheus metrics
    with graceful degradation when dependencies are not available.

    Usage:
        manager = ObservabilityManager.get_instance()
        manager.initialize(config)

        tracer = manager.get_tracer("my-module")
        with tracer.start_as_current_span("my-operation"):
            # ... do work
    """

    _instance: Optional["ObservabilityManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ObservabilityManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    instance._config = None
                    instance._otel_initialized = False
                    instance._prometheus_initialized = False
                    cls._instance = instance
        return cls._instance

    @classmethod
    def get_instance(cls) -> "ObservabilityManager":
        """Get the singleton instance."""
        return cls()

    def initialize(self, config: "ObservabilityConfig") -> None:
        """Initialize observability with configuration.

        Args:
            config: ObservabilityConfig instance from server config
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self._config = config

            # Initialize OpenTelemetry if enabled
            if config.enabled and config.otel_enabled and _OPENTELEMETRY_AVAILABLE:
                try:
                    from foundry_mcp.core.otel import OTelConfig, initialize as init_otel

                    otel_config = OTelConfig(
                        enabled=True,
                        otlp_endpoint=config.otel_endpoint,
                        service_name=config.otel_service_name,
                        sample_rate=config.otel_sample_rate,
                    )
                    init_otel(otel_config)
                    self._otel_initialized = True
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenTelemetry: {e}")

            # Initialize Prometheus if enabled
            if config.enabled and config.prometheus_enabled and _PROMETHEUS_AVAILABLE:
                try:
                    from foundry_mcp.core.prometheus import (
                        PrometheusConfig,
                        get_prometheus_exporter,
                        reset_exporter,
                    )

                    reset_exporter()  # Reset to apply new config
                    prom_config = PrometheusConfig(
                        enabled=True,
                        port=config.prometheus_port,
                        host=config.prometheus_host,
                        namespace=config.prometheus_namespace,
                    )
                    exporter = get_prometheus_exporter(prom_config)
                    if config.prometheus_port > 0:
                        exporter.start_server()
                    self._prometheus_initialized = True
                except Exception as e:
                    logger.warning(f"Failed to initialize Prometheus: {e}")

            self._initialized = True

    def is_tracing_enabled(self) -> bool:
        """Check if OTel tracing is enabled and initialized."""
        return self._otel_initialized

    def is_metrics_enabled(self) -> bool:
        """Check if Prometheus metrics are enabled and initialized."""
        return self._prometheus_initialized

    def get_tracer(self, name: str = __name__) -> Any:
        """Get a tracer instance (real or no-op).

        Args:
            name: Tracer name (typically module __name__)

        Returns:
            Tracer instance
        """
        if self._otel_initialized:
            from foundry_mcp.core.otel import get_tracer

            return get_tracer(name)

        from foundry_mcp.core.otel_stubs import get_noop_tracer

        return get_noop_tracer(name)

    def get_prometheus_exporter(self) -> Any:
        """Get the Prometheus exporter instance.

        Returns:
            PrometheusExporter instance (real or with no-op methods)
        """
        if self._prometheus_initialized:
            from foundry_mcp.core.prometheus import get_prometheus_exporter

            return get_prometheus_exporter()

        # Return a minimal no-op object
        class NoOpExporter:
            def record_tool_invocation(self, *args: Any, **kwargs: Any) -> None:
                pass

            def record_tool_start(self, *args: Any, **kwargs: Any) -> None:
                pass

            def record_tool_end(self, *args: Any, **kwargs: Any) -> None:
                pass

            def record_resource_access(self, *args: Any, **kwargs: Any) -> None:
                pass

            def record_error(self, *args: Any, **kwargs: Any) -> None:
                pass

        return NoOpExporter()

    def shutdown(self) -> None:
        """Shutdown observability providers and flush pending data."""
        if self._otel_initialized:
            try:
                from foundry_mcp.core.otel import shutdown

                shutdown()
            except Exception:
                pass
            self._otel_initialized = False

        self._initialized = False
        self._config = None


# Global manager instance accessor
def get_observability_manager() -> ObservabilityManager:
    """Get the global ObservabilityManager instance."""
    return ObservabilityManager.get_instance()


# =============================================================================
# Sensitive Data Patterns for Redaction
# =============================================================================
# These patterns identify sensitive data that should be redacted from logs,
# error messages, and audit trails. See docs/mcp_best_practices/08-security-trust-boundaries.md

SENSITIVE_PATTERNS: Final[List[Tuple[str, str]]] = [
    # API Keys and Tokens
    (r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{20,})['\"]?", "API_KEY"),
    (
        r"(?i)(secret[_-]?key|secretkey)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{20,})['\"]?",
        "SECRET_KEY",
    ),
    (
        r"(?i)(access[_-]?token|accesstoken)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{20,})['\"]?",
        "ACCESS_TOKEN",
    ),
    (r"(?i)bearer\s+([a-zA-Z0-9_\-\.]+)", "BEARER_TOKEN"),
    # Passwords
    (r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?([^\s'\"]{4,})['\"]?", "PASSWORD"),
    # AWS Credentials
    (r"AKIA[0-9A-Z]{16}", "AWS_ACCESS_KEY"),
    (
        r"(?i)(aws[_-]?secret[_-]?access[_-]?key)\s*[:=]\s*['\"]?([a-zA-Z0-9/+=]{40})['\"]?",
        "AWS_SECRET",
    ),
    # Private Keys
    (r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", "PRIVATE_KEY"),
    # Email Addresses (for PII protection)
    (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "EMAIL"),
    # Social Security Numbers (US)
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
    # Credit Card Numbers (basic pattern)
    (r"\b(?:\d{4}[- ]?){3}\d{4}\b", "CREDIT_CARD"),
    # Phone Numbers (various formats)
    (r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "PHONE"),
    # GitHub/GitLab Tokens
    (r"gh[pousr]_[a-zA-Z0-9]{36,}", "GITHUB_TOKEN"),
    (r"glpat-[a-zA-Z0-9\-]{20,}", "GITLAB_TOKEN"),
    # Generic Base64-encoded secrets (long base64 strings in key contexts)
    (
        r"(?i)(token|secret|key|credential)\s*[:=]\s*['\"]?([a-zA-Z0-9+/]{40,}={0,2})['\"]?",
        "BASE64_SECRET",
    ),
]
"""Patterns for detecting sensitive data that should be redacted.

Each tuple contains:
- regex pattern: The pattern to match sensitive data
- label: A human-readable label for the type of sensitive data

Use with redact_sensitive_data() to sanitize logs and error messages.
"""


def redact_sensitive_data(
    data: Any,
    *,
    patterns: Optional[List[Tuple[str, str]]] = None,
    redaction_format: str = "[REDACTED:{label}]",
    max_depth: int = 10,
) -> Any:
    """Recursively redact sensitive data from strings, dicts, and lists.

    Scans input data for sensitive patterns (API keys, passwords, PII, etc.)
    and replaces matches with redaction markers. Safe for use before logging
    or including data in error messages.

    Args:
        data: The data to redact (string, dict, list, or nested structure)
        patterns: Custom patterns to use (default: SENSITIVE_PATTERNS)
        redaction_format: Format string for redaction markers (uses {label})
        max_depth: Maximum recursion depth to prevent stack overflow

    Returns:
        A copy of the data with sensitive values redacted

    Example:
        >>> data = {"api_key": "sk_live_abc123...", "user": "john"}
        >>> safe_data = redact_sensitive_data(data)
        >>> logger.info("Request data", extra={"data": safe_data})
    """
    if max_depth <= 0:
        return "[MAX_DEPTH_EXCEEDED]"

    check_patterns = patterns if patterns is not None else SENSITIVE_PATTERNS

    def redact_string(text: str) -> str:
        """Redact sensitive patterns from a string."""
        result = text
        for pattern, label in check_patterns:
            replacement = redaction_format.format(label=label)
            result = re.sub(pattern, replacement, result)
        return result

    # Handle different data types
    if isinstance(data, str):
        return redact_string(data)

    elif isinstance(data, dict):
        # Check for sensitive key names and redact their values entirely
        sensitive_keys = {
            "password",
            "passwd",
            "pwd",
            "secret",
            "token",
            "api_key",
            "apikey",
            "api-key",
            "access_token",
            "refresh_token",
            "private_key",
            "secret_key",
            "auth",
            "authorization",
            "credential",
            "credentials",
            "ssn",
            "credit_card",
        }
        result = {}
        for key, value in data.items():
            key_lower = str(key).lower().replace("-", "_")
            if key_lower in sensitive_keys:
                # Redact entire value for known sensitive keys
                result[key] = f"[REDACTED:{key_lower.upper()}]"
            else:
                # Recursively process the value
                result[key] = redact_sensitive_data(
                    value,
                    patterns=check_patterns,
                    redaction_format=redaction_format,
                    max_depth=max_depth - 1,
                )
        return result

    elif isinstance(data, (list, tuple)):
        result_list = [
            redact_sensitive_data(
                item,
                patterns=check_patterns,
                redaction_format=redaction_format,
                max_depth=max_depth - 1,
            )
            for item in data
        ]
        return type(data)(result_list) if isinstance(data, tuple) else result_list

    else:
        # For other types (int, float, bool, None), return as-is
        return data


def redact_for_logging(data: Any) -> str:
    """Convenience function to redact and serialize data for logging.

    Combines redaction with JSON serialization for safe logging.

    Args:
        data: Data to redact and serialize

    Returns:
        JSON string with sensitive data redacted

    Example:
        >>> logger.info(f"Processing request: {redact_for_logging(request_data)}")
    """
    redacted = redact_sensitive_data(data)
    try:
        return json.dumps(redacted, default=str)
    except (TypeError, ValueError):
        return str(redacted)


T = TypeVar("T")


class MetricType(Enum):
    """Types of metrics that can be emitted."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AuditEventType(Enum):
    """Types of audit events for security logging."""

    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    RATE_LIMIT = "rate_limit"
    RESOURCE_ACCESS = "resource_access"
    TOOL_INVOCATION = "tool_invocation"
    PERMISSION_DENIED = "permission_denied"
    CONFIG_CHANGE = "config_change"


@dataclass
class Metric:
    """Structured metric data."""

    name: str
    value: Union[int, float]
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "labels": self.labels,
            "timestamp": self.timestamp,
        }


@dataclass
class AuditEvent:
    """Structured audit event for security logging."""

    event_type: AuditEventType
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    correlation_id: Optional[str] = None
    client_id: Optional[str] = None
    user_id: Optional[str] = None
    ip_address: Optional[str] = None

    def __post_init__(self) -> None:
        """Auto-populate correlation_id and client_id from context if not set."""
        if self.correlation_id is None:
            self.correlation_id = get_correlation_id() or None
        if self.client_id is None:
            ctx_client = get_client_id()
            if ctx_client and ctx_client != "anonymous":
                self.client_id = ctx_client

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "details": self.details,
        }
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        if self.client_id:
            result["client_id"] = self.client_id
        if self.user_id:
            result["user_id"] = self.user_id
        if self.ip_address:
            result["ip_address"] = self.ip_address
        return result


class MetricsCollector:
    """
    Collects and emits metrics to the standard logger and Prometheus.

    Metrics are logged as structured JSON for easy parsing by
    log aggregation systems (e.g., Datadog, Splunk, CloudWatch).
    When Prometheus is enabled, metrics are also exported via the
    Prometheus exporter.
    """

    def __init__(self, prefix: str = "foundry_mcp"):
        self.prefix = prefix
        self._logger = logging.getLogger(f"{__name__}.metrics")

    def emit(self, metric: Metric) -> None:
        """Emit a metric to the logger and Prometheus (if enabled).

        Args:
            metric: The Metric to emit
        """
        # Always emit to structured logger
        self._logger.info(
            f"METRIC: {self.prefix}.{metric.name}", extra={"metric": metric.to_dict()}
        )

        # Emit to Prometheus if enabled
        manager = get_observability_manager()
        if manager.is_metrics_enabled():
            exporter = manager.get_prometheus_exporter()
            # Map our metric types to Prometheus exporter methods
            if metric.metric_type == MetricType.COUNTER:
                # Prometheus counters don't support arbitrary labels easily,
                # so we record as tool invocation if it has tool label
                if "tool" in metric.labels:
                    exporter.record_tool_invocation(
                        metric.labels["tool"],
                        success=metric.labels.get("status") == "success",
                    )
            elif metric.metric_type == MetricType.TIMER:
                # Record duration via tool invocation
                if "tool" in metric.labels:
                    exporter.record_tool_invocation(
                        metric.labels["tool"],
                        success=True,
                        duration_ms=metric.value,
                    )

    def counter(
        self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Emit a counter metric."""
        self.emit(
            Metric(
                name=name,
                value=value,
                metric_type=MetricType.COUNTER,
                labels=labels or {},
            )
        )

    def gauge(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Emit a gauge metric."""
        self.emit(
            Metric(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                labels=labels or {},
            )
        )

    def timer(
        self, name: str, duration_ms: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Emit a timer metric (duration in milliseconds)."""
        self.emit(
            Metric(
                name=name,
                value=duration_ms,
                metric_type=MetricType.TIMER,
                labels=labels or {},
            )
        )

    def histogram(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Emit a histogram metric for distribution tracking."""
        self.emit(
            Metric(
                name=name,
                value=value,
                metric_type=MetricType.HISTOGRAM,
                labels=labels or {},
            )
        )


# Global metrics collector
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics


class AuditLogger:
    """
    Structured audit logging for security events.

    Audit logs are written to a separate logger for easy filtering
    and compliance requirements.
    """

    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.audit")

    def log(self, event: AuditEvent) -> None:
        """Log an audit event."""
        self._logger.info(
            f"AUDIT: {event.event_type.value}", extra={"audit": event.to_dict()}
        )

    def auth_success(self, client_id: Optional[str] = None, **details: Any) -> None:
        """Log successful authentication."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.AUTH_SUCCESS,
                client_id=client_id,
                details=details,
            )
        )

    def auth_failure(
        self,
        reason: str,
        client_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        **details: Any,
    ) -> None:
        """Log failed authentication."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.AUTH_FAILURE,
                client_id=client_id,
                ip_address=ip_address,
                details={"reason": reason, **details},
            )
        )

    def rate_limit(
        self,
        client_id: Optional[str] = None,
        limit: Optional[int] = None,
        **details: Any,
    ) -> None:
        """Log rate limit event."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.RATE_LIMIT,
                client_id=client_id,
                details={"limit": limit, **details},
            )
        )

    def resource_access(
        self, resource_type: str, resource_id: str, action: str = "read", **details: Any
    ) -> None:
        """Log resource access."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.RESOURCE_ACCESS,
                details={
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "action": action,
                    **details,
                },
            )
        )

    def tool_invocation(
        self,
        tool_name: str,
        success: bool = True,
        duration_ms: Optional[float] = None,
        correlation_id: Optional[str] = None,
        **details: Any,
    ) -> None:
        """Log tool invocation."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.TOOL_INVOCATION,
                correlation_id=correlation_id,
                details={
                    "tool": tool_name,
                    "success": success,
                    "duration_ms": duration_ms,
                    **details,
                },
            )
        )


# Global audit logger
_audit = AuditLogger()


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger."""
    return _audit


def audit_log(event_type: str, **details: Any) -> None:
    """
    Convenience function for audit logging.

    Args:
        event_type: Type of event (auth_success, auth_failure, rate_limit,
                    resource_access, tool_invocation, permission_denied, config_change)
        **details: Additional details to include in the audit log
    """
    try:
        event_enum = AuditEventType(event_type)
    except ValueError:
        event_enum = AuditEventType.TOOL_INVOCATION
        details["original_event_type"] = event_type

    _audit.log(AuditEvent(event_type=event_enum, details=details))


def _record_to_metrics_persistence(
    tool_name: str, success: bool, duration_ms: float
) -> None:
    """
    Record tool invocation to metrics persistence for dashboard visibility.

    Fails silently if metrics persistence is not configured.
    """
    try:
        from foundry_mcp.core.metrics_persistence import get_metrics_collector

        collector = get_metrics_collector()
        if collector is not None and collector._config.enabled:
            status = "success" if success else "error"
            collector.record(
                "tool_invocations_total",
                1.0,
                metric_type="counter",
                labels={"tool": tool_name, "status": status},
            )
            collector.record(
                "tool_duration_ms",
                duration_ms,
                metric_type="gauge",
                labels={"tool": tool_name},
            )
    except Exception:
        # Never let metrics persistence failures affect tool execution
        pass


def mcp_tool(
    tool_name: Optional[str] = None, emit_metrics: bool = True, audit: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for MCP tool handlers with observability.

    Automatically:
    - Logs tool invocations
    - Emits latency and status metrics
    - Creates audit log entries
    - Creates OTel spans when tracing is enabled
    - Records Prometheus metrics when metrics are enabled

    Args:
        tool_name: Override tool name (defaults to function name)
        emit_metrics: Whether to emit metrics
        audit: Whether to create audit log entries
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = tool_name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            # Set up request context if not already set
            existing_corr_id = get_correlation_id()
            corr_id = existing_corr_id or generate_correlation_id(prefix="tool")

            # Use context manager if we need to establish context
            if not existing_corr_id:
                with sync_request_context(correlation_id=corr_id):
                    return await _async_tool_impl(
                        func, name, corr_id, emit_metrics, audit, *args, **kwargs
                    )
            else:
                return await _async_tool_impl(
                    func, name, corr_id, emit_metrics, audit, *args, **kwargs
                )

        async def _async_tool_impl(
            func: Callable[..., T],
            name: str,
            corr_id: str,
            emit_metrics: bool,
            audit: bool,
            *args: Any,
            **kwargs: Any,
        ) -> T:
            start = time.perf_counter()
            success = True
            error_msg = None

            # Get observability manager for OTel/Prometheus integration
            manager = get_observability_manager()
            tracer = manager.get_tracer(__name__) if manager.is_tracing_enabled() else None
            prom_exporter = manager.get_prometheus_exporter() if manager.is_metrics_enabled() else None

            # Start Prometheus active operation tracking
            if prom_exporter:
                prom_exporter.record_tool_start(name)

            # Create OTel span if tracing enabled (with correlation_id attribute)
            span_context = None
            if tracer:
                span_context = tracer.start_as_current_span(
                    f"tool:{name}",
                    attributes={
                        "tool.name": name,
                        "tool.type": "mcp_tool",
                        "request.correlation_id": corr_id,
                    },
                )

            try:
                if span_context:
                    with span_context:
                        result = await func(*args, **kwargs)
                else:
                    result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                # Record error in Prometheus
                if prom_exporter:
                    prom_exporter.record_error(name, type(e).__name__)
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000

                # End Prometheus active operation tracking
                if prom_exporter:
                    prom_exporter.record_tool_end(name)
                    prom_exporter.record_tool_invocation(
                        name, success=success, duration_ms=duration_ms
                    )

                if emit_metrics:
                    labels = {"tool": name, "status": "success" if success else "error"}
                    _metrics.counter("tool.invocations", labels=labels)
                    _metrics.timer("tool.latency", duration_ms, labels={"tool": name})

                if audit:
                    _audit.tool_invocation(
                        tool_name=name,
                        success=success,
                        duration_ms=round(duration_ms, 2),
                        error=error_msg,
                        correlation_id=corr_id,
                    )

                # Record to metrics persistence (for dashboard visibility)
                _record_to_metrics_persistence(name, success, duration_ms)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            # Set up request context if not already set
            existing_corr_id = get_correlation_id()
            corr_id = existing_corr_id or generate_correlation_id(prefix="tool")

            # Use context manager if we need to establish context
            if not existing_corr_id:
                with sync_request_context(correlation_id=corr_id):
                    return _sync_tool_impl(
                        func, name, corr_id, emit_metrics, audit, args, kwargs
                    )
            else:
                return _sync_tool_impl(
                    func, name, corr_id, emit_metrics, audit, args, kwargs
                )

        def _sync_tool_impl(
            _wrapped_func: Callable[..., T],
            _tool_name: str,
            _corr_id: str,
            _emit_metrics: bool,
            _do_audit: bool,
            _args: tuple,
            _kwargs: dict,
        ) -> T:
            """Internal implementation for sync tool execution.

            Note: Parameter names are prefixed with underscore to avoid
            conflicts with tool parameter names (e.g., 'name').
            """
            start = time.perf_counter()
            success = True
            error_msg = None

            # Get observability manager for OTel/Prometheus integration
            manager = get_observability_manager()
            tracer = manager.get_tracer(__name__) if manager.is_tracing_enabled() else None
            prom_exporter = manager.get_prometheus_exporter() if manager.is_metrics_enabled() else None

            # Start Prometheus active operation tracking
            if prom_exporter:
                prom_exporter.record_tool_start(_tool_name)

            # Create OTel span if tracing enabled (with correlation_id attribute)
            span_context = None
            if tracer:
                span_context = tracer.start_as_current_span(
                    f"tool:{_tool_name}",
                    attributes={
                        "tool.name": _tool_name,
                        "tool.type": "mcp_tool",
                        "request.correlation_id": _corr_id,
                    },
                )

            try:
                if span_context:
                    with span_context:
                        result = _wrapped_func(*_args, **_kwargs)
                else:
                    result = _wrapped_func(*_args, **_kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                # Record error in Prometheus
                if prom_exporter:
                    prom_exporter.record_error(_tool_name, type(e).__name__)
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000

                # End Prometheus active operation tracking
                if prom_exporter:
                    prom_exporter.record_tool_end(_tool_name)
                    prom_exporter.record_tool_invocation(
                        _tool_name, success=success, duration_ms=duration_ms
                    )

                if _emit_metrics:
                    labels = {"tool": _tool_name, "status": "success" if success else "error"}
                    _metrics.counter("tool.invocations", labels=labels)
                    _metrics.timer("tool.latency", duration_ms, labels={"tool": _tool_name})

                if _do_audit:
                    _audit.tool_invocation(
                        tool_name=_tool_name,
                        success=success,
                        duration_ms=round(duration_ms, 2),
                        error=error_msg,
                        correlation_id=_corr_id,
                    )

                # Record to metrics persistence (for dashboard visibility)
                _record_to_metrics_persistence(_tool_name, success, duration_ms)

        # Return appropriate wrapper based on whether func is async
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def mcp_resource(
    resource_type: Optional[str] = None, emit_metrics: bool = True, audit: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for MCP resource handlers with observability.

    Automatically:
    - Logs resource access
    - Emits latency and status metrics
    - Creates audit log entries
    - Creates OTel spans when tracing is enabled
    - Records Prometheus metrics when metrics are enabled

    Args:
        resource_type: Type of resource (e.g., "spec", "journal")
        emit_metrics: Whether to emit metrics
        audit: Whether to create audit log entries
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        rtype = resource_type or "resource"

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            success = True
            error_msg = None
            resource_id = kwargs.get("spec_id") or kwargs.get("id") or "unknown"

            # Get observability manager for OTel/Prometheus integration
            manager = get_observability_manager()
            tracer = manager.get_tracer(__name__) if manager.is_tracing_enabled() else None
            prom_exporter = manager.get_prometheus_exporter() if manager.is_metrics_enabled() else None

            # Create OTel span if tracing enabled
            span_context = None
            if tracer:
                span_context = tracer.start_as_current_span(
                    f"resource:{rtype}",
                    attributes={
                        "resource.type": rtype,
                        "resource.id": str(resource_id),
                    },
                )

            try:
                if span_context:
                    with span_context:
                        result = await func(*args, **kwargs)
                else:
                    result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                # Record error in Prometheus
                if prom_exporter:
                    prom_exporter.record_error(f"resource:{rtype}", type(e).__name__)
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000

                # Record Prometheus resource access
                if prom_exporter:
                    prom_exporter.record_resource_access(rtype, "read")

                if emit_metrics:
                    labels = {
                        "resource_type": rtype,
                        "status": "success" if success else "error",
                    }
                    _metrics.counter("resource.access", labels=labels)
                    _metrics.timer(
                        "resource.latency", duration_ms, labels={"resource_type": rtype}
                    )

                if audit:
                    _audit.resource_access(
                        resource_type=rtype,
                        resource_id=str(resource_id),
                        action="read",
                        success=success,
                        duration_ms=round(duration_ms, 2),
                        error=error_msg,
                    )

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            success = True
            error_msg = None
            resource_id = kwargs.get("spec_id") or kwargs.get("id") or "unknown"

            # Get observability manager for OTel/Prometheus integration
            manager = get_observability_manager()
            tracer = manager.get_tracer(__name__) if manager.is_tracing_enabled() else None
            prom_exporter = manager.get_prometheus_exporter() if manager.is_metrics_enabled() else None

            # Create OTel span if tracing enabled
            span_context = None
            if tracer:
                span_context = tracer.start_as_current_span(
                    f"resource:{rtype}",
                    attributes={
                        "resource.type": rtype,
                        "resource.id": str(resource_id),
                    },
                )

            try:
                if span_context:
                    with span_context:
                        result = func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                # Record error in Prometheus
                if prom_exporter:
                    prom_exporter.record_error(f"resource:{rtype}", type(e).__name__)
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000

                # Record Prometheus resource access
                if prom_exporter:
                    prom_exporter.record_resource_access(rtype, "read")

                if emit_metrics:
                    labels = {
                        "resource_type": rtype,
                        "status": "success" if success else "error",
                    }
                    _metrics.counter("resource.access", labels=labels)
                    _metrics.timer(
                        "resource.latency", duration_ms, labels={"resource_type": rtype}
                    )

                if audit:
                    _audit.resource_access(
                        resource_type=rtype,
                        resource_id=str(resource_id),
                        action="read",
                        success=success,
                        duration_ms=round(duration_ms, 2),
                        error=error_msg,
                    )

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
