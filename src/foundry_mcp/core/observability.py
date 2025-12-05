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
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


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
    client_id: Optional[str] = None
    user_id: Optional[str] = None
    ip_address: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "details": self.details,
        }
        if self.client_id:
            result["client_id"] = self.client_id
        if self.user_id:
            result["user_id"] = self.user_id
        if self.ip_address:
            result["ip_address"] = self.ip_address
        return result


class MetricsCollector:
    """
    Collects and emits metrics to the standard logger.

    Metrics are logged as structured JSON for easy parsing by
    log aggregation systems (e.g., Datadog, Splunk, CloudWatch).
    """

    def __init__(self, prefix: str = "foundry_mcp"):
        self.prefix = prefix
        self._logger = logging.getLogger(f"{__name__}.metrics")

    def emit(self, metric: Metric) -> None:
        """Emit a metric to the logger."""
        self._logger.info(
            f"METRIC: {self.prefix}.{metric.name}", extra={"metric": metric.to_dict()}
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
        **details: Any,
    ) -> None:
        """Log tool invocation."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.TOOL_INVOCATION,
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


def mcp_tool(
    tool_name: Optional[str] = None, emit_metrics: bool = True, audit: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for MCP tool handlers with observability.

    Automatically:
    - Logs tool invocations
    - Emits latency and status metrics
    - Creates audit log entries

    Args:
        tool_name: Override tool name (defaults to function name)
        emit_metrics: Whether to emit metrics
        audit: Whether to create audit log entries
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = tool_name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            success = True
            error_msg = None

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000

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
                    )

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            success = True
            error_msg = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000

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
                    )

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

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000

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

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000

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
