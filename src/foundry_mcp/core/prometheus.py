"""Prometheus metrics integration with graceful degradation.

This module provides Prometheus metrics integration that gracefully falls back
to no-op operations when the optional prometheus_client dependency is not installed.

Usage:
    from foundry_mcp.core.prometheus import get_prometheus_exporter

    exporter = get_prometheus_exporter()
    exporter.record_tool_invocation("list_specs", success=True, duration_ms=45.2)

    # Optionally start HTTP server for /metrics endpoint
    exporter.start_server(port=9090)
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

# Try to import prometheus_client
try:
    from prometheus_client import (
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

    # Placeholders so type checkers don't complain.
    Counter: Any = None
    Gauge: Any = None
    Histogram: Any = None
    REGISTRY: Any = None
    start_http_server: Any = None

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PrometheusConfig:
    """Configuration for Prometheus metrics.

    Attributes:
        enabled: Whether Prometheus metrics are enabled
        port: HTTP server port for /metrics endpoint (0 = no server)
        host: HTTP server host
        namespace: Metric namespace prefix
    """

    enabled: bool = False
    port: int = 0  # 0 means don't start HTTP server
    host: str = "0.0.0.0"
    namespace: str = "foundry_mcp"

    @classmethod
    def from_env_and_config(
        cls,
        config: Optional[dict[str, Any]] = None,
    ) -> "PrometheusConfig":
        """Load configuration from environment variables and optional config dict.

        Environment variables take precedence over config dict values.

        Env vars:
            PROMETHEUS_ENABLED: "true" or "1" to enable
            PROMETHEUS_PORT: HTTP server port (0 = no server)
            PROMETHEUS_HOST: HTTP server host
            PROMETHEUS_NAMESPACE: Metric namespace

        Args:
            config: Optional dict with config values (typically from TOML)

        Returns:
            PrometheusConfig instance
        """
        config = config or {}

        # Parse enabled from env or config
        env_enabled = os.environ.get("PROMETHEUS_ENABLED", "").lower()
        if env_enabled:
            enabled = env_enabled in ("true", "1", "yes")
        else:
            enabled = config.get("enabled", False)

        # Parse port
        port_str = os.environ.get("PROMETHEUS_PORT")
        if port_str:
            try:
                port = int(port_str)
            except ValueError:
                port = 0
        else:
            port = config.get("port", 0)

        # Parse host
        host = os.environ.get(
            "PROMETHEUS_HOST",
            config.get("host", "0.0.0.0"),
        )

        # Parse namespace
        namespace = os.environ.get(
            "PROMETHEUS_NAMESPACE",
            config.get("namespace", "foundry_mcp"),
        )

        return cls(
            enabled=enabled,
            port=port,
            host=host,
            namespace=namespace,
        )


# =============================================================================
# Prometheus Exporter
# =============================================================================


class PrometheusExporter:
    """Prometheus metrics exporter with graceful degradation.

    When prometheus_client is not installed or metrics are disabled,
    all methods become no-ops that silently do nothing.
    """

    def __init__(self, config: Optional[PrometheusConfig] = None) -> None:
        """Initialize the exporter.

        Args:
            config: Prometheus configuration. If None, loads from env/defaults.
        """
        self._config = config or PrometheusConfig.from_env_and_config()
        self._initialized = False
        self._server_started = False
        self._lock = threading.Lock()

        # Metric instances (set during initialization)
        self._tool_invocations: Any = None
        self._tool_duration: Any = None
        self._tool_errors: Any = None
        self._resource_access: Any = None
        self._active_operations: Any = None

        # Manifest/discovery metrics
        self._manifest_tokens: Any = None
        self._manifest_tool_count: Any = None
        self._feature_flag_state: Any = None

        # Health check metrics
        self._health_status: Any = None
        self._dependency_health: Any = None
        self._health_check_duration: Any = None

        # Auto-initialize if enabled
        if self.is_enabled():
            self._initialize_metrics()

    def is_available(self) -> bool:
        """Check if prometheus_client is installed."""
        return _PROMETHEUS_AVAILABLE

    def is_enabled(self) -> bool:
        """Check if Prometheus metrics are enabled and available."""
        return self._config.enabled and _PROMETHEUS_AVAILABLE

    def _initialize_metrics(self) -> None:
        """Initialize Prometheus metric instances."""
        if self._initialized or not self.is_enabled():
            return

        with self._lock:
            if self._initialized:
                return

            ns = self._config.namespace

            # Tool invocation counter
            self._tool_invocations = Counter(
                f"{ns}_tool_invocations_total",
                "Total number of tool invocations",
                ["tool", "status"],
            )

            # Tool duration histogram
            self._tool_duration = Histogram(
                f"{ns}_tool_duration_seconds",
                "Tool execution duration in seconds",
                ["tool"],
                buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            )

            # Tool error counter
            self._tool_errors = Counter(
                f"{ns}_tool_errors_total",
                "Total number of tool errors",
                ["tool", "error_type"],
            )

            # Resource access counter
            self._resource_access = Counter(
                f"{ns}_resource_access_total",
                "Total number of resource accesses",
                ["resource_type", "action"],
            )

            # Active operations gauge
            self._active_operations = Gauge(
                f"{ns}_active_operations",
                "Number of currently active operations",
                ["operation_type"],
            )

            # Manifest/discovery gauges
            self._manifest_tokens = Gauge(
                f"{ns}_manifest_tokens",
                "Estimated token count for the advertised tool manifest",
                ["manifest"],  # unified|legacy
            )
            self._manifest_tool_count = Gauge(
                f"{ns}_manifest_tool_count",
                "Tool count for the advertised tool manifest",
                ["manifest"],  # unified|legacy
            )
            self._feature_flag_state = Gauge(
                f"{ns}_feature_flag_state",
                "Feature flag state (1=enabled, 0=disabled)",
                ["flag"],
            )

            # Health check metrics
            self._health_status = Gauge(
                f"{ns}_health_status",
                "Current health status (0=unhealthy, 1=degraded, 2=healthy)",
                ["check_type"],  # liveness, readiness, health
            )

            self._dependency_health = Gauge(
                f"{ns}_dependency_health",
                "Dependency health status (0=unhealthy, 1=healthy)",
                ["dependency"],
            )

            self._health_check_duration = Histogram(
                f"{ns}_health_check_duration_seconds",
                "Health check duration in seconds",
                ["check_type"],
                buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            )

            self._initialized = True

    def start_server(
        self, port: Optional[int] = None, host: Optional[str] = None
    ) -> bool:
        """Start the HTTP server for /metrics endpoint.

        Args:
            port: Override port from config
            host: Override host from config

        Returns:
            True if server started, False if already running or not enabled
        """
        if not self.is_enabled():
            return False

        if self._server_started:
            return False

        with self._lock:
            if self._server_started:
                return False

            actual_port = port or self._config.port
            if actual_port <= 0:
                return False

            actual_host = host or self._config.host

            try:
                start_http_server(actual_port, addr=actual_host)
                self._server_started = True
                return True
            except Exception:
                return False

    def record_tool_invocation(
        self,
        tool_name: str,
        *,
        success: bool = True,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Record a tool invocation.

        Args:
            tool_name: Name of the tool
            success: Whether the invocation was successful
            duration_ms: Duration in milliseconds (optional)
        """
        if not self.is_enabled():
            return

        status = "success" if success else "error"
        self._tool_invocations.labels(tool=tool_name, status=status).inc()

        if duration_ms is not None:
            # Convert ms to seconds for Prometheus conventions
            self._tool_duration.labels(tool=tool_name).observe(duration_ms / 1000.0)

    def record_tool_start(self, tool_name: str) -> None:
        """Record tool execution start (increment active operations).

        Args:
            tool_name: Name of the tool
        """
        if not self.is_enabled():
            return

        self._active_operations.labels(operation_type=f"tool:{tool_name}").inc()

    def record_tool_end(self, tool_name: str) -> None:
        """Record tool execution end (decrement active operations).

        Args:
            tool_name: Name of the tool
        """
        if not self.is_enabled():
            return

        self._active_operations.labels(operation_type=f"tool:{tool_name}").dec()

    def record_resource_access(
        self,
        resource_type: str,
        action: str = "read",
    ) -> None:
        """Record a resource access.

        Args:
            resource_type: Type of resource (e.g., "spec", "task", "journal")
            action: Action performed (e.g., "read", "write", "delete")
        """
        if not self.is_enabled():
            return

        self._resource_access.labels(resource_type=resource_type, action=action).inc()

    def record_error(
        self,
        tool_name: str,
        error_type: str = "unknown",
    ) -> None:
        """Record a tool error.

        Args:
            tool_name: Name of the tool
            error_type: Type/category of error
        """
        if not self.is_enabled():
            return

        self._tool_errors.labels(tool=tool_name, error_type=error_type).inc()

    # -------------------------------------------------------------------------
    # Manifest/Discovery Metrics
    # -------------------------------------------------------------------------

    def record_manifest_snapshot(
        self,
        *,
        manifest: str,
        tokens: int,
        tool_count: int,
    ) -> None:
        """Record a manifest snapshot (token count + tool count)."""
        if not self.is_enabled():
            return

        manifest_label = manifest or "unknown"
        self._manifest_tokens.labels(manifest=manifest_label).set(int(tokens))
        self._manifest_tool_count.labels(manifest=manifest_label).set(int(tool_count))

    def record_feature_flag_state(self, flag: str, enabled: bool) -> None:
        """Record feature flag enabled/disabled state."""
        if not self.is_enabled():
            return

        self._feature_flag_state.labels(flag=flag).set(1 if enabled else 0)

    # -------------------------------------------------------------------------
    # Health Check Metrics
    # -------------------------------------------------------------------------

    def record_health_check(
        self,
        check_type: str,
        status: int,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Record a health check result.

        Args:
            check_type: Type of check (liveness, readiness, health)
            status: Health status (0=unhealthy, 1=degraded, 2=healthy)
            duration_seconds: Optional duration of the check in seconds
        """
        if not self.is_enabled():
            return

        self._health_status.labels(check_type=check_type).set(status)

        if duration_seconds is not None:
            self._health_check_duration.labels(check_type=check_type).observe(
                duration_seconds
            )

    def record_dependency_health(
        self,
        dependency: str,
        healthy: bool,
    ) -> None:
        """Record dependency health status.

        Args:
            dependency: Name of the dependency (e.g., specs_dir, otel, prometheus)
            healthy: Whether the dependency is healthy
        """
        if not self.is_enabled():
            return

        self._dependency_health.labels(dependency=dependency).set(1 if healthy else 0)

    def record_health_check_batch(
        self,
        check_type: str,
        status: int,
        dependencies: dict[str, bool],
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Record a complete health check with all dependencies.

        Convenience method to record overall status and all dependency statuses.

        Args:
            check_type: Type of check (liveness, readiness, health)
            status: Health status (0=unhealthy, 1=degraded, 2=healthy)
            dependencies: Dict mapping dependency name to healthy status
            duration_seconds: Optional duration of the check in seconds
        """
        if not self.is_enabled():
            return

        # Record overall status
        self.record_health_check(check_type, status, duration_seconds)

        # Record each dependency
        for dep_name, is_healthy in dependencies.items():
            self.record_dependency_health(dep_name, is_healthy)

    def get_config(self) -> PrometheusConfig:
        """Get the current configuration."""
        return self._config


# =============================================================================
# Singleton Instance
# =============================================================================

_exporter: Optional[PrometheusExporter] = None
_exporter_lock = threading.Lock()


def get_prometheus_exporter(
    config: Optional[PrometheusConfig] = None,
) -> PrometheusExporter:
    """Get the singleton Prometheus exporter instance.

    On first call, initializes with provided config or defaults.
    Subsequent calls return the same instance (config parameter ignored).

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        PrometheusExporter singleton instance
    """
    global _exporter

    if _exporter is None:
        with _exporter_lock:
            if _exporter is None:
                _exporter = PrometheusExporter(config)

    return _exporter


def reset_exporter() -> None:
    """Reset the singleton exporter (mainly for testing)."""
    global _exporter
    with _exporter_lock:
        _exporter = None


# =============================================================================
# Context Manager for Timing
# =============================================================================


class timed_operation:
    """Context manager for timing tool operations.

    Usage:
        with timed_operation("my_tool") as timer:
            # do work
            pass
        # Automatically records duration
    """

    def __init__(
        self, tool_name: str, exporter: Optional[PrometheusExporter] = None
    ) -> None:
        self.tool_name = tool_name
        self.exporter = exporter or get_prometheus_exporter()
        self.start_time: Optional[float] = None
        self.success = True

    def __enter__(self) -> "timed_operation":
        self.start_time = time.perf_counter()
        self.exporter.record_tool_start(self.tool_name)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration_ms = (time.perf_counter() - (self.start_time or 0)) * 1000
        self.success = exc_type is None

        self.exporter.record_tool_end(self.tool_name)
        self.exporter.record_tool_invocation(
            self.tool_name,
            success=self.success,
            duration_ms=duration_ms,
        )

        if exc_type is not None:
            error_type = exc_type.__name__ if exc_type else "unknown"
            self.exporter.record_error(self.tool_name, error_type)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "PrometheusConfig",
    # Exporter
    "PrometheusExporter",
    "get_prometheus_exporter",
    "reset_exporter",
    # Context manager
    "timed_operation",
]
