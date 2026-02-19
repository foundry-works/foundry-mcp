"""Observability Manager singleton.

Thread-safe singleton for managing the observability stack (OpenTelemetry
tracing and Prometheus metrics) with graceful degradation.
"""

import logging
import threading
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from foundry_mcp.config.domains import ObservabilityConfig

logger = logging.getLogger(__name__)


# Optional Dependencies Availability Flags
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
            from foundry_mcp.core.observability.otel import is_enabled

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
                    from foundry_mcp.core.observability.otel import OTelConfig
                    from foundry_mcp.core.observability.otel import initialize as init_otel

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
                    from foundry_mcp.core.observability.prometheus import (
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
            from foundry_mcp.core.observability.otel import get_tracer

            return get_tracer(name)

        from foundry_mcp.core.observability.stubs import get_noop_tracer

        return get_noop_tracer(name)

    def get_prometheus_exporter(self) -> Any:
        """Get the Prometheus exporter instance.

        Returns:
            PrometheusExporter instance (real or with no-op methods)
        """
        if self._prometheus_initialized:
            from foundry_mcp.core.observability.prometheus import get_prometheus_exporter

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
                from foundry_mcp.core.observability.otel import shutdown

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
