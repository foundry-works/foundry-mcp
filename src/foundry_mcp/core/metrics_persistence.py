"""
Metrics persistence collector for the foundry-mcp server.

Hooks into PrometheusExporter to capture metrics and persist them
to disk with time-bucket aggregation.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from foundry_mcp.config import MetricsPersistenceConfig
from foundry_mcp.core.metrics_store import (
    MetricDataPoint,
    FileMetricsStore,
    MetricsStore,
)

logger = logging.getLogger(__name__)


@dataclass
class MetricBucket:
    """
    Aggregated metrics bucket for a time period.

    Collects multiple samples within a time window and aggregates them
    for storage efficiency.
    """

    metric_name: str
    metric_type: str
    labels: dict[str, str]
    bucket_start: datetime
    bucket_end: datetime
    values: list[float] = field(default_factory=list)
    sample_count: int = 0

    def add_sample(self, value: float) -> None:
        """Add a sample to the bucket."""
        self.values.append(value)
        self.sample_count += 1

    def get_aggregated_value(self) -> float:
        """Get the aggregated value based on metric type."""
        if not self.values:
            return 0.0

        if self.metric_type == "counter":
            # For counters, sum all increments
            return sum(self.values)
        elif self.metric_type == "gauge":
            # For gauges, use the last value
            return self.values[-1]
        elif self.metric_type == "histogram":
            # For histograms, store average of observed values
            return sum(self.values) / len(self.values) if self.values else 0.0
        else:
            # Default: sum
            return sum(self.values)

    def to_data_point(self) -> MetricDataPoint:
        """Convert bucket to a MetricDataPoint."""
        return MetricDataPoint(
            metric_name=self.metric_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=self.get_aggregated_value(),
            metric_type=self.metric_type,
            labels=self.labels,
            bucket_start=self.bucket_start.isoformat(),
            bucket_end=self.bucket_end.isoformat(),
            sample_count=self.sample_count,
        )


class MetricsPersistenceCollector:
    """
    Collects metrics and persists them to storage.

    Features:
    - In-memory buffering with periodic flush
    - Time-bucket aggregation to reduce storage
    - Configurable metric filtering
    - Thread-safe operation
    """

    def __init__(
        self,
        config: MetricsPersistenceConfig,
        store: Optional[MetricsStore] = None,
    ) -> None:
        """
        Initialize the collector.

        Args:
            config: Metrics persistence configuration
            store: Optional MetricsStore (uses global singleton if not provided)
        """
        self._config = config
        self._store = store

        # In-memory buffer: key = (metric_name, labels_tuple), value = MetricBucket
        self._buffer: dict[tuple[str, tuple[tuple[str, str], ...]], MetricBucket] = {}
        self._buffer_lock = threading.Lock()

        # Bucket timing - set interval before using _get_bucket_start
        self._bucket_interval = config.bucket_interval_seconds
        self._current_bucket_start = self._get_bucket_start(datetime.now(timezone.utc))

        # Flush timing
        self._last_flush = time.time()
        self._flush_interval = config.flush_interval_seconds

        # Background flush thread
        self._flush_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

        # Start background flush if enabled
        if config.enabled:
            self._start_flush_thread()
            # Note: atexit handler removed to avoid premature shutdown in MCP stdio mode
            # The server's shutdown() method should call _shutdown_flush_thread explicitly

    def _get_store(self) -> MetricsStore:
        """Get the metrics store (lazy initialization)."""
        if self._store is None:
            storage_path = self._config.get_storage_path()
            self._store = FileMetricsStore(storage_path)
        return self._store

    def _get_bucket_start(self, dt: datetime) -> datetime:
        """Get the start of the bucket containing the given datetime."""
        # Round down to nearest bucket interval
        timestamp = dt.timestamp()
        bucket_start_ts = (timestamp // self._bucket_interval) * self._bucket_interval
        return datetime.fromtimestamp(bucket_start_ts, tz=timezone.utc)

    def _get_bucket_key(
        self,
        metric_name: str,
        labels: dict[str, str],
    ) -> tuple[str, tuple[tuple[str, str], ...]]:
        """Create a hashable key for the bucket."""
        # Sort labels for consistent hashing
        labels_tuple = tuple(sorted(labels.items()))
        return (metric_name, labels_tuple)

    def _should_persist(self, metric_name: str) -> bool:
        """Check if this metric should be persisted."""
        if not self._config.enabled:
            return False
        return self._config.should_persist_metric(metric_name)

    def record(
        self,
        metric_name: str,
        value: float,
        metric_type: str = "counter",
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            metric_type: Type of metric (counter, gauge, histogram)
            labels: Label key-value pairs
        """
        if not self._should_persist(metric_name):
            return

        labels = labels or {}
        now = datetime.now(timezone.utc)
        bucket_start = self._get_bucket_start(now)
        bucket_end = datetime.fromtimestamp(
            bucket_start.timestamp() + self._bucket_interval,
            tz=timezone.utc,
        )

        with self._buffer_lock:
            key = self._get_bucket_key(metric_name, labels)

            # Create new bucket if needed or bucket has rolled over
            if key not in self._buffer or self._buffer[key].bucket_start != bucket_start:
                # Flush old bucket if exists
                if key in self._buffer:
                    self._flush_bucket(key)

                self._buffer[key] = MetricBucket(
                    metric_name=metric_name,
                    metric_type=metric_type,
                    labels=labels,
                    bucket_start=bucket_start,
                    bucket_end=bucket_end,
                )

            self._buffer[key].add_sample(value)

    def _flush_bucket(self, key: tuple[str, tuple[tuple[str, str], ...]]) -> None:
        """Flush a single bucket to storage (caller holds lock)."""
        if key not in self._buffer:
            return

        bucket = self._buffer[key]
        if bucket.sample_count == 0:
            return

        try:
            data_point = bucket.to_data_point()
            self._get_store().append(data_point)
        except Exception as e:
            logger.warning(f"Failed to flush metric bucket: {e}")

    def flush(self) -> int:
        """
        Flush all buffered metrics to storage.

        Returns:
            Number of data points flushed
        """
        flushed = 0

        with self._buffer_lock:
            # Collect all buckets with data
            buckets_to_flush = [
                bucket.to_data_point()
                for bucket in self._buffer.values()
                if bucket.sample_count > 0
            ]

            # Clear buffer
            self._buffer.clear()
            self._last_flush = time.time()

        # Flush outside the lock to avoid holding it during I/O
        if buckets_to_flush:
            try:
                self._get_store().append_batch(buckets_to_flush)
                flushed = len(buckets_to_flush)
                logger.debug(f"Flushed {flushed} metric buckets to storage")
            except Exception as e:
                logger.warning(f"Failed to flush metrics batch: {e}")

        return flushed

    def _start_flush_thread(self) -> None:
        """Start the background flush thread."""
        if self._flush_thread is not None:
            return

        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="metrics-persistence-flush",
            daemon=True,
        )
        self._flush_thread.start()

    def _flush_loop(self) -> None:
        """Background loop to periodically flush metrics."""
        while not self._shutdown.wait(timeout=self._flush_interval):
            try:
                self.flush()
            except Exception as e:
                logger.warning(f"Error in metrics flush loop: {e}")

        # Final flush on shutdown
        try:
            self.flush()
        except Exception as e:
            logger.warning(f"Error in final metrics flush: {e}")

    def _shutdown_flush_thread(self) -> None:
        """Shutdown the background flush thread."""
        self._shutdown.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=5.0)
            self._flush_thread = None

    def shutdown(self) -> None:
        """Shutdown the collector, flushing any remaining data."""
        self._shutdown_flush_thread()

    def is_enabled(self) -> bool:
        """Check if persistence is enabled."""
        return self._config.enabled

    def get_buffer_size(self) -> int:
        """Get the current number of buffered buckets."""
        with self._buffer_lock:
            return len(self._buffer)

    def get_sample_count(self) -> int:
        """Get the total number of samples in buffer."""
        with self._buffer_lock:
            return sum(b.sample_count for b in self._buffer.values())


# =============================================================================
# PrometheusExporter Integration
# =============================================================================


def create_persistence_aware_exporter(
    config: MetricsPersistenceConfig,
    collector: Optional[MetricsPersistenceCollector] = None,
) -> "PersistenceAwareExporter":
    """
    Create a PrometheusExporter wrapper that also persists metrics.

    Args:
        config: Metrics persistence configuration
        collector: Optional collector (creates one if not provided)

    Returns:
        PersistenceAwareExporter instance
    """
    from foundry_mcp.core.prometheus import get_prometheus_exporter

    if collector is None:
        collector = MetricsPersistenceCollector(config)

    return PersistenceAwareExporter(
        exporter=get_prometheus_exporter(),
        collector=collector,
    )


class PersistenceAwareExporter:
    """
    Wrapper around PrometheusExporter that also persists metrics.

    Intercepts metric recording calls and forwards them to both
    the underlying Prometheus exporter and the persistence collector.
    """

    def __init__(
        self,
        exporter: Any,  # PrometheusExporter
        collector: MetricsPersistenceCollector,
    ) -> None:
        """
        Initialize the wrapper.

        Args:
            exporter: The underlying PrometheusExporter
            collector: MetricsPersistenceCollector for persistence
        """
        self._exporter = exporter
        self._collector = collector

    @property
    def collector(self) -> MetricsPersistenceCollector:
        """Get the persistence collector."""
        return self._collector

    def record_tool_invocation(
        self,
        tool_name: str,
        *,
        success: bool = True,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Record a tool invocation."""
        # Forward to Prometheus
        self._exporter.record_tool_invocation(
            tool_name, success=success, duration_ms=duration_ms
        )

        # Persist
        status = "success" if success else "error"
        self._collector.record(
            "tool_invocations_total",
            1.0,
            metric_type="counter",
            labels={"tool": tool_name, "status": status},
        )

        if duration_ms is not None:
            self._collector.record(
                "tool_duration_seconds",
                duration_ms / 1000.0,
                metric_type="histogram",
                labels={"tool": tool_name},
            )

    def record_tool_start(self, tool_name: str) -> None:
        """Record tool execution start."""
        self._exporter.record_tool_start(tool_name)
        # Not persisted - gauge at point in time

    def record_tool_end(self, tool_name: str) -> None:
        """Record tool execution end."""
        self._exporter.record_tool_end(tool_name)
        # Not persisted - gauge at point in time

    def record_resource_access(
        self,
        resource_type: str,
        action: str = "read",
    ) -> None:
        """Record a resource access."""
        self._exporter.record_resource_access(resource_type, action)
        # Not in default persist list, but could be configured

    def record_error(
        self,
        tool_name: str,
        error_type: str = "unknown",
    ) -> None:
        """Record a tool error."""
        self._exporter.record_error(tool_name, error_type)

        self._collector.record(
            "tool_errors_total",
            1.0,
            metric_type="counter",
            labels={"tool": tool_name, "error_type": error_type},
        )

    def record_health_check(
        self,
        check_type: str,
        status: int,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Record a health check result."""
        self._exporter.record_health_check(check_type, status, duration_seconds)

        self._collector.record(
            "health_status",
            float(status),
            metric_type="gauge",
            labels={"check_type": check_type},
        )

    def record_dependency_health(
        self,
        dependency: str,
        healthy: bool,
    ) -> None:
        """Record dependency health status."""
        self._exporter.record_dependency_health(dependency, healthy)
        # Not persisted by default

    def record_health_check_batch(
        self,
        check_type: str,
        status: int,
        dependencies: dict[str, bool],
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Record a complete health check with all dependencies."""
        self._exporter.record_health_check_batch(
            check_type, status, dependencies, duration_seconds
        )

        self._collector.record(
            "health_status",
            float(status),
            metric_type="gauge",
            labels={"check_type": check_type},
        )

    # Pass-through methods
    def is_available(self) -> bool:
        """Check if prometheus_client is installed."""
        return self._exporter.is_available()

    def is_enabled(self) -> bool:
        """Check if Prometheus metrics are enabled and available."""
        return self._exporter.is_enabled()

    def start_server(
        self,
        port: Optional[int] = None,
        host: Optional[str] = None,
    ) -> bool:
        """Start the HTTP server for /metrics endpoint."""
        return self._exporter.start_server(port, host)

    def get_config(self) -> Any:
        """Get the current configuration."""
        return self._exporter.get_config()


# =============================================================================
# Global Collector Singleton
# =============================================================================

_collector: Optional[MetricsPersistenceCollector] = None
_collector_lock = threading.Lock()


def get_metrics_collector(
    config: Optional[MetricsPersistenceConfig] = None,
) -> MetricsPersistenceCollector:
    """
    Get the global metrics collector singleton.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        MetricsPersistenceCollector singleton instance
    """
    global _collector

    if _collector is None:
        with _collector_lock:
            if _collector is None:
                if config is None:
                    # Use default disabled config
                    config = MetricsPersistenceConfig(enabled=False)
                _collector = MetricsPersistenceCollector(config)

    return _collector


def reset_metrics_collector() -> None:
    """Reset the global collector (for testing)."""
    global _collector
    with _collector_lock:
        if _collector is not None:
            _collector.shutdown()
            _collector = None


def initialize_metrics_persistence(
    config: MetricsPersistenceConfig,
) -> Optional[MetricsPersistenceCollector]:
    """
    Initialize metrics persistence with the given configuration.

    Should be called during server startup to enable metrics persistence.

    Args:
        config: Metrics persistence configuration

    Returns:
        The initialized collector, or None if disabled
    """
    global _collector

    if not config.enabled:
        logger.debug("Metrics persistence is disabled")
        return None

    with _collector_lock:
        if _collector is not None:
            _collector.shutdown()

        _collector = MetricsPersistenceCollector(config)
        logger.info(
            f"Initialized metrics persistence: "
            f"storage={config.get_storage_path()}, "
            f"bucket_interval={config.bucket_interval_seconds}s, "
            f"flush_interval={config.flush_interval_seconds}s"
        )
        return _collector


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data structures
    "MetricBucket",
    "MetricDataPoint",
    # Collector
    "MetricsPersistenceCollector",
    "get_metrics_collector",
    "reset_metrics_collector",
    "initialize_metrics_persistence",
    # Prometheus integration
    "PersistenceAwareExporter",
    "create_persistence_aware_exporter",
]
