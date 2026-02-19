"""Metrics collection for observability.

Provides structured metric emission to loggers and Prometheus exporters.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Union

from foundry_mcp.core.observability.manager import get_observability_manager

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be emitted."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Structured metric data."""

    name: str
    value: Union[int, float]
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "labels": self.labels,
            "timestamp": self.timestamp,
        }


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
        self._logger.info(f"METRIC: {self.prefix}.{metric.name}", extra={"metric": metric.to_dict()})

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

    def counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
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

    def timer(self, name: str, duration_ms: float, labels: Optional[Dict[str, str]] = None) -> None:
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
