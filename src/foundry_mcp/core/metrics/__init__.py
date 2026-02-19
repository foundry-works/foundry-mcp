"""
Metrics data models, storage, persistence, and registry.

This package consolidates the metrics infrastructure:
- store: MetricDataPoint, MetricsStore ABC, FileMetricsStore
- persistence: MetricBucket, MetricsPersistenceCollector, PersistenceAwareExporter
- registry: MetricCategory, MetricDefinition, METRICS_CATALOG
"""

from foundry_mcp.core.metrics.persistence import (
    MetricBucket,
    MetricsPersistenceCollector,
    PersistenceAwareExporter,
    create_persistence_aware_exporter,
    get_metrics_collector,
    initialize_metrics_persistence,
    reset_metrics_collector,
)
from foundry_mcp.core.metrics.registry import (
    METRICS_CATALOG,
    MetricCategory,
    MetricDefinition,
    MetricType,
    export_catalog_as_markdown,
    get_metric,
    get_metrics_by_category,
    get_metrics_catalog,
)
from foundry_mcp.core.metrics.store import (
    FileMetricsStore,
    MetricDataPoint,
    MetricsStore,
    get_metrics_store,
    reset_metrics_store,
)

__all__ = [
    # store
    "FileMetricsStore",
    "MetricDataPoint",
    "MetricsStore",
    "get_metrics_store",
    "reset_metrics_store",
    # persistence
    "MetricBucket",
    "MetricsPersistenceCollector",
    "PersistenceAwareExporter",
    "create_persistence_aware_exporter",
    "get_metrics_collector",
    "initialize_metrics_persistence",
    "reset_metrics_collector",
    # registry
    "METRICS_CATALOG",
    "MetricCategory",
    "MetricDefinition",
    "MetricType",
    "export_catalog_as_markdown",
    "get_metric",
    "get_metrics_by_category",
    "get_metrics_catalog",
]
