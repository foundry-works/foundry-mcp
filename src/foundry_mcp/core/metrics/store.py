"""
Metrics storage backends for the metrics persistence infrastructure.

Provides abstract base class and concrete implementations for persisting
metric data points to enable time-series analysis across server restarts.
"""

from __future__ import annotations

import fcntl
import json
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricDataPoint:
    """
    A single metric data point or aggregated bucket.

    Attributes:
        metric_name: Name of the metric (e.g., "tool_invocations_total")
        timestamp: ISO 8601 timestamp when recorded
        value: Current value or delta for the bucket
        metric_type: Type of metric (counter, gauge, histogram)
        labels: Label key-value pairs
        bucket_start: Aggregation bucket start time (ISO 8601)
        bucket_end: Aggregation bucket end time (ISO 8601)
        sample_count: Number of samples aggregated in this bucket
    """

    metric_name: str
    timestamp: str
    value: float
    metric_type: str = "counter"
    labels: dict[str, str] = field(default_factory=dict)
    bucket_start: str = ""
    bucket_end: str = ""
    sample_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetricDataPoint":
        """Create from dictionary."""
        return cls(
            metric_name=data.get("metric_name", ""),
            timestamp=data.get("timestamp", ""),
            value=float(data.get("value", 0.0)),
            metric_type=data.get("metric_type", "counter"),
            labels=data.get("labels", {}),
            bucket_start=data.get("bucket_start", ""),
            bucket_end=data.get("bucket_end", ""),
            sample_count=int(data.get("sample_count", 1)),
        )


class MetricsStore(ABC):
    """Abstract base class for metrics storage backends."""

    @abstractmethod
    def append(self, data_point: MetricDataPoint) -> None:
        """
        Append a metric data point to storage.

        Args:
            data_point: The metric data point to store
        """
        pass

    @abstractmethod
    def append_batch(self, data_points: list[MetricDataPoint]) -> None:
        """
        Append multiple metric data points atomically.

        Args:
            data_points: List of metric data points to store
        """
        pass

    @abstractmethod
    def query(
        self,
        *,
        metric_name: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MetricDataPoint]:
        """
        Query metric data points with filtering.

        Args:
            metric_name: Filter by metric name
            labels: Filter by label key-value pairs
            since: ISO 8601 timestamp - include records after this time
            until: ISO 8601 timestamp - include records before this time
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of matching MetricDataPoints
        """
        pass

    @abstractmethod
    def list_metrics(self) -> list[dict[str, Any]]:
        """
        List all persisted metrics with metadata.

        Returns:
            List of metric metadata objects with names and counts
        """
        pass

    @abstractmethod
    def get_summary(
        self,
        metric_name: str,
        *,
        labels: Optional[dict[str, str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Get aggregated statistics for a metric.

        Args:
            metric_name: Name of the metric
            labels: Filter by label key-value pairs
            since: ISO 8601 timestamp - include records after this time
            until: ISO 8601 timestamp - include records before this time

        Returns:
            Dictionary with min, max, avg, sum, count statistics
        """
        pass

    @abstractmethod
    def cleanup(self, retention_days: int, max_records: int) -> int:
        """
        Clean up old records based on retention policy.

        Args:
            retention_days: Delete records older than this many days
            max_records: Maximum number of records to keep

        Returns:
            Number of records deleted
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Get total count of metric data points.

        Returns:
            Total number of stored metric data points
        """
        pass


class FileMetricsStore(MetricsStore):
    """
    JSONL-based metrics storage implementation.

    Stores metrics in append-only JSONL format with separate index file
    for efficient querying. Thread-safe with file locking for concurrent access.

    Directory structure:
        ~/.foundry-mcp/metrics/
            metrics.jsonl    - Append-only metrics log
            index.json       - Metric name -> metadata mapping
    """

    def __init__(self, storage_path: str | Path):
        """
        Initialize the file-based metrics store.

        Args:
            storage_path: Directory path for metrics storage
        """
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.storage_path / "metrics.jsonl"
        self.index_file = self.storage_path / "index.json"

        self._lock = threading.Lock()
        self._index: dict[str, dict[str, Any]] = {}  # metric_name -> metadata
        self._record_count = 0

        # Load index on initialization
        self._load_index()

    def _load_index(self) -> None:
        """Load the index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    data = json.load(f)
                    self._index = data.get("metrics", {})
                    self._record_count = data.get("record_count", 0)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load metrics index, rebuilding: {e}")
                self._rebuild_index()
        else:
            # First run or index deleted - rebuild from metrics file
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild index from the metrics JSONL file."""
        self._index = {}
        self._record_count = 0

        if not self.metrics_file.exists():
            self._save_index()
            return

        try:
            with open(self.metrics_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record_dict = json.loads(line)
                        metric_name = record_dict.get("metric_name", "")
                        timestamp = record_dict.get("timestamp", "")
                        metric_type = record_dict.get("metric_type", "counter")

                        if metric_name:
                            if metric_name not in self._index:
                                self._index[metric_name] = {
                                    "count": 0,
                                    "first_seen": timestamp,
                                    "last_seen": timestamp,
                                    "metric_type": metric_type,
                                    "label_keys": set(),
                                }

                            self._index[metric_name]["count"] += 1
                            self._index[metric_name]["last_seen"] = timestamp

                            # Track label keys
                            labels = record_dict.get("labels", {})
                            if isinstance(labels, dict):
                                self._index[metric_name]["label_keys"].update(labels.keys())

                        self._record_count += 1

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in metrics file at line {self._record_count}")

        except OSError as e:
            logger.error(f"Failed to rebuild metrics index: {e}")

        # Convert label_keys sets to lists for JSON serialization
        for metric_data in self._index.values():
            if isinstance(metric_data.get("label_keys"), set):
                metric_data["label_keys"] = list(metric_data["label_keys"])

        self._save_index()
        logger.info(f"Rebuilt metrics index: {len(self._index)} metrics, {self._record_count} records")

    def _save_index(self) -> None:
        """Save the index to disk."""
        try:
            # Convert label_keys to lists for JSON serialization
            index_copy = {}
            for name, data in self._index.items():
                data_copy = dict(data)
                if isinstance(data_copy.get("label_keys"), set):
                    data_copy["label_keys"] = list(data_copy["label_keys"])
                index_copy[name] = data_copy

            data = {
                "metrics": index_copy,
                "record_count": self._record_count,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            # Atomic write via temp file
            temp_file = self.index_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            temp_file.rename(self.index_file)
        except OSError as e:
            logger.error(f"Failed to save metrics index: {e}")

    def append(self, data_point: MetricDataPoint) -> None:
        """Append a metric data point to storage."""
        with self._lock:
            record_dict = data_point.to_dict()

            # Append to JSONL file with file locking
            try:
                with open(self.metrics_file, "a") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        f.write(json.dumps(record_dict, default=str) + "\n")
                        f.flush()
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            except OSError as e:
                logger.error(f"Failed to append metric data point: {e}")
                return

            # Update index
            metric_name = data_point.metric_name
            if metric_name not in self._index:
                self._index[metric_name] = {
                    "count": 0,
                    "first_seen": data_point.timestamp,
                    "last_seen": data_point.timestamp,
                    "metric_type": data_point.metric_type,
                    "label_keys": set(),
                }

            self._index[metric_name]["count"] += 1
            self._index[metric_name]["last_seen"] = data_point.timestamp

            # Track label keys
            if isinstance(self._index[metric_name].get("label_keys"), list):
                self._index[metric_name]["label_keys"] = set(self._index[metric_name]["label_keys"])
            self._index[metric_name]["label_keys"].update(data_point.labels.keys())

            self._record_count += 1
            self._save_index()

    def append_batch(self, data_points: list[MetricDataPoint]) -> None:
        """Append multiple metric data points atomically."""
        if not data_points:
            return

        with self._lock:
            # Append to JSONL file with file locking
            try:
                with open(self.metrics_file, "a") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        for data_point in data_points:
                            record_dict = data_point.to_dict()
                            f.write(json.dumps(record_dict, default=str) + "\n")
                        f.flush()
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            except OSError as e:
                logger.error(f"Failed to append metric batch: {e}")
                return

            # Update index
            for data_point in data_points:
                metric_name = data_point.metric_name
                if metric_name not in self._index:
                    self._index[metric_name] = {
                        "count": 0,
                        "first_seen": data_point.timestamp,
                        "last_seen": data_point.timestamp,
                        "metric_type": data_point.metric_type,
                        "label_keys": set(),
                    }

                self._index[metric_name]["count"] += 1
                self._index[metric_name]["last_seen"] = data_point.timestamp

                if isinstance(self._index[metric_name].get("label_keys"), list):
                    self._index[metric_name]["label_keys"] = set(self._index[metric_name]["label_keys"])
                self._index[metric_name]["label_keys"].update(data_point.labels.keys())

                self._record_count += 1

            self._save_index()

    def query(
        self,
        *,
        metric_name: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MetricDataPoint]:
        """Query metric data points with filtering."""
        results: list[MetricDataPoint] = []
        skipped = 0

        # Parse time filters
        since_dt = datetime.fromisoformat(since.replace("Z", "+00:00")) if since else None
        until_dt = datetime.fromisoformat(until.replace("Z", "+00:00")) if until else None

        with self._lock:
            if not self.metrics_file.exists():
                return []

            try:
                with open(self.metrics_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            record_dict = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Apply metric name filter
                        if metric_name and record_dict.get("metric_name") != metric_name:
                            continue

                        # Apply label filters
                        if labels:
                            record_labels = record_dict.get("labels", {})
                            if not all(
                                record_labels.get(k) == v
                                for k, v in labels.items()
                            ):
                                continue

                        # Time filters
                        if since_dt or until_dt:
                            try:
                                ts = record_dict.get("timestamp", "")
                                record_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                                if since_dt and record_dt < since_dt:
                                    continue
                                if until_dt and record_dt > until_dt:
                                    continue
                            except (ValueError, TypeError):
                                continue

                        # Apply offset
                        if skipped < offset:
                            skipped += 1
                            continue

                        # Check limit
                        if len(results) >= limit:
                            break

                        results.append(MetricDataPoint.from_dict(record_dict))

            except OSError as e:
                logger.error(f"Failed to query metrics: {e}")

        return results

    def list_metrics(self) -> list[dict[str, Any]]:
        """List all persisted metrics with metadata."""
        with self._lock:
            metrics_list = []
            for name, data in self._index.items():
                label_keys = data.get("label_keys", [])
                if isinstance(label_keys, set):
                    label_keys = list(label_keys)

                metrics_list.append({
                    "metric_name": name,
                    "count": data.get("count", 0),
                    "first_seen": data.get("first_seen"),
                    "last_seen": data.get("last_seen"),
                    "metric_type": data.get("metric_type"),
                    "label_keys": label_keys,
                })

            # Sort by count descending
            metrics_list.sort(key=lambda x: x["count"], reverse=True)
            return metrics_list

    def get_summary(
        self,
        metric_name: str,
        *,
        labels: Optional[dict[str, str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get aggregated statistics for a metric."""
        # Query all matching data points
        data_points = self.query(
            metric_name=metric_name,
            labels=labels,
            since=since,
            until=until,
            limit=100000,  # Get all matching
            offset=0,
        )

        if not data_points:
            return {
                "metric_name": metric_name,
                "count": 0,
                "min": None,
                "max": None,
                "avg": None,
                "sum": None,
                "sample_count": 0,
            }

        values = [dp.value for dp in data_points]
        total_samples = sum(dp.sample_count for dp in data_points)

        return {
            "metric_name": metric_name,
            "count": len(data_points),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values) if values else None,
            "sum": sum(values),
            "sample_count": total_samples,
            "first_timestamp": data_points[0].timestamp if data_points else None,
            "last_timestamp": data_points[-1].timestamp if data_points else None,
        }

    def cleanup(self, retention_days: int, max_records: int) -> int:
        """Clean up old records based on retention policy."""
        with self._lock:
            if not self.metrics_file.exists():
                return 0

            cutoff_dt = datetime.now(timezone.utc) - timedelta(days=retention_days)
            kept_records: list[str] = []
            deleted_count = 0

            try:
                # Read all records
                with open(self.metrics_file, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record_dict = json.loads(line)
                        ts = record_dict.get("timestamp", "")
                        record_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))

                        # Keep if within retention period
                        if record_dt >= cutoff_dt:
                            kept_records.append(line)
                        else:
                            deleted_count += 1

                    except (json.JSONDecodeError, ValueError):
                        # Keep malformed records to avoid data loss
                        kept_records.append(line)

                # Enforce max_records limit (keep most recent)
                if len(kept_records) > max_records:
                    deleted_count += len(kept_records) - max_records
                    kept_records = kept_records[-max_records:]

                # Write back
                temp_file = self.metrics_file.with_suffix(".tmp")
                with open(temp_file, "w") as f:
                    for line in kept_records:
                        f.write(line + "\n")
                temp_file.rename(self.metrics_file)

                # Rebuild index after cleanup
                self._rebuild_index()

                logger.info(f"Cleaned up {deleted_count} metric records")
                return deleted_count

            except OSError as e:
                logger.error(f"Failed to cleanup metrics: {e}")
                return 0

    def count(self) -> int:
        """Get total count of metric data points."""
        with self._lock:
            return self._record_count

    def get_total_count(self, metric_name: Optional[str] = None) -> int:
        """Get total count for a specific metric or all metrics (single source of truth).

        Args:
            metric_name: If provided, returns count for that metric only.
                        If None, returns total count across all metrics.

        Returns:
            Total count of metric records
        """
        with self._lock:
            if metric_name is not None:
                return self._index.get(metric_name, {}).get("count", 0)
            return sum(m.get("count", 0) for m in self._index.values())


# Global store instance
_metrics_store: Optional[MetricsStore] = None
_store_lock = threading.Lock()


def get_metrics_store(storage_path: Optional[str | Path] = None) -> MetricsStore:
    """
    Get the global metrics store instance.

    Args:
        storage_path: Optional path to initialize the store. If not provided
                     on first call, uses default path.

    Returns:
        The MetricsStore instance
    """
    global _metrics_store

    with _store_lock:
        if _metrics_store is None:
            if storage_path is None:
                # Default path
                storage_path = Path.home() / ".foundry-mcp" / "metrics"
            _metrics_store = FileMetricsStore(storage_path)

        return _metrics_store


def reset_metrics_store() -> None:
    """Reset the global metrics store (for testing)."""
    global _metrics_store
    with _store_lock:
        _metrics_store = None
