"""Unit tests for metrics persistence infrastructure."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from foundry_mcp.core.metrics_store import (
    MetricDataPoint,
    FileMetricsStore,
    get_metrics_store,
    reset_metrics_store,
)
from foundry_mcp.core.metrics_persistence import (
    MetricBucket,
    MetricsPersistenceCollector,
    get_metrics_collector,
    reset_metrics_collector,
)
from foundry_mcp.config import MetricsPersistenceConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for metrics storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def metrics_store(temp_storage_dir):
    """Create a FileMetricsStore with temporary storage."""
    return FileMetricsStore(temp_storage_dir)


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global singletons before and after each test."""
    reset_metrics_store()
    reset_metrics_collector()
    yield
    reset_metrics_store()
    reset_metrics_collector()


# =============================================================================
# MetricDataPoint Tests
# =============================================================================


class TestMetricDataPoint:
    """Tests for MetricDataPoint dataclass."""

    def test_create_data_point(self):
        """Test creating a metric data point with required fields."""
        dp = MetricDataPoint(
            metric_name="tool_invocations_total",
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=1.0,
        )

        assert dp.metric_name == "tool_invocations_total"
        assert dp.value == 1.0
        assert dp.metric_type == "counter"  # Default
        assert dp.labels == {}
        assert dp.sample_count == 1

    def test_create_data_point_with_labels(self):
        """Test creating a data point with labels."""
        dp = MetricDataPoint(
            metric_name="tool_duration_seconds",
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=0.5,
            metric_type="histogram",
            labels={"tool": "spec-list", "status": "success"},
        )

        assert dp.labels == {"tool": "spec-list", "status": "success"}
        assert dp.metric_type == "histogram"

    def test_data_point_to_dict(self):
        """Test converting data point to dictionary."""
        dp = MetricDataPoint(
            metric_name="test_metric",
            timestamp="2025-01-01T00:00:00+00:00",
            value=42.0,
            metric_type="gauge",
            labels={"env": "test"},
            bucket_start="2025-01-01T00:00:00+00:00",
            bucket_end="2025-01-01T00:01:00+00:00",
            sample_count=5,
        )

        data = dp.to_dict()
        assert data["metric_name"] == "test_metric"
        assert data["value"] == 42.0
        assert data["metric_type"] == "gauge"
        assert data["labels"] == {"env": "test"}
        assert data["sample_count"] == 5

    def test_data_point_from_dict(self):
        """Test creating data point from dictionary."""
        data = {
            "metric_name": "from_dict_metric",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "value": 123.45,
            "metric_type": "counter",
            "labels": {"key": "value"},
            "sample_count": 10,
        }

        dp = MetricDataPoint.from_dict(data)
        assert dp.metric_name == "from_dict_metric"
        assert dp.value == 123.45
        assert dp.sample_count == 10

    def test_data_point_from_dict_defaults(self):
        """Test from_dict handles missing optional fields."""
        data = {
            "metric_name": "minimal",
            "timestamp": "2025-01-01T00:00:00+00:00",
        }

        dp = MetricDataPoint.from_dict(data)
        assert dp.metric_name == "minimal"
        assert dp.value == 0.0
        assert dp.metric_type == "counter"
        assert dp.labels == {}


# =============================================================================
# FileMetricsStore Tests
# =============================================================================


class TestFileMetricsStore:
    """Tests for FileMetricsStore implementation."""

    def test_init_creates_directory(self, temp_storage_dir):
        """Test that store creates storage directory on init."""
        storage_path = temp_storage_dir / "nested" / "metrics"
        FileMetricsStore(storage_path)
        assert storage_path.exists()

    def test_append_single_data_point(self, metrics_store):
        """Test appending a single data point."""
        dp = MetricDataPoint(
            metric_name="test_counter",
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=1.0,
        )

        metrics_store.append(dp)
        assert metrics_store.count() == 1

    def test_append_batch(self, metrics_store):
        """Test appending multiple data points in a batch."""
        data_points = [
            MetricDataPoint(
                metric_name="batch_metric",
                timestamp=datetime.now(timezone.utc).isoformat(),
                value=float(i),
            )
            for i in range(5)
        ]

        metrics_store.append_batch(data_points)
        assert metrics_store.count() == 5

    def test_append_batch_empty(self, metrics_store):
        """Test appending empty batch does nothing."""
        metrics_store.append_batch([])
        assert metrics_store.count() == 0

    def test_query_all(self, metrics_store):
        """Test querying all data points."""
        for i in range(3):
            dp = MetricDataPoint(
                metric_name="query_test",
                timestamp=datetime.now(timezone.utc).isoformat(),
                value=float(i),
            )
            metrics_store.append(dp)

        results = metrics_store.query()
        assert len(results) == 3

    def test_query_by_metric_name(self, metrics_store):
        """Test querying by metric name."""
        metrics_store.append(MetricDataPoint(
            metric_name="metric_a",
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=1.0,
        ))
        metrics_store.append(MetricDataPoint(
            metric_name="metric_b",
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=2.0,
        ))
        metrics_store.append(MetricDataPoint(
            metric_name="metric_a",
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=3.0,
        ))

        results = metrics_store.query(metric_name="metric_a")
        assert len(results) == 2
        assert all(r.metric_name == "metric_a" for r in results)

    def test_query_by_labels(self, metrics_store):
        """Test querying by label filters."""
        metrics_store.append(MetricDataPoint(
            metric_name="labeled_metric",
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=1.0,
            labels={"env": "prod", "region": "us-east"},
        ))
        metrics_store.append(MetricDataPoint(
            metric_name="labeled_metric",
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=2.0,
            labels={"env": "staging", "region": "us-east"},
        ))
        metrics_store.append(MetricDataPoint(
            metric_name="labeled_metric",
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=3.0,
            labels={"env": "prod", "region": "eu-west"},
        ))

        # Query by single label
        results = metrics_store.query(labels={"env": "prod"})
        assert len(results) == 2

        # Query by multiple labels
        results = metrics_store.query(labels={"env": "prod", "region": "us-east"})
        assert len(results) == 1
        assert results[0].value == 1.0

    def test_query_by_time_range(self, metrics_store):
        """Test querying by time range."""
        now = datetime.now(timezone.utc)
        old_time = (now - timedelta(hours=2)).isoformat()
        recent_time = (now - timedelta(minutes=30)).isoformat()
        current_time = now.isoformat()

        metrics_store.append(MetricDataPoint(
            metric_name="time_metric",
            timestamp=old_time,
            value=1.0,
        ))
        metrics_store.append(MetricDataPoint(
            metric_name="time_metric",
            timestamp=recent_time,
            value=2.0,
        ))
        metrics_store.append(MetricDataPoint(
            metric_name="time_metric",
            timestamp=current_time,
            value=3.0,
        ))

        # Query since 1 hour ago
        since = (now - timedelta(hours=1)).isoformat()
        results = metrics_store.query(since=since)
        assert len(results) == 2

        # Query until 1 hour ago
        until = (now - timedelta(hours=1)).isoformat()
        results = metrics_store.query(until=until)
        assert len(results) == 1
        assert results[0].value == 1.0

    def test_query_with_pagination(self, metrics_store):
        """Test query pagination with limit and offset."""
        for i in range(10):
            metrics_store.append(MetricDataPoint(
                metric_name="paginated_metric",
                timestamp=datetime.now(timezone.utc).isoformat(),
                value=float(i),
            ))

        # Get first 3
        results = metrics_store.query(limit=3)
        assert len(results) == 3

        # Get next 3 with offset
        results = metrics_store.query(limit=3, offset=3)
        assert len(results) == 3

        # Get beyond available
        results = metrics_store.query(limit=5, offset=8)
        assert len(results) == 2

    def test_list_metrics(self, metrics_store):
        """Test listing all metrics with metadata."""
        metrics_store.append(MetricDataPoint(
            metric_name="metric_one",
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=1.0,
            metric_type="counter",
            labels={"tool": "test"},
        ))
        metrics_store.append(MetricDataPoint(
            metric_name="metric_two",
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=2.0,
            metric_type="gauge",
        ))
        metrics_store.append(MetricDataPoint(
            metric_name="metric_one",
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=3.0,
            labels={"tool": "other"},
        ))

        metrics = metrics_store.list_metrics()
        assert len(metrics) == 2

        # Should be sorted by count descending
        assert metrics[0]["metric_name"] == "metric_one"
        assert metrics[0]["count"] == 2
        assert metrics[1]["metric_name"] == "metric_two"
        assert metrics[1]["count"] == 1

        # Check label keys are tracked
        assert "tool" in metrics[0]["label_keys"]

    def test_get_summary(self, metrics_store):
        """Test getting aggregated statistics for a metric."""
        for i in range(1, 6):  # Values 1, 2, 3, 4, 5
            metrics_store.append(MetricDataPoint(
                metric_name="summary_metric",
                timestamp=datetime.now(timezone.utc).isoformat(),
                value=float(i),
                sample_count=2,  # Each represents 2 samples
            ))

        summary = metrics_store.get_summary("summary_metric")
        assert summary["metric_name"] == "summary_metric"
        assert summary["count"] == 5
        assert summary["min"] == 1.0
        assert summary["max"] == 5.0
        assert summary["avg"] == 3.0
        assert summary["sum"] == 15.0
        assert summary["sample_count"] == 10

    def test_get_summary_empty(self, metrics_store):
        """Test summary for non-existent metric."""
        summary = metrics_store.get_summary("nonexistent")
        assert summary["count"] == 0
        assert summary["min"] is None
        assert summary["max"] is None

    def test_get_summary_with_filters(self, metrics_store):
        """Test summary with label and time filters."""
        now = datetime.now(timezone.utc)

        metrics_store.append(MetricDataPoint(
            metric_name="filtered_metric",
            timestamp=(now - timedelta(hours=2)).isoformat(),
            value=10.0,
            labels={"env": "prod"},
        ))
        metrics_store.append(MetricDataPoint(
            metric_name="filtered_metric",
            timestamp=now.isoformat(),
            value=20.0,
            labels={"env": "prod"},
        ))
        metrics_store.append(MetricDataPoint(
            metric_name="filtered_metric",
            timestamp=now.isoformat(),
            value=100.0,
            labels={"env": "staging"},
        ))

        # Summary with label filter
        summary = metrics_store.get_summary(
            "filtered_metric",
            labels={"env": "prod"},
        )
        assert summary["count"] == 2
        assert summary["sum"] == 30.0

        # Summary with time filter
        since = (now - timedelta(hours=1)).isoformat()
        summary = metrics_store.get_summary(
            "filtered_metric",
            since=since,
        )
        assert summary["count"] == 2  # Only recent ones

    def test_cleanup_by_retention(self, metrics_store):
        """Test cleanup removes old records."""
        now = datetime.now(timezone.utc)
        old_time = (now - timedelta(days=10)).isoformat()
        recent_time = now.isoformat()

        # Add old and recent records
        metrics_store.append(MetricDataPoint(
            metric_name="cleanup_metric",
            timestamp=old_time,
            value=1.0,
        ))
        metrics_store.append(MetricDataPoint(
            metric_name="cleanup_metric",
            timestamp=recent_time,
            value=2.0,
        ))

        assert metrics_store.count() == 2

        # Cleanup with 7-day retention
        deleted = metrics_store.cleanup(retention_days=7, max_records=1000)
        assert deleted == 1
        assert metrics_store.count() == 1

    def test_cleanup_by_max_records(self, metrics_store):
        """Test cleanup enforces max records limit."""
        for i in range(10):
            metrics_store.append(MetricDataPoint(
                metric_name="max_test",
                timestamp=datetime.now(timezone.utc).isoformat(),
                value=float(i),
            ))

        assert metrics_store.count() == 10

        # Cleanup with max 5 records
        deleted = metrics_store.cleanup(retention_days=365, max_records=5)
        assert deleted == 5
        assert metrics_store.count() == 5

    def test_persistence_across_reloads(self, temp_storage_dir):
        """Test that data persists across store reloads."""
        # Create store and add data
        store1 = FileMetricsStore(temp_storage_dir)
        store1.append(MetricDataPoint(
            metric_name="persistent_metric",
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=42.0,
        ))
        assert store1.count() == 1

        # Create new store instance with same path
        store2 = FileMetricsStore(temp_storage_dir)
        assert store2.count() == 1

        # Query should return same data
        results = store2.query(metric_name="persistent_metric")
        assert len(results) == 1
        assert results[0].value == 42.0

    def test_index_rebuild_on_corruption(self, temp_storage_dir):
        """Test index is rebuilt if corrupted."""
        store = FileMetricsStore(temp_storage_dir)
        store.append(MetricDataPoint(
            metric_name="rebuild_test",
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=1.0,
        ))

        # Corrupt the index file
        index_file = temp_storage_dir / "index.json"
        with open(index_file, "w") as f:
            f.write("not valid json")

        # Create new store - should rebuild
        store2 = FileMetricsStore(temp_storage_dir)
        assert store2.count() == 1


# =============================================================================
# MetricsPersistenceConfig Tests
# =============================================================================


class TestMetricsPersistenceConfig:
    """Tests for MetricsPersistenceConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MetricsPersistenceConfig()
        assert config.enabled is False
        assert config.retention_days == 7
        assert config.max_records == 100000
        assert config.bucket_interval_seconds == 60
        assert config.flush_interval_seconds == 30
        assert "tool_invocations_total" in config.persist_metrics

    def test_from_toml_dict(self):
        """Test creating config from TOML dict."""
        data = {
            "enabled": True,
            "storage_path": "/custom/path",
            "retention_days": 14,
            "max_records": 50000,
            "bucket_interval_seconds": 120,
            "persist_metrics": ["metric_a", "metric_b"],
        }

        config = MetricsPersistenceConfig.from_toml_dict(data)
        assert config.enabled is True
        assert config.storage_path == "/custom/path"
        assert config.retention_days == 14
        assert config.max_records == 50000
        assert config.bucket_interval_seconds == 120
        assert config.persist_metrics == ["metric_a", "metric_b"]

    def test_from_toml_dict_string_persist_metrics(self):
        """Test handling comma-separated persist_metrics string."""
        data = {
            "persist_metrics": "metric_a, metric_b, metric_c",
        }

        config = MetricsPersistenceConfig.from_toml_dict(data)
        assert config.persist_metrics == ["metric_a", "metric_b", "metric_c"]

    def test_get_storage_path_default(self):
        """Test default storage path."""
        config = MetricsPersistenceConfig()
        path = config.get_storage_path()
        assert "foundry-mcp" in str(path)
        assert "metrics" in str(path)

    def test_get_storage_path_custom(self):
        """Test custom storage path."""
        config = MetricsPersistenceConfig(storage_path="/my/custom/path")
        path = config.get_storage_path()
        assert str(path) == "/my/custom/path"

    def test_should_persist_metric_whitelist(self):
        """Test metric persistence filtering with whitelist."""
        config = MetricsPersistenceConfig(
            persist_metrics=["allowed_metric_1", "allowed_metric_2"]
        )

        assert config.should_persist_metric("allowed_metric_1") is True
        assert config.should_persist_metric("allowed_metric_2") is True
        assert config.should_persist_metric("not_allowed") is False

    def test_should_persist_metric_empty_list(self):
        """Test that empty persist_metrics allows all metrics."""
        config = MetricsPersistenceConfig(persist_metrics=[])

        assert config.should_persist_metric("any_metric") is True
        assert config.should_persist_metric("another_metric") is True


# =============================================================================
# Global Store Tests
# =============================================================================


class TestGlobalMetricsStore:
    """Tests for global metrics store singleton."""

    def test_get_metrics_store_creates_singleton(self, temp_storage_dir):
        """Test that get_metrics_store creates singleton."""
        reset_metrics_store()

        store1 = get_metrics_store(temp_storage_dir)
        store2 = get_metrics_store()  # Should return same instance

        assert store1 is store2

    def test_reset_metrics_store(self, temp_storage_dir):
        """Test resetting the global store."""
        store1 = get_metrics_store(temp_storage_dir)
        reset_metrics_store()

        # After reset, should be None until get_metrics_store called
        # This tests the reset functionality
        store2 = get_metrics_store(temp_storage_dir)
        assert store1 is not store2


# =============================================================================
# MetricBucket Tests
# =============================================================================


class TestMetricBucket:
    """Tests for MetricBucket aggregation."""

    def test_counter_aggregation(self):
        """Test counter aggregation sums values."""
        bucket = MetricBucket(
            metric_name="test_counter",
            metric_type="counter",
            labels={},
            bucket_start=datetime.now(timezone.utc),
            bucket_end=datetime.now(timezone.utc),
        )
        bucket.add_sample(1.0)
        bucket.add_sample(2.0)
        bucket.add_sample(3.0)

        assert bucket.get_aggregated_value() == 6.0
        assert bucket.sample_count == 3

    def test_gauge_aggregation(self):
        """Test gauge aggregation uses last value."""
        bucket = MetricBucket(
            metric_name="test_gauge",
            metric_type="gauge",
            labels={},
            bucket_start=datetime.now(timezone.utc),
            bucket_end=datetime.now(timezone.utc),
        )
        bucket.add_sample(10.0)
        bucket.add_sample(20.0)
        bucket.add_sample(30.0)

        assert bucket.get_aggregated_value() == 30.0

    def test_histogram_aggregation(self):
        """Test histogram aggregation uses average."""
        bucket = MetricBucket(
            metric_name="test_histogram",
            metric_type="histogram",
            labels={},
            bucket_start=datetime.now(timezone.utc),
            bucket_end=datetime.now(timezone.utc),
        )
        bucket.add_sample(1.0)
        bucket.add_sample(2.0)
        bucket.add_sample(3.0)

        assert bucket.get_aggregated_value() == 2.0  # Average

    def test_empty_bucket(self):
        """Test empty bucket returns 0."""
        bucket = MetricBucket(
            metric_name="empty",
            metric_type="counter",
            labels={},
            bucket_start=datetime.now(timezone.utc),
            bucket_end=datetime.now(timezone.utc),
        )
        assert bucket.get_aggregated_value() == 0.0
        assert bucket.sample_count == 0

    def test_to_data_point(self):
        """Test converting bucket to data point."""
        now = datetime.now(timezone.utc)
        bucket = MetricBucket(
            metric_name="dp_test",
            metric_type="counter",
            labels={"env": "test"},
            bucket_start=now,
            bucket_end=now,
        )
        bucket.add_sample(5.0)
        bucket.add_sample(3.0)

        dp = bucket.to_data_point()
        assert dp.metric_name == "dp_test"
        assert dp.value == 8.0  # Sum
        assert dp.metric_type == "counter"
        assert dp.labels == {"env": "test"}
        assert dp.sample_count == 2


# =============================================================================
# MetricsPersistenceCollector Tests
# =============================================================================


class TestMetricsPersistenceCollector:
    """Tests for MetricsPersistenceCollector."""

    def test_record_single_metric(self, temp_storage_dir):
        """Test recording a single metric."""
        config = MetricsPersistenceConfig(
            enabled=True,
            storage_path=str(temp_storage_dir),
            bucket_interval_seconds=60,
            persist_metrics=["test_metric"],
        )
        store = FileMetricsStore(temp_storage_dir)
        collector = MetricsPersistenceCollector(config, store=store)

        collector.record("test_metric", 1.0, "counter", {"key": "value"})

        assert collector.get_buffer_size() == 1
        assert collector.get_sample_count() == 1

        collector.shutdown()

    def test_record_multiple_metrics(self, temp_storage_dir):
        """Test recording multiple different metrics."""
        config = MetricsPersistenceConfig(
            enabled=True,
            storage_path=str(temp_storage_dir),
            bucket_interval_seconds=60,
            persist_metrics=["metric_a", "metric_b"],
        )
        store = FileMetricsStore(temp_storage_dir)
        collector = MetricsPersistenceCollector(config, store=store)

        collector.record("metric_a", 1.0)
        collector.record("metric_b", 2.0)
        collector.record("metric_a", 3.0)  # Same metric, aggregates

        assert collector.get_buffer_size() == 2
        assert collector.get_sample_count() == 3

        collector.shutdown()

    def test_metric_filtering(self, temp_storage_dir):
        """Test that non-whitelisted metrics are not recorded."""
        config = MetricsPersistenceConfig(
            enabled=True,
            storage_path=str(temp_storage_dir),
            persist_metrics=["allowed_metric"],
        )
        store = FileMetricsStore(temp_storage_dir)
        collector = MetricsPersistenceCollector(config, store=store)

        collector.record("allowed_metric", 1.0)
        collector.record("not_allowed", 100.0)

        assert collector.get_buffer_size() == 1

        collector.shutdown()

    def test_flush_to_store(self, temp_storage_dir):
        """Test flushing buffer to store."""
        config = MetricsPersistenceConfig(
            enabled=True,
            storage_path=str(temp_storage_dir),
            persist_metrics=["flush_test"],
        )
        store = FileMetricsStore(temp_storage_dir)
        collector = MetricsPersistenceCollector(config, store=store)

        collector.record("flush_test", 1.0, "counter", {"env": "test"})
        collector.record("flush_test", 2.0, "counter", {"env": "test"})

        flushed = collector.flush()
        assert flushed == 1  # One bucket

        # Buffer should be empty
        assert collector.get_buffer_size() == 0

        # Store should have data
        results = store.query(metric_name="flush_test")
        assert len(results) == 1
        assert results[0].value == 3.0  # Sum of 1 + 2
        assert results[0].sample_count == 2

        collector.shutdown()

    def test_disabled_collector(self, temp_storage_dir):
        """Test disabled collector doesn't record."""
        config = MetricsPersistenceConfig(
            enabled=False,
            storage_path=str(temp_storage_dir),
        )
        store = FileMetricsStore(temp_storage_dir)
        collector = MetricsPersistenceCollector(config, store=store)

        collector.record("any_metric", 100.0)

        assert collector.get_buffer_size() == 0
        assert not collector.is_enabled()

        collector.shutdown()

    def test_labels_create_separate_buckets(self, temp_storage_dir):
        """Test that different labels create separate buckets."""
        config = MetricsPersistenceConfig(
            enabled=True,
            storage_path=str(temp_storage_dir),
            persist_metrics=["labeled_metric"],
        )
        store = FileMetricsStore(temp_storage_dir)
        collector = MetricsPersistenceCollector(config, store=store)

        collector.record("labeled_metric", 1.0, "counter", {"env": "prod"})
        collector.record("labeled_metric", 2.0, "counter", {"env": "staging"})
        collector.record("labeled_metric", 3.0, "counter", {"env": "prod"})

        # Should have 2 buckets (prod and staging)
        assert collector.get_buffer_size() == 2

        collector.flush()

        results = store.query(metric_name="labeled_metric")
        assert len(results) == 2

        collector.shutdown()


# =============================================================================
# Global Collector Tests
# =============================================================================


class TestGlobalMetricsCollector:
    """Tests for global metrics collector singleton."""

    def test_get_metrics_collector_creates_singleton(self, temp_storage_dir):
        """Test that get_metrics_collector creates singleton."""
        reset_metrics_collector()

        config = MetricsPersistenceConfig(
            enabled=True,
            storage_path=str(temp_storage_dir),
        )
        collector1 = get_metrics_collector(config)
        collector2 = get_metrics_collector()

        assert collector1 is collector2

        collector1.shutdown()

    def test_reset_metrics_collector(self, temp_storage_dir):
        """Test resetting the global collector."""
        config = MetricsPersistenceConfig(
            enabled=True,
            storage_path=str(temp_storage_dir),
        )
        collector1 = get_metrics_collector(config)
        reset_metrics_collector()

        collector2 = get_metrics_collector(config)
        assert collector1 is not collector2

        collector2.shutdown()
