"""Data access layer for dashboard.

Wraps MetricsStore, ErrorStore, and other data sources with:
- Streamlit caching (@st.cache_data)
- pandas DataFrame conversion for easy use with st.dataframe
- Graceful handling when stores are disabled
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import streamlit as st

logger = logging.getLogger(__name__)

# Try importing pandas - it's an optional dependency
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


def _get_config():
    """Get foundry-mcp configuration."""
    try:
        from foundry_mcp.config import get_config

        return get_config()
    except Exception as e:
        logger.warning("Could not load config: %s", e)
        return None


def _get_error_store():
    """Get error store instance if enabled."""
    config = _get_config()
    if config is None or not config.error_collection.enabled:
        return None

    try:
        from foundry_mcp.core.error_store import FileErrorStore

        storage_path = config.error_collection.get_storage_path()
        return FileErrorStore(storage_path)
    except Exception as e:
        logger.warning("Could not initialize error store: %s", e)
        return None


def _get_metrics_store():
    """Get metrics store instance if enabled."""
    config = _get_config()
    if config is None or not config.metrics_persistence.enabled:
        return None

    try:
        from foundry_mcp.core.metrics_store import FileMetricsStore

        storage_path = config.metrics_persistence.get_storage_path()
        return FileMetricsStore(storage_path)
    except Exception as e:
        logger.warning("Could not initialize metrics store: %s", e)
        return None


# =============================================================================
# Error Data Functions
# =============================================================================


@st.cache_data(ttl=10)
def get_errors(
    tool_name: Optional[str] = None,
    error_code: Optional[str] = None,
    since_hours: int = 24,
    limit: int = 100,
) -> "pd.DataFrame":
    """Get errors as a DataFrame.

    Args:
        tool_name: Filter by tool name
        error_code: Filter by error code
        since_hours: Hours to look back
        limit: Maximum records to return

    Returns:
        DataFrame with error records, or empty DataFrame if disabled
    """
    if not PANDAS_AVAILABLE:
        return None

    store = _get_error_store()
    if store is None:
        return pd.DataFrame()

    since = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()

    try:
        records = store.query(
            tool_name=tool_name,
            error_code=error_code,
            since=since,
            limit=limit,
        )

        if not records:
            return pd.DataFrame()

        # Convert to list of dicts
        data = []
        for r in records:
            data.append(
                {
                    "id": r.error_id,
                    "timestamp": r.timestamp,
                    "tool_name": r.tool_name,
                    "error_code": r.error_code,
                    "message": r.message,
                    "error_type": r.error_type,
                    "fingerprint": r.fingerprint,
                }
            )

        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    except Exception as e:
        logger.exception("Error querying errors: %s", e)
        return pd.DataFrame()


@st.cache_data(ttl=10)
def get_error_stats() -> dict[str, Any]:
    """Get error statistics."""
    store = _get_error_store()
    if store is None:
        return {"enabled": False, "total": 0}

    try:
        stats = store.get_stats()
        stats["enabled"] = True
        return stats
    except Exception as e:
        logger.exception("Error getting error stats: %s", e)
        return {"enabled": True, "error": str(e)}


@st.cache_data(ttl=30)
def get_error_patterns(min_count: int = 3) -> list[dict]:
    """Get recurring error patterns."""
    store = _get_error_store()
    if store is None:
        return []

    try:
        return store.get_patterns(min_count=min_count)
    except Exception as e:
        logger.exception("Error getting patterns: %s", e)
        return []


def get_error_by_id(error_id: str) -> Optional[dict]:
    """Get a single error by ID (not cached for freshness)."""
    store = _get_error_store()
    if store is None:
        return None

    try:
        record = store.get(error_id)
        if record is None:
            return None

        return {
            "id": record.error_id,
            "timestamp": record.timestamp,
            "tool_name": record.tool_name,
            "error_code": record.error_code,
            "message": record.message,
            "error_type": record.error_type,
            "fingerprint": record.fingerprint,
            "stack_trace": record.stack_trace,
            "context": record.context,
        }
    except Exception as e:
        logger.exception("Error getting error by ID: %s", e)
        return None


# =============================================================================
# Metrics Data Functions
# =============================================================================


@st.cache_data(ttl=5)
def get_metrics_list() -> list[dict]:
    """Get list of available metrics."""
    store = _get_metrics_store()
    if store is None:
        return []

    try:
        return store.list_metrics()
    except Exception as e:
        logger.exception("Error listing metrics: %s", e)
        return []


@st.cache_data(ttl=5)
def get_metrics_timeseries(
    metric_name: str,
    since_hours: int = 24,
    limit: int = 1000,
) -> "pd.DataFrame":
    """Get time-series data for a metric.

    Args:
        metric_name: Name of the metric
        since_hours: Hours to look back
        limit: Maximum data points

    Returns:
        DataFrame with timestamp and value columns
    """
    if not PANDAS_AVAILABLE:
        return None

    store = _get_metrics_store()
    if store is None:
        return pd.DataFrame()

    since = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()

    try:
        data_points = store.query(
            metric_name=metric_name,
            since=since,
            limit=limit,
        )

        if not data_points:
            return pd.DataFrame()

        data = [
            {
                "timestamp": dp.timestamp,
                "value": dp.value,
                "labels": str(dp.labels) if dp.labels else "",
            }
            for dp in data_points
        ]

        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    except Exception as e:
        logger.exception("Error querying metrics: %s", e)
        return pd.DataFrame()


@st.cache_data(ttl=5)
def get_metrics_summary(metric_name: str, since_hours: int = 24) -> dict[str, Any]:
    """Get summary statistics for a metric."""
    store = _get_metrics_store()
    if store is None:
        return {"enabled": False}

    since = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()

    try:
        summary = store.get_summary(metric_name, since=since)
        summary["enabled"] = True
        return summary
    except Exception as e:
        logger.exception("Error getting summary: %s", e)
        return {"enabled": True, "error": str(e)}


# =============================================================================
# Health & Provider Data Functions
# =============================================================================


@st.cache_data(ttl=10)
def get_health_status() -> dict[str, Any]:
    """Get server health status."""
    try:
        from foundry_mcp.core.health import get_health_manager

        manager = get_health_manager()
        result = manager.check_health()

        return {
            "healthy": result.healthy,
            "status": result.status.value if hasattr(result.status, "value") else str(result.status),
            "checks": {
                name: {
                    "healthy": check.healthy,
                    "message": check.message,
                }
                for name, check in (result.checks or {}).items()
            },
        }
    except Exception as e:
        logger.exception("Error getting health: %s", e)
        return {"healthy": False, "error": str(e)}


@st.cache_data(ttl=30)
def get_providers() -> list[dict]:
    """Get list of AI providers with status."""
    try:
        from foundry_mcp.core.providers import describe_providers

        return describe_providers()
    except ImportError:
        # Providers module may not exist yet
        return []
    except Exception as e:
        logger.exception("Error getting providers: %s", e)
        return []


# =============================================================================
# Overview Summary Functions
# =============================================================================


@st.cache_data(ttl=5)
def get_overview_summary() -> dict[str, Any]:
    """Get aggregated overview metrics for dashboard."""
    summary = {
        "total_invocations": 0,
        "invocations_last_hour": 0,
        "active_tools": 0,
        "error_count": 0,
        "error_rate": 0.0,
        "avg_latency_ms": 0.0,
        "health_status": "unknown",
        "provider_count": 0,
    }

    # Get metrics summary
    metrics_list = get_metrics_list()
    for m in metrics_list:
        if m.get("metric_name") == "tool_invocations_total":
            summary["total_invocations"] = m.get("count", 0)

    # Get error stats
    error_stats = get_error_stats()
    if error_stats.get("enabled"):
        summary["error_count"] = error_stats.get("total", 0)

    # Get health
    health = get_health_status()
    summary["health_status"] = "healthy" if health.get("healthy") else "unhealthy"

    # Get providers
    providers = get_providers()
    summary["provider_count"] = len([p for p in providers if p.get("available")])

    return summary
