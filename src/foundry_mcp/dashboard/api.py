"""
REST API endpoints for the dashboard.

Provides JSON endpoints for querying errors, metrics, health, and provider status.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


def setup_api_routes(app: Any) -> None:
    """Setup API routes on the aiohttp application.

    Args:
        app: aiohttp Application instance
    """
    from aiohttp import web

    # Error endpoints
    app.router.add_get("/api/errors", handle_errors_list)
    app.router.add_get("/api/errors/stats", handle_errors_stats)
    app.router.add_get("/api/errors/patterns", handle_errors_patterns)
    app.router.add_get("/api/errors/{error_id}", handle_error_get)

    # Metrics endpoints
    app.router.add_get("/api/metrics", handle_metrics_list)
    app.router.add_get("/api/metrics/summary/{metric_name}", handle_metrics_summary)
    app.router.add_get("/api/metrics/timeseries/{metric_name}", handle_metrics_timeseries)

    # Health endpoint
    app.router.add_get("/api/health", handle_health)

    # Provider endpoints
    app.router.add_get("/api/providers", handle_providers_list)

    # Config endpoint (non-sensitive)
    app.router.add_get("/api/config", handle_config)

    # Overview summary endpoint (aggregated)
    app.router.add_get("/api/overview/summary", handle_overview_summary)


def _json_response(data: Any, status: int = 200) -> Any:
    """Create a JSON response.

    Args:
        data: Data to serialize
        status: HTTP status code

    Returns:
        aiohttp Response
    """
    from aiohttp import web

    return web.json_response(data, status=status)


def _error_response(message: str, status: int = 500) -> Any:
    """Create an error response.

    Args:
        message: Error message
        status: HTTP status code

    Returns:
        aiohttp Response
    """
    return _json_response({"error": message, "success": False}, status=status)


def _get_error_store() -> Optional[Any]:
    """Get the error store instance.

    Returns:
        FileErrorStore instance or None if not available
    """
    try:
        from foundry_mcp.core.error_store import get_error_store
        from foundry_mcp.config import get_config

        config = get_config()
        if not config.error_collection.enabled:
            return None

        return get_error_store(config.error_collection.get_storage_path())
    except Exception as e:
        logger.warning(f"Error getting error store: {e}")
        return None


def _get_metrics_store() -> Optional[Any]:
    """Get the metrics store instance.

    Returns:
        FileMetricsStore instance or None if not available
    """
    try:
        from foundry_mcp.core.metrics_store import get_metrics_store
        from foundry_mcp.config import get_config

        config = get_config()
        if not config.metrics_persistence.enabled:
            return None

        return get_metrics_store(config.metrics_persistence.get_storage_path())
    except Exception as e:
        logger.warning(f"Error getting metrics store: {e}")
        return None


# =============================================================================
# Error Handlers
# =============================================================================


async def handle_errors_list(request: Any) -> Any:
    """List errors with optional filtering.

    Query params:
        tool_name: Filter by tool name
        error_code: Filter by error code
        error_type: Filter by error type
        fingerprint: Filter by fingerprint
        since: ISO timestamp - records after this time
        until: ISO timestamp - records before this time
        limit: Max records (default: 50, max: 100)
        offset: Pagination offset
    """
    store = _get_error_store()
    if store is None:
        return _json_response({
            "errors": [],
            "total": 0,
            "message": "Error collection is disabled or unavailable",
        })

    # Parse query params
    params = request.query
    limit = min(int(params.get("limit", 50)), 100)
    offset = int(params.get("offset", 0))

    try:
        records = store.query(
            tool_name=params.get("tool_name"),
            error_code=params.get("error_code"),
            error_type=params.get("error_type"),
            fingerprint=params.get("fingerprint"),
            since=params.get("since"),
            until=params.get("until"),
            limit=limit,
            offset=offset,
        )

        return _json_response({
            "errors": [r.to_dict() for r in records],
            "total": store.count(),
            "limit": limit,
            "offset": offset,
            "has_more": len(records) == limit,
        })
    except Exception as e:
        logger.exception("Error querying errors")
        return _error_response(f"Failed to query errors: {e}")


async def handle_error_get(request: Any) -> Any:
    """Get a specific error by ID."""
    store = _get_error_store()
    if store is None:
        return _error_response("Error collection is disabled", status=503)

    error_id = request.match_info.get("error_id")
    if not error_id:
        return _error_response("error_id is required", status=400)

    try:
        record = store.get(error_id)
        if record is None:
            return _error_response(f"Error not found: {error_id}", status=404)

        return _json_response({"error": record.to_dict()})
    except Exception as e:
        logger.exception("Error getting error record")
        return _error_response(f"Failed to get error: {e}")


async def handle_errors_stats(request: Any) -> Any:
    """Get aggregated error statistics."""
    store = _get_error_store()
    if store is None:
        return _json_response({
            "total_errors": 0,
            "unique_patterns": 0,
            "by_tool": {},
            "by_error_code": {},
            "message": "Error collection is disabled or unavailable",
        })

    try:
        stats = store.get_stats()
        return _json_response(stats)
    except Exception as e:
        logger.exception("Error getting error stats")
        return _error_response(f"Failed to get error stats: {e}")


async def handle_errors_patterns(request: Any) -> Any:
    """Get recurring error patterns."""
    store = _get_error_store()
    if store is None:
        return _json_response({"patterns": []})

    min_count = int(request.query.get("min_count", 3))

    try:
        patterns = store.get_patterns(min_count=min_count)
        return _json_response({"patterns": patterns})
    except Exception as e:
        logger.exception("Error getting error patterns")
        return _error_response(f"Failed to get error patterns: {e}")


# =============================================================================
# Metrics Handlers
# =============================================================================


async def handle_metrics_list(request: Any) -> Any:
    """List all available metrics."""
    store = _get_metrics_store()
    if store is None:
        return _json_response({
            "metrics": [],
            "message": "Metrics persistence is disabled or unavailable",
        })

    try:
        metrics = store.list_metrics()
        return _json_response({
            "metrics": metrics,
            "total": len(metrics),
        })
    except Exception as e:
        logger.exception("Error listing metrics")
        return _error_response(f"Failed to list metrics: {e}")


async def handle_metrics_summary(request: Any) -> Any:
    """Get summary statistics for a metric."""
    store = _get_metrics_store()
    if store is None:
        return _error_response("Metrics persistence is disabled", status=503)

    metric_name = request.match_info.get("metric_name")
    if not metric_name:
        return _error_response("metric_name is required", status=400)

    # Parse labels from query param (JSON object)
    labels = None
    if "labels" in request.query:
        try:
            labels = json.loads(request.query["labels"])
        except json.JSONDecodeError:
            return _error_response("Invalid labels JSON", status=400)

    try:
        summary = store.get_summary(
            metric_name=metric_name,
            labels=labels,
            since=request.query.get("since"),
            until=request.query.get("until"),
        )
        return _json_response({"summary": summary})
    except Exception as e:
        logger.exception("Error getting metrics summary")
        return _error_response(f"Failed to get metrics summary: {e}")


async def handle_metrics_timeseries(request: Any) -> Any:
    """Get time-series data for a metric."""
    store = _get_metrics_store()
    if store is None:
        return _error_response("Metrics persistence is disabled", status=503)

    metric_name = request.match_info.get("metric_name")
    if not metric_name:
        return _error_response("metric_name is required", status=400)

    # Parse query params
    params = request.query
    limit = min(int(params.get("limit", 100)), 500)

    # Default to last hour if no time range specified
    until = params.get("until")
    since = params.get("since")
    if not since and not until:
        now = datetime.now(timezone.utc)
        since = (now - timedelta(hours=1)).isoformat()

    # Parse labels
    labels = None
    if "labels" in params:
        try:
            labels = json.loads(params["labels"])
        except json.JSONDecodeError:
            return _error_response("Invalid labels JSON", status=400)

    try:
        records = store.query(
            metric_name=metric_name,
            labels=labels,
            since=since,
            until=until,
            limit=limit,
        )

        # Convert to chart-friendly format
        datapoints = [
            {
                "timestamp": r.timestamp,
                "value": r.value,
                "labels": r.labels,
            }
            for r in records
        ]

        return _json_response({
            "metric_name": metric_name,
            "datapoints": datapoints,
            "count": len(datapoints),
        })
    except Exception as e:
        logger.exception("Error getting metrics timeseries")
        return _error_response(f"Failed to get metrics timeseries: {e}")


# =============================================================================
# Health Handler
# =============================================================================


async def handle_health(request: Any) -> Any:
    """Get current health status."""
    try:
        from foundry_mcp.core.health import get_health_manager

        manager = get_health_manager()
        result = manager.check_health()

        # Build dependencies dict from list
        deps = {}
        for dep in result.dependencies:
            deps[dep.name] = {
                "healthy": dep.healthy,
                "message": dep.message,
                "latency_ms": dep.latency_ms,
            }

        return _json_response({
            "status": result.status.value,
            "is_healthy": result.is_healthy,
            "dependencies": deps,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except ImportError:
        return _json_response({
            "status": "unknown",
            "is_healthy": True,
            "message": "Health manager not available",
        })
    except Exception as e:
        logger.exception("Error checking health")
        return _json_response({
            "status": "unhealthy",
            "is_healthy": False,
            "error": str(e),
        })


# =============================================================================
# Provider Handler
# =============================================================================


async def handle_providers_list(request: Any) -> Any:
    """List AI providers and their status."""
    try:
        from foundry_mcp.core.providers import (
            describe_providers,
        )

        # describe_providers() returns dicts with id, description, available, etc.
        all_providers = describe_providers()

        providers = []
        for prov in all_providers:
            providers.append({
                "id": prov.get("id", ""),
                "available": prov.get("available", False),
                "description": prov.get("description", ""),
                "tags": prov.get("tags", []),
            })

        return _json_response({
            "providers": providers,
            "total": len(providers),
        })
    except ImportError:
        return _json_response({
            "providers": [],
            "message": "Provider registry not available",
        })
    except Exception as e:
        logger.exception("Error listing providers")
        return _error_response(f"Failed to list providers: {e}")


# =============================================================================
# Config Handler
# =============================================================================


async def handle_config(request: Any) -> Any:
    """Get dashboard configuration (non-sensitive)."""
    config = request.app.get("config")
    if config is None:
        return _json_response({})

    return _json_response({
        "refresh_interval_ms": config.refresh_interval_ms,
        "host": config.host,
        "port": config.port,
    })


# =============================================================================
# Overview Summary Handler
# =============================================================================


async def handle_overview_summary(request: Any) -> Any:
    """Get aggregated overview metrics for dashboard summary cards.

    Returns a single response with all overview metrics to reduce API calls.
    """
    result = {
        "invocations": {"total": 0, "last_hour": 0},
        "active_tools": {"count": 0},
        "health": {"status": "unknown", "deps_ok": 0, "deps_total": 0},
        "latency": {"avg_ms": None},
        "errors": {"total": 0, "last_hour": 0, "failure_rate_pct": 0.0},
        "providers": {"available": 0, "total": 0, "names": []},
    }

    # Get invocation metrics
    metrics_store = _get_metrics_store()
    if metrics_store is not None:
        try:
            # Get total invocations
            summary = metrics_store.get_summary(metric_name="tool_invocations_total")
            if summary:
                result["invocations"]["total"] = int(summary.get("count", 0))

            # Get invocations in last hour
            now = datetime.now(timezone.utc)
            hour_ago = (now - timedelta(hours=1)).isoformat()
            recent_summary = metrics_store.get_summary(
                metric_name="tool_invocations_total",
                since=hour_ago,
            )
            if recent_summary:
                result["invocations"]["last_hour"] = int(recent_summary.get("count", 0))

            # Get active tools (unique tool names from recent metrics)
            records = metrics_store.query(
                metric_name="tool_invocations_total",
                since=hour_ago,
                limit=500,
            )
            unique_tools = set()
            for r in records:
                if r.labels and "tool" in r.labels:
                    unique_tools.add(r.labels["tool"])
            result["active_tools"]["count"] = len(unique_tools)

            # Get latency
            latency_summary = metrics_store.get_summary(metric_name="tool_duration_ms")
            if latency_summary and latency_summary.get("avg") is not None:
                result["latency"]["avg_ms"] = round(latency_summary["avg"], 1)
        except Exception as e:
            logger.warning(f"Error getting metrics for overview: {e}")

    # Get error stats
    error_store = _get_error_store()
    if error_store is not None:
        try:
            stats = error_store.get_stats()
            result["errors"]["total"] = stats.get("total_errors", 0)
            result["errors"]["last_hour"] = stats.get("errors_last_hour", 0)

            # Calculate failure rate if we have invocation data
            total_invocations = result["invocations"]["total"]
            total_errors = result["errors"]["total"]
            if total_invocations > 0:
                result["errors"]["failure_rate_pct"] = round(
                    (total_errors / total_invocations) * 100, 2
                )
        except Exception as e:
            logger.warning(f"Error getting error stats for overview: {e}")

    # Get health status
    try:
        from foundry_mcp.core.health import get_health_manager

        manager = get_health_manager()
        health_result = manager.check_health()
        result["health"]["status"] = health_result.status.value
        deps_ok = sum(1 for d in health_result.dependencies if d.healthy)
        result["health"]["deps_ok"] = deps_ok
        result["health"]["deps_total"] = len(health_result.dependencies)
    except Exception as e:
        logger.warning(f"Error getting health for overview: {e}")

    # Get provider status
    try:
        from foundry_mcp.core.providers import describe_providers

        all_providers = describe_providers()
        available = [p for p in all_providers if p.get("available", False)]
        result["providers"]["total"] = len(all_providers)
        result["providers"]["available"] = len(available)
        result["providers"]["names"] = [p.get("id", "") for p in available[:3]]
    except Exception as e:
        logger.warning(f"Error getting providers for overview: {e}")

    return _json_response(result)
