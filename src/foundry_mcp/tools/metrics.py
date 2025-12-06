"""MCP tools for metrics data introspection.

Provides tools to query, analyze, and explore persisted metrics data
for debugging and system monitoring across server restarts.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.responses import success_response, error_response, ErrorCode
from foundry_mcp.core.pagination import (
    encode_cursor,
    decode_cursor,
    paginated_response,
    normalize_page_size,
    CursorError,
)

logger = logging.getLogger(__name__)


def register_metrics_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """Register metrics introspection tools.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="metrics-query",
        description="""
        Query historical metrics with time-range and label filtering.

        Retrieves persisted metric data points matching the specified filters.
        Returns paginated results sorted by timestamp.

        WHEN TO USE:
        - Investigating metric trends over time
        - Analyzing tool invocation patterns
        - Examining error rates for specific tools
        - Debugging performance issues across restarts

        Args:
            metric_name: Filter by metric name (e.g., "tool_invocations_total")
            labels: JSON object of label key-value pairs to filter by
            since: ISO 8601 timestamp - include records after this time
            until: ISO 8601 timestamp - include records before this time
            limit: Maximum number of records to return (default: 100, max: 1000)
            cursor: Pagination cursor from previous response

        Returns:
            JSON object with matching metric data points and pagination metadata
        """,
    )
    def metrics_query(
        metric_name: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> dict[str, Any]:
        """Query historical metrics with filtering."""
        if not config.metrics_persistence.enabled:
            return error_response(
                error_code=ErrorCode.UNAVAILABLE,
                message="Metrics persistence is disabled",
                details={"config_key": "metrics_persistence.enabled"},
            )

        try:
            from foundry_mcp.core.metrics_store import get_metrics_store

            store = get_metrics_store(config.metrics_persistence.get_storage_path())

            # Handle pagination
            page_size = normalize_page_size(limit)
            offset = 0
            if cursor:
                try:
                    cursor_data = decode_cursor(cursor)
                    offset = cursor_data.get("offset", 0)
                except CursorError as e:
                    return error_response(
                        error_code=ErrorCode.VALIDATION_ERROR,
                        message=f"Invalid cursor: {e}",
                    )

            # Query metrics from store
            records = store.query(
                metric_name=metric_name,
                labels=labels,
                since=since,
                until=until,
                limit=page_size + 1,  # +1 to detect if there are more
                offset=offset,
            )

            # Check if there are more results
            has_more = len(records) > page_size
            if has_more:
                records = records[:page_size]

            # Generate next cursor if needed
            next_cursor = None
            if has_more:
                next_cursor = encode_cursor({"offset": offset + page_size})

            # Convert records to dicts
            metrics_dicts = [record.to_dict() for record in records]

            return paginated_response(
                items=metrics_dicts,
                total_count=store.count(),
                cursor=next_cursor,
                limit=page_size,
                item_key="metrics",
            )

        except Exception as e:
            logger.exception("Error querying metrics")
            return error_response(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to query metrics: {e}",
            )

    @canonical_tool(
        mcp,
        canonical_name="metrics-list",
        description="""
        List all persisted metrics with metadata.

        Returns a list of all metric names that have been persisted, along with
        their counts, first/last seen timestamps, and available label keys.

        WHEN TO USE:
        - Discovering what metrics are available
        - Understanding metric coverage
        - Finding metrics for investigation
        - Checking persistence health

        Args:
            limit: Maximum number of metrics to return (default: 100, max: 1000)
            cursor: Pagination cursor from previous response

        Returns:
            JSON object with list of metrics and pagination metadata
        """,
    )
    def metrics_list(
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> dict[str, Any]:
        """List all persisted metrics with metadata."""
        if not config.metrics_persistence.enabled:
            return error_response(
                error_code=ErrorCode.UNAVAILABLE,
                message="Metrics persistence is disabled",
            )

        try:
            from foundry_mcp.core.metrics_store import get_metrics_store

            store = get_metrics_store(config.metrics_persistence.get_storage_path())

            # Get all metrics
            all_metrics = store.list_metrics()

            # Handle pagination
            page_size = normalize_page_size(limit)
            offset = 0
            if cursor:
                try:
                    cursor_data = decode_cursor(cursor)
                    offset = cursor_data.get("offset", 0)
                except CursorError as e:
                    return error_response(
                        error_code=ErrorCode.VALIDATION_ERROR,
                        message=f"Invalid cursor: {e}",
                    )

            # Paginate results
            total_count = len(all_metrics)
            end_idx = offset + page_size
            metrics_page = all_metrics[offset:end_idx]

            # Generate next cursor if there are more
            next_cursor = None
            if end_idx < total_count:
                next_cursor = encode_cursor({"offset": end_idx})

            return paginated_response(
                items=metrics_page,
                total_count=total_count,
                cursor=next_cursor,
                limit=page_size,
                item_key="metrics",
            )

        except Exception as e:
            logger.exception("Error listing metrics")
            return error_response(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to list metrics: {e}",
            )

    @canonical_tool(
        mcp,
        canonical_name="metrics-summary",
        description="""
        Get aggregated statistics (min/max/avg/count) for a metric.

        Calculates summary statistics for the specified metric, optionally
        filtered by labels and time range.

        WHEN TO USE:
        - Getting overview statistics for a metric
        - Calculating average response times
        - Finding min/max values over a period
        - Generating reports

        Args:
            metric_name: Name of the metric to summarize (required)
            labels: JSON object of label key-value pairs to filter by
            since: ISO 8601 timestamp - include records after this time
            until: ISO 8601 timestamp - include records before this time

        Returns:
            JSON object with min, max, avg, sum, and count statistics
        """,
    )
    def metrics_summary(
        metric_name: str,
        labels: Optional[dict[str, str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get aggregated statistics for a metric."""
        if not config.metrics_persistence.enabled:
            return error_response(
                error_code=ErrorCode.UNAVAILABLE,
                message="Metrics persistence is disabled",
            )

        if not metric_name:
            return error_response(
                error_code=ErrorCode.VALIDATION_ERROR,
                message="metric_name is required",
            )

        try:
            from foundry_mcp.core.metrics_store import get_metrics_store

            store = get_metrics_store(config.metrics_persistence.get_storage_path())

            summary = store.get_summary(
                metric_name=metric_name,
                labels=labels,
                since=since,
                until=until,
            )

            return success_response(data={"summary": summary})

        except Exception as e:
            logger.exception("Error getting metrics summary")
            return error_response(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to get metrics summary: {e}",
            )

    @canonical_tool(
        mcp,
        canonical_name="metrics-cleanup",
        description="""
        Clean up old metric records based on retention policy.

        Removes metric records older than the retention period and
        enforces the maximum record count limit.

        WHEN TO USE:
        - Periodic maintenance of metrics storage
        - Freeing disk space from old records
        - Applying new retention settings

        Args:
            retention_days: Delete records older than this (default: from config)
            max_records: Maximum records to keep (default: from config)
            dry_run: Preview cleanup without deleting (default: False)

        Returns:
            JSON object with cleanup results:
            - deleted_count: Number of records deleted (or would be deleted)
            - dry_run: Whether this was a dry run
        """,
    )
    def metrics_cleanup(
        retention_days: Optional[int] = None,
        max_records: Optional[int] = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Clean up old metric records."""
        if not config.metrics_persistence.enabled:
            return error_response(
                error_code=ErrorCode.UNAVAILABLE,
                message="Metrics persistence is disabled",
            )

        # Use config defaults if not specified
        effective_retention = retention_days or config.metrics_persistence.retention_days
        effective_max = max_records or config.metrics_persistence.max_records

        try:
            from foundry_mcp.core.metrics_store import get_metrics_store

            store = get_metrics_store(config.metrics_persistence.get_storage_path())

            if dry_run:
                # For dry run, just return current count vs what would remain
                current_count = store.count()
                return success_response(
                    data={
                        "current_count": current_count,
                        "retention_days": effective_retention,
                        "max_records": effective_max,
                        "dry_run": True,
                        "message": "Dry run - no records deleted",
                    },
                )

            deleted_count = store.cleanup(
                retention_days=effective_retention,
                max_records=effective_max,
            )

            return success_response(
                data={
                    "deleted_count": deleted_count,
                    "retention_days": effective_retention,
                    "max_records": effective_max,
                    "dry_run": False,
                },
            )

        except Exception as e:
            logger.exception("Error cleaning up metrics")
            return error_response(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to cleanup metrics: {e}",
            )

    @canonical_tool(
        mcp,
        canonical_name="metrics-stats",
        description="""
        Get overall metrics persistence statistics.

        Returns high-level statistics about the metrics store including
        total records, unique metrics, and storage health information.

        WHEN TO USE:
        - Checking metrics persistence health
        - Understanding storage usage
        - Monitoring metrics collection coverage

        Returns:
            JSON object with storage statistics
        """,
    )
    def metrics_stats() -> dict[str, Any]:
        """Get overall metrics persistence statistics."""
        if not config.metrics_persistence.enabled:
            return error_response(
                error_code=ErrorCode.UNAVAILABLE,
                message="Metrics persistence is disabled",
            )

        try:
            from foundry_mcp.core.metrics_store import get_metrics_store

            store = get_metrics_store(config.metrics_persistence.get_storage_path())

            metrics_list = store.list_metrics()
            total_records = store.count()

            # Calculate summary stats
            unique_metrics = len(metrics_list)
            total_samples = sum(m.get("count", 0) for m in metrics_list)

            return success_response(
                data={
                    "total_records": total_records,
                    "unique_metrics": unique_metrics,
                    "total_samples": total_samples,
                    "metrics_by_name": {
                        m["metric_name"]: m["count"]
                        for m in metrics_list
                    },
                    "storage_path": str(config.metrics_persistence.get_storage_path()),
                    "retention_days": config.metrics_persistence.retention_days,
                    "max_records": config.metrics_persistence.max_records,
                },
            )

        except Exception as e:
            logger.exception("Error getting metrics stats")
            return error_response(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to get metrics stats: {e}",
            )
