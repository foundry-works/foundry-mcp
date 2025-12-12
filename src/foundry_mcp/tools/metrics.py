"""MCP tools for metrics data introspection.

Provides tools to query, analyze, and explore persisted metrics data
for debugging and system monitoring across server restarts.
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.tools.unified.metrics import legacy_metrics_action


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
    ) -> dict:
        """Query historical metrics with filtering."""

        return legacy_metrics_action(
            "query",
            config=config,
            metric_name=metric_name,
            labels=labels,
            since=since,
            until=until,
            limit=limit,
            cursor=cursor,
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
    ) -> dict:
        """List all persisted metrics with metadata."""

        return legacy_metrics_action(
            "list",
            config=config,
            limit=limit,
            cursor=cursor,
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
    ) -> dict:
        """Get aggregated statistics for a metric."""

        return legacy_metrics_action(
            "summary",
            config=config,
            metric_name=metric_name,
            labels=labels,
            since=since,
            until=until,
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
    ) -> dict:
        """Clean up old metric records."""

        return legacy_metrics_action(
            "cleanup",
            config=config,
            retention_days=retention_days,
            max_records=max_records,
            dry_run=dry_run,
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
    def metrics_stats() -> dict:
        """Get overall metrics persistence statistics."""

        return legacy_metrics_action("stats", config=config)
