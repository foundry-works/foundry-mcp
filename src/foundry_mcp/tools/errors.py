"""MCP tools for error data introspection.

Provides tools to query, analyze, and explore collected error data
for debugging and system improvement.
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.tools.unified.error import legacy_error_action


def register_error_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """Register error introspection tools.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="error-list",
        description="""
        Query error records with filtering and pagination.

        Filter errors by tool_name, error_code, error_type, fingerprint,
        provider_id, or time range. Returns paginated results.

        WHEN TO USE:
        - Investigating recent errors for a specific tool
        - Finding errors from a particular AI provider
        - Exploring errors matching a specific fingerprint
        - Debugging issues within a time window

        Args:
            tool_name: Filter by tool name
            error_code: Filter by error code (e.g., "VALIDATION_ERROR")
            error_type: Filter by error type (e.g., "validation")
            fingerprint: Filter by error fingerprint
            provider_id: Filter by AI provider ID
            since: ISO 8601 timestamp - include errors after this time
            until: ISO 8601 timestamp - include errors before this time
            limit: Maximum number of records to return (default: 100, max: 1000)
            cursor: Pagination cursor from previous response

        Returns:
            JSON object with matching error records and pagination metadata
        """,
    )
    def error_list(
        tool_name: Optional[str] = None,
        error_code: Optional[str] = None,
        error_type: Optional[str] = None,
        fingerprint: Optional[str] = None,
        provider_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> dict:
        """Query error records with filtering."""

        return legacy_error_action(
            action="list",
            config=config,
            tool_name=tool_name,
            error_code=error_code,
            error_type=error_type,
            fingerprint=fingerprint,
            provider_id=provider_id,
            since=since,
            until=until,
            limit=limit,
            cursor=cursor,
        )

    @canonical_tool(
        mcp,
        canonical_name="error-get",
        description="""
        Get detailed information about a specific error by ID.

        Retrieves full error record including stack trace, input summary,
        and all metadata for a specific error occurrence.

        WHEN TO USE:
        - Investigating a specific error occurrence
        - Getting full context for debugging
        - Examining stack traces for root cause analysis

        Args:
            error_id: The error ID to retrieve (format: err_<uuid>)

        Returns:
            JSON object with complete error record
        """,
    )
    def error_get(error_id: str) -> dict:
        """Get detailed error record by ID."""

        return legacy_error_action(action="get", config=config, error_id=error_id)

    @canonical_tool(
        mcp,
        canonical_name="error-stats",
        description="""
        Get aggregated error statistics.

        Returns error counts grouped by tool, error_code, and shows
        top error patterns (fingerprints) by occurrence count.

        WHEN TO USE:
        - Getting an overview of error distribution
        - Identifying most problematic tools
        - Finding most common error patterns
        - Monitoring error trends

        Returns:
            JSON object with aggregated statistics including:
            - total_errors: Total number of error records
            - unique_patterns: Number of unique error fingerprints
            - by_tool: Error counts per tool
            - by_error_code: Error counts per error code
            - top_patterns: Most frequent error patterns
        """,
    )
    def error_stats() -> dict:
        """Get aggregated error statistics."""

        return legacy_error_action(action="stats", config=config)

    @canonical_tool(
        mcp,
        canonical_name="error-patterns",
        description="""
        Get recurring error patterns (fingerprints with multiple occurrences).

        Identifies error patterns that occur repeatedly, useful for finding
        systemic issues that need investigation.

        WHEN TO USE:
        - Finding recurring issues that need attention
        - Identifying patterns for automated handling
        - Prioritizing debugging efforts
        - Monitoring for regression patterns

        Args:
            min_count: Minimum occurrence count to include (default: 3)

        Returns:
            JSON object with list of recurring patterns including:
            - fingerprint: Error signature
            - count: Number of occurrences
            - tool_name: Tool that generated the error
            - error_code: Error classification
            - first_seen/last_seen: Occurrence timestamps
            - sample_ids: Recent error IDs for investigation
        """,
    )
    def error_patterns(min_count: int = 3) -> dict:
        """Get recurring error patterns."""

        return legacy_error_action(
            action="patterns",
            config=config,
            min_count=min_count,
        )

    @canonical_tool(
        mcp,
        canonical_name="error-cleanup",
        description="""
        Clean up old error records based on retention policy.

        Removes error records older than the retention period and
        enforces the maximum error count limit.

        WHEN TO USE:
        - Periodic maintenance of error storage
        - Freeing disk space from old records
        - Applying new retention settings

        Args:
            retention_days: Delete records older than this (default: from config)
            max_errors: Maximum records to keep (default: from config)
            dry_run: Preview cleanup without deleting (default: False)

        Returns:
            JSON object with cleanup results:
            - deleted_count: Number of records deleted (or would be deleted)
            - dry_run: Whether this was a dry run
        """,
    )
    def error_cleanup(
        retention_days: Optional[int] = None,
        max_errors: Optional[int] = None,
        dry_run: bool = False,
    ) -> dict:
        """Clean up old error records."""

        return legacy_error_action(
            action="cleanup",
            config=config,
            retention_days=retention_days,
            max_errors=max_errors,
            dry_run=dry_run,
        )
