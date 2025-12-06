"""MCP tools for error data introspection.

Provides tools to query, analyze, and explore collected error data
for debugging and system improvement.
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
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
)

logger = logging.getLogger(__name__)


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
    ) -> dict[str, Any]:
        """Query error records with filtering."""
        # Check if error collection is enabled
        if not config.error_collection.enabled:
            return error_response(
                error_code=ErrorCode.UNAVAILABLE,
                message="Error collection is disabled",
                details={"config_key": "error_collection.enabled"},
            )

        try:
            from foundry_mcp.core.error_collection import get_error_collector

            collector = get_error_collector()
            if not collector.is_enabled():
                return error_response(
                    error_code=ErrorCode.UNAVAILABLE,
                    message="Error collector is not enabled",
                )

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

            # Query errors from store
            store = collector.store
            records = store.query(
                tool_name=tool_name,
                error_code=error_code,
                error_type=error_type,
                fingerprint=fingerprint,
                provider_id=provider_id,
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
            error_dicts = [record.to_dict() for record in records]

            return paginated_response(
                items=error_dicts,
                total_count=store.count(),
                cursor=next_cursor,
                limit=page_size,
                item_key="errors",
            )

        except Exception as e:
            logger.exception("Error querying errors")
            return error_response(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to query errors: {e}",
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
    def error_get(error_id: str) -> dict[str, Any]:
        """Get detailed error record by ID."""
        if not config.error_collection.enabled:
            return error_response(
                error_code=ErrorCode.UNAVAILABLE,
                message="Error collection is disabled",
            )

        if not error_id:
            return error_response(
                error_code=ErrorCode.VALIDATION_ERROR,
                message="error_id is required",
            )

        try:
            from foundry_mcp.core.error_collection import get_error_collector

            collector = get_error_collector()
            store = collector.store
            record = store.get(error_id)

            if record is None:
                return error_response(
                    error_code=ErrorCode.NOT_FOUND,
                    message=f"Error record not found: {error_id}",
                )

            return success_response(
                data={"error": record.to_dict()},
            )

        except Exception as e:
            logger.exception("Error retrieving error record")
            return error_response(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to retrieve error: {e}",
            )

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
    def error_stats() -> dict[str, Any]:
        """Get aggregated error statistics."""
        if not config.error_collection.enabled:
            return error_response(
                error_code=ErrorCode.UNAVAILABLE,
                message="Error collection is disabled",
            )

        try:
            from foundry_mcp.core.error_collection import get_error_collector

            collector = get_error_collector()
            store = collector.store
            stats = store.get_stats()

            return success_response(data=stats)

        except Exception as e:
            logger.exception("Error getting error stats")
            return error_response(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to get error stats: {e}",
            )

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
    def error_patterns(min_count: int = 3) -> dict[str, Any]:
        """Get recurring error patterns."""
        if not config.error_collection.enabled:
            return error_response(
                error_code=ErrorCode.UNAVAILABLE,
                message="Error collection is disabled",
            )

        if min_count < 1:
            min_count = 1

        try:
            from foundry_mcp.core.error_collection import get_error_collector

            collector = get_error_collector()
            store = collector.store
            patterns = store.get_patterns(min_count=min_count)

            return success_response(
                data={
                    "patterns": patterns,
                    "pattern_count": len(patterns),
                    "min_count_filter": min_count,
                },
            )

        except Exception as e:
            logger.exception("Error getting error patterns")
            return error_response(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to get error patterns: {e}",
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
    ) -> dict[str, Any]:
        """Clean up old error records."""
        if not config.error_collection.enabled:
            return error_response(
                error_code=ErrorCode.UNAVAILABLE,
                message="Error collection is disabled",
            )

        # Use config defaults if not specified
        effective_retention = retention_days or config.error_collection.retention_days
        effective_max = max_errors or config.error_collection.max_errors

        try:
            from foundry_mcp.core.error_collection import get_error_collector

            collector = get_error_collector()
            store = collector.store

            if dry_run:
                # For dry run, just return current count vs what would remain
                current_count = store.count()
                return success_response(
                    data={
                        "current_count": current_count,
                        "retention_days": effective_retention,
                        "max_errors": effective_max,
                        "dry_run": True,
                        "message": "Dry run - no records deleted",
                    },
                )

            deleted_count = store.cleanup(
                retention_days=effective_retention,
                max_errors=effective_max,
            )

            return success_response(
                data={
                    "deleted_count": deleted_count,
                    "retention_days": effective_retention,
                    "max_errors": effective_max,
                    "dry_run": False,
                },
            )

        except Exception as e:
            logger.exception("Error cleaning up errors")
            return error_response(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to cleanup errors: {e}",
            )
