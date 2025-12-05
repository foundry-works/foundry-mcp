"""
Utility tools for foundry-mcp.

Provides MCP tools for miscellaneous operations like cache management
and schema exports. Uses direct Python API calls to core modules.
"""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response, sanitize_error_message
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.cache import CacheManager
from foundry_mcp.core.observability import (
    get_metrics,
    get_audit_logger,
    mcp_tool,
)

logger = logging.getLogger(__name__)

# Metrics singleton for utility tools
_metrics = get_metrics()


def register_utility_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register utility tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="sdd-cache-manage",
    )
    @mcp_tool(tool_name="sdd-cache-manage", emit_metrics=True, audit=True)
    def sdd_cache_manage(
        action: str,
        spec_id: Optional[str] = None,
        review_type: Optional[str] = None,
    ) -> dict:
        """
        Manage SDD CLI cache entries.

        Provides operations to inspect and manage the SDD consultation cache
        used for AI consultation results.

        Args:
            action: Cache operation - 'info' to get stats, 'clear' to remove entries
            spec_id: Optional spec ID filter for clear operation
            review_type: Optional review type filter ('fidelity' or 'plan')

        Returns:
            JSON object with:
            - For 'info': cache statistics (total entries, active entries, size)
            - For 'clear': count of deleted entries and filters applied

        WHEN TO USE:
        - Check cache status and statistics
        - Clear expired or stale cache entries
        - Free up disk space from consultation cache
        - Debug caching issues
        """
        start_time = time.perf_counter()

        try:
            # Validate action
            if action not in ("info", "clear"):
                return asdict(
                    error_response(
                        f"Invalid action: {action}. Must be 'info' or 'clear'.",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Use 'info' to get cache stats or 'clear' to remove entries",
                    )
                )

            # Validate review_type if provided
            if review_type and review_type not in ("fidelity", "plan"):
                return asdict(
                    error_response(
                        f"Invalid review_type: {review_type}. Must be 'fidelity' or 'plan'.",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Use 'fidelity' or 'plan' for review_type filter",
                    )
                )

            # Use CacheManager directly
            cache_manager = CacheManager()

            if action == "info":
                cache_stats = cache_manager.get_stats()
                duration_ms = (time.perf_counter() - start_time) * 1000
                _metrics.timer("utility.cache_manage.duration_ms", duration_ms, labels={"action": "info"})

                return asdict(
                    success_response(
                        action="info",
                        cache=cache_stats,
                        duration_ms=round(duration_ms, 2),
                    )
                )
            else:  # clear
                entries_deleted = cache_manager.clear(spec_id=spec_id, review_type=review_type)
                duration_ms = (time.perf_counter() - start_time) * 1000
                _metrics.timer("utility.cache_manage.duration_ms", duration_ms, labels={"action": "clear"})

                return asdict(
                    success_response(
                        action="clear",
                        entries_deleted=entries_deleted,
                        filters={
                            "spec_id": spec_id,
                            "review_type": review_type,
                        },
                        duration_ms=round(duration_ms, 2),
                    )
                )

        except Exception as e:
            logger.exception("Error in cache management")
            return asdict(
                error_response(
                    sanitize_error_message(e, context="cache management"),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                    remediation="Check logs for details",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="spec-schema-export",
    )
    @mcp_tool(tool_name="spec-schema-export", emit_metrics=True, audit=True)
    def spec_schema_export(
        schema_type: str = "spec",
    ) -> dict:
        """
        Export JSON schemas for SDD specifications and tasks.

        Returns the JSON Schema used to validate SDD specification files.
        Useful for schema validation, IDE integration, and documentation.

        Args:
            schema_type: Type of schema to export ('spec' for full spec schema)

        Returns:
            JSON object with:
            - schema: The JSON Schema definition
            - schema_type: Type of schema returned
            - schema_url: Reference URL for the schema

        WHEN TO USE:
        - Get schema for validation tools
        - IDE/editor schema configuration
        - Documentation generation
        - Understanding spec file structure
        """
        start_time = time.perf_counter()

        try:
            # Validate schema_type
            if schema_type not in ("spec",):
                return asdict(
                    error_response(
                        f"Invalid schema_type: {schema_type}. Must be 'spec'.",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Use 'spec' for the full specification schema",
                    )
                )

            # Load schema from package resources
            # Schema is bundled with the package at src/foundry_mcp/schemas/sdd-spec-schema.json
            schema_path = Path(__file__).parent.parent / "schemas" / "sdd-spec-schema.json"

            if not schema_path.exists():
                return asdict(
                    error_response(
                        "Schema file not found",
                        error_code="NOT_FOUND",
                        error_type="internal",
                        data={"schema_path": str(schema_path)},
                        remediation="Ensure foundry-mcp package is properly installed",
                    )
                )

            with open(schema_path, "r") as f:
                schema_data = json.load(f)

            duration_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer("utility.schema_export.duration_ms", duration_ms, labels={"schema_type": schema_type})

            return asdict(
                success_response(
                    schema=schema_data,
                    schema_type=schema_type,
                    schema_url="https://github.com/sdd-toolkit/sdd-spec-schema",
                    duration_ms=round(duration_ms, 2),
                )
            )

        except json.JSONDecodeError as e:
            logger.exception("Error parsing schema file")
            return asdict(
                error_response(
                    sanitize_error_message(e, context="schema export"),
                    error_code="PARSE_ERROR",
                    error_type="internal",
                    remediation="Check schema file is valid JSON",
                )
            )
        except Exception as e:
            logger.exception("Error exporting schema")
            return asdict(
                error_response(
                    sanitize_error_message(e, context="schema export"),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                    remediation="Check logs for details",
                )
            )

    logger.debug("Registered utility tools: sdd-cache-manage, spec-schema-export")
