"""
Utility tools for foundry-mcp.

Provides MCP tools for miscellaneous operations like cache management
and schema exports.
"""

import logging
import subprocess
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.resilience import (
    CircuitBreaker,
    CircuitBreakerError,
)
from foundry_mcp.core.observability import (
    get_metrics,
    get_audit_logger,
    mcp_tool,
)

logger = logging.getLogger(__name__)

# Circuit breaker for utility operations
_utility_breaker = CircuitBreaker(
    name="utilities",
    failure_threshold=5,
    recovery_timeout=30.0,
)


def _run_sdd_command(args: list) -> Dict[str, Any]:
    """Run an SDD CLI command and return parsed JSON output.

    Args:
        args: Command arguments to pass to sdd CLI

    Returns:
        Dict with parsed JSON output or error info
    """
    try:
        result = subprocess.run(
            ["foundry-cli"] + args + ["--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            import json
            try:
                return {"success": True, "data": json.loads(result.stdout)}
            except json.JSONDecodeError:
                return {"success": True, "data": {"raw_output": result.stdout}}
        else:
            return {
                "success": False,
                "error": result.stderr or result.stdout or "Command failed"
            }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Command timed out after 30 seconds"}
    except FileNotFoundError:
        return {"success": False, "error": "foundry-cli not found. Ensure foundry-mcp is installed."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def register_utility_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register utility tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """
    metrics = get_metrics()
    audit = get_audit_logger()

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
            # Circuit breaker check
            if not _utility_breaker.can_execute():
                status = _utility_breaker.get_status()
                metrics.counter(
                    "utility.circuit_breaker_open",
                    labels={"tool": "sdd-cache-manage"},
                )
                return asdict(
                    error_response(
                        "Utility operations temporarily unavailable",
                        data={
                            "retry_after_seconds": status.get("retry_after_seconds"),
                            "breaker_state": status.get("state"),
                        },
                    )
                )

            # Validate action
            if action not in ("info", "clear"):
                return asdict(
                    error_response(
                        f"Invalid action: {action}. Must be 'info' or 'clear'."
                    )
                )

            # Validate review_type if provided
            if review_type and review_type not in ("fidelity", "plan"):
                return asdict(
                    error_response(
                        f"Invalid review_type: {review_type}. Must be 'fidelity' or 'plan'."
                    )
                )

            # Build command arguments
            cmd_args = ["cache", action]

            if action == "clear":
                if spec_id:
                    cmd_args.extend(["--spec-id", spec_id])
                if review_type:
                    cmd_args.extend(["--review-type", review_type])

            # Execute command
            result = _run_sdd_command(cmd_args)

            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.timer(
                "utility.cache_manage_time",
                duration_ms,
                labels={"action": action},
            )

            if not result["success"]:
                _utility_breaker.record_failure()
                return asdict(error_response(result["error"]))

            _utility_breaker.record_success()

            # Format response based on action
            if action == "info":
                cache_data = result["data"]
                return asdict(
                    success_response(
                        action="info",
                        cache=cache_data,
                        telemetry={"duration_ms": round(duration_ms, 2)},
                    )
                )
            else:  # clear
                cache_data = result["data"]
                return asdict(
                    success_response(
                        action="clear",
                        entries_deleted=cache_data.get("entries_deleted", 0),
                        filters={
                            "spec_id": spec_id,
                            "review_type": review_type,
                        },
                        telemetry={"duration_ms": round(duration_ms, 2)},
                    )
                )

        except CircuitBreakerError as e:
            logger.warning(f"Circuit breaker open for utilities: {e}")
            return asdict(
                error_response(
                    "Utility operations temporarily unavailable",
                    data={"retry_after_seconds": e.retry_after},
                )
            )
        except Exception as e:
            _utility_breaker.record_failure()
            logger.error(f"Error in cache management: {e}")
            return asdict(error_response(str(e)))

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
            # Circuit breaker check
            if not _utility_breaker.can_execute():
                status = _utility_breaker.get_status()
                metrics.counter(
                    "utility.circuit_breaker_open",
                    labels={"tool": "spec-schema-export"},
                )
                return asdict(
                    error_response(
                        "Utility operations temporarily unavailable",
                        data={
                            "retry_after_seconds": status.get("retry_after_seconds"),
                            "breaker_state": status.get("state"),
                        },
                    )
                )

            # Validate schema_type
            if schema_type not in ("spec",):
                return asdict(
                    error_response(
                        f"Invalid schema_type: {schema_type}. Must be 'spec'."
                    )
                )

            # Execute schema command
            result = _run_sdd_command(["schema"])

            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.timer(
                "utility.schema_export_time",
                duration_ms,
                labels={"schema_type": schema_type},
            )

            if not result["success"]:
                _utility_breaker.record_failure()
                return asdict(error_response(result["error"]))

            _utility_breaker.record_success()

            schema_data = result["data"]

            return asdict(
                success_response(
                    schema=schema_data,
                    schema_type=schema_type,
                    schema_url="https://github.com/sdd-toolkit/sdd-spec-schema",
                    telemetry={"duration_ms": round(duration_ms, 2)},
                )
            )

        except CircuitBreakerError as e:
            logger.warning(f"Circuit breaker open for utilities: {e}")
            return asdict(
                error_response(
                    "Utility operations temporarily unavailable",
                    data={"retry_after_seconds": e.retry_after},
                )
            )
        except Exception as e:
            _utility_breaker.record_failure()
            logger.error(f"Error exporting schema: {e}")
            return asdict(error_response(str(e)))

    logger.debug("Registered utility tools: sdd-cache-manage, spec-schema-export")
