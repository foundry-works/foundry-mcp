"""
Documentation generation tools for foundry-mcp.

Provides MCP tools for generating human-facing documentation bundles
from SDD specifications. Wraps the `sdd doc` CLI to produce markdown
and HTML documentation artifacts.
"""

import logging
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    mcp_tool,
)

logger = logging.getLogger(__name__)

# Circuit breaker for documentation generation operations
_doc_breaker = CircuitBreaker(
    name="documentation",
    failure_threshold=3,
    recovery_timeout=60.0,
)

# Default timeout for doc generation (can take longer for large specs)
DOC_GENERATION_TIMEOUT = 120


def _run_sdd_doc_command(
    args: List[str],
    timeout: int = DOC_GENERATION_TIMEOUT,
) -> Dict[str, Any]:
    """Run an SDD doc CLI command and return parsed JSON output.

    Args:
        args: Command arguments to pass to sdd doc CLI
        timeout: Command timeout in seconds

    Returns:
        Dict with parsed JSON output or error info
    """
    try:
        result = subprocess.run(
            ["sdd", "doc"] + args + ["--json"],
            capture_output=True,
            text=True,
            timeout=timeout,
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
                "error": result.stderr or result.stdout or "Command failed",
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Command timed out after {timeout} seconds",
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "sdd CLI not found. Ensure sdd-toolkit is installed.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _run_sdd_render_command(
    args: List[str],
    timeout: int = DOC_GENERATION_TIMEOUT,
) -> Dict[str, Any]:
    """Run an SDD render CLI command and return parsed JSON output.

    Args:
        args: Command arguments to pass to sdd render CLI
        timeout: Command timeout in seconds

    Returns:
        Dict with parsed JSON output or error info
    """
    try:
        result = subprocess.run(
            ["sdd", "render"] + args + ["--json"],
            capture_output=True,
            text=True,
            timeout=timeout,
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
                "error": result.stderr or result.stdout or "Command failed",
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Command timed out after {timeout} seconds",
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "sdd CLI not found. Ensure sdd-toolkit is installed.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def register_documentation_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register documentation generation tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """
    metrics = get_metrics()

    @canonical_tool(
        mcp,
        canonical_name="spec-doc",
    )
    @mcp_tool(tool_name="spec-doc", emit_metrics=True, audit=True)
    def spec_doc(
        spec_id: str,
        output_format: str = "markdown",
        output_path: Optional[str] = None,
        include_progress: bool = True,
        include_journal: bool = False,
        mode: str = "basic",
        workspace: Optional[str] = None,
    ) -> dict:
        """
        Generate human-facing documentation bundle from a specification.

        Creates formatted documentation artifacts (markdown/HTML) from
        an SDD specification. Wraps `sdd render` to produce documentation
        suitable for stakeholders, project tracking, and archive.

        Args:
            spec_id: Specification ID to document
            output_format: Output format - 'markdown' or 'md' (HTML planned)
            output_path: Custom output path (default: specs/.human-readable/<spec_id>.md)
            include_progress: Include visual progress bars and stats
            include_journal: Include recent journal entries in output
            mode: Rendering mode - 'basic' (fast) or 'enhanced' (AI features)
            workspace: Optional workspace path (defaults to config)

        Returns:
            JSON object with:
            - output_path: Path where documentation was written
            - format: Format of generated documentation
            - spec_id: Specification ID
            - title: Specification title
            - stats: Documentation statistics (sections, tasks, etc.)

        WHEN TO USE:
        - Generate stakeholder-friendly documentation from specs
        - Create progress reports for project tracking
        - Archive completed specifications in readable format
        - Export specs for external sharing or review
        """
        start_time = time.perf_counter()

        try:
            # Circuit breaker check
            if not _doc_breaker.can_execute():
                status = _doc_breaker.get_status()
                metrics.counter(
                    "documentation.circuit_breaker_open",
                    labels={"tool": "spec-doc"},
                )
                return asdict(
                    error_response(
                        "Documentation generation temporarily unavailable",
                        error_code="UNAVAILABLE",
                        error_type="unavailable",
                        data={
                            "retry_after_seconds": status.get("retry_after_seconds"),
                            "breaker_state": status.get("state"),
                        },
                        remediation="Wait and retry. The service is recovering from errors.",
                    )
                )

            # Validate output_format
            if output_format not in ("markdown", "md"):
                return asdict(
                    error_response(
                        f"Invalid output_format: {output_format}. "
                        "Supported formats: 'markdown', 'md'",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Use 'markdown' or 'md' as output_format.",
                    )
                )

            # Validate mode
            if mode not in ("basic", "enhanced"):
                return asdict(
                    error_response(
                        f"Invalid mode: {mode}. Must be 'basic' or 'enhanced'.",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Use 'basic' for fast rendering or 'enhanced' for AI features.",
                    )
                )

            # Build command arguments for sdd render
            cmd_args = [spec_id, "--format", output_format, "--mode", mode]

            if output_path:
                cmd_args.extend(["--output", output_path])

            if workspace:
                cmd_args.extend(["--path", workspace])

            # Execute render command
            result = _run_sdd_render_command(cmd_args)

            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.timer(
                "documentation.spec_doc_time",
                duration_ms,
                labels={"mode": mode, "format": output_format},
            )

            if not result["success"]:
                _doc_breaker.record_failure()

                # Check for specific error patterns
                error_msg = result["error"]
                if "not found" in error_msg.lower():
                    return asdict(
                        error_response(
                            f"Specification not found: {spec_id}",
                            error_code="SPEC_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the spec ID exists using spec-list.",
                        )
                    )

                return asdict(error_response(error_msg))

            _doc_breaker.record_success()

            # Parse the successful response
            doc_data = result["data"]

            # Build response with documentation metadata
            response_data = {
                "spec_id": spec_id,
                "format": output_format,
                "mode": mode,
                "output_path": doc_data.get(
                    "output_path",
                    f"specs/.human-readable/{spec_id}.md",
                ),
            }

            # Add optional fields if present
            if "title" in doc_data:
                response_data["title"] = doc_data["title"]

            if "stats" in doc_data:
                response_data["stats"] = doc_data["stats"]
            elif "total_tasks" in doc_data:
                response_data["stats"] = {
                    "total_tasks": doc_data.get("total_tasks", 0),
                    "completed_tasks": doc_data.get("completed_tasks", 0),
                    "total_sections": doc_data.get("total_sections", 0),
                }

            if include_progress and "progress_percentage" in doc_data:
                response_data["progress_percentage"] = doc_data["progress_percentage"]

            return asdict(
                success_response(
                    **response_data,
                    telemetry={"duration_ms": round(duration_ms, 2)},
                )
            )

        except CircuitBreakerError as e:
            logger.warning(f"Circuit breaker open for documentation: {e}")
            return asdict(
                error_response(
                    "Documentation generation temporarily unavailable",
                    error_code="UNAVAILABLE",
                    error_type="unavailable",
                    data={"retry_after_seconds": e.retry_after},
                    remediation="Wait and retry. The service is recovering from errors.",
                )
            )
        except Exception as e:
            _doc_breaker.record_failure()
            logger.error(f"Error generating documentation: {e}")
            return asdict(
                error_response(
                    str(e),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                )
            )

    logger.debug("Registered documentation tools: spec-doc")
