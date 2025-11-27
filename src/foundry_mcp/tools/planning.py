"""
Planning tools for foundry-mcp.

Provides MCP tools for task planning and execution utilities,
including plan formatting, phase management, time reporting,
and spec state reconciliation.

Resilience features:
- Circuit breaker for SDD CLI calls (opens after 5 consecutive failures)
- Timing metrics for all tool invocations
- Configurable timeout (default 30s per operation)
"""

import json
import logging
import subprocess
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import audit_log, get_metrics
from foundry_mcp.core.resilience import (
    CircuitBreaker,
    CircuitBreakerError,
    MEDIUM_TIMEOUT,
)

logger = logging.getLogger(__name__)

# Metrics singleton for planning tools
_metrics = get_metrics()

# Circuit breaker for SDD CLI operations
# Opens after 5 consecutive failures, recovers after 30 seconds
_sdd_cli_breaker = CircuitBreaker(
    name="sdd_cli_planning",
    failure_threshold=5,
    recovery_timeout=30.0,
    half_open_max_calls=3,
)

# Default timeout for CLI operations (30 seconds)
CLI_TIMEOUT: float = MEDIUM_TIMEOUT


def _run_sdd_command(
    cmd: list,
    tool_name: str,
    timeout: float = CLI_TIMEOUT,
) -> subprocess.CompletedProcess:
    """
    Execute an SDD CLI command with circuit breaker protection and timing.

    Args:
        cmd: Command list to execute
        tool_name: Name of the calling tool (for metrics)
        timeout: Timeout in seconds

    Returns:
        CompletedProcess result from subprocess.run

    Raises:
        CircuitBreakerError: If circuit breaker is open
        subprocess.TimeoutExpired: If command times out
        FileNotFoundError: If SDD CLI is not found
    """
    # Check circuit breaker
    if not _sdd_cli_breaker.can_execute():
        status = _sdd_cli_breaker.get_status()
        _metrics.counter(f"planning.{tool_name}", labels={"status": "circuit_open"})
        raise CircuitBreakerError(
            f"SDD CLI circuit breaker is open (retry after {status.get('retry_after_seconds', 0):.1f}s)",
            breaker_name="sdd_cli_planning",
            state=_sdd_cli_breaker.state,
            retry_after=status.get("retry_after_seconds"),
        )

    start_time = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Record success or failure based on return code
        if result.returncode == 0:
            _sdd_cli_breaker.record_success()
        else:
            # Non-zero return code counts as a failure for circuit breaker
            _sdd_cli_breaker.record_failure()

        return result

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        # These are infrastructure failures that should trip the circuit breaker
        _sdd_cli_breaker.record_failure()
        raise
    finally:
        # Record timing metrics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _metrics.timer(f"planning.{tool_name}.duration_ms", elapsed_ms)


def register_planning_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register planning tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="plan-format",
    )
    def plan_format(
        spec_id: str,
        task_id: str,
        path: Optional[str] = None,
    ) -> dict:
        """
        Pretty-print a task plan for sharing and review.

        Wraps the SDD CLI format-plan command to generate a human-readable
        formatted output of a task plan, suitable for sharing with team
        members or including in documentation.

        WHEN TO USE:
        - Sharing task plans with team members
        - Generating documentation from task plans
        - Reviewing task structure before implementation
        - Creating readable summaries of complex tasks

        Args:
            spec_id: Specification ID containing the task
            task_id: Task ID to format
            path: Project root path (default: current directory)

        Returns:
            JSON object with formatted plan:
            - formatted: Human-readable plan text
            - task_id: The task ID that was formatted
            - spec_id: The specification ID
        """
        tool_name = "plan_format"
        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            if not task_id:
                return asdict(error_response(
                    "task_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a task_id parameter",
                ))

            # Build command
            cmd = ["sdd", "format-plan", spec_id, task_id, "--json"]

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="plan-format",
                action="format_plan",
                spec_id=spec_id,
                task_id=task_id,
            )

            # Execute the command with resilience
            result = _run_sdd_command(cmd, tool_name)

            # Parse the JSON output
            if result.returncode == 0:
                try:
                    output_data = json.loads(result.stdout) if result.stdout.strip() else {}
                except json.JSONDecodeError:
                    output_data = {"formatted": result.stdout}

                # Build response data
                data: Dict[str, Any] = {
                    "spec_id": spec_id,
                    "task_id": task_id,
                    "formatted": output_data.get("formatted", output_data.get("plan", result.stdout)),
                }

                # Include additional fields if available
                if "title" in output_data:
                    data["title"] = output_data["title"]
                if "status" in output_data:
                    data["status"] = output_data["status"]

                # Track metrics
                _metrics.counter(f"planning.{tool_name}", labels={"status": "success"})

                return asdict(success_response(
                    data=data,
                    message="Plan formatted successfully",
                ))
            else:
                # Handle specific error cases
                stderr = result.stderr.strip()

                if "not found" in stderr.lower():
                    error_code = "NOT_FOUND"
                    remediation = "Ensure the spec and task IDs exist"
                else:
                    error_code = "FORMAT_FAILED"
                    remediation = "Check the spec_id and task_id"

                _metrics.counter(f"planning.{tool_name}", labels={"status": "error", "code": error_code})

                return asdict(error_response(
                    stderr or "Plan formatting failed",
                    error_code=error_code,
                    error_type="planning",
                    remediation=remediation,
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="resilience",
                remediation="Wait for circuit breaker recovery, then retry",
            ))

        except subprocess.TimeoutExpired:
            _metrics.counter(f"planning.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Plan formatting timed out after {CLI_TIMEOUT}s",
                error_code="TIMEOUT",
                error_type="timeout",
                remediation="Try again or check system resources",
            ))

        except FileNotFoundError:
            _metrics.counter(f"planning.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found. Ensure 'sdd' is installed and in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="configuration",
                remediation="Install SDD toolkit: pip install claude-sdd-toolkit",
            ))

        except Exception as e:
            logger.exception(f"Unexpected error in {tool_name}")
            _metrics.counter(f"planning.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="phase-list",
    )
    def phase_list(
        spec_id: str,
        path: Optional[str] = None,
    ) -> dict:
        """
        Enumerate all phases in a specification.

        Wraps the SDD CLI list-phases command to return all phases
        in a specification with their completion status and progress.

        WHEN TO USE:
        - Getting an overview of spec structure
        - Checking phase completion status
        - Planning which phase to work on next
        - Understanding spec organization

        Args:
            spec_id: Specification ID to enumerate phases for
            path: Project root path (default: current directory)

        Returns:
            JSON object with phase list:
            - phases: Array of phase objects with id, title, status, progress
            - total_phases: Total number of phases
            - completed_phases: Number of completed phases
        """
        tool_name = "phase_list"
        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            # Build command
            cmd = ["sdd", "list-phases", spec_id, "--json"]

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="phase-list",
                action="list_phases",
                spec_id=spec_id,
            )

            # Execute the command with resilience
            result = _run_sdd_command(cmd, tool_name)

            # Parse the JSON output
            if result.returncode == 0:
                try:
                    output_data = json.loads(result.stdout) if result.stdout.strip() else {}
                except json.JSONDecodeError:
                    output_data = {}

                # Build response data
                phases = output_data.get("phases", [])
                data: Dict[str, Any] = {
                    "spec_id": spec_id,
                    "phases": phases,
                    "total_phases": len(phases),
                    "completed_phases": sum(1 for p in phases if p.get("status") == "completed"),
                }

                # Track metrics
                _metrics.counter(f"planning.{tool_name}", labels={"status": "success"})

                return asdict(success_response(
                    data=data,
                    message=f"Found {len(phases)} phases",
                ))
            else:
                # Handle specific error cases
                stderr = result.stderr.strip()

                if "not found" in stderr.lower():
                    error_code = "NOT_FOUND"
                    remediation = "Ensure the spec_id exists in specs/active or specs/pending"
                else:
                    error_code = "LIST_FAILED"
                    remediation = "Check the spec_id and try again"

                _metrics.counter(f"planning.{tool_name}", labels={"status": "error", "code": error_code})

                return asdict(error_response(
                    stderr or "Failed to list phases",
                    error_code=error_code,
                    error_type="planning",
                    remediation=remediation,
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="resilience",
                remediation="Wait for circuit breaker recovery, then retry",
            ))

        except subprocess.TimeoutExpired:
            _metrics.counter(f"planning.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Phase listing timed out after {CLI_TIMEOUT}s",
                error_code="TIMEOUT",
                error_type="timeout",
                remediation="Try again or check system resources",
            ))

        except FileNotFoundError:
            _metrics.counter(f"planning.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found. Ensure 'sdd' is installed and in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="configuration",
                remediation="Install SDD toolkit: pip install claude-sdd-toolkit",
            ))

        except Exception as e:
            logger.exception(f"Unexpected error in {tool_name}")
            _metrics.counter(f"planning.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))
