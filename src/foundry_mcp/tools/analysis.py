"""
Analysis tools for foundry-mcp.

Provides MCP tools for analyzing SDD specifications and performing
structural/quality analysis on specs and their dependencies.

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

# Metrics singleton for analysis tools
_metrics = get_metrics()

# Circuit breaker for SDD CLI operations
# Opens after 5 consecutive failures, recovers after 30 seconds
_sdd_cli_breaker = CircuitBreaker(
    name="sdd_cli_analysis",
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
        _metrics.counter(f"analysis.{tool_name}", labels={"status": "circuit_open"})
        raise CircuitBreakerError(
            f"SDD CLI circuit breaker is open (retry after {status.get('retry_after_seconds', 0):.1f}s)",
            breaker_name="sdd_cli_analysis",
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
        _metrics.timer(f"analysis.{tool_name}.duration_ms", elapsed_ms)


def register_analysis_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register analysis tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="spec-analyze",
    )
    def spec_analyze(
        directory: Optional[str] = None,
        path: Optional[str] = None,
    ) -> dict:
        """
        Perform deep structural and quality analysis on SDD specifications.

        Wraps the SDD CLI analyze command to inspect the specification
        directory and provide heuristics about spec health, structure,
        and documentation availability.

        WHEN TO USE:
        - Evaluating spec health before starting work
        - Checking for documentation availability
        - Understanding project SDD structure
        - Auditing spec organization and quality

        Args:
            directory: Directory to analyze (default: current directory)
            path: Project root path (default: current directory)

        Returns:
            JSON object with analysis results:
            - directory: Analyzed directory path
            - has_specs: Whether specs directory exists
            - documentation_available: Whether generated docs exist
            - Additional heuristics from sdd analyze
        """
        tool_name = "spec_analyze"
        try:
            # Build command
            cmd = ["foundry-cli", "analyze", "--json"]

            if directory:
                cmd.append(directory)

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-analyze",
                action="analyze_specs",
                directory=directory or ".",
            )

            # Execute the command with resilience
            result = _run_sdd_command(cmd, tool_name)

            # Parse the JSON output
            if result.returncode == 0:
                try:
                    output_data = json.loads(result.stdout) if result.stdout.strip() else {}
                except json.JSONDecodeError:
                    output_data = {"raw_output": result.stdout}

                # Track metrics
                _metrics.counter(f"analysis.{tool_name}", labels={"status": "success"})

                return asdict(success_response(
                    data=output_data,
                    message="Specification analysis completed",
                ))
            else:
                # Handle specific error cases
                stderr = result.stderr.strip()

                if "not found" in stderr.lower():
                    error_code = "NOT_FOUND"
                    remediation = "Ensure the directory exists and contains SDD specifications"
                else:
                    error_code = "ANALYSIS_FAILED"
                    remediation = "Check the directory path and ensure sdd is properly configured"

                _metrics.counter(f"analysis.{tool_name}", labels={"status": "error", "code": error_code})

                return asdict(error_response(
                    stderr or "Analysis failed",
                    error_code=error_code,
                    error_type="analysis",
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
            _metrics.counter(f"analysis.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Analysis timed out after {CLI_TIMEOUT}s",
                error_code="TIMEOUT",
                error_type="timeout",
                remediation="Try analyzing a smaller directory or increase timeout",
            ))

        except FileNotFoundError:
            _metrics.counter(f"analysis.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found. Ensure 'sdd' is installed and in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="configuration",
                remediation="Install SDD toolkit: pip install claude-sdd-toolkit",
            ))

        except Exception as e:
            logger.exception(f"Unexpected error in {tool_name}")
            _metrics.counter(f"analysis.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="review-parse-feedback",
    )
    def review_parse_feedback(
        spec_id: str,
        review_path: str,
        output_path: Optional[str] = None,
        path: Optional[str] = None,
    ) -> dict:
        """
        Transform review feedback into structured modification actions.

        Wraps the SDD CLI parse-review command to convert review markdown
        or JSON into a structured suggestions file that can be applied
        to modify the specification.

        WHEN TO USE:
        - Processing AI-generated review feedback
        - Converting review notes into actionable changes
        - Preparing review feedback for automated application
        - Transforming unstructured feedback into spec modifications

        Args:
            spec_id: Specification ID being reviewed
            review_path: Path to review report file (.md or .json)
            output_path: Output path for suggestions JSON (default: <review>.suggestions.json)
            path: Project root path (default: current directory)

        Returns:
            JSON object with parsing results:
            - suggestions_count: Number of suggestions extracted
            - output_file: Path to generated suggestions file
            - categories: Breakdown of suggestion types
        """
        tool_name = "review_parse_feedback"
        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            if not review_path:
                return asdict(error_response(
                    "review_path is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a path to the review file (.md or .json)",
                ))

            # Build command
            cmd = ["foundry-cli", "parse-review", spec_id, "--review", review_path, "--json"]

            if output_path:
                cmd.extend(["--output", output_path])

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="review-parse-feedback",
                action="parse_review",
                spec_id=spec_id,
                review_path=review_path,
            )

            # Execute the command with resilience
            result = _run_sdd_command(cmd, tool_name)

            # Parse the JSON output
            if result.returncode == 0:
                try:
                    output_data = json.loads(result.stdout) if result.stdout.strip() else {}
                except json.JSONDecodeError:
                    output_data = {"raw_output": result.stdout}

                # Track metrics
                _metrics.counter(f"analysis.{tool_name}", labels={"status": "success"})

                return asdict(success_response(
                    data=output_data,
                    message="Review feedback parsed successfully",
                ))
            else:
                # Handle specific error cases
                stderr = result.stderr.strip()

                if "not found" in stderr.lower():
                    error_code = "NOT_FOUND"
                    remediation = "Ensure the review file and spec exist"
                elif "invalid" in stderr.lower() or "parse" in stderr.lower():
                    error_code = "PARSE_ERROR"
                    remediation = "Check the review file format (.md or .json)"
                else:
                    error_code = "PARSE_FAILED"
                    remediation = "Check the review file and spec_id"

                _metrics.counter(f"analysis.{tool_name}", labels={"status": "error", "code": error_code})

                return asdict(error_response(
                    stderr or "Review parsing failed",
                    error_code=error_code,
                    error_type="analysis",
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
            _metrics.counter(f"analysis.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Review parsing timed out after {CLI_TIMEOUT}s",
                error_code="TIMEOUT",
                error_type="timeout",
                remediation="Try with a smaller review file",
            ))

        except FileNotFoundError:
            _metrics.counter(f"analysis.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found. Ensure 'sdd' is installed and in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="configuration",
                remediation="Install SDD toolkit: pip install claude-sdd-toolkit",
            ))

        except Exception as e:
            logger.exception(f"Unexpected error in {tool_name}")
            _metrics.counter(f"analysis.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="spec-analyze-deps",
    )
    def spec_analyze_deps(
        spec_id: str,
        bottleneck_threshold: Optional[int] = None,
        path: Optional[str] = None,
    ) -> dict:
        """
        Inspect dependency graph health for an SDD specification.

        Wraps the SDD CLI analyze-deps command to analyze dependency
        relationships between tasks, identify bottlenecks, and surface
        potential blocking issues.

        WHEN TO USE:
        - Identifying blocking tasks before starting work
        - Finding bottlenecks in the dependency graph
        - Understanding critical path for completion
        - Auditing spec structure for dependency issues

        Args:
            spec_id: Specification ID or path to analyze
            bottleneck_threshold: Minimum tasks blocked to flag as bottleneck (default: 3)
            path: Project root path (default: current directory)

        Returns:
            JSON object with dependency analysis:
            - dependency_count: Total number of dependencies
            - bottlenecks: Tasks blocking many others
            - circular_deps: Any circular dependency issues
            - critical_path: Tasks on the longest dependency chain
        """
        tool_name = "spec_analyze_deps"
        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter (e.g., my-feature-spec)",
                ))

            # Build command
            cmd = ["foundry-cli", "analyze-deps", spec_id, "--json"]

            if bottleneck_threshold is not None:
                cmd.extend(["--bottleneck-threshold", str(bottleneck_threshold)])

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-analyze-deps",
                action="analyze_dependencies",
                spec_id=spec_id,
            )

            # Execute the command with resilience
            result = _run_sdd_command(cmd, tool_name)

            # Parse the JSON output
            if result.returncode == 0:
                try:
                    output_data = json.loads(result.stdout) if result.stdout.strip() else {}
                except json.JSONDecodeError:
                    output_data = {"raw_output": result.stdout}

                # Track metrics
                _metrics.counter(f"analysis.{tool_name}", labels={"status": "success"})

                return asdict(success_response(
                    data=output_data,
                    message="Dependency analysis completed",
                ))
            else:
                # Handle specific error cases
                stderr = result.stderr.strip()

                if "not found" in stderr.lower():
                    error_code = "NOT_FOUND"
                    remediation = "Ensure the spec exists in specs/active or specs/pending"
                elif "circular" in stderr.lower():
                    error_code = "CIRCULAR_DEPENDENCY"
                    remediation = "Review and fix circular dependencies in the spec"
                else:
                    error_code = "ANALYSIS_FAILED"
                    remediation = "Check the spec_id and ensure the spec file is valid"

                _metrics.counter(f"analysis.{tool_name}", labels={"status": "error", "code": error_code})

                return asdict(error_response(
                    stderr or "Dependency analysis failed",
                    error_code=error_code,
                    error_type="analysis",
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
            _metrics.counter(f"analysis.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Dependency analysis timed out after {CLI_TIMEOUT}s",
                error_code="TIMEOUT",
                error_type="timeout",
                remediation="Try analyzing a smaller spec or increase timeout",
            ))

        except FileNotFoundError:
            _metrics.counter(f"analysis.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found. Ensure 'sdd' is installed and in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="configuration",
                remediation="Install SDD toolkit: pip install claude-sdd-toolkit",
            ))

        except Exception as e:
            logger.exception(f"Unexpected error in {tool_name}")
            _metrics.counter(f"analysis.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))
