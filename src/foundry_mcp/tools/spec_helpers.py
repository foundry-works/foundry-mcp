"""
Spec helper tools for foundry-mcp.

Provides MCP tools for spec discovery, validation, and analysis.
These tools wrap SDD CLI commands to provide file relationship discovery,
pattern matching, dependency cycle detection, and path validation.

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
from typing import Any, Dict, List, Optional

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

# Metrics singleton for spec helper tools
_metrics = get_metrics()

# Circuit breaker for SDD CLI operations
# Opens after 5 consecutive failures, recovers after 30 seconds
_sdd_cli_breaker = CircuitBreaker(
    name="sdd_cli",
    failure_threshold=5,
    recovery_timeout=30.0,
    half_open_max_calls=3,
)

# Default timeout for CLI operations (30 seconds)
CLI_TIMEOUT: float = MEDIUM_TIMEOUT


def _run_sdd_command(
    cmd: List[str],
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
        _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "circuit_open"})
        raise CircuitBreakerError(
            f"SDD CLI circuit breaker is open (retry after {status.get('retry_after_seconds', 0):.1f}s)",
            breaker_name="sdd_cli",
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
        _metrics.timer(f"spec_helpers.{tool_name}.duration_ms", elapsed_ms)


def register_spec_helper_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register spec helper tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="spec-find-related-files",
    )
    def spec_find_related_files(
        file_path: str,
        spec_id: Optional[str] = None,
        include_metadata: bool = False,
    ) -> dict:
        """
        Locate files referenced by a spec node or related to a source file.

        Wraps the SDD CLI find-related-files command to discover relationships
        between source files and specification nodes. Returns file paths that
        are referenced in spec metadata or structurally related.

        WHEN TO USE:
        - Finding files associated with a spec task
        - Discovering file relationships before making changes
        - Understanding spec-to-code mappings
        - Validating that referenced files exist

        Args:
            file_path: Source file path to find relationships for
            spec_id: Optional spec ID to narrow search scope
            include_metadata: Include additional metadata about relationships

        Returns:
            JSON object with related file information:
            - file_path: The queried file path
            - related_files: List of related file objects with paths and relationship types
            - spec_references: List of specs that reference this file
            - total_count: Number of related files found
        """
        tool_name = "find_related_files"
        try:
            # Build command
            cmd = ["sdd", "find-related-files", file_path, "--json"]

            if spec_id:
                cmd.extend(["--spec-id", spec_id])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-find-related-files",
                action="find_related",
                file_path=file_path,
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
                related_files: List[Dict[str, Any]] = output_data.get("related_files", [])
                spec_references: List[Dict[str, Any]] = output_data.get("spec_references", [])

                data: Dict[str, Any] = {
                    "file_path": file_path,
                    "related_files": related_files,
                    "spec_references": spec_references,
                    "total_count": len(related_files),
                }

                if include_metadata:
                    data["metadata"] = {
                        "command": " ".join(cmd),
                        "exit_code": result.returncode,
                    }

                # Track metrics
                _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "success"})

                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "error"})

                return asdict(error_response(
                    f"Failed to find related files: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the file path exists and SDD CLI is available",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try with a smaller scope or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in spec-find-related-files")
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="spec-find-patterns",
    )
    def spec_find_patterns(
        pattern: str,
        directory: Optional[str] = None,
        include_metadata: bool = False,
    ) -> dict:
        """
        Search specs and codebase for structural or code patterns.

        Wraps the SDD CLI find-pattern command to search across spec contents
        and source files using glob patterns. Returns matching files and locations.

        WHEN TO USE:
        - Finding files matching a specific pattern (e.g., "*.spec.ts")
        - Searching for structural patterns in the codebase
        - Discovering test files or configuration files
        - Auditing file organization

        Args:
            pattern: Glob pattern to search for (e.g., "*.ts", "src/**/*.spec.ts")
            directory: Optional directory to scope the search
            include_metadata: Include additional metadata about the search

        Returns:
            JSON object with pattern match results:
            - pattern: The search pattern used
            - matches: List of matching file paths
            - total_count: Number of matches found
        """
        tool_name = "find_patterns"
        try:
            # Build command
            cmd = ["sdd", "find-pattern", pattern, "--json"]

            if directory:
                cmd.extend(["--directory", directory])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-find-patterns",
                action="find_patterns",
                pattern=pattern,
                directory=directory,
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
                matches: List[str] = output_data.get("matches", output_data.get("files", []))

                data: Dict[str, Any] = {
                    "pattern": pattern,
                    "matches": matches,
                    "total_count": len(matches),
                }

                if directory:
                    data["directory"] = directory

                if include_metadata:
                    data["metadata"] = {
                        "command": " ".join(cmd),
                        "exit_code": result.returncode,
                    }

                # Track metrics
                _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "success"})

                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "error"})

                return asdict(error_response(
                    f"Failed to find patterns: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the pattern is valid and SDD CLI is available",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try with a more specific pattern or smaller scope",
            ))
        except FileNotFoundError:
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in spec-find-patterns")
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="spec-detect-cycles",
    )
    def spec_detect_cycles(
        spec_id: str,
        include_metadata: bool = False,
    ) -> dict:
        """
        Detect cyclic dependencies in a specification's task dependency graph.

        Wraps the SDD CLI find-circular-deps command to analyze task dependencies
        and identify any circular references that would prevent task completion.

        WHEN TO USE:
        - Validating a specification before starting implementation
        - Debugging blocked tasks that can't be started
        - Auditing dependency structure after spec modifications
        - Ensuring task graph is acyclic before phase planning

        Args:
            spec_id: The specification ID to analyze
            include_metadata: Include additional metadata about the analysis

        Returns:
            JSON object with cycle detection results:
            - spec_id: The analyzed specification ID
            - has_cycles: Boolean indicating if cycles were detected
            - cycles: List of detected cycles (each cycle is a list of task IDs)
            - cycle_count: Number of cycles detected
            - affected_tasks: List of task IDs involved in cycles
        """
        tool_name = "detect_cycles"
        try:
            # Build command
            cmd = ["sdd", "find-circular-deps", spec_id, "--json"]

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-detect-cycles",
                action="detect_cycles",
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
                cycles: List[List[str]] = output_data.get("cycles", [])
                affected_tasks: List[str] = output_data.get("affected_tasks", [])

                # If affected_tasks not provided, derive from cycles
                if not affected_tasks and cycles:
                    seen = set()
                    for cycle in cycles:
                        seen.update(cycle)
                    affected_tasks = list(seen)

                data: Dict[str, Any] = {
                    "spec_id": spec_id,
                    "has_cycles": len(cycles) > 0,
                    "cycles": cycles,
                    "cycle_count": len(cycles),
                    "affected_tasks": affected_tasks,
                }

                if include_metadata:
                    data["metadata"] = {
                        "command": " ".join(cmd),
                        "exit_code": result.returncode,
                    }

                # Track metrics
                _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "success"})

                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "error"})

                return asdict(error_response(
                    f"Failed to detect cycles: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the spec_id exists and SDD CLI is available",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try with a smaller specification or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in spec-detect-cycles")
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="spec-validate-paths",
    )
    def spec_validate_paths(
        paths: List[str],
        base_directory: Optional[str] = None,
        include_metadata: bool = False,
    ) -> dict:
        """
        Validate that file paths exist on disk.

        Wraps the SDD CLI validate-paths command to check that file references
        in specifications or code actually exist in the filesystem.

        WHEN TO USE:
        - Validating spec file references before implementation
        - Auditing broken file references after refactoring
        - Pre-flight checks before large-scale changes
        - Ensuring spec metadata file_path entries are current

        Args:
            paths: List of file paths to validate
            base_directory: Optional base directory for resolving relative paths
            include_metadata: Include additional metadata about the validation

        Returns:
            JSON object with path validation results:
            - paths_checked: Number of paths validated
            - valid_paths: List of paths that exist
            - invalid_paths: List of paths that do not exist
            - all_valid: Boolean indicating if all paths are valid
            - valid_count: Number of valid paths
            - invalid_count: Number of invalid paths
        """
        tool_name = "validate_paths"
        try:
            # Build command
            cmd = ["sdd", "validate-paths", "--json"] + paths

            if base_directory:
                cmd.extend(["--base-directory", base_directory])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-validate-paths",
                action="validate_paths",
                path_count=len(paths),
                base_directory=base_directory,
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
                valid_paths: List[str] = output_data.get("valid_paths", output_data.get("valid", []))
                invalid_paths: List[str] = output_data.get("invalid_paths", output_data.get("invalid", []))

                data: Dict[str, Any] = {
                    "paths_checked": len(paths),
                    "valid_paths": valid_paths,
                    "invalid_paths": invalid_paths,
                    "all_valid": len(invalid_paths) == 0,
                    "valid_count": len(valid_paths),
                    "invalid_count": len(invalid_paths),
                }

                if base_directory:
                    data["base_directory"] = base_directory

                if include_metadata:
                    data["metadata"] = {
                        "command": " ".join(cmd),
                        "exit_code": result.returncode,
                    }

                # Track metrics
                _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "success"})

                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "error"})

                return asdict(error_response(
                    f"Failed to validate paths: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that paths are valid and SDD CLI is available",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try with fewer paths or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in spec-validate-paths")
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))
