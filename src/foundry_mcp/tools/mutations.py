"""
Mutation tools for foundry-mcp.

Provides MCP tools for batch modifications to SDD specifications.
These tools wrap SDD CLI commands for applying plans, verification,
and metadata mutations.

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
from foundry_mcp.core.modifications import apply_modifications, load_modifications_file
from foundry_mcp.core.spec import find_specs_directory

logger = logging.getLogger(__name__)

# Metrics singleton for mutation tools
_metrics = get_metrics()

# Circuit breaker for SDD CLI operations
# Opens after 5 consecutive failures, recovers after 30 seconds
_sdd_cli_breaker = CircuitBreaker(
    name="sdd_cli_mutations",
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
        _metrics.counter(f"mutations.{tool_name}", labels={"status": "circuit_open"})
        raise CircuitBreakerError(
            f"SDD CLI circuit breaker is open (retry after {status.get('retry_after_seconds', 0):.1f}s)",
            breaker_name="sdd_cli_mutations",
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
        _metrics.timer(f"mutations.{tool_name}.duration_ms", elapsed_ms)


def register_mutation_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register mutation tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="spec-apply-plan",
    )
    def spec_apply_plan(
        spec_id: str,
        modifications_file: str,
        dry_run: bool = False,
        output_file: Optional[str] = None,
        path: Optional[str] = None,
    ) -> dict:
        """
        Apply bulk structural edits from a diff/plan JSON file.

        Wraps the SDD CLI apply-modifications command to batch apply changes
        to a specification from a JSON file. The modifications file contains
        structured changes like task additions, updates, removals, and
        metadata modifications.

        WHEN TO USE:
        - Applying review feedback to a specification
        - Batch updating multiple tasks from parsed review output
        - Migrating specs with automated transformations
        - Applying AI-generated spec modifications
        - Implementing bulk changes from sdd parse-review output

        Args:
            spec_id: Specification ID to modify
            modifications_file: Path to JSON file containing modifications
            dry_run: Preview changes without applying them
            output_file: Output path for modified spec (default: overwrite original)
            path: Project root path (default: current directory)

        Returns:
            JSON object with modification results:
            - spec_id: The specification ID
            - modifications_applied: Number of modifications applied
            - modifications_skipped: Number of modifications skipped
            - changes: Array of change summaries
            - dry_run: Whether this was a dry run
            - output_path: Path where modified spec was written
        """
        tool_name = "spec_apply_plan"
        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            if not modifications_file:
                return asdict(error_response(
                    "modifications_file is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a path to the modifications JSON file",
                ))

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-apply-plan",
                action="apply_modifications",
                spec_id=spec_id,
                modifications_file=modifications_file,
                dry_run=dry_run,
            )

            # Load modifications from file
            try:
                modifications = load_modifications_file(modifications_file)
            except FileNotFoundError:
                _metrics.counter(f"mutations.{tool_name}", labels={"status": "file_not_found"})
                return asdict(error_response(
                    f"Modifications file not found: {modifications_file}",
                    error_code="FILE_NOT_FOUND",
                    error_type="not_found",
                    remediation="Verify the modifications file path exists",
                ))
            except json.JSONDecodeError as e:
                _metrics.counter(f"mutations.{tool_name}", labels={"status": "invalid_json"})
                return asdict(error_response(
                    f"Invalid modifications JSON format: {str(e)}",
                    error_code="VALIDATION_ERROR",
                    error_type="validation",
                    remediation="Ensure the modifications file contains valid JSON",
                ))

            # Find specs directory
            specs_dir = find_specs_directory(path)
            if not specs_dir:
                _metrics.counter(f"mutations.{tool_name}", labels={"status": "specs_not_found"})
                return asdict(error_response(
                    "Could not find specs directory",
                    error_code="SPECS_NOT_FOUND",
                    error_type="not_found",
                    remediation="Ensure you are in a project with a specs/ directory",
                ))

            # Apply modifications using direct Python API
            try:
                applied, skipped, changes = apply_modifications(
                    spec_id=spec_id,
                    modifications=modifications,
                    specs_dir=specs_dir,
                    dry_run=dry_run,
                )
            except ValueError as e:
                _metrics.counter(f"mutations.{tool_name}", labels={"status": "spec_not_found"})
                return asdict(error_response(
                    str(e),
                    error_code="SPEC_NOT_FOUND",
                    error_type="not_found",
                    remediation="Verify the spec ID exists using spec-list",
                ))

            # Build response data
            data: Dict[str, Any] = {
                "spec_id": spec_id,
                "modifications_applied": applied,
                "modifications_skipped": skipped,
                "changes": changes,
                "dry_run": dry_run,
            }

            # Include output path if specified
            if output_file:
                data["output_path"] = output_file

            # Include warnings if any modifications were skipped
            warnings = []
            if skipped > 0:
                warnings.append(f"{skipped} modifications were skipped")

            # Track metrics
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "success", "dry_run": str(dry_run)})

            if warnings:
                return asdict(success_response(data, warnings=warnings))
            return asdict(success_response(data))

        except Exception as e:
            logger.exception("Unexpected error in spec-apply-plan")
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="verification-add",
    )
    def verification_add(
        spec_id: str,
        verify_id: str,
        result: str,
        command: Optional[str] = None,
        output: Optional[str] = None,
        issues: Optional[str] = None,
        notes: Optional[str] = None,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Add verification result to a task or spec verification node.

        Wraps the SDD CLI add-verification command to record verification
        results including test outcomes, command output, and issues found.

        WHEN TO USE:
        - Recording test execution results
        - Documenting manual verification outcomes
        - Adding verification data to verify-type tasks
        - Capturing CI/CD pipeline verification results

        Args:
            spec_id: Specification ID containing the verification
            verify_id: Verification ID (e.g., verify-1-1)
            result: Verification result (PASSED, FAILED, PARTIAL)
            command: Command that was run for verification
            output: Command output or test results
            issues: Issues found during verification
            notes: Additional notes about the verification
            dry_run: Preview changes without applying them
            path: Project root path (default: current directory)

        Returns:
            JSON object with verification results:
            - spec_id: The specification ID
            - verify_id: The verification ID
            - result: Verification result
            - command: Command run (if provided)
            - dry_run: Whether this was a dry run
        """
        tool_name = "verification_add"
        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            if not verify_id:
                return asdict(error_response(
                    "verify_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a verify_id parameter (e.g., verify-1-1)",
                ))

            # Validate result
            valid_results = ("PASSED", "FAILED", "PARTIAL")
            if result not in valid_results:
                return asdict(error_response(
                    f"Invalid result '{result}'. Must be one of: {', '.join(valid_results)}",
                    error_code="VALIDATION_ERROR",
                    error_type="validation",
                    remediation=f"Use one of: {', '.join(valid_results)}",
                ))

            # Build command
            cmd = ["foundry-cli", "add-verification", spec_id, verify_id, result, "--json"]

            if command:
                cmd.extend(["--command", command])

            if output:
                cmd.extend(["--output", output])

            if issues:
                cmd.extend(["--issues", issues])

            if notes:
                cmd.extend(["--notes", notes])

            if dry_run:
                cmd.append("--dry-run")

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="verification-add",
                action="add_verification",
                spec_id=spec_id,
                verify_id=verify_id,
                result=result,
                dry_run=dry_run,
            )

            # Execute the command with resilience
            result_proc = _run_sdd_command(cmd, tool_name)

            # Parse the JSON output
            if result_proc.returncode == 0:
                try:
                    output_data = json.loads(result_proc.stdout) if result_proc.stdout.strip() else {}
                except json.JSONDecodeError:
                    output_data = {}

                # Build response data
                data: Dict[str, Any] = {
                    "spec_id": spec_id,
                    "verify_id": verify_id,
                    "result": result,
                    "dry_run": dry_run,
                }

                if command:
                    data["command"] = command

                if output_data.get("timestamp"):
                    data["timestamp"] = output_data["timestamp"]

                # Track metrics
                _metrics.counter(f"mutations.{tool_name}", labels={"status": "success", "result": result})

                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result_proc.stderr.strip() if result_proc.stderr else "Command failed"
                _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})

                # Check for common errors
                if "not found" in error_msg.lower():
                    if "spec" in error_msg.lower():
                        return asdict(error_response(
                            f"Specification '{spec_id}' not found",
                            error_code="SPEC_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the spec ID exists using spec-list",
                        ))
                    elif "verify" in error_msg.lower() or verify_id in error_msg:
                        return asdict(error_response(
                            f"Verification '{verify_id}' not found in spec",
                            error_code="NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the verification ID exists in the specification",
                        ))

                return asdict(error_response(
                    f"Failed to add verification: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the spec and verification ID exist",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try again or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in verification-add")
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="verification-execute",
    )
    def verification_execute(
        spec_id: str,
        verify_id: str,
        record: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Execute verification steps defined in a verification node.

        Wraps the SDD CLI execute-verify command to run the verification
        commands defined in a verification task and capture their output.
        Optionally records the result back to the spec.

        WHEN TO USE:
        - Running automated verification tests
        - Executing verification commands from a spec
        - Validating task completion with defined checks
        - Running CI/CD verification steps

        Args:
            spec_id: Specification ID containing the verification
            verify_id: Verification ID to execute (e.g., verify-1-1)
            record: Automatically record result to spec
            path: Project root path (default: current directory)

        Returns:
            JSON object with execution results:
            - spec_id: The specification ID
            - verify_id: The verification ID
            - result: Execution result (PASSED, FAILED, PARTIAL)
            - command: Command that was executed
            - output: Command output
            - exit_code: Command exit code
            - recorded: Whether result was recorded to spec
        """
        tool_name = "verification_execute"
        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            if not verify_id:
                return asdict(error_response(
                    "verify_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a verify_id parameter (e.g., verify-1-1)",
                ))

            # Build command
            cmd = ["foundry-cli", "execute-verify", spec_id, verify_id, "--json"]

            if record:
                cmd.append("--record")

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="verification-execute",
                action="execute_verification",
                spec_id=spec_id,
                verify_id=verify_id,
                record=record,
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
                data: Dict[str, Any] = {
                    "spec_id": spec_id,
                    "verify_id": verify_id,
                    "result": output_data.get("result", output_data.get("status", "UNKNOWN")),
                    "recorded": record,
                }

                if output_data.get("command"):
                    data["command"] = output_data["command"]

                if output_data.get("output"):
                    data["output"] = output_data["output"]

                if "exit_code" in output_data:
                    data["exit_code"] = output_data["exit_code"]

                if output_data.get("timestamp"):
                    data["timestamp"] = output_data["timestamp"]

                # Track metrics
                _metrics.counter(f"mutations.{tool_name}", labels={
                    "status": "success",
                    "result": data.get("result", "UNKNOWN"),
                })

                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})

                # Check for common errors
                if "not found" in error_msg.lower():
                    if "spec" in error_msg.lower():
                        return asdict(error_response(
                            f"Specification '{spec_id}' not found",
                            error_code="SPEC_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the spec ID exists using spec-list",
                        ))
                    elif "verify" in error_msg.lower() or verify_id in error_msg:
                        return asdict(error_response(
                            f"Verification '{verify_id}' not found in spec",
                            error_code="NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the verification ID exists in the specification",
                        ))

                if "no command" in error_msg.lower():
                    return asdict(error_response(
                        f"Verification '{verify_id}' has no command defined",
                        error_code="NO_COMMAND",
                        error_type="validation",
                        remediation="Add a command to the verification before executing",
                    ))

                return asdict(error_response(
                    f"Failed to execute verification: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the spec and verification ID exist",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try again or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in verification-execute")
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="verification-format-summary",
    )
    def verification_format_summary(
        json_file: Optional[str] = None,
        json_input: Optional[str] = None,
        path: Optional[str] = None,
    ) -> dict:
        """
        Format verification results into a human-readable summary.

        Wraps the SDD CLI format-verification-summary command to produce
        formatted summaries of verification results from JSON data.

        WHEN TO USE:
        - Generating human-readable test reports
        - Formatting verification results for display
        - Creating summary reports from raw verification data
        - Preparing verification output for documentation

        Args:
            json_file: Path to JSON file with verification results
            json_input: JSON string with verification results (alternative to file)
            path: Project root path (default: current directory)

        Returns:
            JSON object with formatted summary:
            - summary: Human-readable summary text
            - total_verifications: Total number of verifications
            - passed: Number of passed verifications
            - failed: Number of failed verifications
            - partial: Number of partial verifications
        """
        tool_name = "verification_format_summary"
        try:
            # Validate that exactly one input source is provided
            if not json_file and not json_input:
                return asdict(error_response(
                    "Either json_file or json_input is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide either json_file (path) or json_input (JSON string)",
                ))

            if json_file and json_input:
                return asdict(error_response(
                    "Only one of json_file or json_input should be provided",
                    error_code="VALIDATION_ERROR",
                    error_type="validation",
                    remediation="Provide either json_file or json_input, not both",
                ))

            # Build command
            cmd = ["foundry-cli", "format-verification-summary", "--json"]

            if json_file:
                cmd.extend(["--json-file", json_file])
            elif json_input:
                cmd.extend(["--json-input", json_input])

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="verification-format-summary",
                action="format_summary",
                has_file=bool(json_file),
                has_input=bool(json_input),
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
                data: Dict[str, Any] = {
                    "summary": output_data.get("summary", output_data.get("formatted", "")),
                }

                # Include stats if available
                if "total_verifications" in output_data:
                    data["total_verifications"] = output_data["total_verifications"]
                if "passed" in output_data:
                    data["passed"] = output_data["passed"]
                if "failed" in output_data:
                    data["failed"] = output_data["failed"]
                if "partial" in output_data:
                    data["partial"] = output_data["partial"]

                # Include raw results if available
                if "results" in output_data:
                    data["results"] = output_data["results"]

                # Track metrics
                _metrics.counter(f"mutations.{tool_name}", labels={"status": "success"})

                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})

                # Check for common errors
                if "not found" in error_msg.lower() and json_file:
                    return asdict(error_response(
                        f"JSON file not found: {json_file}",
                        error_code="FILE_NOT_FOUND",
                        error_type="not_found",
                        remediation="Verify the JSON file path exists",
                    ))

                if "invalid" in error_msg.lower() and "json" in error_msg.lower():
                    return asdict(error_response(
                        "Invalid JSON format in input",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Ensure the input contains valid JSON",
                    ))

                return asdict(error_response(
                    f"Failed to format verification summary: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the input JSON is valid",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try again or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in verification-format-summary")
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="task-update-estimate",
    )
    def task_update_estimate(
        spec_id: str,
        task_id: str,
        hours: Optional[float] = None,
        complexity: Optional[str] = None,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Update effort/time estimates for a task.

        Wraps the SDD CLI update-estimate command to modify estimated_hours
        and complexity metadata for a task.

        WHEN TO USE:
        - Adjusting task estimates based on new information
        - Recording actual vs estimated time
        - Updating complexity assessments mid-implementation
        - Re-estimating work after scope changes

        Args:
            spec_id: Specification ID containing the task
            task_id: Task ID to update
            hours: Estimated hours (float)
            complexity: Complexity level (low, medium, high)
            dry_run: Preview changes without saving
            path: Project root path (default: current directory)

        Returns:
            JSON object with update results:
            - spec_id: The specification ID
            - task_id: The task ID
            - hours: Updated hours estimate (if provided)
            - complexity: Updated complexity (if provided)
            - previous_hours: Previous hours estimate
            - previous_complexity: Previous complexity
            - dry_run: Whether this was a dry run
        """
        tool_name = "task_update_estimate"
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

            # Validate at least one update field is provided
            if hours is None and complexity is None:
                return asdict(error_response(
                    "At least one of hours or complexity must be provided",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide hours and/or complexity to update",
                ))

            # Validate complexity if provided
            if complexity and complexity not in ("low", "medium", "high"):
                return asdict(error_response(
                    f"Invalid complexity '{complexity}'. Must be one of: low, medium, high",
                    error_code="VALIDATION_ERROR",
                    error_type="validation",
                    remediation="Use one of: low, medium, high",
                ))

            # Build command
            cmd = ["foundry-cli", "update-estimate", spec_id, task_id, "--json"]

            if hours is not None:
                cmd.extend(["--hours", str(hours)])

            if complexity:
                cmd.extend(["--complexity", complexity])

            if dry_run:
                cmd.append("--dry-run")

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="task-update-estimate",
                action="update_estimate",
                spec_id=spec_id,
                task_id=task_id,
                hours=hours,
                complexity=complexity,
                dry_run=dry_run,
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
                data: Dict[str, Any] = {
                    "spec_id": spec_id,
                    "task_id": task_id,
                    "dry_run": dry_run,
                }

                if hours is not None:
                    data["hours"] = hours
                if complexity:
                    data["complexity"] = complexity

                # Include previous values if available
                if "previous_hours" in output_data:
                    data["previous_hours"] = output_data["previous_hours"]
                if "previous_complexity" in output_data:
                    data["previous_complexity"] = output_data["previous_complexity"]

                # Track metrics
                _metrics.counter(f"mutations.{tool_name}", labels={"status": "success"})

                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})

                # Check for common errors
                if "not found" in error_msg.lower():
                    if "spec" in error_msg.lower():
                        return asdict(error_response(
                            f"Specification '{spec_id}' not found",
                            error_code="SPEC_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the spec ID exists using spec-list",
                        ))
                    elif "task" in error_msg.lower() or task_id in error_msg:
                        return asdict(error_response(
                            f"Task '{task_id}' not found in spec",
                            error_code="TASK_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the task ID exists in the specification",
                        ))

                return asdict(error_response(
                    f"Failed to update estimate: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the spec and task ID exist",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try again or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in task-update-estimate")
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="task-update-metadata",
    )
    def task_update_metadata(
        spec_id: str,
        task_id: str,
        file_path: Optional[str] = None,
        description: Optional[str] = None,
        task_category: Optional[str] = None,
        actual_hours: Optional[float] = None,
        status_note: Optional[str] = None,
        verification_type: Optional[str] = None,
        command: Optional[str] = None,
        metadata_json: Optional[str] = None,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Update arbitrary metadata fields for a task.

        Wraps the SDD CLI update-task-metadata command to mutate per-task
        metadata payloads. Supports both named fields and custom JSON metadata.

        WHEN TO USE:
        - Updating task file paths after implementation
        - Recording actual hours spent
        - Adding status notes or completion notes
        - Setting verification type and commands
        - Storing custom metadata fields

        Args:
            spec_id: Specification ID containing the task
            task_id: Task ID to update
            file_path: File path for this task
            description: Task description
            task_category: Task category (implementation, testing, etc.)
            actual_hours: Actual hours spent on task
            status_note: Status note or completion note
            verification_type: Verification type (auto, manual, none)
            command: Command executed
            metadata_json: JSON string with custom metadata fields
            dry_run: Preview changes without saving
            path: Project root path (default: current directory)

        Returns:
            JSON object with update results:
            - spec_id: The specification ID
            - task_id: The task ID
            - fields_updated: List of fields that were updated
            - dry_run: Whether this was a dry run
        """
        tool_name = "task_update_metadata"
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

            # Validate at least one update field is provided
            has_update = any([
                file_path, description, task_category, actual_hours is not None,
                status_note, verification_type, command, metadata_json
            ])
            if not has_update:
                return asdict(error_response(
                    "At least one metadata field must be provided",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide at least one field to update",
                ))

            # Validate verification_type if provided
            if verification_type and verification_type not in ("auto", "manual", "none"):
                return asdict(error_response(
                    f"Invalid verification_type '{verification_type}'. Must be one of: auto, manual, none",
                    error_code="VALIDATION_ERROR",
                    error_type="validation",
                    remediation="Use one of: auto, manual, none",
                ))

            # Build command
            cmd = ["foundry-cli", "update-task-metadata", spec_id, task_id, "--json"]

            fields_updated = []
            if file_path:
                cmd.extend(["--file-path", file_path])
                fields_updated.append("file_path")
            if description:
                cmd.extend(["--description", description])
                fields_updated.append("description")
            if task_category:
                cmd.extend(["--task-category", task_category])
                fields_updated.append("task_category")
            if actual_hours is not None:
                cmd.extend(["--actual-hours", str(actual_hours)])
                fields_updated.append("actual_hours")
            if status_note:
                cmd.extend(["--status-note", status_note])
                fields_updated.append("status_note")
            if verification_type:
                cmd.extend(["--verification-type", verification_type])
                fields_updated.append("verification_type")
            if command:
                cmd.extend(["--command", command])
                fields_updated.append("command")
            if metadata_json:
                cmd.extend(["--metadata", metadata_json])
                fields_updated.append("custom_metadata")

            if dry_run:
                cmd.append("--dry-run")

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="task-update-metadata",
                action="update_metadata",
                spec_id=spec_id,
                task_id=task_id,
                fields_updated=fields_updated,
                dry_run=dry_run,
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
                data: Dict[str, Any] = {
                    "spec_id": spec_id,
                    "task_id": task_id,
                    "fields_updated": fields_updated,
                    "dry_run": dry_run,
                }

                # Include previous values if available
                if "previous_values" in output_data:
                    data["previous_values"] = output_data["previous_values"]

                # Track metrics
                _metrics.counter(f"mutations.{tool_name}", labels={
                    "status": "success",
                    "field_count": str(len(fields_updated)),
                })

                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})

                # Check for common errors
                if "not found" in error_msg.lower():
                    if "spec" in error_msg.lower():
                        return asdict(error_response(
                            f"Specification '{spec_id}' not found",
                            error_code="SPEC_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the spec ID exists using spec-list",
                        ))
                    elif "task" in error_msg.lower() or task_id in error_msg:
                        return asdict(error_response(
                            f"Task '{task_id}' not found in spec",
                            error_code="TASK_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the task ID exists in the specification",
                        ))

                if "invalid" in error_msg.lower() and "json" in error_msg.lower():
                    return asdict(error_response(
                        "Invalid JSON in metadata_json parameter",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Ensure metadata_json contains valid JSON",
                    ))

                return asdict(error_response(
                    f"Failed to update metadata: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the spec and task ID exist",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try again or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in task-update-metadata")
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="spec-sync-metadata",
    )
    def spec_sync_metadata(
        spec_id: str,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Synchronize spec metadata across stores.

        Wraps the SDD CLI sync-metadata command to push/pull spec metadata
        between locations, ensuring consistency across different storage
        backends or documentation caches.

        WHEN TO USE:
        - Syncing spec metadata after bulk edits
        - Refreshing documentation cache with latest spec data
        - Ensuring consistency between spec file and derived artifacts
        - Propagating metadata changes to downstream consumers

        Args:
            spec_id: Specification ID to sync
            dry_run: Preview changes without applying them
            path: Project root path (default: current directory)

        Returns:
            JSON object with sync results:
            - spec_id: The specification ID
            - synced: Whether sync was successful
            - changes: List of metadata changes synced
            - dry_run: Whether this was a dry run
        """
        tool_name = "spec_sync_metadata"
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
            cmd = ["foundry-cli", "sync-metadata", spec_id, "--json"]

            if dry_run:
                cmd.append("--dry-run")

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-sync-metadata",
                action="sync_metadata",
                spec_id=spec_id,
                dry_run=dry_run,
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
                data: Dict[str, Any] = {
                    "spec_id": spec_id,
                    "synced": True,
                    "dry_run": dry_run,
                }

                # Include changes if available
                if "changes" in output_data:
                    data["changes"] = output_data["changes"]
                elif "synced_fields" in output_data:
                    data["changes"] = output_data["synced_fields"]

                # Include source/target info if available
                if "source" in output_data:
                    data["source"] = output_data["source"]
                if "target" in output_data:
                    data["target"] = output_data["target"]

                # Include warnings if any
                warnings = []
                if output_data.get("warnings"):
                    warnings = output_data["warnings"]

                # Track metrics
                _metrics.counter(f"mutations.{tool_name}", labels={
                    "status": "success",
                    "dry_run": str(dry_run),
                })

                if warnings:
                    return asdict(success_response(data, warnings=warnings))
                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})

                # Check for common errors
                if "not found" in error_msg.lower():
                    return asdict(error_response(
                        f"Specification '{spec_id}' not found",
                        error_code="SPEC_NOT_FOUND",
                        error_type="not_found",
                        remediation="Verify the spec ID exists using spec-list",
                    ))

                if "no changes" in error_msg.lower():
                    # No changes to sync is not an error
                    return asdict(success_response({
                        "spec_id": spec_id,
                        "synced": True,
                        "changes": [],
                        "dry_run": dry_run,
                        "message": "No metadata changes to sync",
                    }))

                if "permission" in error_msg.lower() or "access" in error_msg.lower():
                    return asdict(error_response(
                        f"Permission denied while syncing metadata: {error_msg}",
                        error_code="PERMISSION_DENIED",
                        error_type="validation",
                        remediation="Check file permissions and access rights",
                    ))

                return asdict(error_response(
                    f"Failed to sync metadata: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the spec exists and is accessible",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try again or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in spec-sync-metadata")
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))