"""
Git integration tools for foundry-mcp.

Provides MCP tools for git-related SDD operations including task commits
and bulk journaling.

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

# Metrics singleton for git integration tools
_metrics = get_metrics()

# Circuit breaker for SDD CLI operations
# Opens after 5 consecutive failures, recovers after 30 seconds
_sdd_cli_breaker = CircuitBreaker(
    name="sdd_cli_git_integration",
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
        _metrics.counter(f"git_integration.{tool_name}", labels={"status": "circuit_open"})
        raise CircuitBreakerError(
            f"SDD CLI circuit breaker is open (retry after {status.get('retry_after_seconds', 0):.1f}s)",
            breaker_name="sdd_cli_git_integration",
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
        _metrics.timer(f"git_integration.{tool_name}.duration_ms", elapsed_ms)


def register_git_integration_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register git integration tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="task-create-commit",
    )
    def task_create_commit(
        spec_id: str,
        task_id: str,
        skip_status_check: bool = False,
        force: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Generate a git commit for task-scoped changes.

        Wraps the SDD CLI create-task-commit command to create a git commit
        with proper task context including spec ID, task ID, and task metadata
        in the commit message.

        WHEN TO USE:
        - Creating commits for completed task work
        - Generating structured commit messages with task context
        - Maintaining traceability between commits and spec tasks
        - Automating commit creation in CI/CD pipelines

        Args:
            spec_id: Specification ID containing the task
            task_id: Task ID to create commit for
            skip_status_check: Skip checking if task is completed
            force: Force commit even if task is not completed
            path: Project root path (default: current directory)

        Returns:
            JSON object with commit results:
            - spec_id: The specification ID
            - task_id: The task ID
            - commit_hash: The created commit hash
            - commit_message: The generated commit message
            - files_committed: List of files included in commit
        """
        tool_name = "task_create_commit"
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
            cmd = ["foundry-cli", "create-task-commit", spec_id, task_id, "--json"]

            if skip_status_check:
                cmd.append("--skip-status-check")

            if force:
                cmd.append("--force")

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="task-create-commit",
                action="create_commit",
                spec_id=spec_id,
                task_id=task_id,
                force=force,
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
                }

                # Include commit details
                if "commit_hash" in output_data:
                    data["commit_hash"] = output_data["commit_hash"]
                elif "hash" in output_data:
                    data["commit_hash"] = output_data["hash"]

                if "commit_message" in output_data:
                    data["commit_message"] = output_data["commit_message"]
                elif "message" in output_data:
                    data["commit_message"] = output_data["message"]

                if "files_committed" in output_data:
                    data["files_committed"] = output_data["files_committed"]
                elif "files" in output_data:
                    data["files_committed"] = output_data["files"]

                # Include warnings if any
                warnings = []
                if output_data.get("warnings"):
                    warnings = output_data["warnings"]

                # Track metrics
                _metrics.counter(f"git_integration.{tool_name}", labels={"status": "success"})

                if warnings:
                    return asdict(success_response(data, warnings=warnings))
                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"git_integration.{tool_name}", labels={"status": "error"})

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

                if "not completed" in error_msg.lower():
                    return asdict(error_response(
                        f"Task '{task_id}' is not completed",
                        error_code="TASK_NOT_COMPLETED",
                        error_type="validation",
                        remediation="Complete the task first or use force=True to override",
                    ))

                if "nothing to commit" in error_msg.lower():
                    return asdict(error_response(
                        "No changes to commit",
                        error_code="NO_CHANGES",
                        error_type="validation",
                        remediation="Make changes to files before creating a commit",
                    ))

                if "git" in error_msg.lower() and "not a repository" in error_msg.lower():
                    return asdict(error_response(
                        "Not a git repository",
                        error_code="NOT_GIT_REPO",
                        error_type="validation",
                        remediation="Initialize a git repository first",
                    ))

                return asdict(error_response(
                    f"Failed to create commit: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the spec and task exist and git is configured",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"git_integration.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try again or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"git_integration.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in task-create-commit")
            _metrics.counter(f"git_integration.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="journal-bulk-add",
    )
    def journal_bulk_add(
        spec_id: str,
        tasks: Optional[str] = None,
        template: Optional[str] = None,
        template_author: Optional[str] = None,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Add multiple journal entries in one shot.

        Wraps the SDD CLI bulk-journal command to add journal entries
        to multiple tasks at once. Can use templates for consistent
        journal entry formatting.

        WHEN TO USE:
        - Journaling multiple completed tasks at once
        - Applying consistent journal templates across tasks
        - Bulk updating unjournaled tasks
        - Automating journal entry creation

        Args:
            spec_id: Specification ID to journal
            tasks: Comma-separated list of task IDs (if omitted, journals all unjournaled tasks)
            template: Journal template to apply (completion, decision, blocker)
            template_author: Override author for templated entries
            dry_run: Preview journal entries without saving
            path: Project root path (default: current directory)

        Returns:
            JSON object with journaling results:
            - spec_id: The specification ID
            - tasks_journaled: Number of tasks that received journal entries
            - task_ids: List of task IDs that were journaled
            - template_used: Template that was applied (if any)
            - dry_run: Whether this was a dry run
        """
        tool_name = "journal_bulk_add"
        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            # Validate template if provided
            valid_templates = ("completion", "decision", "blocker")
            if template and template not in valid_templates:
                return asdict(error_response(
                    f"Invalid template '{template}'. Must be one of: {', '.join(valid_templates)}",
                    error_code="VALIDATION_ERROR",
                    error_type="validation",
                    remediation=f"Use one of: {', '.join(valid_templates)}",
                ))

            # Build command
            cmd = ["foundry-cli", "bulk-journal", spec_id, "--json"]

            if tasks:
                cmd.extend(["--tasks", tasks])

            if template:
                cmd.extend(["--template", template])

            if template_author:
                cmd.extend(["--template-author", template_author])

            if dry_run:
                cmd.append("--dry-run")

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="journal-bulk-add",
                action="bulk_journal",
                spec_id=spec_id,
                tasks=tasks,
                template=template,
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
                    "dry_run": dry_run,
                }

                # Include journaling results
                if "tasks_journaled" in output_data:
                    data["tasks_journaled"] = output_data["tasks_journaled"]
                elif "count" in output_data:
                    data["tasks_journaled"] = output_data["count"]
                elif "journaled" in output_data:
                    data["tasks_journaled"] = len(output_data["journaled"]) if isinstance(output_data["journaled"], list) else output_data["journaled"]

                if "task_ids" in output_data:
                    data["task_ids"] = output_data["task_ids"]
                elif "journaled" in output_data and isinstance(output_data["journaled"], list):
                    data["task_ids"] = output_data["journaled"]

                if template:
                    data["template_used"] = template

                # Include warnings if any
                warnings = []
                if output_data.get("warnings"):
                    warnings = output_data["warnings"]
                if output_data.get("skipped"):
                    warnings.append(f"{len(output_data['skipped'])} tasks were skipped")

                # Track metrics
                _metrics.counter(f"git_integration.{tool_name}", labels={
                    "status": "success",
                    "dry_run": str(dry_run),
                    "has_template": str(bool(template)),
                })

                if warnings:
                    return asdict(success_response(data, warnings=warnings))
                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"git_integration.{tool_name}", labels={"status": "error"})

                # Check for common errors
                if "not found" in error_msg.lower():
                    if "spec" in error_msg.lower():
                        return asdict(error_response(
                            f"Specification '{spec_id}' not found",
                            error_code="SPEC_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the spec ID exists using spec-list",
                        ))

                if "no tasks" in error_msg.lower() or "nothing to journal" in error_msg.lower():
                    return asdict(success_response({
                        "spec_id": spec_id,
                        "tasks_journaled": 0,
                        "task_ids": [],
                        "dry_run": dry_run,
                        "message": "No unjournaled tasks found",
                    }))

                return asdict(error_response(
                    f"Failed to bulk journal: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the spec exists and tasks are valid",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"git_integration.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try again or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"git_integration.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in journal-bulk-add")
            _metrics.counter(f"git_integration.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))
