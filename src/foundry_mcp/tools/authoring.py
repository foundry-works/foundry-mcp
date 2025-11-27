"""
Authoring tools for foundry-mcp.

Provides MCP tools for creating and modifying SDD specifications.
These tools wrap SDD CLI commands for spec creation, task management,
and metadata operations.

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

# Metrics singleton for authoring tools
_metrics = get_metrics()

# Circuit breaker for SDD CLI operations
# Opens after 5 consecutive failures, recovers after 30 seconds
_sdd_cli_breaker = CircuitBreaker(
    name="sdd_cli_authoring",
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
        _metrics.counter(f"authoring.{tool_name}", labels={"status": "circuit_open"})
        raise CircuitBreakerError(
            f"SDD CLI circuit breaker is open (retry after {status.get('retry_after_seconds', 0):.1f}s)",
            breaker_name="sdd_cli_authoring",
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
        _metrics.timer(f"authoring.{tool_name}.duration_ms", elapsed_ms)


def register_authoring_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register authoring tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="spec-create",
    )
    def spec_create(
        name: str,
        template: Optional[str] = None,
        category: Optional[str] = None,
        path: Optional[str] = None,
    ) -> dict:
        """
        Scaffold a brand-new SDD specification from scratch.

        Wraps the SDD CLI create command to generate a new specification file
        with the default hierarchy structure. The specification will be created
        in the specs/pending directory.

        WHEN TO USE:
        - Starting a new feature implementation
        - Creating a specification for a refactoring effort
        - Setting up a decision record or investigation spec
        - Initializing a project with SDD methodology

        Args:
            name: Specification name (will be used to generate spec ID)
            template: Template to use (simple, medium, complex, security). Default: medium
            category: Default task category (investigation, implementation, refactoring, decision, research)
            path: Project root path (default: current directory)

        Returns:
            JSON object with creation results:
            - spec_id: The generated specification ID
            - spec_path: Path to the created specification file
            - template: Template used for creation
            - category: Task category applied
            - structure: Overview of the generated spec structure
        """
        tool_name = "spec_create"
        try:
            # Build command
            cmd = ["sdd", "create", name, "--json"]

            if template:
                if template not in ("simple", "medium", "complex", "security"):
                    return asdict(error_response(
                        f"Invalid template '{template}'. Must be one of: simple, medium, complex, security",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Use one of: simple, medium, complex, security",
                    ))
                cmd.extend(["--template", template])

            if category:
                if category not in ("investigation", "implementation", "refactoring", "decision", "research"):
                    return asdict(error_response(
                        f"Invalid category '{category}'. Must be one of: investigation, implementation, refactoring, decision, research",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Use one of: investigation, implementation, refactoring, decision, research",
                    ))
                cmd.extend(["--category", category])

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-create",
                action="create_spec",
                name=name,
                template=template,
                category=category,
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
                    "spec_id": output_data.get("spec_id", output_data.get("id")),
                    "spec_path": output_data.get("spec_path", output_data.get("path")),
                    "template": template or "medium",
                    "name": name,
                }

                if category:
                    data["category"] = category

                # Include structure info if available
                if "structure" in output_data:
                    data["structure"] = output_data["structure"]
                elif "phases" in output_data:
                    data["structure"] = {
                        "phases": len(output_data.get("phases", [])),
                        "tasks": output_data.get("task_count", 0),
                    }

                # Track metrics
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "success"})

                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})

                # Check for common errors
                if "already exists" in error_msg.lower():
                    return asdict(error_response(
                        f"A specification with name '{name}' already exists",
                        error_code="DUPLICATE_ENTRY",
                        error_type="conflict",
                        remediation="Use a different name or update the existing spec",
                    ))

                return asdict(error_response(
                    f"Failed to create specification: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the SDD CLI is available and the project path is valid",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try again or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in spec-create")
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="spec-template",
    )
    def spec_template(
        action: str,
        template_name: Optional[str] = None,
        path: Optional[str] = None,
    ) -> dict:
        """
        Emit opinionated templates/snippets for spec sections.

        Wraps the SDD CLI template command to list available templates,
        show template contents, or apply templates to generate spec sections.
        Provides phase, task, and subtask templates for common patterns.

        WHEN TO USE:
        - Discovering available spec templates
        - Viewing template structure before applying
        - Generating consistent spec sections (phases, tasks, subtasks)
        - Applying best-practice patterns to specifications

        Args:
            action: Action to perform (list, show, apply)
            template_name: Template name (required for show/apply actions)
            path: Project root path (default: current directory)

        Returns:
            JSON object with template results:
            - For 'list': templates array with name and description
            - For 'show': template content and structure
            - For 'apply': generated content and instructions
        """
        tool_name = "spec_template"
        try:
            # Validate action
            valid_actions = ("list", "show", "apply")
            if action not in valid_actions:
                return asdict(error_response(
                    f"Invalid action '{action}'. Must be one of: {', '.join(valid_actions)}",
                    error_code="VALIDATION_ERROR",
                    error_type="validation",
                    remediation=f"Use one of: {', '.join(valid_actions)}",
                ))

            # Validate template_name for show/apply
            if action in ("show", "apply") and not template_name:
                return asdict(error_response(
                    f"template_name is required for '{action}' action",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a template_name parameter",
                ))

            # Build command
            cmd = ["sdd", "template", action, "--json"]

            if template_name:
                cmd.append(template_name)

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-template",
                action=action,
                template_name=template_name,
            )

            # Execute the command with resilience
            result = _run_sdd_command(cmd, tool_name)

            # Parse the JSON output
            if result.returncode == 0:
                try:
                    output_data = json.loads(result.stdout) if result.stdout.strip() else {}
                except json.JSONDecodeError:
                    output_data = {}

                # Build response based on action
                data: Dict[str, Any] = {
                    "action": action,
                }

                if action == "list":
                    data["templates"] = output_data.get("templates", [])
                    data["total_count"] = len(data["templates"])
                elif action == "show":
                    data["template_name"] = template_name
                    data["content"] = output_data.get("content", output_data.get("template", {}))
                    data["description"] = output_data.get("description", "")
                elif action == "apply":
                    data["template_name"] = template_name
                    data["generated"] = output_data.get("generated", output_data.get("content", {}))
                    data["instructions"] = output_data.get("instructions", "")

                # Track metrics
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "success", "action": action})

                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "error", "action": action})

                # Check for common errors
                if "not found" in error_msg.lower() and template_name:
                    return asdict(error_response(
                        f"Template '{template_name}' not found",
                        error_code="NOT_FOUND",
                        error_type="not_found",
                        remediation="Use 'list' action to see available templates",
                    ))

                return asdict(error_response(
                    f"Failed to execute template action: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the SDD CLI is available",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try again or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in spec-template")
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="task-add",
    )
    def task_add(
        spec_id: str,
        parent: str,
        title: str,
        description: Optional[str] = None,
        task_type: Optional[str] = None,
        hours: Optional[float] = None,
        position: Optional[int] = None,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Add a new task to an SDD specification.

        Wraps the SDD CLI add-task command to insert a new task node into the
        specification hierarchy. Tasks can be added as children of phases or
        other tasks.

        WHEN TO USE:
        - Adding new work items to an existing specification
        - Expanding scope during implementation
        - Creating subtasks for detailed work breakdown
        - Adding verification tasks to phases

        Args:
            spec_id: Specification ID to add task to
            parent: Parent node ID (e.g., phase-1, task-2-1)
            title: Task title
            description: Optional task description
            task_type: Task type (task, subtask, verify). Default: task
            hours: Estimated hours for the task
            position: Position in parent's children list (0-based)
            dry_run: Preview changes without saving
            path: Project root path (default: current directory)

        Returns:
            JSON object with task creation results:
            - task_id: The generated task ID
            - parent: Parent node ID
            - title: Task title
            - type: Task type
            - position: Position in hierarchy
            - dry_run: Whether this was a dry run
        """
        tool_name = "task_add"
        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            if not parent:
                return asdict(error_response(
                    "parent is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a parent node ID (e.g., phase-1, task-2-1)",
                ))

            if not title:
                return asdict(error_response(
                    "title is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a task title",
                ))

            # Validate task_type if provided
            if task_type and task_type not in ("task", "subtask", "verify"):
                return asdict(error_response(
                    f"Invalid task_type '{task_type}'. Must be one of: task, subtask, verify",
                    error_code="VALIDATION_ERROR",
                    error_type="validation",
                    remediation="Use one of: task, subtask, verify",
                ))

            # Build command
            cmd = ["sdd", "add-task", spec_id, "--parent", parent, "--title", title, "--json"]

            if description:
                cmd.extend(["--description", description])

            if task_type:
                cmd.extend(["--type", task_type])

            if hours is not None:
                cmd.extend(["--hours", str(hours)])

            if position is not None:
                cmd.extend(["--position", str(position)])

            if dry_run:
                cmd.append("--dry-run")

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="task-add",
                action="add_task",
                spec_id=spec_id,
                parent=parent,
                title=title,
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
                    "task_id": output_data.get("task_id", output_data.get("id")),
                    "parent": parent,
                    "title": title,
                    "type": task_type or "task",
                    "dry_run": dry_run,
                }

                if position is not None:
                    data["position"] = position

                if description:
                    data["description"] = description

                if hours is not None:
                    data["hours"] = hours

                # Track metrics
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "success", "dry_run": str(dry_run)})

                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})

                # Check for common errors
                if "not found" in error_msg.lower():
                    if "spec" in error_msg.lower():
                        return asdict(error_response(
                            f"Specification '{spec_id}' not found",
                            error_code="SPEC_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the spec ID exists using spec-list",
                        ))
                    elif "parent" in error_msg.lower() or parent in error_msg:
                        return asdict(error_response(
                            f"Parent node '{parent}' not found in spec",
                            error_code="NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the parent node ID exists in the specification",
                        ))

                return asdict(error_response(
                    f"Failed to add task: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the spec and parent node exist",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try again or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in task-add")
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="task-remove",
    )
    def task_remove(
        spec_id: str,
        task_id: str,
        cascade: bool = False,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Remove a task from an SDD specification.

        Wraps the SDD CLI remove-task command to delete a task node from the
        specification hierarchy. Can optionally cascade to remove child tasks.

        WHEN TO USE:
        - Removing work items that are no longer needed
        - Cleaning up abandoned or duplicate tasks
        - Reducing spec complexity during refactoring
        - Pruning completed branches from hierarchy

        Args:
            spec_id: Specification ID containing the task
            task_id: Task ID to remove
            cascade: Also remove all child tasks recursively
            dry_run: Preview changes without saving
            path: Project root path (default: current directory)

        Returns:
            JSON object with removal results:
            - task_id: The removed task ID
            - spec_id: The specification ID
            - cascade: Whether cascade deletion was used
            - children_removed: Number of child tasks removed (if cascade)
            - dry_run: Whether this was a dry run
        """
        tool_name = "task_remove"
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
            cmd = ["sdd", "remove-task", spec_id, task_id, "--json"]

            if cascade:
                cmd.append("--cascade")

            if dry_run:
                cmd.append("--dry-run")

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="task-remove",
                action="remove_task",
                spec_id=spec_id,
                task_id=task_id,
                cascade=cascade,
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
                    "task_id": task_id,
                    "spec_id": spec_id,
                    "cascade": cascade,
                    "dry_run": dry_run,
                }

                if cascade:
                    data["children_removed"] = output_data.get("children_removed", 0)

                # Track metrics
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "success", "cascade": str(cascade)})

                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})

                # Check for common errors
                if "not found" in error_msg.lower():
                    if "spec" in error_msg.lower():
                        return asdict(error_response(
                            f"Specification '{spec_id}' not found",
                            error_code="SPEC_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the spec ID exists using spec-list",
                        ))
                    else:
                        return asdict(error_response(
                            f"Task '{task_id}' not found in spec",
                            error_code="TASK_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the task ID exists in the specification",
                        ))

                if "has children" in error_msg.lower():
                    return asdict(error_response(
                        f"Task '{task_id}' has children. Use cascade=True to remove recursively",
                        error_code="CONFLICT",
                        error_type="conflict",
                        remediation="Set cascade=True to remove task and all children",
                    ))

                return asdict(error_response(
                    f"Failed to remove task: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the spec and task exist",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try again or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in task-remove")
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="assumption-add",
    )
    def assumption_add(
        spec_id: str,
        text: str,
        assumption_type: Optional[str] = None,
        author: Optional[str] = None,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Add an assumption to an SDD specification.

        Wraps the SDD CLI add-assumption command to append assumptions to
        the spec's assumptions array. Assumptions document constraints and
        requirements that inform the implementation.

        WHEN TO USE:
        - Documenting project constraints
        - Recording requirements that affect design decisions
        - Capturing environmental assumptions
        - Adding prerequisite conditions

        Args:
            spec_id: Specification ID to add assumption to
            text: Assumption text/description
            assumption_type: Type of assumption (constraint, requirement)
            author: Author who added the assumption
            dry_run: Preview assumption without saving
            path: Project root path (default: current directory)

        Returns:
            JSON object with assumption results:
            - spec_id: The specification ID
            - assumption_id: Generated assumption identifier
            - text: The assumption text
            - type: Assumption type
            - author: Author if provided
            - dry_run: Whether this was a dry run
        """
        tool_name = "assumption_add"
        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            if not text:
                return asdict(error_response(
                    "text is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide assumption text",
                ))

            # Validate assumption_type if provided
            if assumption_type and assumption_type not in ("constraint", "requirement"):
                return asdict(error_response(
                    f"Invalid assumption_type '{assumption_type}'. Must be one of: constraint, requirement",
                    error_code="VALIDATION_ERROR",
                    error_type="validation",
                    remediation="Use one of: constraint, requirement",
                ))

            # Build command
            cmd = ["sdd", "add-assumption", spec_id, text, "--json"]

            if assumption_type:
                cmd.extend(["--type", assumption_type])

            if author:
                cmd.extend(["--author", author])

            if dry_run:
                cmd.append("--dry-run")

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="assumption-add",
                action="add_assumption",
                spec_id=spec_id,
                assumption_type=assumption_type,
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
                    "assumption_id": output_data.get("assumption_id", output_data.get("id")),
                    "text": text,
                    "type": assumption_type or "constraint",
                    "dry_run": dry_run,
                }

                if author:
                    data["author"] = author

                # Track metrics
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "success"})

                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})

                if "not found" in error_msg.lower():
                    return asdict(error_response(
                        f"Specification '{spec_id}' not found",
                        error_code="SPEC_NOT_FOUND",
                        error_type="not_found",
                        remediation="Verify the spec ID exists using spec-list",
                    ))

                return asdict(error_response(
                    f"Failed to add assumption: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the spec exists",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try again or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in assumption-add")
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="assumption-list",
    )
    def assumption_list(
        spec_id: str,
        assumption_type: Optional[str] = None,
        path: Optional[str] = None,
    ) -> dict:
        """
        List assumptions in an SDD specification.

        Wraps the SDD CLI list-assumptions command to retrieve assumptions
        from a specification. Can filter by assumption type.

        WHEN TO USE:
        - Reviewing project constraints before implementation
        - Auditing requirements that affect design
        - Checking existing assumptions before adding new ones
        - Generating reports of project constraints

        Args:
            spec_id: Specification ID to list assumptions from
            assumption_type: Filter by type (constraint, requirement)
            path: Project root path (default: current directory)

        Returns:
            JSON object with assumptions list:
            - spec_id: The specification ID
            - assumptions: Array of assumption objects
            - total_count: Number of assumptions
            - filter_type: Filter applied if any
        """
        tool_name = "assumption_list"
        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            # Validate assumption_type if provided
            if assumption_type and assumption_type not in ("constraint", "requirement"):
                return asdict(error_response(
                    f"Invalid assumption_type '{assumption_type}'. Must be one of: constraint, requirement",
                    error_code="VALIDATION_ERROR",
                    error_type="validation",
                    remediation="Use one of: constraint, requirement",
                ))

            # Build command
            cmd = ["sdd", "list-assumptions", spec_id, "--json"]

            if assumption_type:
                cmd.extend(["--type", assumption_type])

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="assumption-list",
                action="list_assumptions",
                spec_id=spec_id,
                assumption_type=assumption_type,
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
                assumptions = output_data.get("assumptions", [])
                data: Dict[str, Any] = {
                    "spec_id": spec_id,
                    "assumptions": assumptions,
                    "total_count": len(assumptions),
                }

                if assumption_type:
                    data["filter_type"] = assumption_type

                # Track metrics
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "success"})

                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})

                if "not found" in error_msg.lower():
                    return asdict(error_response(
                        f"Specification '{spec_id}' not found",
                        error_code="SPEC_NOT_FOUND",
                        error_type="not_found",
                        remediation="Verify the spec ID exists using spec-list",
                    ))

                return asdict(error_response(
                    f"Failed to list assumptions: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the spec exists",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try again or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in assumption-list")
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="revision-add",
    )
    def revision_add(
        spec_id: str,
        version: str,
        changes: str,
        author: Optional[str] = None,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Append a revision history entry to an SDD specification.

        Wraps the SDD CLI add-revision command to add revision entries
        to the spec's revision_history array.

        WHEN TO USE:
        - Recording spec version changes
        - Documenting significant spec modifications
        - Tracking spec evolution over time
        - Adding changelog entries

        Args:
            spec_id: Specification ID
            version: Revision version (e.g., 1.1, 2.0)
            changes: Summary of changes
            author: Revision author
            dry_run: Preview revision without saving
            path: Project root path (default: current directory)

        Returns:
            JSON object with revision results:
            - spec_id: The specification ID
            - version: The revision version
            - changes: Changes summary
            - author: Author if provided
            - dry_run: Whether this was a dry run
        """
        tool_name = "revision_add"
        try:
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            if not version:
                return asdict(error_response(
                    "version is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a version parameter (e.g., 1.1, 2.0)",
                ))

            if not changes:
                return asdict(error_response(
                    "changes is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a changes summary",
                ))

            cmd = ["sdd", "add-revision", spec_id, version, changes, "--json"]

            if author:
                cmd.extend(["--author", author])

            if dry_run:
                cmd.append("--dry-run")

            if path:
                cmd.extend(["--path", path])

            audit_log(
                "tool_invocation",
                tool="revision-add",
                action="add_revision",
                spec_id=spec_id,
                version=version,
                dry_run=dry_run,
            )

            result = _run_sdd_command(cmd, tool_name)

            if result.returncode == 0:
                try:
                    output_data = json.loads(result.stdout) if result.stdout.strip() else {}
                except json.JSONDecodeError:
                    output_data = {}

                data: Dict[str, Any] = {
                    "spec_id": spec_id,
                    "version": version,
                    "changes": changes,
                    "dry_run": dry_run,
                }

                if author:
                    data["author"] = author

                _metrics.counter(f"authoring.{tool_name}", labels={"status": "success"})
                return asdict(success_response(data))
            else:
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})

                if "not found" in error_msg.lower():
                    return asdict(error_response(
                        f"Specification '{spec_id}' not found",
                        error_code="SPEC_NOT_FOUND",
                        error_type="not_found",
                        remediation="Verify the spec ID exists using spec-list",
                    ))

                return asdict(error_response(
                    f"Failed to add revision: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the spec exists",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try again or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in revision-add")
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="spec-update-frontmatter",
    )
    def spec_update_frontmatter(
        spec_id: str,
        key: str,
        value: str,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Mutate top-level metadata blocks in an SDD specification.

        Wraps the SDD CLI update-frontmatter command to update frontmatter
        fields like title, status, version, category, or any top-level
        metadata key in the specification.

        WHEN TO USE:
        - Updating spec title or description
        - Changing spec status (draft, active, completed)
        - Updating version numbers
        - Modifying spec category or priority
        - Setting custom frontmatter fields

        Args:
            spec_id: Specification ID to update
            key: Frontmatter key to update (e.g., title, status, version)
            value: New value for the key
            dry_run: Preview changes without saving
            path: Project root path (default: current directory)

        Returns:
            JSON object with update results:
            - spec_id: The specification ID
            - key: The updated key
            - value: The new value
            - previous_value: Previous value if available
            - dry_run: Whether this was a dry run
        """
        tool_name = "spec_update_frontmatter"
        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            if not key:
                return asdict(error_response(
                    "key is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a frontmatter key (e.g., title, status, version)",
                ))

            if value is None:
                return asdict(error_response(
                    "value is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a value for the frontmatter key",
                ))

            # Build command
            cmd = ["sdd", "update-frontmatter", spec_id, key, value, "--json"]

            if dry_run:
                cmd.append("--dry-run")

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-update-frontmatter",
                action="update_frontmatter",
                spec_id=spec_id,
                key=key,
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
                    "key": key,
                    "value": value,
                    "dry_run": dry_run,
                }

                # Include previous value if available
                if "previous_value" in output_data:
                    data["previous_value"] = output_data["previous_value"]
                elif "old_value" in output_data:
                    data["previous_value"] = output_data["old_value"]

                # Track metrics
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "success"})

                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})

                # Check for common errors
                if "not found" in error_msg.lower():
                    if "spec" in error_msg.lower():
                        return asdict(error_response(
                            f"Specification '{spec_id}' not found",
                            error_code="SPEC_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the spec ID exists using spec-list",
                        ))
                    elif "key" in error_msg.lower():
                        return asdict(error_response(
                            f"Frontmatter key '{key}' not found or invalid",
                            error_code="INVALID_KEY",
                            error_type="validation",
                            remediation="Use a valid frontmatter key (e.g., title, status, version)",
                        ))

                if "invalid" in error_msg.lower() and "value" in error_msg.lower():
                    return asdict(error_response(
                        f"Invalid value '{value}' for key '{key}'",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Check the expected format for this frontmatter key",
                    ))

                return asdict(error_response(
                    f"Failed to update frontmatter: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the spec exists and key/value are valid",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try again or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in spec-update-frontmatter")
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))
