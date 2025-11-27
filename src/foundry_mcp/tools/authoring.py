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
