"""
Authoring tools for foundry-mcp.

Provides MCP tools for creating and modifying SDD specifications.
These tools use the core library APIs directly for spec creation,
task management, and metadata operations.

Resilience features:
- Timing metrics for all tool invocations
- Input validation for all parameters
"""

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import (
    success_response,
    error_response,
    sanitize_error_message,
)
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import audit_log, get_metrics
from foundry_mcp.core.spec import (
    find_specs_directory,
    create_spec,
    add_assumption,
    add_phase,
    remove_phase,
    list_assumptions,
    add_revision,
    update_frontmatter,
    TEMPLATES,
    CATEGORIES,
)
from foundry_mcp.core.task import (
    add_task,
    remove_task,
)

logger = logging.getLogger(__name__)

# Metrics singleton for authoring tools
_metrics = get_metrics()


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
    def spec_create_tool(
        name: str,
        template: Optional[str] = None,
        category: Optional[str] = None,
        path: Optional[str] = None,
    ) -> dict:
        """
        Scaffold a brand-new SDD specification from scratch.

        Generates a new specification file with the default hierarchy structure.
        The specification will be created in the specs/pending directory.

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
        start_time = time.perf_counter()

        try:
            # Validate required parameters
            if not name:
                return asdict(
                    error_response(
                        "name is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a specification name",
                    )
                )

            # Validate template if provided
            effective_template = template or "medium"
            if effective_template not in TEMPLATES:
                return asdict(
                    error_response(
                        f"Invalid template '{effective_template}'. Must be one of: {', '.join(TEMPLATES)}",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation=f"Use one of: {', '.join(TEMPLATES)}",
                    )
                )

            # Validate category if provided
            effective_category = category or "implementation"
            if effective_category not in CATEGORIES:
                return asdict(
                    error_response(
                        f"Invalid category '{effective_category}'. Must be one of: {', '.join(CATEGORIES)}",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation=f"Use one of: {', '.join(CATEGORIES)}",
                    )
                )

            # Find specs directory
            specs_dir = (
                find_specs_directory(path)
                if path
                else (config.specs_dir or find_specs_directory())
            )

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-create",
                action="create_spec",
                name=name,
                template=effective_template,
                category=effective_category,
            )

            # Call the core function
            result, error = create_spec(
                name=name,
                template=effective_template,
                category=effective_category,
                specs_dir=specs_dir,
            )

            # Record timing metrics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer(f"authoring.{tool_name}.duration_ms", elapsed_ms)

            if error:
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})

                # Check for common errors
                if "already exists" in error.lower():
                    return asdict(
                        error_response(
                            f"A specification with name '{name}' already exists",
                            error_code="DUPLICATE_ENTRY",
                            error_type="conflict",
                            remediation="Use a different name or update the existing spec",
                        )
                    )

                if "no specs directory" in error.lower():
                    return asdict(
                        error_response(
                            error,
                            error_code="NOT_FOUND",
                            error_type="not_found",
                            remediation="Create a specs directory or provide a valid path",
                        )
                    )

                return asdict(
                    error_response(
                        f"Failed to create specification: {error}",
                        error_code="COMMAND_FAILED",
                        error_type="internal",
                        remediation="Check that the project path is valid",
                    )
                )

            # Build response data
            data: Dict[str, Any] = {
                "spec_id": result.get("spec_id"),
                "spec_path": result.get("spec_path"),
                "template": effective_template,
                "name": name,
            }

            if category:
                data["category"] = effective_category

            # Include structure info if available
            if "structure" in result:
                data["structure"] = result["structure"]

            # Track metrics
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "success"})

            return asdict(success_response(data))

        except Exception as e:
            logger.exception("Unexpected error in spec-create")
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})
            return asdict(
                error_response(
                    sanitize_error_message(e, context="authoring"),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                    remediation="Check logs for details",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="spec-template",
    )
    def spec_template_tool(
        action: str,
        template_name: Optional[str] = None,
        path: Optional[str] = None,
    ) -> dict:
        """
        Emit opinionated templates/snippets for spec sections.

        Lists available templates, shows template contents, or provides
        template information for spec sections.

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
        start_time = time.perf_counter()

        try:
            # Validate action
            valid_actions = ("list", "show", "apply")
            if action not in valid_actions:
                return asdict(
                    error_response(
                        f"Invalid action '{action}'. Must be one of: {', '.join(valid_actions)}",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation=f"Use one of: {', '.join(valid_actions)}",
                    )
                )

            # Validate template_name for show/apply
            if action in ("show", "apply") and not template_name:
                return asdict(
                    error_response(
                        f"template_name is required for '{action}' action",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a template_name parameter",
                    )
                )

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-template",
                action=action,
                template_name=template_name,
            )

            # Record timing metrics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer(f"authoring.{tool_name}.duration_ms", elapsed_ms)

            # Build response based on action
            data: Dict[str, Any] = {
                "action": action,
            }

            if action == "list":
                # Return list of available templates
                data["templates"] = [
                    {
                        "name": "simple",
                        "description": "Minimal spec with 1 phase and basic tasks",
                    },
                    {
                        "name": "medium",
                        "description": "Standard spec with 2-3 phases (default)",
                    },
                    {
                        "name": "complex",
                        "description": "Multi-phase spec with groups and subtasks",
                    },
                    {
                        "name": "security",
                        "description": "Security-focused spec with audit tasks",
                    },
                ]
                data["total_count"] = len(data["templates"])
            elif action == "show":
                # Validate template exists
                if template_name not in TEMPLATES:
                    _metrics.counter(
                        f"authoring.{tool_name}",
                        labels={"status": "error", "action": action},
                    )
                    return asdict(
                        error_response(
                            f"Template '{template_name}' not found",
                            error_code="NOT_FOUND",
                            error_type="not_found",
                            remediation=f"Use 'list' action to see available templates. Valid: {', '.join(TEMPLATES)}",
                        )
                    )
                data["template_name"] = template_name
                data["content"] = {
                    "name": template_name,
                    "description": f"Template structure for '{template_name}' specs",
                    "usage": f"Use spec-create with template='{template_name}' to create a spec from this template",
                }
            elif action == "apply":
                # Validate template exists
                if template_name not in TEMPLATES:
                    _metrics.counter(
                        f"authoring.{tool_name}",
                        labels={"status": "error", "action": action},
                    )
                    return asdict(
                        error_response(
                            f"Template '{template_name}' not found",
                            error_code="NOT_FOUND",
                            error_type="not_found",
                            remediation=f"Use 'list' action to see available templates. Valid: {', '.join(TEMPLATES)}",
                        )
                    )
                data["template_name"] = template_name
                data["generated"] = {
                    "template": template_name,
                    "message": f"Use spec-create with template='{template_name}' to create a new spec",
                }
                data["instructions"] = (
                    f"Call spec-create with name='your-spec-name' and template='{template_name}'"
                )

            # Track metrics
            _metrics.counter(
                f"authoring.{tool_name}", labels={"status": "success", "action": action}
            )

            return asdict(success_response(data))

        except Exception as e:
            logger.exception("Unexpected error in spec-template")
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})
            return asdict(
                error_response(
                    sanitize_error_message(e, context="authoring"),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                    remediation="Check logs for details",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="task-add",
    )
    def task_add_tool(
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

        Inserts a new task node into the specification hierarchy. Tasks can be
        added as children of phases or other tasks.

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
            dry_run: Preview changes without saving (not supported in direct API)
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
        start_time = time.perf_counter()

        try:
            # Validate required parameters
            if not spec_id:
                return asdict(
                    error_response(
                        "spec_id is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a spec_id parameter",
                    )
                )

            if not parent:
                return asdict(
                    error_response(
                        "parent is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a parent node ID (e.g., phase-1, task-2-1)",
                    )
                )

            if not title:
                return asdict(
                    error_response(
                        "title is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a task title",
                    )
                )

            # Validate task_type if provided
            effective_task_type = task_type or "task"
            if effective_task_type not in ("task", "subtask", "verify"):
                return asdict(
                    error_response(
                        f"Invalid task_type '{effective_task_type}'. Must be one of: task, subtask, verify",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Use one of: task, subtask, verify",
                    )
                )

            # Find specs directory
            specs_dir = (
                find_specs_directory(path)
                if path
                else (config.specs_dir or find_specs_directory())
            )

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

            # Note: dry_run is not supported in direct API - document this
            if dry_run:
                # Return preview without actually adding
                _metrics.counter(
                    f"authoring.{tool_name}",
                    labels={"status": "success", "dry_run": "true"},
                )
                return asdict(
                    success_response(
                        {
                            "task_id": "(preview)",
                            "parent": parent,
                            "title": title,
                            "type": effective_task_type,
                            "dry_run": True,
                            "note": "Dry run - no changes made",
                        }
                    )
                )

            # Call the core function
            result, error = add_task(
                spec_id=spec_id,
                parent_id=parent,
                title=title,
                description=description,
                task_type=effective_task_type,
                estimated_hours=hours,
                position=position,
                specs_dir=specs_dir,
            )

            # Record timing metrics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer(f"authoring.{tool_name}.duration_ms", elapsed_ms)

            if error:
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})

                # Check for common errors
                if "not found" in error.lower():
                    if "spec" in error.lower():
                        return asdict(
                            error_response(
                                f"Specification '{spec_id}' not found",
                                error_code="SPEC_NOT_FOUND",
                                error_type="not_found",
                                remediation="Verify the spec ID exists using spec-list",
                            )
                        )
                    elif "parent" in error.lower() or parent in error:
                        return asdict(
                            error_response(
                                f"Parent node '{parent}' not found in spec",
                                error_code="NOT_FOUND",
                                error_type="not_found",
                                remediation="Verify the parent node ID exists in the specification",
                            )
                        )

                return asdict(
                    error_response(
                        f"Failed to add task: {error}",
                        error_code="COMMAND_FAILED",
                        error_type="internal",
                        remediation="Check that the spec and parent node exist",
                    )
                )

            # Build response data
            data: Dict[str, Any] = {
                "task_id": result.get("task_id"),
                "parent": parent,
                "title": title,
                "type": effective_task_type,
                "dry_run": False,
            }

            if position is not None:
                data["position"] = position

            if description:
                data["description"] = description

            if hours is not None:
                data["hours"] = hours

            # Track metrics
            _metrics.counter(
                f"authoring.{tool_name}",
                labels={"status": "success", "dry_run": "false"},
            )

            return asdict(success_response(data))

        except Exception as e:
            logger.exception("Unexpected error in task-add")
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})
            return asdict(
                error_response(
                    sanitize_error_message(e, context="authoring"),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                    remediation="Check logs for details",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="phase-add",
    )
    def phase_add_tool(
        spec_id: str,
        title: str,
        description: Optional[str] = None,
        purpose: Optional[str] = None,
        estimated_hours: Optional[float] = None,
        position: Optional[int] = None,
        link_previous: bool = True,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """Add a new phase to spec-root, including verification scaffolding."""
        tool_name = "phase_add"
        start_time = time.perf_counter()

        try:
            if not spec_id:
                return asdict(
                    error_response(
                        "spec_id is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a spec_id parameter",
                    )
                )

            if not title:
                return asdict(
                    error_response(
                        "title is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a phase title",
                    )
                )

            if estimated_hours is not None and estimated_hours < 0:
                return asdict(
                    error_response(
                        "estimated_hours must be non-negative",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Provide a non-negative value",
                    )
                )

            specs_dir = (
                find_specs_directory(path)
                if path
                else (config.specs_dir or find_specs_directory())
            )

            audit_log(
                "tool_invocation",
                tool="phase-add",
                action="add_phase",
                spec_id=spec_id,
                title=title,
                dry_run=dry_run,
                link_previous=link_previous,
            )

            if dry_run:
                _metrics.counter(
                    f"authoring.{tool_name}",
                    labels={"status": "success", "dry_run": "true"},
                )
                return asdict(
                    success_response(
                        {
                            "phase_id": "(preview)",
                            "title": title,
                            "dry_run": True,
                            "note": "Dry run - no changes made",
                        }
                    )
                )

            if specs_dir is None:
                return asdict(
                    error_response(
                        "No specs directory found. Use specs_dir parameter or set SDD_SPECS_DIR.",
                        error_code="NOT_FOUND",
                        error_type="not_found",
                        remediation="Use --specs-dir or set SDD_SPECS_DIR",
                    )
                )

            result, error = add_phase(
                spec_id=spec_id,
                title=title,
                description=description,
                purpose=purpose,
                estimated_hours=estimated_hours,
                position=position,
                link_previous=link_previous,
                specs_dir=specs_dir,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer(f"authoring.{tool_name}.duration_ms", elapsed_ms)

            if error:
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})
                lowered = error.lower()
                if "specification" in lowered and "not found" in lowered:
                    return asdict(
                        error_response(
                            f"Specification '{spec_id}' not found",
                            error_code="SPEC_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the spec ID exists using spec-list",
                        )
                    )
                return asdict(
                    error_response(
                        f"Failed to add phase: {error}",
                        error_code="COMMAND_FAILED",
                        error_type="internal",
                        remediation="Check input parameters and retry",
                    )
                )

            _metrics.counter(f"authoring.{tool_name}", labels={"status": "success"})
            return asdict(
                success_response(
                    {
                        "spec_id": spec_id,
                        "dry_run": False,
                        **(result or {}),
                        "telemetry": {"duration_ms": round(elapsed_ms, 2)},
                    }
                )
            )

        except Exception as e:
            logger.exception("Unexpected error in phase-add")
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})
            return asdict(
                error_response(
                    sanitize_error_message(e, context="authoring"),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                    remediation="Check logs for details",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="phase-remove",
    )
    def phase_remove_tool(
        spec_id: str,
        phase_id: str,
        force: bool = False,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Remove a phase and all its children from an SDD specification.

        Deletes a phase node and all its child tasks/verifications from the hierarchy.
        Handles re-linking of adjacent phases when dependency chains exist.

        WHEN TO USE:
        - Removing phases that are no longer needed
        - Cleaning up abandoned or duplicate phases
        - Restructuring specification scope
        - Pruning completed phases from hierarchy

        Args:
            spec_id: Specification ID containing the phase
            phase_id: Phase ID to remove (e.g., phase-1, phase-2)
            force: Remove even if phase contains non-completed tasks (default: False)
            dry_run: Preview changes without saving
            path: Project root path (default: current directory)

        Returns:
            JSON object with removal results:
            - spec_id: The specification ID
            - phase_id: The removed phase ID
            - phase_title: Title of the removed phase
            - children_removed: Number of child tasks/verifications removed
            - total_tasks_removed: Total task count removed
            - force: Whether force flag was used
            - relinked: Info about re-linked adjacent phases (if applicable)
            - dry_run: Whether this was a dry run
        """
        tool_name = "phase_remove"
        start_time = time.perf_counter()

        try:
            # Validate required parameters
            if not spec_id:
                return asdict(
                    error_response(
                        "spec_id is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a spec_id parameter",
                    )
                )

            if not phase_id:
                return asdict(
                    error_response(
                        "phase_id is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a phase_id parameter (e.g., phase-1)",
                    )
                )

            # Find specs directory
            specs_dir = (
                find_specs_directory(path)
                if path
                else (config.specs_dir or find_specs_directory())
            )

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="phase-remove",
                action="remove_phase",
                spec_id=spec_id,
                phase_id=phase_id,
                force=force,
                dry_run=dry_run,
            )

            # Handle dry_run preview
            if dry_run:
                _metrics.counter(
                    f"authoring.{tool_name}",
                    labels={"status": "success", "force": str(force)},
                )
                return asdict(
                    success_response(
                        {
                            "spec_id": spec_id,
                            "phase_id": phase_id,
                            "force": force,
                            "dry_run": True,
                            "note": "Dry run - no changes made",
                        }
                    )
                )

            # Call the core function
            result, error = remove_phase(
                spec_id=spec_id,
                phase_id=phase_id,
                force=force,
                specs_dir=specs_dir,
            )

            # Record timing metrics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer(f"authoring.{tool_name}.duration_ms", elapsed_ms)

            if error:
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})

                # Check for common errors
                lowered = error.lower()
                if "not found" in lowered:
                    if "spec" in lowered:
                        return asdict(
                            error_response(
                                f"Specification '{spec_id}' not found",
                                error_code="SPEC_NOT_FOUND",
                                error_type="not_found",
                                remediation="Verify the spec ID exists using spec-list",
                            )
                        )
                    elif "phase" in lowered:
                        return asdict(
                            error_response(
                                f"Phase '{phase_id}' not found in spec",
                                error_code="PHASE_NOT_FOUND",
                                error_type="not_found",
                                remediation="Verify the phase ID exists in the specification",
                            )
                        )

                if "is not a phase" in lowered:
                    return asdict(
                        error_response(
                            f"Node '{phase_id}' is not a phase",
                            error_code="VALIDATION_ERROR",
                            error_type="validation",
                            remediation="Use task-remove for task nodes, phase-remove is only for phases",
                        )
                    )

                if "non-completed" in lowered or "has" in lowered and "task" in lowered:
                    return asdict(
                        error_response(
                            f"Phase '{phase_id}' has non-completed tasks. Use force=True to remove anyway",
                            error_code="CONFLICT",
                            error_type="conflict",
                            remediation="Set force=True to remove phase with active tasks",
                        )
                    )

                return asdict(
                    error_response(
                        f"Failed to remove phase: {error}",
                        error_code="COMMAND_FAILED",
                        error_type="internal",
                        remediation="Check that the spec and phase exist",
                    )
                )

            # Build response data
            data: Dict[str, Any] = {
                "spec_id": spec_id,
                "phase_id": phase_id,
                "phase_title": result.get("phase_title", ""),
                "children_removed": result.get("children_removed", 0),
                "total_tasks_removed": result.get("total_tasks_removed", 0),
                "force": force,
                "dry_run": False,
            }

            # Include relinked info if phases were re-linked
            if "relinked" in result:
                data["relinked"] = result["relinked"]

            # Track metrics
            _metrics.counter(
                f"authoring.{tool_name}",
                labels={"status": "success", "force": str(force)},
            )

            return asdict(success_response(data))

        except Exception as e:
            logger.exception("Unexpected error in phase-remove")
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})
            return asdict(
                error_response(
                    sanitize_error_message(e, context="authoring"),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                    remediation="Check logs for details",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="task-remove",
    )
    def task_remove_tool(
        spec_id: str,
        task_id: str,
        cascade: bool = False,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Remove a task from an SDD specification.

        Deletes a task node from the specification hierarchy. Can optionally
        cascade to remove child tasks.

        WHEN TO USE:
        - Removing work items that are no longer needed
        - Cleaning up abandoned or duplicate tasks
        - Reducing spec complexity during refactoring
        - Pruning completed branches from hierarchy

        Args:
            spec_id: Specification ID containing the task
            task_id: Task ID to remove
            cascade: Also remove all child tasks recursively
            dry_run: Preview changes without saving (not supported in direct API)
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
        start_time = time.perf_counter()

        try:
            # Validate required parameters
            if not spec_id:
                return asdict(
                    error_response(
                        "spec_id is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a spec_id parameter",
                    )
                )

            if not task_id:
                return asdict(
                    error_response(
                        "task_id is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a task_id parameter",
                    )
                )

            # Find specs directory
            specs_dir = (
                find_specs_directory(path)
                if path
                else (config.specs_dir or find_specs_directory())
            )

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

            # Note: dry_run is not supported in direct API - document this
            if dry_run:
                # Return preview without actually removing
                _metrics.counter(
                    f"authoring.{tool_name}",
                    labels={"status": "success", "cascade": str(cascade)},
                )
                return asdict(
                    success_response(
                        {
                            "task_id": task_id,
                            "spec_id": spec_id,
                            "cascade": cascade,
                            "dry_run": True,
                            "note": "Dry run - no changes made",
                        }
                    )
                )

            # Call the core function
            result, error = remove_task(
                spec_id=spec_id,
                task_id=task_id,
                cascade=cascade,
                specs_dir=specs_dir,
            )

            # Record timing metrics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer(f"authoring.{tool_name}.duration_ms", elapsed_ms)

            if error:
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})

                # Check for common errors
                if "not found" in error.lower():
                    if "spec" in error.lower():
                        return asdict(
                            error_response(
                                f"Specification '{spec_id}' not found",
                                error_code="SPEC_NOT_FOUND",
                                error_type="not_found",
                                remediation="Verify the spec ID exists using spec-list",
                            )
                        )
                    else:
                        return asdict(
                            error_response(
                                f"Task '{task_id}' not found in spec",
                                error_code="TASK_NOT_FOUND",
                                error_type="not_found",
                                remediation="Verify the task ID exists in the specification",
                            )
                        )

                if (
                    "has children" in error.lower()
                    or "has" in error.lower()
                    and "children" in error.lower()
                ):
                    return asdict(
                        error_response(
                            f"Task '{task_id}' has children. Use cascade=True to remove recursively",
                            error_code="CONFLICT",
                            error_type="conflict",
                            remediation="Set cascade=True to remove task and all children",
                        )
                    )

                return asdict(
                    error_response(
                        f"Failed to remove task: {error}",
                        error_code="COMMAND_FAILED",
                        error_type="internal",
                        remediation="Check that the spec and task exist",
                    )
                )

            # Build response data
            data: Dict[str, Any] = {
                "task_id": task_id,
                "spec_id": spec_id,
                "cascade": cascade,
                "dry_run": False,
            }

            if cascade:
                data["children_removed"] = result.get("children_removed", 0)

            # Track metrics
            _metrics.counter(
                f"authoring.{tool_name}",
                labels={"status": "success", "cascade": str(cascade)},
            )

            return asdict(success_response(data))

        except Exception as e:
            logger.exception("Unexpected error in task-remove")
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})
            return asdict(
                error_response(
                    sanitize_error_message(e, context="authoring"),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                    remediation="Check logs for details",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="assumption-add",
    )
    def assumption_add_tool(
        spec_id: str,
        text: str,
        assumption_type: Optional[str] = None,
        author: Optional[str] = None,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Add an assumption to an SDD specification.

        Appends assumptions to the spec's assumptions array. Assumptions document
        constraints and requirements that inform the implementation.

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
            dry_run: Preview assumption without saving (not supported in direct API)
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
        start_time = time.perf_counter()

        try:
            # Validate required parameters
            if not spec_id:
                return asdict(
                    error_response(
                        "spec_id is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a spec_id parameter",
                    )
                )

            if not text:
                return asdict(
                    error_response(
                        "text is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide assumption text",
                    )
                )

            # Validate assumption_type if provided
            effective_type = assumption_type or "constraint"
            if effective_type not in ("constraint", "requirement"):
                return asdict(
                    error_response(
                        f"Invalid assumption_type '{effective_type}'. Must be one of: constraint, requirement",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Use one of: constraint, requirement",
                    )
                )

            # Find specs directory
            specs_dir = (
                find_specs_directory(path)
                if path
                else (config.specs_dir or find_specs_directory())
            )

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="assumption-add",
                action="add_assumption",
                spec_id=spec_id,
                assumption_type=effective_type,
                dry_run=dry_run,
            )

            # Note: dry_run is not supported in direct API
            if dry_run:
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "success"})
                return asdict(
                    success_response(
                        {
                            "spec_id": spec_id,
                            "assumption_id": "(preview)",
                            "text": text,
                            "type": effective_type,
                            "dry_run": True,
                            "note": "Dry run - no changes made",
                        }
                    )
                )

            # Call the core function
            result, error = add_assumption(
                spec_id=spec_id,
                text=text,
                assumption_type=effective_type,
                author=author,
                specs_dir=specs_dir,
            )

            # Record timing metrics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer(f"authoring.{tool_name}.duration_ms", elapsed_ms)

            if error:
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})

                if "not found" in error.lower():
                    return asdict(
                        error_response(
                            f"Specification '{spec_id}' not found",
                            error_code="SPEC_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the spec ID exists using spec-list",
                        )
                    )

                return asdict(
                    error_response(
                        f"Failed to add assumption: {error}",
                        error_code="COMMAND_FAILED",
                        error_type="internal",
                        remediation="Check that the spec exists",
                    )
                )

            # Build response data
            data: Dict[str, Any] = {
                "spec_id": spec_id,
                "assumption_id": result.get("assumption_id"),
                "text": text,
                "type": effective_type,
                "dry_run": False,
            }

            if author:
                data["author"] = author

            # Track metrics
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "success"})

            return asdict(success_response(data))

        except Exception as e:
            logger.exception("Unexpected error in assumption-add")
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})
            return asdict(
                error_response(
                    sanitize_error_message(e, context="authoring"),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                    remediation="Check logs for details",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="assumption-list",
    )
    def assumption_list_tool(
        spec_id: str,
        assumption_type: Optional[str] = None,
        path: Optional[str] = None,
    ) -> dict:
        """
        List assumptions in an SDD specification.

        Retrieves assumptions from a specification. Can filter by assumption type.

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
        start_time = time.perf_counter()

        try:
            # Validate required parameters
            if not spec_id:
                return asdict(
                    error_response(
                        "spec_id is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a spec_id parameter",
                    )
                )

            # Validate assumption_type if provided
            if assumption_type and assumption_type not in ("constraint", "requirement"):
                return asdict(
                    error_response(
                        f"Invalid assumption_type '{assumption_type}'. Must be one of: constraint, requirement",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Use one of: constraint, requirement",
                    )
                )

            # Find specs directory
            specs_dir = (
                find_specs_directory(path)
                if path
                else (config.specs_dir or find_specs_directory())
            )

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="assumption-list",
                action="list_assumptions",
                spec_id=spec_id,
                assumption_type=assumption_type,
            )

            # Call the core function
            result, error = list_assumptions(
                spec_id=spec_id,
                assumption_type=assumption_type,
                specs_dir=specs_dir,
            )

            # Record timing metrics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer(f"authoring.{tool_name}.duration_ms", elapsed_ms)

            if error:
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})

                if "not found" in error.lower():
                    return asdict(
                        error_response(
                            f"Specification '{spec_id}' not found",
                            error_code="SPEC_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the spec ID exists using spec-list",
                        )
                    )

                return asdict(
                    error_response(
                        f"Failed to list assumptions: {error}",
                        error_code="COMMAND_FAILED",
                        error_type="internal",
                        remediation="Check that the spec exists",
                    )
                )

            # Build response data
            assumptions = result.get("assumptions", [])
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

        except Exception as e:
            logger.exception("Unexpected error in assumption-list")
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})
            return asdict(
                error_response(
                    sanitize_error_message(e, context="authoring"),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                    remediation="Check logs for details",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="revision-add",
    )
    def revision_add_tool(
        spec_id: str,
        version: str,
        changes: str,
        author: Optional[str] = None,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Append a revision history entry to an SDD specification.

        Adds revision entries to the spec's revision_history array.

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
            dry_run: Preview revision without saving (not supported in direct API)
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
        start_time = time.perf_counter()

        try:
            if not spec_id:
                return asdict(
                    error_response(
                        "spec_id is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a spec_id parameter",
                    )
                )

            if not version:
                return asdict(
                    error_response(
                        "version is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a version parameter (e.g., 1.1, 2.0)",
                    )
                )

            if not changes:
                return asdict(
                    error_response(
                        "changes is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a changes summary",
                    )
                )

            # Find specs directory
            specs_dir = (
                find_specs_directory(path)
                if path
                else (config.specs_dir or find_specs_directory())
            )

            audit_log(
                "tool_invocation",
                tool="revision-add",
                action="add_revision",
                spec_id=spec_id,
                version=version,
                dry_run=dry_run,
            )

            # Note: dry_run is not supported in direct API
            if dry_run:
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "success"})
                data: Dict[str, Any] = {
                    "spec_id": spec_id,
                    "version": version,
                    "changes": changes,
                    "dry_run": True,
                    "note": "Dry run - no changes made",
                }
                if author:
                    data["author"] = author
                return asdict(success_response(data))

            # Call the core function
            result, error = add_revision(
                spec_id=spec_id,
                version=version,
                changelog=changes,
                author=author,
                specs_dir=specs_dir,
            )

            # Record timing metrics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer(f"authoring.{tool_name}.duration_ms", elapsed_ms)

            if error:
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})

                if "not found" in error.lower():
                    return asdict(
                        error_response(
                            f"Specification '{spec_id}' not found",
                            error_code="SPEC_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the spec ID exists using spec-list",
                        )
                    )

                return asdict(
                    error_response(
                        f"Failed to add revision: {error}",
                        error_code="COMMAND_FAILED",
                        error_type="internal",
                        remediation="Check that the spec exists",
                    )
                )

            data = {
                "spec_id": spec_id,
                "version": version,
                "changes": changes,
                "dry_run": False,
            }

            if author:
                data["author"] = author

            _metrics.counter(f"authoring.{tool_name}", labels={"status": "success"})
            return asdict(success_response(data))

        except Exception as e:
            logger.exception("Unexpected error in revision-add")
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})
            return asdict(
                error_response(
                    sanitize_error_message(e, context="authoring"),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                    remediation="Check logs for details",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="spec-update-frontmatter",
    )
    def spec_update_frontmatter_tool(
        spec_id: str,
        key: str,
        value: str,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Mutate top-level metadata blocks in an SDD specification.

        Updates frontmatter fields like title, status, version, category, or any
        top-level metadata key in the specification.

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
            dry_run: Preview changes without saving (not supported in direct API)
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
        start_time = time.perf_counter()

        try:
            # Validate required parameters
            if not spec_id:
                return asdict(
                    error_response(
                        "spec_id is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a spec_id parameter",
                    )
                )

            if not key:
                return asdict(
                    error_response(
                        "key is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a frontmatter key (e.g., title, status, version)",
                    )
                )

            if value is None:
                return asdict(
                    error_response(
                        "value is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a value for the frontmatter key",
                    )
                )

            # Find specs directory
            specs_dir = (
                find_specs_directory(path)
                if path
                else (config.specs_dir or find_specs_directory())
            )

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-update-frontmatter",
                action="update_frontmatter",
                spec_id=spec_id,
                key=key,
                dry_run=dry_run,
            )

            # Note: dry_run is not supported in direct API
            if dry_run:
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "success"})
                return asdict(
                    success_response(
                        {
                            "spec_id": spec_id,
                            "key": key,
                            "value": value,
                            "dry_run": True,
                            "note": "Dry run - no changes made",
                        }
                    )
                )

            # Call the core function
            result, error = update_frontmatter(
                spec_id=spec_id,
                key=key,
                value=value,
                specs_dir=specs_dir,
            )

            # Record timing metrics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer(f"authoring.{tool_name}.duration_ms", elapsed_ms)

            if error:
                _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})

                # Check for common errors
                if "not found" in error.lower():
                    if "spec" in error.lower():
                        return asdict(
                            error_response(
                                f"Specification '{spec_id}' not found",
                                error_code="SPEC_NOT_FOUND",
                                error_type="not_found",
                                remediation="Verify the spec ID exists using spec-list",
                            )
                        )
                    elif "key" in error.lower():
                        return asdict(
                            error_response(
                                f"Frontmatter key '{key}' not found or invalid",
                                error_code="INVALID_KEY",
                                error_type="validation",
                                remediation="Use a valid frontmatter key (e.g., title, status, version)",
                            )
                        )

                if "dedicated function" in error.lower():
                    return asdict(
                        error_response(
                            error,
                            error_code="VALIDATION_ERROR",
                            error_type="validation",
                            remediation="Use assumption-add or revision-add for those fields",
                        )
                    )

                return asdict(
                    error_response(
                        f"Failed to update frontmatter: {error}",
                        error_code="COMMAND_FAILED",
                        error_type="internal",
                        remediation="Check that the spec exists and key/value are valid",
                    )
                )

            # Build response data
            data: Dict[str, Any] = {
                "spec_id": spec_id,
                "key": key,
                "value": value,
                "dry_run": False,
            }

            # Include previous value if available
            if "previous_value" in result:
                data["previous_value"] = result["previous_value"]

            # Track metrics
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "success"})

            return asdict(success_response(data))

        except Exception as e:
            logger.exception("Unexpected error in spec-update-frontmatter")
            _metrics.counter(f"authoring.{tool_name}", labels={"status": "error"})
            return asdict(
                error_response(
                    sanitize_error_message(e, context="authoring"),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                    remediation="Check logs for details",
                )
            )

    logger.debug(
        "Registered authoring tools: spec-create, spec-template, task-add, phase-add, phase-remove, task-remove, assumption-add, assumption-list, revision-add, spec-update-frontmatter"
    )
