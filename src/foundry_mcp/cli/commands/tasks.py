"""Task management commands for SDD CLI.

Provides commands for discovering, querying, and updating tasks in specifications.
"""

from typing import Any, Dict, Optional

import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit, emit_error, emit_success
from foundry_mcp.cli.resilience import (
    handle_keyboard_interrupt,
    MEDIUM_TIMEOUT,
    with_sync_timeout,
)
from foundry_mcp.cli.registry import get_context

logger = get_cli_logger()
from foundry_mcp.core.spec import load_spec, find_spec_file, get_node
from foundry_mcp.core.journal import (
    mark_blocked,
    save_journal,
    unblock,
    update_task_status,
)
from foundry_mcp.core.task import (
    check_dependencies,
    get_next_task,
    get_parent_context,
    get_phase_context,
    get_previous_sibling,
    get_task_journal_summary,
    prepare_task,
)


@click.group("tasks")
def tasks() -> None:
    """Task management commands."""
    pass


@tasks.command("next")
@click.argument("spec_id")
@click.pass_context
@cli_command("tasks-next")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Task discovery timed out")
def next_task(ctx: click.Context, spec_id: str) -> None:
    """Find the next actionable task in a specification.

    SPEC_ID is the specification identifier.
    """
    cli_ctx = get_context(ctx)
    specs_dir = cli_ctx.specs_dir

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set SDD_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set SDD_SPECS_DIR"},
        )

    # Load the spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs list",
            details={"spec_id": spec_id, "specs_dir": str(specs_dir)},
        )

    # Find the next task
    result = get_next_task(spec_data)

    if result:
        task_id, task_data = result
        emit_success({
            "found": True,
            "spec_id": spec_id,
            "task_id": task_id,
            "title": task_data.get("title", ""),
            "type": task_data.get("type", "task"),
            "status": task_data.get("status", "pending"),
            "metadata": task_data.get("metadata", {}),
        })
    else:
        # Check if spec is complete or blocked
        hierarchy = spec_data.get("hierarchy", {})
        all_tasks = [
            node
            for node in hierarchy.values()
            if isinstance(node, dict) and node.get("type") in ("task", "subtask", "verify")
        ]
        completed = sum(1 for t in all_tasks if t.get("status") == "completed")
        pending = sum(1 for t in all_tasks if t.get("status") == "pending")

        if pending == 0 and completed > 0:
            emit_success({
                "found": False,
                "spec_id": spec_id,
                "spec_complete": True,
                "message": "All tasks completed",
            })
        else:
            emit_success({
                "found": False,
                "spec_id": spec_id,
                "spec_complete": False,
                "message": "No actionable tasks (tasks may be blocked)",
            })


@tasks.command("prepare")
@click.argument("spec_id")
@click.argument("task_id", required=False)
@click.pass_context
@cli_command("tasks-prepare")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Task preparation timed out")
def prepare_task_cmd(ctx: click.Context, spec_id: str, task_id: Optional[str] = None) -> None:
    """Prepare complete context for task implementation.

    SPEC_ID is the specification identifier.
    TASK_ID is optional; auto-discovers next task if not provided.
    """
    cli_ctx = get_context(ctx)
    specs_dir = cli_ctx.specs_dir

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set SDD_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set SDD_SPECS_DIR"},
        )

    # Use core prepare_task function
    result = prepare_task(spec_id, specs_dir, task_id)

    # Check if result indicates an error (from error_response)
    if result.get("success") is False:
        error = result.get("error", {})
        emit_error(
            error.get("message", "Task preparation failed"),
            code=error.get("code", "INTERNAL_ERROR"),
            error_type="internal",
            remediation="Check the spec file exists and is valid",
            details={"spec_id": spec_id, "task_id": task_id},
        )

    # Extract data from success response
    data = result.get("data", result)

    emit_success({
        "spec_id": spec_id,
        "task_id": data.get("task_id"),
        "spec_complete": data.get("spec_complete", False),
        "task_data": data.get("task_data"),
        "dependencies": data.get("dependencies"),
        "context": data.get("context"),
    })


@tasks.command("info")
@click.argument("spec_id")
@click.argument("task_id")
@click.option("--include-context/--no-context", default=True, help="Include task context.")
@click.pass_context
@cli_command("tasks-info")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Task info lookup timed out")
def task_info_cmd(
    ctx: click.Context,
    spec_id: str,
    task_id: str,
    include_context: bool,
) -> None:
    """Get detailed information about a specific task.

    SPEC_ID is the specification identifier.
    TASK_ID is the task identifier.
    """
    cli_ctx = get_context(ctx)
    specs_dir = cli_ctx.specs_dir

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set SDD_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set SDD_SPECS_DIR"},
        )

    # Load the spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs list",
            details={"spec_id": spec_id, "specs_dir": str(specs_dir)},
        )

    # Get task data
    task_data = get_node(spec_data, task_id)
    if task_data is None:
        emit_error(
            f"Task not found: {task_id}",
            code="TASK_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the task ID exists using: sdd tasks info <spec_id> --list",
            details={"spec_id": spec_id, "task_id": task_id},
        )

    # Check dependencies
    deps = check_dependencies(spec_data, task_id)

    result: Dict[str, Any] = {
        "spec_id": spec_id,
        "task_id": task_id,
        "title": task_data.get("title", ""),
        "type": task_data.get("type", "task"),
        "status": task_data.get("status", "pending"),
        "metadata": task_data.get("metadata", {}),
        "children": task_data.get("children", []),
        "dependencies": deps,
    }

    # Add context if requested
    if include_context:
        result["context"] = {
            "previous_sibling": get_previous_sibling(spec_data, task_id),
            "parent_task": get_parent_context(spec_data, task_id),
            "phase": get_phase_context(spec_data, task_id),
            "task_journal": get_task_journal_summary(spec_data, task_id),
        }

    emit_success(result)


@tasks.command("update-status")
@click.argument("spec_id")
@click.argument("task_id")
@click.argument("status", type=click.Choice(["pending", "in_progress", "completed", "blocked"]))
@click.option("--note", "-n", help="Optional note about the status change.")
@click.pass_context
@cli_command("tasks-update-status")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Status update timed out")
def update_status_cmd(
    ctx: click.Context,
    spec_id: str,
    task_id: str,
    status: str,
    note: Optional[str],
) -> None:
    """Update a task's status.

    SPEC_ID is the specification identifier.
    TASK_ID is the task identifier.
    STATUS is one of: pending, in_progress, completed, blocked.
    """
    cli_ctx = get_context(ctx)
    specs_dir = cli_ctx.specs_dir

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set SDD_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set SDD_SPECS_DIR"},
        )

    # Find and load spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs list",
            details={"spec_id": spec_id, "specs_dir": str(specs_dir)},
        )

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Failed to load specification: {spec_id}",
            code="INTERNAL_ERROR",
            error_type="internal",
            remediation="Check that the spec file is valid JSON",
            details={"spec_id": spec_id},
        )

    # Update status
    success = update_task_status(spec_data, task_id, status, note)
    if not success:
        emit_error(
            f"Failed to update task status: {task_id}",
            code="INTERNAL_ERROR",
            error_type="internal",
            remediation="Verify the task ID exists and the status transition is valid",
            details={"task_id": task_id, "status": status},
        )

    # Save changes
    if not save_journal(spec_data, str(spec_path), create_backup=True):
        emit_error(
            "Failed to save spec file",
            code="INTERNAL_ERROR",
            error_type="internal",
            remediation="Check file permissions and disk space",
            details={"path": str(spec_path)},
        )

    emit_success({
        "spec_id": spec_id,
        "task_id": task_id,
        "status": status,
        "note": note,
    })


@tasks.command("block")
@click.argument("spec_id")
@click.argument("task_id")
@click.option("--reason", "-r", required=True, help="Description of the blocker.")
@click.option(
    "--type", "-t", "blocker_type",
    type=click.Choice(["dependency", "technical", "resource", "decision"]),
    default="dependency",
    help="Type of blocker.",
)
@click.option("--ticket", help="Optional ticket/issue reference.")
@click.pass_context
@cli_command("tasks-block")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Block task timed out")
def block_task_cmd(
    ctx: click.Context,
    spec_id: str,
    task_id: str,
    reason: str,
    blocker_type: str,
    ticket: Optional[str],
) -> None:
    """Mark a task as blocked.

    SPEC_ID is the specification identifier.
    TASK_ID is the task identifier.
    """
    cli_ctx = get_context(ctx)
    specs_dir = cli_ctx.specs_dir

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set SDD_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set SDD_SPECS_DIR"},
        )

    # Find and load spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs list",
            details={"spec_id": spec_id},
        )

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Failed to load specification: {spec_id}",
            code="INTERNAL_ERROR",
            error_type="internal",
            remediation="Check that the spec file is valid JSON",
            details={"spec_id": spec_id},
        )

    # Mark blocked
    success = mark_blocked(spec_data, task_id, reason, blocker_type, ticket)
    if not success:
        emit_error(
            f"Failed to block task: {task_id}",
            code="INTERNAL_ERROR",
            error_type="internal",
            remediation="Verify the task ID exists",
            details={"task_id": task_id},
        )

    # Save changes
    if not save_journal(spec_data, str(spec_path), create_backup=True):
        emit_error(
            "Failed to save spec file",
            code="INTERNAL_ERROR",
            error_type="internal",
            remediation="Check file permissions and disk space",
            details={"path": str(spec_path)},
        )

    emit_success({
        "spec_id": spec_id,
        "task_id": task_id,
        "status": "blocked",
        "blocker_type": blocker_type,
        "reason": reason,
        "ticket": ticket,
    })


@tasks.command("unblock")
@click.argument("spec_id")
@click.argument("task_id")
@click.option("--resolution", "-r", help="Description of how blocker was resolved.")
@click.option(
    "--status", "-s",
    type=click.Choice(["pending", "in_progress"]),
    default="pending",
    help="Status after unblocking.",
)
@click.pass_context
@cli_command("tasks-unblock")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Unblock task timed out")
def unblock_task_cmd(
    ctx: click.Context,
    spec_id: str,
    task_id: str,
    resolution: Optional[str],
    status: str,
) -> None:
    """Unblock a task.

    SPEC_ID is the specification identifier.
    TASK_ID is the task identifier.
    """
    cli_ctx = get_context(ctx)
    specs_dir = cli_ctx.specs_dir

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set SDD_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set SDD_SPECS_DIR"},
        )

    # Find and load spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs list",
            details={"spec_id": spec_id},
        )

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Failed to load specification: {spec_id}",
            code="INTERNAL_ERROR",
            error_type="internal",
            remediation="Check that the spec file is valid JSON",
            details={"spec_id": spec_id},
        )

    # Unblock
    success = unblock(spec_data, task_id, resolution, status)
    if not success:
        emit_error(
            f"Failed to unblock task: {task_id}",
            code="CONFLICT",
            error_type="conflict",
            remediation="Verify the task is currently blocked",
            details={"task_id": task_id, "hint": "Task may not be blocked"},
        )

    # Save changes
    if not save_journal(spec_data, str(spec_path), create_backup=True):
        emit_error(
            "Failed to save spec file",
            code="INTERNAL_ERROR",
            error_type="internal",
            remediation="Check file permissions and disk space",
            details={"path": str(spec_path)},
        )

    emit_success({
        "spec_id": spec_id,
        "task_id": task_id,
        "status": status,
        "resolution": resolution,
    })
