"""Modification commands for SDD CLI.

Provides commands for modifying SDD specifications including:
- Applying bulk modifications from JSON files
- Adding/removing tasks
- Managing assumptions and revisions
- Updating frontmatter metadata
"""

import json
import time
from typing import Optional

import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error, emit_success
from foundry_mcp.cli.registry import get_context
from foundry_mcp.cli.resilience import (
    MEDIUM_TIMEOUT,
    with_sync_timeout,
    handle_keyboard_interrupt,
)
from foundry_mcp.core.modifications import apply_modifications, load_modifications_file
from foundry_mcp.core.task import add_task, remove_task
from foundry_mcp.core.spec import (
    add_assumption,
    add_phase,
    add_revision,
    update_frontmatter,
)

logger = get_cli_logger()

# Default timeout for modification operations
MODIFY_TIMEOUT = 60


@click.group("modify")
def modify_group() -> None:
    """Spec modification commands."""
    pass


@modify_group.command("apply")
@click.argument("spec_id")
@click.argument("modifications_file", type=click.Path(exists=True))
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview changes without applying.",
)
@click.option(
    "--output",
    "output_file",
    type=click.Path(),
    help="Output path for modified spec (default: overwrite original).",
)
@click.pass_context
@cli_command("apply")
@handle_keyboard_interrupt()
@with_sync_timeout(MODIFY_TIMEOUT, "Apply modifications timed out")
def modify_apply_cmd(
    ctx: click.Context,
    spec_id: str,
    modifications_file: str,
    dry_run: bool,
    output_file: Optional[str],
) -> None:
    """Apply bulk modifications from a JSON file.

    SPEC_ID is the specification identifier.
    MODIFICATIONS_FILE is the path to the JSON modifications file.

    The modifications file should contain structured changes like
    task additions, updates, removals, and metadata modifications.
    """
    start_time = time.perf_counter()
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
        return

    try:
        # Load modifications from file
        modifications = load_modifications_file(modifications_file)

        # Apply modifications using native Python API
        applied, skipped, changes = apply_modifications(
            spec_id=spec_id,
            modifications=modifications,
            specs_dir=specs_dir,
            dry_run=dry_run,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success(
            {
                "spec_id": spec_id,
                "dry_run": dry_run,
                "modifications_applied": applied,
                "modifications_skipped": skipped,
                "changes": changes,
                "output_path": output_file if output_file else str(specs_dir),
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            }
        )

    except FileNotFoundError as e:
        emit_error(
            str(e),
            code="FILE_NOT_FOUND",
            error_type="validation",
            remediation="Check that the modifications file and spec exist",
            details={
                "spec_id": spec_id,
                "modifications_file": modifications_file,
            },
        )
    except json.JSONDecodeError as e:
        emit_error(
            f"Invalid JSON in modifications file: {e}",
            code="INVALID_JSON",
            error_type="validation",
            remediation="Check that the modifications file is valid JSON",
            details={
                "spec_id": spec_id,
                "modifications_file": modifications_file,
            },
        )
    except ValueError as e:
        emit_error(
            str(e),
            code="APPLY_FAILED",
            error_type="internal",
            remediation="Verify spec exists and modifications are valid",
            details={
                "spec_id": spec_id,
                "modifications_file": modifications_file,
            },
        )


# Phase subgroup
@modify_group.group("phase")
def modify_phase_group() -> None:
    """Phase modification commands."""
    pass


@modify_phase_group.command("add")
@click.argument("spec_id")
@click.option(
    "--title",
    required=True,
    help="Phase title.",
)
@click.option(
    "--description",
    help="Phase description or scope.",
)
@click.option(
    "--purpose",
    help="Phase purpose metadata.",
)
@click.option(
    "--estimated-hours",
    "estimated_hours",
    type=float,
    help="Estimated hours for this phase.",
)
@click.option(
    "--position",
    type=int,
    help="Insertion index under spec-root children (0-based).",
)
@click.option(
    "--link-previous/--no-link-previous",
    default=True,
    show_default=True,
    help="Automatically block on the previous phase when appending.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview changes without applying.",
)
@click.pass_context
@cli_command("phase-add")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Add phase timed out")
def modify_phase_add_cmd(
    ctx: click.Context,
    spec_id: str,
    title: str,
    description: Optional[str],
    purpose: Optional[str],
    estimated_hours: Optional[float],
    position: Optional[int],
    link_previous: bool,
    dry_run: bool,
) -> None:
    """Add a new phase to a specification."""
    start_time = time.perf_counter()
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
        return

    if dry_run:
        duration_ms = (time.perf_counter() - start_time) * 1000
        emit_success(
            {
                "spec_id": spec_id,
                "title": title,
                "dry_run": True,
                "preview": {
                    "action": "add_phase",
                    "description": description,
                    "purpose": purpose,
                    "estimated_hours": estimated_hours,
                    "position": position,
                    "link_previous": link_previous,
                },
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            }
        )
        return

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

    duration_ms = (time.perf_counter() - start_time) * 1000

    if error:
        emit_error(
            f"Add phase failed: {error}",
            code="ADD_FAILED",
            error_type="internal",
            remediation="Check that the spec exists and parameters are valid",
            details={
                "spec_id": spec_id,
                "title": title,
            },
        )
        return

    emit_success(
        {
            "spec_id": spec_id,
            "dry_run": False,
            **result,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        }
    )


# Task subgroup
@modify_group.group("task")
def modify_task_group() -> None:
    """Task modification commands."""
    pass


@modify_task_group.command("add")
@click.argument("spec_id")
@click.option(
    "--parent",
    required=True,
    help="Parent node ID (e.g., phase-1, task-2-1).",
)
@click.option(
    "--title",
    required=True,
    help="Task title.",
)
@click.option(
    "--description",
    help="Task description.",
)
@click.option(
    "--type",
    "task_type",
    type=click.Choice(["task", "subtask", "verify"]),
    default="task",
    help="Task type.",
)
@click.option(
    "--hours",
    type=float,
    help="Estimated hours.",
)
@click.option(
    "--position",
    type=int,
    help="Position in parent's children list (0-based).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview changes without applying.",
)
@click.pass_context
@cli_command("task-add")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Add task timed out")
def modify_task_add_cmd(
    ctx: click.Context,
    spec_id: str,
    parent: str,
    title: str,
    description: Optional[str],
    task_type: str,
    hours: Optional[float],
    position: Optional[int],
    dry_run: bool,
) -> None:
    """Add a new task to a specification.

    SPEC_ID is the specification identifier.
    """
    start_time = time.perf_counter()
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
        return

    if dry_run:
        # For dry_run, we just validate and preview without saving
        # The native add_task function doesn't support dry_run directly,
        # so we emit a preview response
        duration_ms = (time.perf_counter() - start_time) * 1000
        emit_success(
            {
                "spec_id": spec_id,
                "parent": parent,
                "title": title,
                "type": task_type,
                "dry_run": True,
                "preview": {
                    "action": "add_task",
                    "description": description,
                    "estimated_hours": hours,
                    "position": position,
                },
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            }
        )
        return

    # Use native add_task function
    result, error = add_task(
        spec_id=spec_id,
        parent_id=parent,
        title=title,
        description=description,
        task_type=task_type,
        estimated_hours=hours,
        position=position,
        specs_dir=specs_dir,
    )

    duration_ms = (time.perf_counter() - start_time) * 1000

    if error:
        emit_error(
            f"Add task failed: {error}",
            code="ADD_FAILED",
            error_type="internal",
            remediation="Check that the parent node exists and parameters are valid",
            details={
                "spec_id": spec_id,
                "parent": parent,
            },
        )
        return

    emit_success(
        {
            "spec_id": spec_id,
            "parent": parent,
            "title": title,
            "type": task_type,
            "dry_run": False,
            **result,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        }
    )


@modify_task_group.command("remove")
@click.argument("spec_id")
@click.argument("task_id")
@click.option(
    "--cascade",
    is_flag=True,
    help="Also remove all child tasks recursively.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview changes without applying.",
)
@click.pass_context
@cli_command("task-remove")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Remove task timed out")
def modify_task_remove_cmd(
    ctx: click.Context,
    spec_id: str,
    task_id: str,
    cascade: bool,
    dry_run: bool,
) -> None:
    """Remove a task from a specification.

    SPEC_ID is the specification identifier.
    TASK_ID is the task to remove.
    """
    start_time = time.perf_counter()
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
        return

    if dry_run:
        # For dry_run, emit a preview response
        duration_ms = (time.perf_counter() - start_time) * 1000
        emit_success(
            {
                "spec_id": spec_id,
                "task_id": task_id,
                "cascade": cascade,
                "dry_run": True,
                "preview": {
                    "action": "remove_task",
                    "cascade": cascade,
                },
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            }
        )
        return

    # Use native remove_task function
    result, error = remove_task(
        spec_id=spec_id,
        task_id=task_id,
        cascade=cascade,
        specs_dir=specs_dir,
    )

    duration_ms = (time.perf_counter() - start_time) * 1000

    if error:
        emit_error(
            f"Remove task failed: {error}",
            code="REMOVE_FAILED",
            error_type="internal",
            remediation="Verify task exists and spec structure is valid",
            details={
                "spec_id": spec_id,
                "task_id": task_id,
            },
        )
        return

    emit_success(
        {
            "spec_id": spec_id,
            "task_id": task_id,
            "cascade": cascade,
            "dry_run": False,
            **result,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        }
    )


@modify_group.command("assumption")
@click.argument("spec_id")
@click.option(
    "--text",
    required=True,
    help="Assumption text/description.",
)
@click.option(
    "--type",
    "assumption_type",
    type=click.Choice(["constraint", "requirement"]),
    help="Type of assumption.",
)
@click.option(
    "--author",
    help="Author who added the assumption.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview changes without applying.",
)
@click.pass_context
@cli_command("assumption")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Add assumption timed out")
def modify_assumption_cmd(
    ctx: click.Context,
    spec_id: str,
    text: str,
    assumption_type: Optional[str],
    author: Optional[str],
    dry_run: bool,
) -> None:
    """Add an assumption to a specification.

    SPEC_ID is the specification identifier.
    """
    start_time = time.perf_counter()
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
        return

    if dry_run:
        # For dry_run, emit a preview response
        duration_ms = (time.perf_counter() - start_time) * 1000
        emit_success(
            {
                "spec_id": spec_id,
                "text": text,
                "type": assumption_type or "constraint",
                "dry_run": True,
                "preview": {
                    "action": "add_assumption",
                    "author": author,
                },
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            }
        )
        return

    # Use native add_assumption function
    result, error = add_assumption(
        spec_id=spec_id,
        text=text,
        assumption_type=assumption_type or "constraint",
        author=author,
        specs_dir=specs_dir,
    )

    duration_ms = (time.perf_counter() - start_time) * 1000

    if error:
        emit_error(
            f"Add assumption failed: {error}",
            code="ADD_FAILED",
            error_type="internal",
            remediation="Verify spec exists and assumption format is valid",
            details={
                "spec_id": spec_id,
            },
        )
        return

    emit_success(
        {
            "spec_id": spec_id,
            "text": text,
            "type": assumption_type or "constraint",
            "dry_run": False,
            **result,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        }
    )


@modify_group.command("revision")
@click.argument("spec_id")
@click.option(
    "--version",
    required=True,
    help="Revision version (e.g., 1.1, 2.0).",
)
@click.option(
    "--changes",
    required=True,
    help="Summary of changes.",
)
@click.option(
    "--author",
    help="Revision author.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview changes without applying.",
)
@click.pass_context
@cli_command("revision")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Add revision timed out")
def modify_revision_cmd(
    ctx: click.Context,
    spec_id: str,
    version: str,
    changes: str,
    author: Optional[str],
    dry_run: bool,
) -> None:
    """Add a revision history entry to a specification.

    SPEC_ID is the specification identifier.
    """
    start_time = time.perf_counter()
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
        return

    if dry_run:
        # For dry_run, emit a preview response
        duration_ms = (time.perf_counter() - start_time) * 1000
        emit_success(
            {
                "spec_id": spec_id,
                "version": version,
                "changes": changes,
                "dry_run": True,
                "preview": {
                    "action": "add_revision",
                    "author": author,
                },
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            }
        )
        return

    # Use native add_revision function
    result, error = add_revision(
        spec_id=spec_id,
        version=version,
        changelog=changes,
        author=author,
        specs_dir=specs_dir,
    )

    duration_ms = (time.perf_counter() - start_time) * 1000

    if error:
        emit_error(
            f"Add revision failed: {error}",
            code="ADD_FAILED",
            error_type="internal",
            remediation="Verify spec exists and revision format is valid",
            details={
                "spec_id": spec_id,
            },
        )
        return

    emit_success(
        {
            "spec_id": spec_id,
            "version": version,
            "changes": changes,
            "dry_run": False,
            **result,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        }
    )


@modify_group.command("frontmatter")
@click.argument("spec_id")
@click.option(
    "--key",
    required=True,
    help="Frontmatter key to update (e.g., title, status, version).",
)
@click.option(
    "--value",
    required=True,
    help="New value for the key.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview changes without applying.",
)
@click.pass_context
@cli_command("frontmatter")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Update frontmatter timed out")
def modify_frontmatter_cmd(
    ctx: click.Context,
    spec_id: str,
    key: str,
    value: str,
    dry_run: bool,
) -> None:
    """Update frontmatter metadata in a specification.

    SPEC_ID is the specification identifier.
    """
    start_time = time.perf_counter()
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
        return

    if dry_run:
        # For dry_run, emit a preview response
        duration_ms = (time.perf_counter() - start_time) * 1000
        emit_success(
            {
                "spec_id": spec_id,
                "key": key,
                "value": value,
                "dry_run": True,
                "preview": {
                    "action": "update_frontmatter",
                },
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            }
        )
        return

    # Use native update_frontmatter function
    result, error = update_frontmatter(
        spec_id=spec_id,
        key=key,
        value=value,
        specs_dir=specs_dir,
    )

    duration_ms = (time.perf_counter() - start_time) * 1000

    if error:
        emit_error(
            f"Update frontmatter failed: {error}",
            code="UPDATE_FAILED",
            error_type="internal",
            remediation="Verify spec exists and frontmatter key is valid",
            details={
                "spec_id": spec_id,
                "key": key,
            },
        )
        return

    emit_success(
        {
            "spec_id": spec_id,
            "key": key,
            "value": value,
            "dry_run": False,
            **result,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        }
    )
