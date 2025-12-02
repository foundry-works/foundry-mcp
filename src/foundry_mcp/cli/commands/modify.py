"""Modification commands for SDD CLI.

Provides commands for modifying SDD specifications including:
- Applying bulk modifications from JSON files
- Adding/removing tasks
- Managing assumptions and revisions
- Updating frontmatter metadata
"""

import json
import subprocess
import time
from pathlib import Path
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
@cli_command("modify-apply")
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

        emit_success({
            "spec_id": spec_id,
            "dry_run": dry_run,
            "modifications_applied": applied,
            "modifications_skipped": skipped,
            "changes": changes,
            "output_path": output_file if output_file else str(specs_dir),
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

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
@cli_command("modify-task-add")
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
        emit_success({
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
        })
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

    emit_success({
        "spec_id": spec_id,
        "parent": parent,
        "title": title,
        "type": task_type,
        "dry_run": False,
        **result,
        "telemetry": {"duration_ms": round(duration_ms, 2)},
    })


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
@cli_command("modify-task-remove")
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
        emit_success({
            "spec_id": spec_id,
            "task_id": task_id,
            "cascade": cascade,
            "dry_run": True,
            "preview": {
                "action": "remove_task",
                "cascade": cascade,
            },
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })
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

    emit_success({
        "spec_id": spec_id,
        "task_id": task_id,
        "cascade": cascade,
        "dry_run": False,
        **result,
        "telemetry": {"duration_ms": round(duration_ms, 2)},
    })


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
@cli_command("modify-assumption")
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

    # Build command
    cmd = ["sdd", "add-assumption", spec_id, "--text", text, "--json"]

    if assumption_type:
        cmd.extend(["--type", assumption_type])
    if author:
        cmd.extend(["--author", author])
    if dry_run:
        cmd.append("--dry-run")
    if specs_dir:
        cmd.extend(["--path", str(specs_dir.parent)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=MEDIUM_TIMEOUT,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Add assumption failed"
            emit_error(
                f"Add assumption failed: {error_msg}",
                code="ADD_FAILED",
                error_type="internal",
                remediation="Verify spec exists and assumption format is valid",
                details={
                    "spec_id": spec_id,
                    "exit_code": result.returncode,
                },
            )
            return

        # Parse output
        try:
            assumption_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            assumption_data = {"raw_output": result.stdout}

        emit_success({
            "spec_id": spec_id,
            "text": text,
            "type": assumption_type,
            "dry_run": dry_run,
            **assumption_data,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

    except subprocess.TimeoutExpired:
        emit_error(
            f"Add assumption timed out after {MEDIUM_TIMEOUT}s",
            code="TIMEOUT",
            error_type="internal",
            remediation="Try again or check system resources",
            details={"spec_id": spec_id},
        )
    except FileNotFoundError:
        emit_error(
            "SDD CLI not found",
            code="CLI_NOT_FOUND",
            error_type="internal",
            remediation="Ensure 'sdd' is installed and in PATH",
            details={"hint": "Ensure 'sdd' is installed and in PATH"},
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
@cli_command("modify-revision")
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

    # Build command
    cmd = ["sdd", "add-revision", spec_id, "--version", version, "--changes", changes, "--json"]

    if author:
        cmd.extend(["--author", author])
    if dry_run:
        cmd.append("--dry-run")
    if specs_dir:
        cmd.extend(["--path", str(specs_dir.parent)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=MEDIUM_TIMEOUT,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Add revision failed"
            emit_error(
                f"Add revision failed: {error_msg}",
                code="ADD_FAILED",
                error_type="internal",
                remediation="Verify spec exists and revision format is valid",
                details={
                    "spec_id": spec_id,
                    "exit_code": result.returncode,
                },
            )
            return

        # Parse output
        try:
            revision_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            revision_data = {"raw_output": result.stdout}

        emit_success({
            "spec_id": spec_id,
            "version": version,
            "changes": changes,
            "dry_run": dry_run,
            **revision_data,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

    except subprocess.TimeoutExpired:
        emit_error(
            f"Add revision timed out after {MEDIUM_TIMEOUT}s",
            code="TIMEOUT",
            error_type="internal",
            remediation="Try again or check system resources",
            details={"spec_id": spec_id},
        )
    except FileNotFoundError:
        emit_error(
            "SDD CLI not found",
            code="CLI_NOT_FOUND",
            error_type="internal",
            remediation="Ensure 'sdd' is installed and in PATH",
            details={"hint": "Ensure 'sdd' is installed and in PATH"},
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
@cli_command("modify-frontmatter")
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

    # Build command
    cmd = ["sdd", "update-frontmatter", spec_id, "--key", key, "--value", value, "--json"]

    if dry_run:
        cmd.append("--dry-run")
    if specs_dir:
        cmd.extend(["--path", str(specs_dir.parent)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=MEDIUM_TIMEOUT,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Update frontmatter failed"
            emit_error(
                f"Update frontmatter failed: {error_msg}",
                code="UPDATE_FAILED",
                error_type="internal",
                remediation="Verify spec exists and frontmatter key is valid",
                details={
                    "spec_id": spec_id,
                    "key": key,
                    "exit_code": result.returncode,
                },
            )
            return

        # Parse output
        try:
            frontmatter_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            frontmatter_data = {"raw_output": result.stdout}

        emit_success({
            "spec_id": spec_id,
            "key": key,
            "value": value,
            "dry_run": dry_run,
            **frontmatter_data,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

    except subprocess.TimeoutExpired:
        emit_error(
            f"Update frontmatter timed out after {MEDIUM_TIMEOUT}s",
            code="TIMEOUT",
            error_type="internal",
            remediation="Try again or check system resources",
            details={"spec_id": spec_id},
        )
    except FileNotFoundError:
        emit_error(
            "SDD CLI not found",
            code="CLI_NOT_FOUND",
            error_type="internal",
            remediation="Ensure 'sdd' is installed and in PATH",
            details={"hint": "Ensure 'sdd' is installed and in PATH"},
        )
