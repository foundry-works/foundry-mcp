"""Spec lifecycle commands for SDD CLI.

Provides commands for spec lifecycle transitions: activate, complete, archive.
"""


import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error, emit_success
from foundry_mcp.cli.resilience import (
    handle_keyboard_interrupt,
    MEDIUM_TIMEOUT,
    with_sync_timeout,
)
from foundry_mcp.cli.registry import get_context

logger = get_cli_logger()
from foundry_mcp.core.lifecycle import (
    activate_spec,
    archive_spec,
    complete_spec,
    get_lifecycle_state,
    move_spec,
)


@click.group("lifecycle")
def lifecycle() -> None:
    """Spec lifecycle management commands."""
    pass


@lifecycle.command("activate")
@click.argument("spec_id")
@click.pass_context
@cli_command("activate")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Spec activation timed out")
def activate_spec_cmd(ctx: click.Context, spec_id: str) -> None:
    """Activate a specification (move from pending to active).

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

    result = activate_spec(spec_id, specs_dir)

    if not result.success:
        emit_error(
            result.error or "Failed to activate spec",
            code="CONFLICT",
            error_type="conflict",
            remediation="Verify the spec exists in pending folder and is ready for activation",
            details={"spec_id": spec_id, "from_folder": result.from_folder},
        )

    emit_success(
        {
            "spec_id": spec_id,
            "from_folder": result.from_folder,
            "to_folder": result.to_folder,
            "old_path": result.old_path,
            "new_path": result.new_path,
        }
    )


@lifecycle.command("complete")
@click.argument("spec_id")
@click.option(
    "--force", "-f", is_flag=True, help="Force completion even with incomplete tasks."
)
@click.pass_context
@cli_command("complete")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Spec completion timed out")
def complete_spec_cmd(ctx: click.Context, spec_id: str, force: bool) -> None:
    """Mark a specification as completed.

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

    result = complete_spec(spec_id, specs_dir, force=force)

    if not result.success:
        emit_error(
            result.error or "Failed to complete spec",
            code="CONFLICT",
            error_type="conflict",
            remediation="Verify all tasks are completed, or use --force to complete anyway",
            details={
                "spec_id": spec_id,
                "from_folder": result.from_folder,
                "force": force,
            },
        )

    emit_success(
        {
            "spec_id": spec_id,
            "from_folder": result.from_folder,
            "to_folder": result.to_folder,
            "old_path": result.old_path,
            "new_path": result.new_path,
        }
    )


@lifecycle.command("archive")
@click.argument("spec_id")
@click.pass_context
@cli_command("archive")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Spec archival timed out")
def archive_spec_cmd(ctx: click.Context, spec_id: str) -> None:
    """Archive a specification.

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

    result = archive_spec(spec_id, specs_dir)

    if not result.success:
        emit_error(
            result.error or "Failed to archive spec",
            code="CONFLICT",
            error_type="conflict",
            remediation="Verify the spec exists and is in a state that can be archived",
            details={"spec_id": spec_id, "from_folder": result.from_folder},
        )

    emit_success(
        {
            "spec_id": spec_id,
            "from_folder": result.from_folder,
            "to_folder": result.to_folder,
            "old_path": result.old_path,
            "new_path": result.new_path,
        }
    )


@lifecycle.command("move")
@click.argument("spec_id")
@click.argument(
    "to_folder", type=click.Choice(["pending", "active", "completed", "archived"])
)
@click.pass_context
@cli_command("move")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Spec move timed out")
def move_spec_cmd(ctx: click.Context, spec_id: str, to_folder: str) -> None:
    """Move a specification between status folders.

    SPEC_ID is the specification identifier.
    TO_FOLDER is one of: pending, active, completed, archived.
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

    result = move_spec(spec_id, to_folder, specs_dir)

    if not result.success:
        emit_error(
            result.error or "Failed to move spec",
            code="CONFLICT",
            error_type="conflict",
            remediation="Verify the spec exists and the transition is valid",
            details={
                "spec_id": spec_id,
                "from_folder": result.from_folder,
                "to_folder": to_folder,
            },
        )

    emit_success(
        {
            "spec_id": spec_id,
            "from_folder": result.from_folder,
            "to_folder": result.to_folder,
            "old_path": result.old_path,
            "new_path": result.new_path,
        }
    )


@lifecycle.command("state")
@click.argument("spec_id")
@click.pass_context
@cli_command("state")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Lifecycle state lookup timed out")
def lifecycle_state_cmd(ctx: click.Context, spec_id: str) -> None:
    """Get the current lifecycle state of a specification.

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

    state = get_lifecycle_state(spec_id, specs_dir)

    if state is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs list",
            details={"spec_id": spec_id},
        )

    emit_success(
        {
            "spec_id": state.spec_id,
            "folder": state.folder,
            "status": state.status,
            "progress_percentage": state.progress_percentage,
            "total_tasks": state.total_tasks,
            "completed_tasks": state.completed_tasks,
            "can_complete": state.can_complete,
            "can_archive": state.can_archive,
        }
    )
