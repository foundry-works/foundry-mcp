"""Journal management commands for SDD CLI.

Provides commands for managing journal entries in specifications.
"""

from typing import Optional

import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error, emit_success
from foundry_mcp.cli.registry import get_context
from foundry_mcp.cli.resilience import (
    FAST_TIMEOUT,
    MEDIUM_TIMEOUT,
    handle_keyboard_interrupt,
    with_sync_timeout,
)
from foundry_mcp.core.spec import load_spec, find_spec_file
from foundry_mcp.core.journal import (
    add_journal_entry,
    find_unjournaled_tasks,
    get_journal_entries,
    save_journal,
)

logger = get_cli_logger()


@click.group("journal")
def journal() -> None:
    """Journal entry management."""
    pass


@journal.command("add")
@click.argument("spec_id")
@click.option("--title", "-t", required=True, help="Entry title.")
@click.option("--content", "-c", required=True, help="Entry content.")
@click.option(
    "--type",
    "-T",
    "entry_type",
    type=click.Choice(["note", "decision", "blocker", "deviation", "status_change"]),
    default="note",
    help="Type of journal entry.",
)
@click.option("--task-id", help="Associated task ID (optional).")
@click.pass_context
@cli_command("add")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Journal add timed out")
def journal_add_cmd(
    ctx: click.Context,
    spec_id: str,
    title: str,
    content: str,
    entry_type: str,
    task_id: Optional[str],
) -> None:
    """Add a journal entry to a specification.

    SPEC_ID is the specification identifier.

    Examples:
        sdd journal add my-spec --title "Progress update" --content "Completed phase 1"
        sdd journal add my-spec -t "Decision" -c "Using Redis for cache" --type decision
        sdd journal add my-spec -t "Task note" -c "Found edge case" --task-id task-2-1
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
        return

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
        return

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Failed to load specification: {spec_id}",
            code="INTERNAL_ERROR",
            error_type="internal",
            remediation="Check that the spec file is valid JSON",
            details={"spec_id": spec_id},
        )
        return

    # Validate task_id if provided
    if task_id:
        hierarchy = spec_data.get("hierarchy", {})
        if task_id not in hierarchy:
            emit_error(
                f"Task not found: {task_id}",
                code="TASK_NOT_FOUND",
                error_type="not_found",
                remediation="Verify the task ID exists using: sdd task-info <spec_id> <task_id>",
                details={"spec_id": spec_id, "task_id": task_id},
            )
            return

    # Add journal entry
    entry = add_journal_entry(
        spec_data,
        title=title,
        content=content,
        entry_type=entry_type,
        task_id=task_id,
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
        return

    emit_success(
        {
            "spec_id": spec_id,
            "entry": {
                "timestamp": entry.timestamp,
                "entry_type": entry.entry_type,
                "title": entry.title,
                "content": entry.content,
                "task_id": entry.task_id,
            },
        }
    )


@journal.command("list")
@click.argument("spec_id")
@click.option("--task-id", help="Filter by task ID.")
@click.option(
    "--type",
    "-T",
    "entry_type",
    type=click.Choice(["note", "decision", "blocker", "deviation", "status_change"]),
    help="Filter by entry type.",
)
@click.option("--limit", "-l", type=int, help="Limit number of entries returned.")
@click.pass_context
@cli_command("list")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Journal list timed out")
def journal_list_cmd(
    ctx: click.Context,
    spec_id: str,
    task_id: Optional[str],
    entry_type: Optional[str],
    limit: Optional[int],
) -> None:
    """List journal entries from a specification.

    SPEC_ID is the specification identifier.

    Returns entries sorted by timestamp (most recent first).

    Examples:
        sdd journal list my-spec
        sdd journal list my-spec --task-id task-2-1
        sdd journal list my-spec --type decision --limit 5
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
        return

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs list",
            details={"spec_id": spec_id, "specs_dir": str(specs_dir)},
        )
        return

    # Get journal entries
    entries = get_journal_entries(
        spec_data,
        task_id=task_id,
        entry_type=entry_type,
        limit=limit,
    )

    emit_success(
        {
            "spec_id": spec_id,
            "entry_count": len(entries),
            "filters": {
                "task_id": task_id,
                "entry_type": entry_type,
                "limit": limit,
            },
            "entries": [
                {
                    "timestamp": e.timestamp,
                    "entry_type": e.entry_type,
                    "title": e.title,
                    "content": e.content,
                    "task_id": e.task_id,
                    "author": e.author,
                }
                for e in entries
            ],
        }
    )


@journal.command("unjournaled")
@click.argument("spec_id")
@click.pass_context
@cli_command("unjournaled")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Journal unjournaled lookup timed out")
def journal_unjournaled_cmd(
    ctx: click.Context,
    spec_id: str,
) -> None:
    """List completed tasks that need journal entries.

    SPEC_ID is the specification identifier.

    Returns tasks that are marked as completed but have the
    needs_journaling flag set to true.

    Examples:
        sdd journal unjournaled my-spec
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
        return

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs list",
            details={"spec_id": spec_id, "specs_dir": str(specs_dir)},
        )
        return

    # Find unjournaled tasks
    unjournaled = find_unjournaled_tasks(spec_data)

    emit_success(
        {
            "spec_id": spec_id,
            "count": len(unjournaled),
            "tasks": unjournaled,
        }
    )


@journal.command("get")
@click.argument("spec_id")
@click.option("--task-id", help="Filter by task ID.")
@click.option(
    "--last", "-n", "last_n", type=int, help="Get last N entries (most recent)."
)
@click.pass_context
@cli_command("get")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Journal get timed out")
def journal_get_cmd(
    ctx: click.Context,
    spec_id: str,
    task_id: Optional[str],
    last_n: Optional[int],
) -> None:
    """Get journal entries for a specification or task.

    SPEC_ID is the specification identifier.

    Retrieves journal entries, optionally filtered by task.
    Use --last to limit to the N most recent entries.

    Examples:
        sdd get-journal my-spec
        sdd get-journal my-spec --task-id task-2-1
        sdd get-journal my-spec --last 5
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
        return

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs list",
            details={"spec_id": spec_id, "specs_dir": str(specs_dir)},
        )
        return

    # Get journal entries
    entries = get_journal_entries(
        spec_data,
        task_id=task_id,
        limit=last_n,
    )

    emit_success(
        {
            "spec_id": spec_id,
            "task_id": task_id,
            "entry_count": len(entries),
            "entries": [
                {
                    "timestamp": e.timestamp,
                    "entry_type": e.entry_type,
                    "title": e.title,
                    "content": e.content,
                    "task_id": e.task_id,
                    "author": e.author,
                }
                for e in entries
            ],
        }
    )
