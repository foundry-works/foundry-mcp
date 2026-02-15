"""Audit log management commands.

Provides commands for verifying audit ledger integrity and viewing audit history.
"""

from pathlib import Path
from typing import Optional

import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error, emit_success
from foundry_mcp.cli.resilience import MEDIUM_TIMEOUT, handle_keyboard_interrupt, with_sync_timeout

logger = get_cli_logger()


@click.group("audit")
def audit() -> None:
    """Audit ledger management commands."""
    pass


@audit.command("verify")
@click.option("--spec-id", required=True, help="Spec ID to verify audit ledger for")
@click.option(
    "--workspace",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Workspace path (default: current directory)",
)
@click.pass_context
@cli_command("verify")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Audit verification timed out")
def verify_cmd(
    ctx: click.Context,
    spec_id: str,
    workspace: Optional[Path],
) -> None:
    """Verify audit ledger hash chain integrity.

    Walks the hash-linked audit ledger and reports any tampering detected.
    Broken chains produce a warning but do not block operations.

    Examples:
        foundry audit verify --spec-id my-spec-001
        foundry audit verify --spec-id my-spec-001 --workspace /path/to/project
    """
    from foundry_mcp.core.autonomy.audit import verify_chain

    workspace_path = workspace or Path.cwd()

    result = verify_chain(spec_id=spec_id, workspace_path=workspace_path)

    if result.valid:
        emit_success(
            {
                "spec_id": spec_id,
                "valid": True,
                "total_entries": result.total_entries,
                "warnings": result.warnings,
            }
        )
    else:
        emit_error(
            f"Audit chain broken at sequence {result.divergence_point}: {result.divergence_type}",
            code="AUDIT_CHAIN_BROKEN",
            remediation=result.divergence_detail,
        )


@audit.command("list")
@click.option("--spec-id", required=True, help="Spec ID to list audit events for")
@click.option("--limit", type=int, default=50, help="Maximum events to return")
@click.option("--event-type", help="Filter by event type (step_issued, pause, etc.)")
@click.option(
    "--workspace",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Workspace path (default: current directory)",
)
@click.pass_context
@cli_command("list")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Audit list timed out")
def list_cmd(
    ctx: click.Context,
    spec_id: str,
    limit: int,
    event_type: Optional[str],
    workspace: Optional[Path],
) -> None:
    """List audit events for a spec.

    Events are returned in reverse chronological order (newest first).

    Examples:
        foundry audit list --spec-id my-spec-001
        foundry audit list --spec-id my-spec-001 --event-type pause --limit 10
    """
    from foundry_mcp.core.autonomy.audit import AuditEventType, AuditLedger

    workspace_path = workspace or Path.cwd()

    # Parse event type if provided
    filter_type = None
    if event_type:
        try:
            filter_type = AuditEventType(event_type)
        except ValueError:
            valid_types = [e.value for e in AuditEventType]
            emit_error(
                f"Invalid event type '{event_type}'",
                code="INVALID_EVENT_TYPE",
                remediation=f"Valid types: {', '.join(valid_types)}",
            )
            return

    ledger = AuditLedger(spec_id=spec_id, workspace_path=workspace_path)
    entries = ledger.get_entries(limit=limit, event_type=filter_type)

    emit_success(
        {
            "spec_id": spec_id,
            "count": len(entries),
            "total_in_ledger": ledger.get_entry_count(),
            "events": [
                {
                    "sequence": e.sequence,
                    "timestamp": e.timestamp,
                    "event_type": e.event_type.value,
                    "action": e.action,
                    "step_id": e.step_id,
                    "phase_id": e.phase_id,
                    "task_id": e.task_id,
                }
                for e in entries
            ],
        }
    )


@audit.command("path")
@click.option("--spec-id", required=True, help="Spec ID to get ledger path for")
@click.option(
    "--workspace",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Workspace path (default: current directory)",
)
@click.pass_context
@cli_command("path")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Audit path timed out")
def path_cmd(
    ctx: click.Context,
    spec_id: str,
    workspace: Optional[Path],
) -> None:
    """Show the path to a spec's audit ledger.

    Examples:
        foundry audit path --spec-id my-spec-001
    """
    from foundry_mcp.core.autonomy.audit import get_ledger_path

    workspace_path = workspace or Path.cwd()
    ledger_path = get_ledger_path(spec_id=spec_id, workspace_path=workspace_path)

    emit_success(
        {
            "spec_id": spec_id,
            "ledger_path": str(ledger_path),
            "exists": ledger_path.exists(),
        }
    )
