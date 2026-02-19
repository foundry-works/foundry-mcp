"""Foundry watch command for real-time monitoring of autonomous sessions.

Assembles session state, audit events, and spec progress from disk,
then presents them via a Rich Live dashboard or simple streaming output.
"""

import json
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error
from foundry_mcp.cli.registry import get_context
from foundry_mcp.cli.resilience import (
    FAST_TIMEOUT,
    handle_keyboard_interrupt,
    with_sync_timeout,
)

logger = get_cli_logger()

_TERMINAL_STATUSES = frozenset({"paused", "completed", "ended", "failed"})


def _assemble_watch_data(
    specs_dir: Path, spec_id: str, max_events: int
) -> dict[str, Any]:
    """Assemble session state, audit events, and spec progress.

    Args:
        specs_dir: Path to the specs directory.
        spec_id: Spec identifier.
        max_events: Maximum number of recent audit events to include.

    Returns:
        Dictionary with session, audit, and progress data.

    Raises:
        SystemExit: Via emit_error when session or spec not found.
    """
    from foundry_mcp.core.autonomy.audit import AuditLedger
    from foundry_mcp.core.autonomy.memory import AutonomyStorage
    from foundry_mcp.core.progress import recalculate_progress
    from foundry_mcp.core.spec.io import load_spec

    workspace_path = specs_dir.parent
    storage = AutonomyStorage(workspace_path=workspace_path)

    # Resolve active session
    session_id = storage.get_active_session(spec_id)
    if session_id is None:
        emit_error(
            f"No active session for spec: {spec_id}",
            code="NOT_FOUND",
            error_type="not_found",
            remediation="Start an autonomous session first, or check the spec ID",
            details={"spec_id": spec_id},
        )

    session = storage.load(session_id)
    if session is None:
        emit_error(
            f"Failed to load session: {session_id}",
            code="NOT_FOUND",
            error_type="not_found",
            remediation="The session file may be corrupted or missing",
            details={"spec_id": spec_id, "session_id": session_id},
        )

    # Load audit events
    try:
        ledger = AuditLedger(spec_id=spec_id, workspace_path=workspace_path)
        events = ledger.get_entries(limit=max_events)
        audit_data = [
            {
                "sequence": e.sequence,
                "timestamp": e.timestamp,
                "event_type": e.event_type.value,
                "action": e.action,
                "task_id": e.task_id,
                "phase_id": e.phase_id,
            }
            for e in events
        ]
    except Exception as exc:
        logger.warning(f"Failed to load audit events: {exc}")
        audit_data = []

    # Load spec progress
    spec_data = load_spec(spec_id, specs_dir)
    progress_data: dict[str, Any] = {}
    if spec_data:
        recalculate_progress(spec_data)
        root = spec_data.get("hierarchy", {}).get("spec-root", {})
        progress_data = {
            "total_tasks": root.get("total_tasks", 0),
            "completed_tasks": root.get("completed_tasks", 0),
            "status": root.get("status", "unknown"),
        }

    session_dump = session.model_dump(by_alias=True)

    # Extract heartbeat age
    heartbeat_at = session.context.last_heartbeat_at
    heartbeat_age_s: Optional[float] = None
    if heartbeat_at is not None:
        delta = datetime.now(timezone.utc) - heartbeat_at
        heartbeat_age_s = delta.total_seconds()

    # Extract last step info
    last_step: Optional[dict[str, Any]] = None
    if session.last_step_issued is not None:
        step = session.last_step_issued
        issued_delta = datetime.now(timezone.utc) - step.issued_at
        last_step = {
            "step_id": step.step_id[:12],
            "type": step.type.value,
            "task_id": step.task_id,
            "phase_id": step.phase_id,
            "seconds_ago": round(issued_delta.total_seconds(), 1),
        }

    return {
        "session": {
            "id": session_dump.get("id"),
            "spec_id": session_dump.get("spec_id"),
            "status": session_dump.get("status"),
            "active_phase_id": session.active_phase_id,
            "last_task_id": session.last_task_id,
            "context_usage_pct": session.context.context_usage_pct,
            "heartbeat_age_s": heartbeat_age_s,
            "pause_reason": session_dump.get("pause_reason"),
            "created_at": session_dump.get("created_at"),
            "updated_at": session_dump.get("updated_at"),
            "tasks_completed": session.counters.tasks_completed,
            "consecutive_errors": session.counters.consecutive_errors,
        },
        "last_step": last_step,
        "audit_events": audit_data,
        "progress": progress_data,
    }


def _build_dashboard(data: dict[str, Any]) -> Any:
    """Build a Rich Layout with status, step, and events panels.

    Args:
        data: Assembled watch data from _assemble_watch_data.

    Returns:
        Rich Layout renderable.
    """
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table

    session = data["session"]
    last_step = data.get("last_step")
    events = data.get("audit_events", [])
    progress = data.get("progress", {})

    # -- Status panel --
    total = progress.get("total_tasks", 0)
    completed = progress.get("completed_tasks", 0)
    pct = round((completed / total * 100) if total > 0 else 0, 1)
    bar_filled = int(pct / 5)  # 20-char bar
    bar_empty = 20 - bar_filled
    progress_bar = f"[green]{'█' * bar_filled}[/green][dim]{'░' * bar_empty}[/dim] {completed}/{total} ({pct}%)"

    status_val = session.get("status", "unknown")
    status_color = {
        "running": "green",
        "paused": "yellow",
        "completed": "blue",
        "failed": "red",
        "ended": "dim",
    }.get(status_val, "white")

    heartbeat = session.get("heartbeat_age_s")
    heartbeat_str = f"{heartbeat:.0f}s ago" if heartbeat is not None else "n/a"
    ctx_pct = session.get("context_usage_pct", 0)
    ctx_color = "green" if ctx_pct < 60 else ("yellow" if ctx_pct < 85 else "red")

    status_lines = [
        f"  Session   [{status_color}]{session.get('id', 'unknown')}[/{status_color}]",
        f"  Status    [{status_color}]{status_val}[/{status_color}]",
        f"  Phase     {session.get('active_phase_id') or 'none'}",
        f"  Progress  {progress_bar}",
        f"  Context   [{ctx_color}]{ctx_pct}%[/{ctx_color}]",
        f"  Heartbeat {heartbeat_str}",
    ]
    if session.get("pause_reason"):
        status_lines.append(f"  Paused    {session['pause_reason']}")
    if session.get("consecutive_errors", 0) > 0:
        status_lines.append(f"  Errors    [red]{session['consecutive_errors']} consecutive[/red]")

    status_panel = Panel(
        "\n".join(status_lines),
        title="[bold]Session Status[/bold]",
        border_style=status_color,
    )

    # -- Current step panel --
    if last_step:
        step_lines = [
            f"  Type      {last_step['type']}",
            f"  Task      {last_step.get('task_id') or 'n/a'}",
            f"  Step ID   {last_step['step_id']}",
            f"  Issued    {last_step['seconds_ago']}s ago",
        ]
    else:
        step_lines = ["  [dim]No step currently issued[/dim]"]

    step_panel = Panel(
        "\n".join(step_lines),
        title="[bold]Current Step[/bold]",
        border_style="cyan",
    )

    # -- Events table --
    events_table = Table(title="Recent Events", expand=True, show_lines=False)
    events_table.add_column("Time", style="dim", width=20)
    events_table.add_column("Event", width=18)
    events_table.add_column("Task", width=14)
    events_table.add_column("Action", ratio=1)

    for evt in reversed(events):  # most recent first
        ts = evt.get("timestamp", "")
        if len(ts) > 19:
            ts = ts[11:19]  # HH:MM:SS
        events_table.add_row(
            ts,
            evt.get("event_type", ""),
            evt.get("task_id") or evt.get("phase_id") or "",
            evt.get("action", ""),
        )

    events_panel = Panel(events_table, border_style="blue")

    # -- Assemble layout --
    layout = Layout()
    layout.split_column(
        Layout(name="top", size=max(len(status_lines) + 2, 8)),
        Layout(name="middle", size=max(len(step_lines) + 2, 5)),
        Layout(name="bottom"),
    )
    layout["top"].update(status_panel)
    layout["middle"].update(step_panel)
    layout["bottom"].update(events_panel)

    return layout


@contextmanager
def _raw_terminal() -> Iterator[None]:
    """Context manager to put terminal in raw mode for non-blocking key reads.

    Restores original terminal settings on exit.
    """
    if not sys.stdin.isatty():
        yield
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _read_key_nonblocking() -> Optional[str]:
    """Read a single keypress without blocking.

    Returns:
        The key character, or None if no key was pressed.
    """
    if not sys.stdin.isatty():
        return None

    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if readable:
        return sys.stdin.read(1)
    return None


def _write_stop_signal(specs_dir: Path, spec_id: str) -> Path:
    """Write a stop signal file for the given spec.

    Reuses the same signal file format as the stop command.

    Args:
        specs_dir: Path to the specs directory.
        spec_id: Spec identifier.

    Returns:
        Path to the written signal file.
    """
    signal_dir = specs_dir / ".autonomy" / "signals"
    signal_dir.mkdir(parents=True, exist_ok=True)

    signal_file = signal_dir / f"{spec_id}.stop"
    payload = {
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "requested_by": "foundry-watch",
        "reason": "operator_stop",
    }
    signal_file.write_text(json.dumps(payload, indent=2))
    return signal_file


def _run_live_dashboard(
    specs_dir: Path, spec_id: str, interval: float, max_events: int
) -> None:
    """Run the Rich Live dashboard polling loop.

    Polls session data every `interval` seconds and updates the display.
    Auto-exits when the session reaches a terminal state.

    Args:
        specs_dir: Path to the specs directory.
        spec_id: Spec identifier.
        interval: Refresh interval in seconds.
        max_events: Maximum number of audit events to display.
    """
    from rich.console import Console
    from rich.live import Live

    console = Console()

    data = _assemble_watch_data(specs_dir, spec_id, max_events=max_events)
    layout = _build_dashboard(data)

    with _raw_terminal(), Live(layout, console=console, refresh_per_second=1, screen=False) as live:
        while True:
            status = data["session"].get("status", "")
            if status in _TERMINAL_STATUSES:
                break

            # Non-blocking key poll within the sleep interval
            deadline = time.monotonic() + interval
            quit_requested = False
            while time.monotonic() < deadline:
                key = _read_key_nonblocking()
                if key == "q":
                    quit_requested = True
                    break
                if key == "s":
                    _write_stop_signal(specs_dir, spec_id)
                    logger.info(f"Stop signal written for {spec_id}")
                time.sleep(0.1)

            if quit_requested:
                break

            try:
                data = _assemble_watch_data(specs_dir, spec_id, max_events=max_events)
            except SystemExit:
                break

            layout = _build_dashboard(data)
            live.update(layout)


def _run_simple_stream(
    specs_dir: Path, spec_id: str, interval: float, max_events: int
) -> None:
    """Run simple streaming mode for pipe-friendly output.

    Prints a header line with session info, then tails the audit ledger
    printing one line per new event. No Rich escape codes.

    Args:
        specs_dir: Path to the specs directory.
        spec_id: Spec identifier.
        interval: Poll interval in seconds.
        max_events: Maximum events for initial fetch.
    """
    from foundry_mcp.core.autonomy.audit import AuditLedger
    from foundry_mcp.core.autonomy.memory import AutonomyStorage

    workspace_path = specs_dir.parent
    storage = AutonomyStorage(workspace_path=workspace_path)

    session_id = storage.get_active_session(spec_id)
    if session_id is None:
        emit_error(
            f"No active session for spec: {spec_id}",
            code="NOT_FOUND",
            error_type="not_found",
            remediation="Start an autonomous session first, or check the spec ID",
            details={"spec_id": spec_id},
        )

    session = storage.load(session_id)
    if session is None:
        emit_error(
            f"Failed to load session: {session_id}",
            code="NOT_FOUND",
            error_type="not_found",
            remediation="The session file may be corrupted or missing",
            details={"spec_id": spec_id, "session_id": session_id},
        )

    # Print header
    click.echo(
        f"session={session.id} spec={spec_id} status={session.status.value} "
        f"phase={session.active_phase_id or 'none'}"
    )

    # Initial event fetch to set baseline sequence
    last_seq = 0
    try:
        ledger = AuditLedger(spec_id=spec_id, workspace_path=workspace_path)
        initial_events = ledger.get_entries(limit=max_events)
        if initial_events:
            last_seq = initial_events[-1].sequence
            for evt in initial_events:
                _print_event_line(evt)
    except Exception as exc:
        logger.warning(f"Failed to load initial events: {exc}")

    # Tail loop
    while True:
        session = storage.load(session_id)
        if session is None or session.status.value in _TERMINAL_STATUSES:
            status_val = session.status.value if session else "unknown"
            click.echo(f"--- session {status_val} ---")
            break

        time.sleep(interval)

        try:
            ledger = AuditLedger(spec_id=spec_id, workspace_path=workspace_path)
            new_events = ledger.get_entries(limit=50, since_sequence=last_seq)
            for evt in new_events:
                _print_event_line(evt)
                last_seq = evt.sequence
        except Exception as exc:
            logger.warning(f"Failed to poll events: {exc}")


def _print_event_line(evt: Any) -> None:
    """Print a single audit event as a plain-text line."""
    ts = evt.timestamp
    if len(ts) > 19:
        ts = ts[:19]
    task_or_phase = evt.task_id or evt.phase_id or ""
    click.echo(f"{ts}  {evt.event_type.value:<18s}  {task_or_phase:<16s}  {evt.action}")


@click.command("watch")
@click.argument("spec_id")
@click.option(
    "--interval",
    "-n",
    type=float,
    default=2.0,
    show_default=True,
    help="Refresh interval in seconds.",
)
@click.option(
    "--events",
    type=int,
    default=10,
    show_default=True,
    help="Number of recent audit events to display.",
)
@click.option(
    "--simple",
    is_flag=True,
    help="Use simple streaming output instead of Rich Live dashboard.",
)
@click.pass_context
@cli_command("watch")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Watch command timed out")
def watch_cmd(
    ctx: click.Context,
    spec_id: str,
    interval: float,
    events: int,
    simple: bool,
) -> None:
    """Monitor an autonomous session for SPEC_ID in real time.

    Displays session state, audit events, and spec progress.
    Press Ctrl+C to exit.
    """
    cli_ctx = get_context(ctx)
    specs_dir = cli_ctx.specs_dir

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set FOUNDRY_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set FOUNDRY_SPECS_DIR"},
        )

    from foundry_mcp.core.spec import resolve_spec_file

    spec_path = resolve_spec_file(spec_id, specs_dir)
    if spec_path is None:
        emit_error(
            f"Spec not found: {spec_id}",
            code="NOT_FOUND",
            error_type="not_found",
            remediation="Check the spec ID and ensure the spec exists",
            details={"spec_id": spec_id},
        )

    if simple:
        _run_simple_stream(specs_dir, spec_id, interval, max_events=events)
    else:
        _run_live_dashboard(specs_dir, spec_id, interval, max_events=events)
