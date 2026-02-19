"""Foundry stop command for gracefully stopping autonomous sessions.

Writes a signal file that the orchestrator checks at Step 7b to trigger
a clean PAUSE without modifying session state directly.
"""

import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error, emit_success
from foundry_mcp.cli.registry import get_context
from foundry_mcp.cli.resilience import (
    FAST_TIMEOUT,
    handle_keyboard_interrupt,
    with_sync_timeout,
)

logger = get_cli_logger()


@click.command("stop")
@click.argument("spec_id")
@click.option("--force", is_flag=True, help="Force-kill the session process.")
@click.option("--wait", is_flag=True, help="Poll until session is paused/stopped.")
@click.option(
    "--timeout",
    type=int,
    default=120,
    show_default=True,
    help="Timeout in seconds for --wait mode.",
)
@click.pass_context
@cli_command("stop")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Stop command timed out")
def stop_cmd(
    ctx: click.Context,
    spec_id: str,
    force: bool,
    wait: bool,
    timeout: int,
) -> None:
    """Gracefully stop an autonomous session for SPEC_ID.

    Writes a signal file that the orchestrator consumes on its next step
    cycle, triggering a clean PAUSE with reason 'user'.
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

    # Write signal file
    signal_dir = specs_dir / ".autonomy" / "signals"
    signal_dir.mkdir(parents=True, exist_ok=True)

    signal_file = signal_dir / f"{spec_id}.stop"
    payload = {
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "requested_by": "foundry-cli",
        "reason": "operator_stop",
    }

    try:
        signal_file.write_text(json.dumps(payload, indent=2))
    except OSError as e:
        emit_error(
            f"Failed to write signal file: {e}",
            code="IO_ERROR",
            error_type="io",
            remediation="Check file system permissions for the specs directory",
            details={"signal_path": str(signal_file)},
        )

    killed_pids: list[int] = []
    if force:
        killed_pids = _force_kill_processes(spec_id)

    if wait:
        final_status = _wait_for_stop(specs_dir, spec_id, timeout)
        emit_success(
            {
                "spec_id": spec_id,
                "signal_file": str(signal_file),
                "action": "stopped" if final_status else "timeout",
                "killed_pids": killed_pids if force else None,
                "final_status": final_status,
                "message": (
                    f"Session reached status '{final_status}'."
                    if final_status
                    else f"Timed out after {timeout}s waiting for session to stop."
                ),
            }
        )
    else:
        emit_success(
            {
                "spec_id": spec_id,
                "signal_file": str(signal_file),
                "action": "force_stop" if force else "stop_requested",
                "killed_pids": killed_pids if force else None,
                "message": (
                    f"Stop signal written for spec {spec_id}. "
                    + (
                        f"Force-killed {len(killed_pids)} process(es)."
                        if force
                        else "The session will pause on the next orchestrator cycle."
                    )
                ),
            }
        )


def _force_kill_processes(spec_id: str) -> list[int]:
    """Find and SIGTERM processes matching the spec_id.

    Uses pgrep to find processes with the spec_id in their command line,
    then sends SIGTERM to each.

    Returns:
        List of PIDs that were successfully signaled.
    """
    killed: list[int] = []

    try:
        result = subprocess.run(
            ["pgrep", "-f", f"foundry.*{spec_id}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.info(f"No matching processes found for spec {spec_id}")
            return killed

        pids = [int(p) for p in result.stdout.strip().split("\n") if p.strip()]
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        logger.warning(f"Failed to search for processes: {e}")
        return killed

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
            logger.info(f"Sent SIGTERM to PID {pid}")
        except ProcessLookupError:
            logger.info(f"PID {pid} already exited")
        except PermissionError:
            logger.warning(f"Permission denied sending SIGTERM to PID {pid}")

    return killed


_TERMINAL_STATUSES = frozenset({"paused", "completed", "ended", "failed"})
_POLL_INTERVAL = 2  # seconds


def _wait_for_stop(specs_dir: Path, spec_id: str, timeout: int) -> str | None:
    """Poll session status until it reaches a terminal state or times out.

    Args:
        specs_dir: Path to the specs directory.
        spec_id: Spec identifier.
        timeout: Maximum seconds to wait.

    Returns:
        Final session status string, or None on timeout.
    """
    from foundry_mcp.core.autonomy.memory import AutonomyStorage

    workspace_path = specs_dir.parent
    storage = AutonomyStorage(workspace_path=workspace_path)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        session_id = storage.get_active_session(spec_id)
        if session_id is None:
            return "no_active_session"

        session = storage.load(session_id)
        if session is None:
            return "no_active_session"

        status = session.status.value
        if status in _TERMINAL_STATUSES:
            return status

        time.sleep(_POLL_INTERVAL)

    return None
