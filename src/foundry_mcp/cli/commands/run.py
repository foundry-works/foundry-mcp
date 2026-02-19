"""Foundry run command for launching autonomous sessions in tmux.

Creates a tmux session with the agent in one pane and the watcher in another,
providing the best DX for running autonomous sessions.
"""

import shutil
import subprocess

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


@click.command("run")
@click.argument("spec_id")
@click.option(
    "--posture",
    type=click.Choice(["unattended", "supervised"], case_sensitive=False),
    default="unattended",
    show_default=True,
    help="Autonomy posture for the agent.",
)
@click.option(
    "--detach",
    "-d",
    is_flag=True,
    help="Detach from the tmux session after creation.",
)
@click.option(
    "--layout",
    type=click.Choice(["horizontal", "vertical"], case_sensitive=False),
    default="vertical",
    show_default=True,
    help="Pane split direction (vertical = top/bottom, horizontal = side-by-side).",
)
@click.pass_context
@cli_command("run")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Run command timed out")
def run_cmd(
    ctx: click.Context,
    spec_id: str,
    posture: str,
    detach: bool,
    layout: str,
) -> None:
    """Launch an autonomous session for SPEC_ID in a tmux session.

    Creates a tmux session with two panes: the agent running claude in one
    pane and the foundry watcher in the other.
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

    # Verify tmux is installed
    if shutil.which("tmux") is None:
        emit_error(
            "tmux is not installed or not found in PATH",
            code="DEPENDENCY_MISSING",
            error_type="validation",
            remediation="Install tmux: apt install tmux (Debian/Ubuntu), brew install tmux (macOS)",
            details={"dependency": "tmux"},
        )

    # Validate spec exists
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

    # Generate tmux session name (tmux names can't contain dots/colons)
    session_name = f"foundry-{spec_id[:30]}".replace(".", "-").replace(":", "-")

    # Check if session already exists
    check_result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        capture_output=True,
        text=True,
    )
    if check_result.returncode == 0:
        emit_error(
            f"tmux session '{session_name}' already exists",
            code="ALREADY_EXISTS",
            error_type="validation",
            remediation=f"Attach with: tmux attach -t {session_name}  |  Kill with: tmux kill-session -t {session_name}",
            details={"session_name": session_name},
        )

    # Build agent command
    agent_cmd = (
        f"FOUNDRY_MCP_ROLE=autonomy_runner "
        f"FOUNDRY_MCP_AUTONOMY_POSTURE={posture} "
        f"claude -p '/foundry-implement-auto {spec_id}' --dangerously-skip-permissions"
    )

    # Build watcher command (delay to let the agent start first)
    watcher_cmd = f"sleep 5 && foundry watch {spec_id}"

    # Split direction flag: -v = vertical (top/bottom), -h = horizontal (side-by-side)
    split_flag = "-v" if layout == "vertical" else "-h"

    try:
        # Create tmux session with agent pane
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, agent_cmd],
            check=True,
            capture_output=True,
            text=True,
        )

        # Split window for watcher pane
        subprocess.run(
            ["tmux", "split-window", "-t", session_name, split_flag, watcher_cmd],
            check=True,
            capture_output=True,
            text=True,
        )

        # Select the agent pane (first pane)
        subprocess.run(
            ["tmux", "select-pane", "-t", f"{session_name}:0.0"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        emit_error(
            f"Failed to create tmux session: {e}",
            code="TMUX_ERROR",
            error_type="io",
            remediation="Check tmux is running correctly and try again",
            details={"stderr": e.stderr, "returncode": e.returncode},
        )

    reattach_cmd = f"tmux attach -t {session_name}"

    if not detach:
        # Attach to the session (replaces this process)
        emit_success(
            {
                "spec_id": spec_id,
                "session_name": session_name,
                "posture": posture,
                "layout": layout,
                "action": "attaching",
                "reattach_command": reattach_cmd,
                "message": f"Attaching to tmux session '{session_name}'...",
            }
        )
        # exec into tmux attach — this replaces the current process
        subprocess.run(["tmux", "attach", "-t", session_name])
    else:
        emit_success(
            {
                "spec_id": spec_id,
                "session_name": session_name,
                "posture": posture,
                "layout": layout,
                "action": "detached",
                "reattach_command": reattach_cmd,
                "message": (
                    f"Autonomous session started in detached tmux session '{session_name}'. "
                    f"Reattach with: {reattach_cmd}"
                ),
            }
        )
