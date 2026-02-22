"""Foundry run command for launching autonomous sessions.

Launches claude in interactive mode with the foundry-implement-auto skill,
giving the user full visibility into what the agent is doing.
"""

import os
import shutil
import signal

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


@click.command("run")
@click.argument("spec_id")
@click.option(
    "--posture",
    type=click.Choice(["unattended", "supervised"], case_sensitive=False),
    default="unattended",
    show_default=True,
    help="Autonomy posture for the agent.",
)
@click.pass_context
@cli_command("run")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Run command timed out")
def run_cmd(
    ctx: click.Context,
    spec_id: str,
    posture: str,
) -> None:
    """Launch an autonomous session for SPEC_ID.

    Starts claude in interactive mode with the foundry-implement-auto skill.
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
        return

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
        return

    # Verify claude is installed
    claude_bin = shutil.which("claude")
    if claude_bin is None:
        emit_error(
            "claude CLI is not installed or not found in PATH",
            code="DEPENDENCY_MISSING",
            error_type="validation",
            remediation="Install Claude Code: https://docs.anthropic.com/en/docs/claude-code",
            details={"dependency": "claude"},
        )
        return

    # Build the command
    cmd = [claude_bin, f"/foundry-implement-auto {spec_id}"]
    if posture == "unattended":
        cmd.append("--dangerously-skip-permissions")

    # Set up environment
    env = os.environ.copy()
    env["FOUNDRY_MCP_ROLE"] = "autonomy_runner"
    env["FOUNDRY_MCP_AUTONOMY_POSTURE"] = posture

    # Cancel the SIGALRM timeout before launching interactive claude
    signal.alarm(0)

    # Replace this process with claude
    os.execvpe(claude_bin, cmd, env)
