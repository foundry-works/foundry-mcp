"""Session management commands for SDD CLI.

Provides commands for session tracking, context limits, and consultation monitoring.
"""

import os
import secrets
from typing import Optional

import click

from foundry_mcp.cli.agent import agent_gated, get_agent_type
from foundry_mcp.cli.context import (
    ContextTracker,
    get_context_tracker,
    get_session_status,
    record_consultation,
    start_cli_session,
)
from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error, emit_success
from foundry_mcp.cli.registry import get_context
from foundry_mcp.cli.resilience import (
    FAST_TIMEOUT,
    handle_keyboard_interrupt,
    with_sync_timeout,
)

logger = get_cli_logger()


# Valid work modes
WORK_MODES = frozenset({"single", "autonomous"})
DEFAULT_WORK_MODE = "single"


@click.group("session")
def session() -> None:
    """Session and context management commands."""
    pass


@session.command("start")
@click.option("--id", "session_id", help="Custom session ID.")
@click.option("--max-consultations", type=int, help="Maximum LLM consultations allowed.")
@click.option("--max-tokens", type=int, help="Maximum context tokens allowed.")
@click.pass_context
@cli_command("session-start")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Session start timed out")
def start_session_cmd(
    ctx: click.Context,
    session_id: Optional[str],
    max_consultations: Optional[int],
    max_tokens: Optional[int],
) -> None:
    """Start a new CLI session with optional limits.

    Sessions track consultation usage and context budget for LLM workflows.
    """
    from foundry_mcp.cli.context import SessionLimits

    tracker = get_context_tracker()

    # Build limits if any overrides provided
    limits = None
    if max_consultations is not None or max_tokens is not None:
        limits = SessionLimits(
            max_consultations=max_consultations or 50,
            max_context_tokens=max_tokens or 100000,
        )

    session = tracker.start_session(session_id=session_id, limits=limits)

    emit_success({
        "session_id": session.session_id,
        "started_at": session.started_at,
        "limits": {
            "max_consultations": session.limits.max_consultations,
            "max_context_tokens": session.limits.max_context_tokens,
        },
    })


@session.command("status")
@click.pass_context
@cli_command("session-status")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Session status lookup timed out")
def session_status_cmd(ctx: click.Context) -> None:
    """Get current session status and usage."""
    status = get_session_status()
    emit_success(status)


@session.command("record")
@click.option("--tokens", type=int, default=0, help="Estimated tokens used.")
@click.pass_context
@cli_command("session-record")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Record consultation timed out")
def record_consultation_cmd(ctx: click.Context, tokens: int) -> None:
    """Record an LLM consultation.

    Tracks consultation count and token usage against session limits.
    """
    result = record_consultation(tokens)
    emit_success(result)


@session.command("reset")
@click.pass_context
@cli_command("session-reset")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Session reset timed out")
def reset_session_cmd(ctx: click.Context) -> None:
    """Reset the current session."""
    tracker = get_context_tracker()
    tracker.reset()
    emit_success({"message": "Session reset"})


@session.command("limits")
@click.pass_context
@cli_command("session-limits")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Limits lookup timed out")
def show_limits_cmd(ctx: click.Context) -> None:
    """Show current session limits and remaining budget."""
    tracker = get_context_tracker()
    session = tracker.get_session()

    if session is None:
        emit_success({
            "active": False,
            "message": "No active session. Use 'sdd session start' to begin.",
            "default_limits": {
                "max_consultations": tracker._default_limits.max_consultations,
                "max_context_tokens": tracker._default_limits.max_context_tokens,
                "warn_at_percentage": tracker._default_limits.warn_at_percentage,
            },
        })
    else:
        emit_success({
            "active": True,
            "session_id": session.session_id,
            "limits": {
                "max_consultations": session.limits.max_consultations,
                "max_context_tokens": session.limits.max_context_tokens,
                "warn_at_percentage": session.limits.warn_at_percentage,
            },
            "usage": {
                "consultations_used": session.stats.consultation_count,
                "consultations_remaining": session.consultations_remaining,
                "tokens_used": session.stats.estimated_tokens_used,
                "tokens_remaining": session.tokens_remaining,
            },
            "status": {
                "consultation_percentage": round(session.consultation_usage_percentage, 1),
                "token_percentage": round(session.token_usage_percentage, 1),
                "should_warn": session.should_warn,
                "at_limit": session.at_limit,
            },
        })


@session.command("capabilities")
@click.pass_context
@cli_command("session-capabilities")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Capabilities lookup timed out")
def session_capabilities_cmd(ctx: click.Context) -> None:
    """Show CLI capabilities and feature flags.

    Returns a manifest of available features, commands, and their status
    for AI coding assistants to understand available functionality.
    """
    from foundry_mcp.cli.flags import flags_for_discovery, get_cli_flags
    from foundry_mcp.cli.main import cli

    cli_ctx = get_context(ctx)

    # Get registered command groups
    command_groups = {}
    for name, cmd in cli.commands.items():
        if hasattr(cmd, "commands"):  # It's a group
            command_groups[name] = {
                "type": "group",
                "subcommands": list(cmd.commands.keys()),
            }
        else:
            command_groups[name] = {"type": "command"}

    # Get feature flags
    flag_registry = get_cli_flags()
    flags = flags_for_discovery()

    # Known CLI capabilities
    capabilities = {
        "json_output": True,  # All output is JSON
        "spec_driven": True,  # SDD methodology supported
        "feature_flags": True,  # Feature flag system available
        "session_tracking": True,  # Session/context tracking
        "rate_limiting": True,  # Rate limiting built-in
    }

    emit_success({
        "version": "0.1.0",
        "name": "foundry-cli",
        "capabilities": capabilities,
        "feature_flags": flags,
        "command_groups": list(command_groups.keys()),
        "command_count": len(cli.commands),
        "specs_dir": str(cli_ctx.specs_dir) if cli_ctx.specs_dir else None,
    })


def get_work_mode() -> str:
    """Get the configured work mode from environment.

    Work mode controls how sdd-next executes tasks:
    - "single": Execute one task at a time, pause for approval
    - "autonomous": Execute all tasks in a phase without pausing

    Set via MCP server config:
        "env": {"FOUNDRY_MCP_WORK_MODE": "autonomous"}

    Returns:
        Work mode string ("single" or "autonomous").
    """
    env_mode = os.environ.get("FOUNDRY_MCP_WORK_MODE", "")
    mode = env_mode.lower().strip()
    return mode if mode in WORK_MODES else DEFAULT_WORK_MODE


@session.command("work-mode")
@click.pass_context
@cli_command("session-work-mode")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Work mode lookup timed out")
def work_mode_cmd(ctx: click.Context) -> None:
    """Get the current work mode for task execution.

    Work mode is configured via FOUNDRY_MCP_WORK_MODE environment variable
    in the MCP server configuration.

    Modes:
    - single: Execute one task at a time, pause for approval
    - autonomous: Execute all tasks in a phase without pausing
    """
    mode = get_work_mode()
    agent = get_agent_type()

    emit_success({
        "work_mode": mode,
        "agent_type": agent,
        "modes_available": list(WORK_MODES),
        "configured_via": "FOUNDRY_MCP_WORK_MODE",
    })


@session.command("token-usage")
@agent_gated("claude-code")
@click.option("--session-marker", help="Session marker from generate-marker command.")
@click.pass_context
@cli_command("session-token-usage")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Token usage lookup timed out")
def token_usage_cmd(ctx: click.Context, session_marker: Optional[str]) -> None:
    """Monitor token and context usage (Claude Code only).

    Parses Claude Code transcript files to extract token usage metrics.
    Requires agent_type=claude-code in MCP configuration.

    Use --session-marker to filter to a specific session.
    """
    # Note: Full implementation requires transcript parsing logic
    # For now, return a placeholder indicating the feature is available
    emit_success({
        "available": True,
        "agent_type": "claude-code",
        "session_marker": session_marker,
        "message": "Token usage tracking available. Full metrics require transcript access.",
        "hint": "Use generate-marker to create a session marker for tracking.",
    })


@session.command("generate-marker")
@agent_gated("claude-code")
@click.pass_context
@cli_command("session-generate-marker")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Marker generation timed out")
def generate_marker_cmd(ctx: click.Context) -> None:
    """Generate a session marker for transcript identification (Claude Code only).

    Creates a unique marker that can be used to identify and filter
    transcript entries for token usage tracking.

    Requires agent_type=claude-code in MCP configuration.
    """
    marker = f"SESSION_MARKER_{secrets.token_hex(4).upper()}"

    emit_success({
        "marker": marker,
        "usage": "Include this marker in your prompts to track context usage.",
        "hint": "Pass to 'session token-usage --session-marker' to filter metrics.",
    })
