"""Session management commands for SDD CLI.

Provides commands for session tracking, context limits, and consultation monitoring.
"""

import os
import secrets
from pathlib import Path
from typing import List, Optional

import click

from foundry_mcp.cli.agent import agent_gated, get_agent_type
from foundry_mcp.cli.transcript import find_transcript_by_marker, parse_transcript

TRANSCRIPT_OPT_IN_ENV = "FOUNDRY_MCP_ALLOW_TRANSCRIPTS"
from foundry_mcp.cli.context import (
    get_context_tracker,
    get_session_status,
    record_consultation,
)
from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error, emit_success
from foundry_mcp.cli.registry import get_context
from foundry_mcp.cli.resilience import (
    FAST_TIMEOUT,
    MEDIUM_TIMEOUT,
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
@click.option(
    "--max-consultations", type=int, help="Maximum LLM consultations allowed."
)
@click.option("--max-tokens", type=int, help="Maximum context tokens allowed.")
@click.pass_context
@cli_command("start")
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

    emit_success(
        {
            "session_id": session.session_id,
            "started_at": session.started_at,
            "limits": {
                "max_consultations": session.limits.max_consultations,
                "max_context_tokens": session.limits.max_context_tokens,
            },
        }
    )


@session.command("status")
@click.pass_context
@cli_command("status")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Session status lookup timed out")
def session_status_cmd(ctx: click.Context) -> None:
    """Get current session status and usage."""
    status = get_session_status()
    emit_success(status)


@session.command("record")
@click.option("--tokens", type=int, default=0, help="Estimated tokens used.")
@click.pass_context
@cli_command("record")
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
@cli_command("reset")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Session reset timed out")
def reset_session_cmd(ctx: click.Context) -> None:
    """Reset the current session."""
    tracker = get_context_tracker()
    tracker.reset()
    emit_success({"message": "Session reset"})


@session.command("limits")
@click.pass_context
@cli_command("limits")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Limits lookup timed out")
def show_limits_cmd(ctx: click.Context) -> None:
    """Show current session limits and remaining budget."""
    tracker = get_context_tracker()
    session = tracker.get_session()

    if session is None:
        emit_success(
            {
                "active": False,
                "message": "No active session. Use 'sdd session start' to begin.",
                "default_limits": {
                    "max_consultations": tracker._default_limits.max_consultations,
                    "max_context_tokens": tracker._default_limits.max_context_tokens,
                    "warn_at_percentage": tracker._default_limits.warn_at_percentage,
                },
            }
        )
    else:
        emit_success(
            {
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
                    "consultation_percentage": round(
                        session.consultation_usage_percentage, 1
                    ),
                    "token_percentage": round(session.token_usage_percentage, 1),
                    "should_warn": session.should_warn,
                    "at_limit": session.at_limit,
                },
            }
        )


@session.command("capabilities")
@click.pass_context
@cli_command("capabilities")
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
    get_cli_flags()
    flags = flags_for_discovery()

    # Known CLI capabilities
    capabilities = {
        "json_output": True,  # All output is JSON
        "spec_driven": True,  # SDD methodology supported
        "feature_flags": True,  # Feature flag system available
        "session_tracking": True,  # Session/context tracking
        "rate_limiting": True,  # Rate limiting built-in
    }

    emit_success(
        {
            "version": "0.1.0",
            "name": "foundry-cli",
            "capabilities": capabilities,
            "feature_flags": flags,
            "command_groups": list(command_groups.keys()),
            "command_count": len(cli.commands),
            "specs_dir": str(cli_ctx.specs_dir) if cli_ctx.specs_dir else None,
        }
    )


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
@cli_command("work-mode")
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

    emit_success(
        {
            "work_mode": mode,
            "agent_type": agent,
            "modes_available": list(WORK_MODES),
            "configured_via": "FOUNDRY_MCP_WORK_MODE",
        }
    )


@session.command("token-usage")
@agent_gated("claude-code")
@click.option("--session-marker", help="Session marker from generate-marker command.")
@click.pass_context
@cli_command("token-usage")
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
    emit_success(
        {
            "available": True,
            "agent_type": "claude-code",
            "session_marker": session_marker,
            "message": "Token usage tracking available. Full metrics require transcript access.",
            "hint": "Use generate-marker to create a session marker for tracking.",
        }
    )


@session.command("generate-marker")
@agent_gated("claude-code")
@click.pass_context
@cli_command("generate-marker")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Marker generation timed out")
def generate_marker_cmd(ctx: click.Context) -> None:
    """Generate a session marker for transcript identification (Claude Code only).

    Creates a unique marker that can be used to identify and filter
    transcript entries for token usage tracking.

    Requires agent_type=claude-code in MCP configuration.
    """
    marker = f"SESSION_MARKER_{secrets.token_hex(4).upper()}"

    emit_success(
        {
            "marker": marker,
            "usage": "Include this marker in your prompts to track context usage.",
            "hint": "Pass to 'session token-usage --session-marker' to filter metrics.",
        }
    )


@session.command("context")
@agent_gated("claude-code")
@click.option(
    "--session-marker",
    required=True,
    help="Session marker from generate-marker command for context tracking.",
)
@click.option(
    "--check-limits",
    is_flag=True,
    help="Include limit checking and recommendations.",
)
@click.option(
    "--transcript-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Explicit directory containing transcript JSONL files.",
)
@click.option(
    "--allow-home-transcripts",
    is_flag=True,
    help="Allow scanning ~/.claude/projects for transcripts (requires opt-in).",
)
@click.pass_context
@cli_command("context")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Context check timed out")
def context_cmd(
    ctx: click.Context,
    session_marker: str,
    check_limits: bool,
    transcript_dir: Optional[Path],
    allow_home_transcripts: bool,
) -> None:
    """Check current context usage percentage (Claude Code only).

    Completes the two-step context tracking contract:
    1. Run 'sdd session generate-marker' to get a marker
    2. Run 'sdd session context --session-marker <marker>' to check usage

    The session marker is logged to the transcript and used to calculate
    context percentage by analyzing conversation length.

    Example:
        sdd session generate-marker
        # Returns: SESSION_MARKER_ABCD1234
        sdd session context --session-marker SESSION_MARKER_ABCD1234
        # Returns: {"context_percentage_used": 45}
    """
    # Validate marker format
    if not session_marker.startswith("SESSION_MARKER_"):
        emit_error(
            "Invalid session marker format",
            code="INVALID_MARKER",
            error_type="validation",
            remediation="Use a marker from 'sdd session generate-marker'",
            details={"provided_marker": session_marker},
        )
        return

    transcript_dirs: Optional[List[Path]] = None
    if transcript_dir is not None:
        resolved_dir = transcript_dir.expanduser().resolve()
        if not resolved_dir.exists() or not resolved_dir.is_dir():
            emit_error(
                "Transcript directory not found",
                code="VALIDATION_ERROR",
                error_type="validation",
                remediation="Pass a directory containing transcript JSONL files",
                details={"transcript_dir": str(transcript_dir)},
            )
            return
        transcript_dirs = [resolved_dir]

    allow_home_search = allow_home_transcripts or bool(
        os.environ.get(TRANSCRIPT_OPT_IN_ENV, "").strip()
    )

    if transcript_dirs is None and not allow_home_search:
        emit_error(
            "Transcript access disabled",
            code="TRANSCRIPTS_DISABLED",
            error_type="forbidden",
            remediation=(
                "Pass --transcript-dir, use --allow-home-transcripts, or set FOUNDRY_MCP_ALLOW_TRANSCRIPTS=1"
            ),
            details={"session_marker": session_marker},
        )
        return

    # Find transcript containing the session marker
    transcript_path = find_transcript_by_marker(
        Path.cwd(),
        session_marker,
        search_dirs=transcript_dirs,
        allow_home_search=allow_home_search,
    )
    if transcript_path is None:
        emit_error(
            "Could not find transcript containing marker",
            code="TRANSCRIPT_NOT_FOUND",
            error_type="not_found",
            remediation=(
                "Ensure you run 'sdd session generate-marker' first, then wait for "
                "the marker to be logged before running 'sdd session context'."
            ),
            details={
                "session_marker": session_marker,
                "cwd": str(Path.cwd()),
            },
        )
        return

    # Parse the transcript to get token metrics
    metrics = parse_transcript(transcript_path)
    if metrics is None:
        emit_error(
            "Failed to parse transcript file",
            code="PARSE_ERROR",
            error_type="internal",
            remediation="Check that the transcript file is valid JSONL.",
            details={"transcript_path": str(transcript_path)},
        )
        return

    # Calculate context percentage (default max context: 155,000 tokens)
    max_context = 155000
    context_percentage = round(metrics.context_percentage(max_context))
    recommendations = []

    if check_limits:
        if context_percentage >= 85:
            recommendations.append(
                "Context at or above 85%. Consider '/clear' and '/sdd-begin'."
            )
        elif context_percentage >= 70:
            recommendations.append("Context above 70%. Monitor usage closely.")

    result = {"context_percentage_used": int(context_percentage)}

    if check_limits:
        result["session_marker"] = session_marker
        result["recommendations"] = recommendations
        result["threshold_warning"] = 85
        result["threshold_stop"] = 90
        result["context_length"] = metrics.context_length
        result["max_context"] = max_context

    emit_success(result)
