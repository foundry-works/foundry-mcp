"""SDD CLI - Native command-line interface for Spec-Driven Development.

This CLI provides JSON-only output designed for AI coding assistants.
All commands emit structured JSON to stdout for reliable parsing.
"""

from foundry_mcp.cli.config import CLIContext, create_context
from foundry_mcp.cli.flags import (
    CLIFlagRegistry,
    apply_cli_flag_overrides,
    flags_for_discovery,
    get_cli_flags,
    with_flag_options,
)
from foundry_mcp.cli.logging import (
    CLILogContext,
    cli_command,
    get_cli_logger,
    get_request_id,
    set_request_id,
)
from foundry_mcp.cli.main import cli
from foundry_mcp.cli.output import emit, emit_error, emit_success
from foundry_mcp.cli.registry import get_context, set_context
from foundry_mcp.cli.resilience import (
    FAST_TIMEOUT,
    MEDIUM_TIMEOUT,
    SLOW_TIMEOUT,
    with_sync_timeout,
    cli_retryable,
    handle_keyboard_interrupt,
)
from foundry_mcp.cli.context import (
    ContextSession,
    ContextTracker,
    get_context_tracker,
    get_session_status,
    record_consultation,
    start_cli_session,
)

__all__ = [
    # Entry point
    "cli",
    # Context
    "CLIContext",
    "create_context",
    "get_context",
    "set_context",
    # Output
    "emit",
    "emit_error",
    "emit_success",
    # Feature flags
    "CLIFlagRegistry",
    "apply_cli_flag_overrides",
    "flags_for_discovery",
    "get_cli_flags",
    "with_flag_options",
    # Logging
    "CLILogContext",
    "cli_command",
    "get_cli_logger",
    "get_request_id",
    "set_request_id",
    # Resilience
    "FAST_TIMEOUT",
    "MEDIUM_TIMEOUT",
    "SLOW_TIMEOUT",
    "with_sync_timeout",
    "cli_retryable",
    "handle_keyboard_interrupt",
    # Session/Context tracking
    "ContextSession",
    "ContextTracker",
    "get_context_tracker",
    "get_session_status",
    "record_consultation",
    "start_cli_session",
]
