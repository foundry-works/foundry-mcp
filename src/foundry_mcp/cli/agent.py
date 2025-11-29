"""Agent detection and feature gating for CLI commands.

Provides mechanisms to detect which AI coding assistant is running the CLI
and gate features that are only available for specific agents.

Agent types:
- claude-code: Anthropic's Claude Code (has transcript access)
- cursor: Cursor IDE
- generic: Unknown/default agent

Configuration:
- Set FOUNDRY_MCP_AGENT_TYPE in MCP server config (env section)
- The CLI inherits this from the MCP server environment
"""

import os
from functools import wraps
from typing import Callable, TypeVar

from foundry_mcp.cli.output import emit_success

# Valid agent types
AGENT_TYPES = frozenset({"claude-code", "cursor", "generic"})
DEFAULT_AGENT_TYPE = "generic"

F = TypeVar("F", bound=Callable)


def get_agent_type() -> str:
    """Get the configured agent type from environment.

    Set via MCP server config:
        "env": {"FOUNDRY_MCP_AGENT_TYPE": "claude-code"}

    Returns:
        Agent type string (claude-code, cursor, generic).
    """
    env_agent = os.environ.get("FOUNDRY_MCP_AGENT_TYPE", "")
    agent = env_agent.lower().strip()
    return agent if agent in AGENT_TYPES else DEFAULT_AGENT_TYPE


def agent_gated(required_agent: str) -> Callable[[F], F]:
    """Decorator for agent-specific commands.

    When the current agent type doesn't match the required agent,
    returns a success response with a warning indicating the feature
    is unavailable, rather than failing.

    Args:
        required_agent: The agent type required for this command (e.g., "claude-code").

    Returns:
        Decorated function that gates execution by agent type.

    Example:
        @session.command("token-usage")
        @agent_gated("claude-code")
        def token_usage_cmd():
            # Only runs when agent_type == "claude-code"
            ...
    """

    def decorator(f: F) -> F:
        @wraps(f)
        def wrapper(*args, **kwargs):
            current_agent = get_agent_type()

            if current_agent != required_agent:
                emit_success(
                    {
                        "available": False,
                        "reason": f"This feature requires agent_type='{required_agent}'",
                        "current_agent": current_agent,
                        "hint": f"Set FOUNDRY_MCP_AGENT_TYPE={required_agent} or configure in foundry-mcp.toml",
                    },
                    meta={
                        "warning": f"Feature unavailable for agent_type='{current_agent}'"
                    },
                )
                return

            return f(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def is_claude_code() -> bool:
    """Check if the current agent is Claude Code.

    Returns:
        True if agent_type is "claude-code".
    """
    return get_agent_type() == "claude-code"
