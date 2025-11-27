"""MCP tools for foundry-mcp."""

from foundry_mcp.tools.queries import register_query_tools
from foundry_mcp.tools.tasks import register_task_tools
from foundry_mcp.tools.validation import register_validation_tools
from foundry_mcp.tools.journal import register_journal_tools
from foundry_mcp.tools.environment import register_environment_tools

__all__ = [
    "register_query_tools",
    "register_task_tools",
    "register_validation_tools",
    "register_journal_tools",
    "register_environment_tools",
]
