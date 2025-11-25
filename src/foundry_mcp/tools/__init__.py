"""MCP tools for foundry-mcp."""

from foundry_mcp.tools.queries import register_query_tools
from foundry_mcp.tools.tasks import register_task_tools

__all__ = ["register_query_tools", "register_task_tools"]
