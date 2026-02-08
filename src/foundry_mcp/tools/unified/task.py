"""Unified task router — delegates to task_handlers/ package.

This module is a backward-compatible shim. All handler logic now lives in
``foundry_mcp.tools.unified.task_handlers``.
"""

from foundry_mcp.tools.unified.task_handlers import register_unified_task_tool  # noqa: F401

__all__ = [
    "register_unified_task_tool",
]
