"""Naming helpers for MCP tool registration."""

from __future__ import annotations

from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from foundry_mcp.core.observability import mcp_tool


def canonical_tool(
    mcp: FastMCP,
    *,
    canonical_name: str,
    **tool_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a tool under its canonical name."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return mcp.tool(name=canonical_name, **tool_kwargs)(
            mcp_tool(tool_name=canonical_name)(func)
        )

    return decorator
