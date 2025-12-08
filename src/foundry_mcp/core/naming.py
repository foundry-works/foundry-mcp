"""Naming helpers for MCP tool registration."""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from foundry_mcp.core.observability import mcp_tool

logger = logging.getLogger(__name__)


def canonical_tool(
    mcp: FastMCP,
    *,
    canonical_name: str,
    **tool_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a tool under its canonical name.

    This decorator wraps the tool function to:
    1. Register it with FastMCP under the canonical name
    2. Apply observability instrumentation via mcp_tool
    3. Collect error data when exceptions occur

    Args:
        mcp: FastMCP instance
        canonical_name: The canonical name for the tool
        **tool_kwargs: Additional kwargs passed to mcp.tool()

    Returns:
        Decorated function registered as an MCP tool
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if asyncio.iscoroutinefunction(func):
            # Async function - use async wrapper
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                """Async wrapper for async underlying functions."""
                start_time = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    _collect_tool_error(
                        tool_name=canonical_name,
                        error=e,
                        input_params=kwargs,
                        duration_ms=duration_ms,
                    )
                    raise

            wrapper = async_wrapper
        else:
            # Sync function - use sync wrapper to preserve sync behavior
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                """Sync wrapper for sync underlying functions."""
                start_time = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    _collect_tool_error(
                        tool_name=canonical_name,
                        error=e,
                        input_params=kwargs,
                        duration_ms=duration_ms,
                    )
                    raise

            wrapper = sync_wrapper

        # Apply mcp_tool first, then register with FastMCP
        instrumented = mcp_tool(tool_name=canonical_name)(wrapper)
        return mcp.tool(name=canonical_name, **tool_kwargs)(instrumented)

    return decorator


def _collect_tool_error(
    tool_name: str,
    error: Exception,
    input_params: dict[str, Any],
    duration_ms: float,
) -> None:
    """Collect error data for later introspection.

    Uses lazy import to avoid circular dependencies and only
    collects if error collection is enabled.

    Args:
        tool_name: Name of the tool that raised the error
        error: The exception that was raised
        input_params: Input parameters passed to the tool
        duration_ms: Duration in milliseconds before error
    """
    try:
        # Lazy import to avoid circular dependencies
        from foundry_mcp.config import get_config

        config = get_config()
        if not config.error_collection.enabled:
            return

        from foundry_mcp.core.error_collection import get_error_collector

        collector = get_error_collector()
        collector.collect_tool_error(
            tool_name=tool_name,
            error=error,
            input_params=input_params,
            duration_ms=duration_ms,
        )
    except Exception as collect_error:
        # Never let error collection failures affect tool execution
        logger.debug(f"Error collection failed for {tool_name}: {collect_error}")
