"""Unified task router — delegates to task_handlers/ package.

This module is a backward-compatible shim. All handler logic now lives in
``foundry_mcp.tools.unified.task_handlers``.

The dispatch function is defined here (not imported) so that tests can patch
``task._TASK_ROUTER`` and have it take effect.
"""

from __future__ import annotations

from typing import Any, Dict

from foundry_mcp.config import ServerConfig
from foundry_mcp.tools.unified.common import dispatch_with_standard_errors
from foundry_mcp.tools.unified.task_handlers import (  # noqa: F401
    _TASK_ROUTER,
    register_unified_task_tool,
)
from foundry_mcp.tools.unified.task_handlers._helpers import (  # noqa: F401
    _metric as _metric,
)
from foundry_mcp.tools.unified.common import build_request_id  # noqa: F401


def _request_id() -> str:
    return build_request_id("task")


def _dispatch_task_action(
    *, action: str, payload: Dict[str, Any], config: ServerConfig
) -> dict:
    return dispatch_with_standard_errors(
        _TASK_ROUTER, "task", action, config=config, **payload
    )


__all__ = [
    "register_unified_task_tool",
]
