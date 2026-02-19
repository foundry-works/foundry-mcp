"""Unified task router — delegates to task_handlers/ package.

This module is a backward-compatible shim. All handler logic now lives in
``foundry_mcp.tools.unified.task_handlers``.

The dispatch function is defined here (not imported) so that tests can patch
``task._TASK_ROUTER`` and have it take effect.
"""

from __future__ import annotations

from typing import Any, Dict

from foundry_mcp.config.server import ServerConfig
from foundry_mcp.tools.unified.common import (
    build_request_id,  # noqa: F401
    dispatch_with_standard_errors,
)
from foundry_mcp.tools.unified.task_handlers import (  # noqa: F401
    _TASK_ROUTER,
    register_unified_task_tool,
)
from foundry_mcp.tools.unified.task_handlers._helpers import (  # noqa: F401
    _attach_deprecation_metadata,
    _attach_session_step_loop_metadata,
    _emit_legacy_action_warning,
    _normalize_task_action_shape,
)
from foundry_mcp.tools.unified.task_handlers._helpers import (
    _metric as _metric,
)


def _request_id() -> str:
    return build_request_id("task")


def _dispatch_task_action(*, action: str, payload: Dict[str, Any], config: ServerConfig) -> dict:
    request_id = payload.get("request_id")
    if not isinstance(request_id, str):
        request_id = build_request_id("task")
    normalized_action, normalized_payload, deprecation, error = _normalize_task_action_shape(
        action=action,
        payload=payload,
        request_id=request_id,
    )
    if error is not None:
        return error

    _emit_legacy_action_warning(normalized_action, deprecation)
    response = dispatch_with_standard_errors(
        _TASK_ROUTER, "task", normalized_action, config=config, **normalized_payload
    )
    response = _attach_deprecation_metadata(response, deprecation)
    return _attach_session_step_loop_metadata(normalized_action, response)


__all__ = [
    "register_unified_task_tool",
]
