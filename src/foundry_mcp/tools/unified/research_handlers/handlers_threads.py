"""Thread management handlers: list, get, delete."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from foundry_mcp.core.research.models.enums import ThreadStatus
from foundry_mcp.core.research.workflows import ChatWorkflow
from foundry_mcp.core.responses.types import (
    ErrorCode,
    ErrorType,
)
from foundry_mcp.core.responses.builders import (
    error_response,
    success_response,
)
from foundry_mcp.tools.unified.param_schema import Str, validate_payload

from ._helpers import _get_config, _get_memory

# ---------------------------------------------------------------------------
# Declarative validation schemas
# ---------------------------------------------------------------------------

_THREAD_LIST_SCHEMA = {
    "status": Str(choices=frozenset(s.value for s in ThreadStatus)),
}

_THREAD_GET_SCHEMA = {
    "thread_id": Str(required=True),
}

_THREAD_DELETE_SCHEMA = {
    "thread_id": Str(required=True),
}


def _handle_thread_list(
    *,
    status: Optional[str] = None,
    limit: int = 50,
    **kwargs: Any,
) -> dict:
    """Handle thread-list action."""
    payload = {"status": status}
    err = validate_payload(payload, _THREAD_LIST_SCHEMA, tool_name="research", action="thread-list")
    if err:
        return err

    thread_status = ThreadStatus(status) if status else None

    config = _get_config()
    workflow = ChatWorkflow(config.research, _get_memory())
    threads = workflow.list_threads(status=thread_status, limit=limit)

    return asdict(
        success_response(
            data={
                "threads": threads,
                "count": len(threads),
            }
        )
    )


def _handle_thread_get(
    *,
    thread_id: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Handle thread-get action."""
    payload = {"thread_id": thread_id}
    err = validate_payload(payload, _THREAD_GET_SCHEMA, tool_name="research", action="thread-get")
    if err:
        return err

    config = _get_config()
    workflow = ChatWorkflow(config.research, _get_memory())
    thread = workflow.get_thread(thread_id)

    if not thread:
        return asdict(
            error_response(
                f"Thread '{thread_id}' not found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Use thread-list to find valid thread IDs",
            )
        )

    return asdict(success_response(data=thread))


def _handle_thread_delete(
    *,
    thread_id: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Handle thread-delete action."""
    payload = {"thread_id": thread_id}
    err = validate_payload(payload, _THREAD_DELETE_SCHEMA, tool_name="research", action="thread-delete")
    if err:
        return err

    config = _get_config()
    workflow = ChatWorkflow(config.research, _get_memory())
    deleted = workflow.delete_thread(thread_id)

    if not deleted:
        return asdict(
            error_response(
                f"Thread '{thread_id}' not found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Use thread-list to find valid thread IDs",
            )
        )

    return asdict(
        success_response(
            data={
                "deleted": True,
                "thread_id": thread_id,
            }
        )
    )
