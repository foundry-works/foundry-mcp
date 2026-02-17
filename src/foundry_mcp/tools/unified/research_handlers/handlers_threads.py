"""Thread management handlers: list, get, delete."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from foundry_mcp.core.research.models import ThreadStatus
from foundry_mcp.core.research.workflows import ChatWorkflow
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
)

from ._helpers import _get_config, _get_memory, _validation_error


def _handle_thread_list(
    *,
    status: Optional[str] = None,
    limit: int = 50,
    **kwargs: Any,
) -> dict:
    """Handle thread-list action."""
    thread_status = None
    if status:
        try:
            thread_status = ThreadStatus(status)
        except ValueError:
            valid = [s.value for s in ThreadStatus]
            return _validation_error(
                field="status",
                action="thread-list",
                message=f"Invalid value. Valid: {valid}",
            )

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
    if not thread_id:
        return _validation_error(field="thread_id", action="thread-get", message="Required")

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
    if not thread_id:
        return _validation_error(field="thread_id", action="thread-delete", message="Required")

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
