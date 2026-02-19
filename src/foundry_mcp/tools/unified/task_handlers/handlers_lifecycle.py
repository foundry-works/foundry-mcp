"""Lifecycle action handlers: start, complete, block, unblock, update-status, list-blocked."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from foundry_mcp.config.server import ServerConfig
from foundry_mcp.core.journal import (
    add_journal_entry,
    get_blocker_info,
    list_blocked_tasks,
    mark_blocked,
    update_task_status,
)
from foundry_mcp.core.journal import (
    unblock as unblock_task,
)
from foundry_mcp.core.pagination import (
    CursorError,
    decode_cursor,
    encode_cursor,
    normalize_page_size,
    paginated_response,
)
from foundry_mcp.core.progress import (
    get_progress_summary,
    sync_computed_fields,
    update_parent_status,
)
from foundry_mcp.core.responses.builders import (
    error_response,
    success_response,
)
from foundry_mcp.core.responses.types import (
    ErrorCode,
    ErrorType,
)
from foundry_mcp.core.spec import save_spec
from foundry_mcp.core.task import check_dependencies
from foundry_mcp.tools.unified.param_schema import Str, validate_payload
from foundry_mcp.tools.unified.task_handlers._helpers import (
    _ALLOWED_STATUS,
    _TASK_DEFAULT_PAGE_SIZE,
    _TASK_MAX_PAGE_SIZE,
    _check_autonomy_write_lock,
    _load_spec_data,
    _metric,
    _metrics,
    _pagination_warnings,
    _request_id,
    _resolve_specs_dir,
)

# ---------------------------------------------------------------------------
# Declarative validation schemas
# ---------------------------------------------------------------------------

_UPDATE_STATUS_SCHEMA = {
    "spec_id": Str(required=True),
    "task_id": Str(required=True),
    "status": Str(required=True, choices=frozenset(_ALLOWED_STATUS)),
    "note": Str(),
}

_START_SCHEMA = {
    "spec_id": Str(required=True),
    "task_id": Str(required=True),
    "note": Str(),
}

_COMPLETE_SCHEMA = {
    "spec_id": Str(required=True),
    "task_id": Str(required=True),
    "completion_note": Str(required=True),
}

_BLOCK_SCHEMA = {
    "spec_id": Str(required=True),
    "task_id": Str(required=True),
    "reason": Str(required=True),
    "blocker_type": Str(choices=frozenset({"dependency", "technical", "resource", "decision"})),
    "ticket": Str(),
}

_UNBLOCK_SCHEMA = {
    "spec_id": Str(required=True),
    "task_id": Str(required=True),
    "resolution": Str(),
}

_LIST_BLOCKED_SCHEMA = {
    "spec_id": Str(required=True),
}


def _handle_update_status(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "update-status"

    err = validate_payload(payload, _UPDATE_STATUS_SCHEMA, tool_name="task", action=action, request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    task_id = payload["task_id"]
    status = payload["status"]
    note = payload.get("note")
    bypass_autonomy_lock = payload.get("bypass_autonomy_lock", False)
    bypass_reason = payload.get("bypass_reason")
    workspace = payload.get("workspace")

    # Check autonomy write-lock before proceeding with protected mutation
    lock_error = _check_autonomy_write_lock(
        spec_id=spec_id,
        workspace=workspace,
        bypass_autonomy_lock=bool(bypass_autonomy_lock),
        bypass_reason=bypass_reason,
        request_id=request_id,
        config=config,
    )
    if lock_error:
        return lock_error

    specs_dir, _specs_err = _resolve_specs_dir(config, workspace)
    spec_data, error = _load_spec_data(spec_id, specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    hierarchy = spec_data.get("hierarchy", {})
    task_key = task_id
    if task_key not in hierarchy:
        return asdict(
            error_response(
                f"Task not found: {task_key}",
                error_code=ErrorCode.TASK_NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Verify the task ID exists in the hierarchy",
                request_id=request_id,
            )
        )

    start = time.perf_counter()
    updated = update_task_status(spec_data, task_key, status, note=None)
    if not updated:
        return asdict(
            error_response(
                f"Failed to update task status for {task_key}",
                error_code=ErrorCode.CONFLICT,
                error_type=ErrorType.CONFLICT,
                remediation="Confirm the task exists and the status is valid",
                request_id=request_id,
            )
        )

    update_parent_status(spec_data, task_key)

    if note:
        add_journal_entry(
            spec_data,
            title=f"Status changed to {status}",
            content=note,
            entry_type="status_change",
            task_id=task_key,
            author="foundry-mcp",
        )

    if specs_dir is None or not save_spec(spec_id, spec_data, specs_dir):
        return asdict(
            error_response(
                "Failed to save spec",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check filesystem permissions and retry",
                request_id=request_id,
            )
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    response = success_response(
        spec_id=spec_id,
        task_id=task_key,
        new_status=status,
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})
    return asdict(response)


def _handle_start(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "start"

    err = validate_payload(payload, _START_SCHEMA, tool_name="task", action=action, request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    task_id = payload["task_id"]
    note = payload.get("note")
    bypass_autonomy_lock = payload.get("bypass_autonomy_lock", False)
    bypass_reason = payload.get("bypass_reason")
    workspace = payload.get("workspace")

    # Check autonomy write-lock before proceeding with protected mutation
    lock_error = _check_autonomy_write_lock(
        spec_id=spec_id,
        workspace=workspace,
        bypass_autonomy_lock=bool(bypass_autonomy_lock),
        bypass_reason=bypass_reason,
        request_id=request_id,
        config=config,
    )
    if lock_error:
        return lock_error

    specs_dir, _specs_err = _resolve_specs_dir(config, workspace)
    spec_data, error = _load_spec_data(spec_id, specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    start = time.perf_counter()
    deps = check_dependencies(spec_data, task_id)
    if not deps.get("can_start", False):
        blockers = [b.get("title", b.get("id", "")) for b in (deps.get("blocked_by") or []) if isinstance(b, dict)]
        return asdict(
            error_response(
                "Task is blocked by: " + ", ".join([b for b in blockers if b]),
                error_code=ErrorCode.CONFLICT,
                error_type=ErrorType.CONFLICT,
                remediation="Resolve blocking tasks then retry",
                details={"blocked_by": deps.get("blocked_by")},
                request_id=request_id,
            )
        )

    updated = update_task_status(spec_data, task_id, "in_progress", note=None)
    if not updated:
        return asdict(
            error_response(
                f"Failed to start task: {task_id}",
                error_code=ErrorCode.CONFLICT,
                error_type=ErrorType.CONFLICT,
                remediation="Confirm the task exists and is not blocked",
                request_id=request_id,
            )
        )

    update_parent_status(spec_data, task_id)
    sync_computed_fields(spec_data)

    if note:
        add_journal_entry(
            spec_data,
            title=f"Task Started: {task_id}",
            content=note,
            entry_type="status_change",
            task_id=task_id,
            author="foundry-mcp",
        )

    if specs_dir is None or not save_spec(spec_id, spec_data, specs_dir):
        return asdict(
            error_response(
                "Failed to save spec",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check filesystem permissions and retry",
                request_id=request_id,
            )
        )

    task_data = spec_data.get("hierarchy", {}).get(task_id, {})
    started_at = task_data.get("metadata", {}).get("started_at")
    elapsed_ms = (time.perf_counter() - start) * 1000
    response = success_response(
        spec_id=spec_id,
        task_id=task_id,
        started_at=started_at,
        title=task_data.get("title", ""),
        type=task_data.get("type", "task"),
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})
    return asdict(response)


def _handle_complete(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "complete"

    err = validate_payload(payload, _COMPLETE_SCHEMA, tool_name="task", action=action, request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    task_id = payload["task_id"]
    completion_note = payload["completion_note"]
    bypass_autonomy_lock = payload.get("bypass_autonomy_lock", False)
    bypass_reason = payload.get("bypass_reason")
    workspace = payload.get("workspace")

    # Check autonomy write-lock before proceeding with protected mutation
    lock_error = _check_autonomy_write_lock(
        spec_id=spec_id,
        workspace=workspace,
        bypass_autonomy_lock=bool(bypass_autonomy_lock),
        bypass_reason=bypass_reason,
        request_id=request_id,
        config=config,
    )
    if lock_error:
        return lock_error

    specs_dir, _specs_err = _resolve_specs_dir(config, workspace)
    spec_data, error = _load_spec_data(spec_id, specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    start = time.perf_counter()
    updated = update_task_status(spec_data, task_id, "completed", note=None)
    if not updated:
        return asdict(
            error_response(
                f"Failed to complete task: {task_id}",
                error_code=ErrorCode.CONFLICT,
                error_type=ErrorType.CONFLICT,
                remediation="Confirm the task exists and is not already completed",
                request_id=request_id,
            )
        )

    update_parent_status(spec_data, task_id)
    sync_computed_fields(spec_data)

    task_data = spec_data.get("hierarchy", {}).get(task_id, {})

    # Determine if commit is suggested based on git cadence config
    suggest_commit = False
    commit_scope: Optional[str] = None
    commit_message_hint: Optional[str] = None

    if config.git.enabled:
        cadence = config.git.commit_cadence
        hierarchy = spec_data.get("hierarchy", {})

        if cadence == "task":
            suggest_commit = True
            commit_scope = "task"
            commit_message_hint = f"task: {task_data.get('title', task_id)}"
        elif cadence == "phase":
            # Check if parent phase just completed
            parent_id = task_data.get("parent")
            if parent_id:
                parent_data = hierarchy.get(parent_id, {})
                # Only suggest commit if parent is a phase and is now completed
                if parent_data.get("type") == "phase" and parent_data.get("status") == "completed":
                    suggest_commit = True
                    commit_scope = "phase"
                    commit_message_hint = f"phase: {parent_data.get('title', parent_id)}"
    add_journal_entry(
        spec_data,
        title=f"Task Completed: {task_data.get('title', task_id)}",
        content=completion_note,
        entry_type="status_change",
        task_id=task_id,
        author="foundry-mcp",
    )

    if specs_dir is None or not save_spec(spec_id, spec_data, specs_dir):
        return asdict(
            error_response(
                "Failed to save spec",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check filesystem permissions and retry",
                request_id=request_id,
            )
        )

    completed_at = task_data.get("metadata", {}).get("completed_at")

    # Note: Autonomous session tracking has been migrated to the autonomy module.
    # Batch mode is now handled by AutonomousSessionState in core/autonomy/models.py.

    progress = get_progress_summary(spec_data)
    elapsed_ms = (time.perf_counter() - start) * 1000

    response_kwargs: Dict[str, Any] = {
        "spec_id": spec_id,
        "task_id": task_id,
        "completed_at": completed_at,
        "progress": {
            "completed_tasks": progress.get("completed_tasks", 0),
            "total_tasks": progress.get("total_tasks", 0),
            "percentage": progress.get("percentage", 0),
        },
        "suggest_commit": suggest_commit,
        "commit_scope": commit_scope,
        "commit_message_hint": commit_message_hint,
        "request_id": request_id,
        "telemetry": {"duration_ms": round(elapsed_ms, 2)},
    }

    response = success_response(**response_kwargs)
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})
    return asdict(response)


def _handle_block(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "block"

    # Default blocker_type before validation so choices check works
    if payload.get("blocker_type") is None:
        payload["blocker_type"] = "dependency"

    err = validate_payload(payload, _BLOCK_SCHEMA, tool_name="task", action=action, request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    task_id = payload["task_id"]
    reason = payload["reason"]
    blocker_type = payload["blocker_type"]
    ticket = payload.get("ticket")
    bypass_autonomy_lock = payload.get("bypass_autonomy_lock", False)
    bypass_reason = payload.get("bypass_reason")
    workspace = payload.get("workspace")

    # Check autonomy write-lock before proceeding with protected mutation
    lock_error = _check_autonomy_write_lock(
        spec_id=spec_id,
        workspace=workspace,
        bypass_autonomy_lock=bool(bypass_autonomy_lock),
        bypass_reason=bypass_reason,
        request_id=request_id,
        config=config,
    )
    if lock_error:
        return lock_error

    specs_dir, _specs_err = _resolve_specs_dir(config, workspace)
    spec_data, error = _load_spec_data(spec_id, specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    start = time.perf_counter()
    blocked = mark_blocked(
        spec_data,
        task_id,
        reason,
        blocker_type=blocker_type,
        ticket=ticket,
    )
    if not blocked:
        return asdict(
            error_response(
                f"Task not found: {task_id}",
                error_code=ErrorCode.TASK_NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Verify the task ID exists in the hierarchy",
                request_id=request_id,
            )
        )

    add_journal_entry(
        spec_data,
        title=f"Task Blocked: {task_id}",
        content=f"Blocker ({blocker_type}): {reason}" + (f" [Ticket: {ticket}]" if ticket else ""),
        entry_type="blocker",
        task_id=task_id,
        author="foundry-mcp",
    )
    sync_computed_fields(spec_data)

    if specs_dir is None or not save_spec(spec_id, spec_data, specs_dir):
        return asdict(
            error_response(
                "Failed to save spec",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check filesystem permissions and retry",
                request_id=request_id,
            )
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    response = success_response(
        spec_id=spec_id,
        task_id=task_id,
        blocker_type=blocker_type,
        reason=reason,
        ticket=ticket,
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})
    return asdict(response)


def _handle_unblock(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "unblock"

    err = validate_payload(payload, _UNBLOCK_SCHEMA, tool_name="task", action=action, request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    task_id = payload["task_id"]
    resolution = payload.get("resolution")
    bypass_autonomy_lock = payload.get("bypass_autonomy_lock", False)
    bypass_reason = payload.get("bypass_reason")
    workspace = payload.get("workspace")

    # Check autonomy write-lock before proceeding with protected mutation
    lock_error = _check_autonomy_write_lock(
        spec_id=spec_id,
        workspace=workspace,
        bypass_autonomy_lock=bool(bypass_autonomy_lock),
        bypass_reason=bypass_reason,
        request_id=request_id,
        config=config,
    )
    if lock_error:
        return lock_error

    specs_dir, _specs_err = _resolve_specs_dir(config, workspace)
    spec_data, error = _load_spec_data(spec_id, specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    start = time.perf_counter()
    blocker = get_blocker_info(spec_data, task_id)
    if blocker is None:
        return asdict(
            error_response(
                f"Task {task_id} is not blocked",
                error_code=ErrorCode.CONFLICT,
                error_type=ErrorType.CONFLICT,
                remediation="Confirm the task is currently blocked before unblocking",
                request_id=request_id,
            )
        )

    unblocked = unblock_task(spec_data, task_id, resolution)
    if not unblocked:
        return asdict(
            error_response(
                f"Failed to unblock task: {task_id}",
                error_code=ErrorCode.CONFLICT,
                error_type=ErrorType.CONFLICT,
                remediation="Confirm the task exists and is currently blocked",
                request_id=request_id,
            )
        )

    add_journal_entry(
        spec_data,
        title=f"Task Unblocked: {task_id}",
        content=f"Resolved: {resolution if resolution else 'Blocker resolved'}",
        entry_type="note",
        task_id=task_id,
        author="foundry-mcp",
    )
    sync_computed_fields(spec_data)

    if specs_dir is None or not save_spec(spec_id, spec_data, specs_dir):
        return asdict(
            error_response(
                "Failed to save spec",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check filesystem permissions and retry",
                request_id=request_id,
            )
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    response = success_response(
        spec_id=spec_id,
        task_id=task_id,
        previous_blocker={
            "type": blocker.blocker_type,
            "description": blocker.description,
        },
        resolution=resolution or "Blocker resolved",
        new_status="pending",
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})
    return asdict(response)


def _handle_list_blocked(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "list-blocked"

    err = validate_payload(payload, _LIST_BLOCKED_SCHEMA, tool_name="task", action=action, request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    cursor = payload.get("cursor")
    limit = payload.get("limit")

    page_size = normalize_page_size(
        limit,
        default=_TASK_DEFAULT_PAGE_SIZE,
        maximum=_TASK_MAX_PAGE_SIZE,
    )

    start_after_id = None
    if cursor:
        try:
            cursor_data = decode_cursor(cursor)
            start_after_id = cursor_data.get("last_id")
        except CursorError as exc:
            return asdict(
                error_response(
                    f"Invalid cursor: {exc.reason or exc}",
                    error_code=ErrorCode.INVALID_FORMAT,
                    error_type=ErrorType.VALIDATION,
                    request_id=request_id,
                )
            )

    workspace = payload.get("workspace")
    specs_dir, _specs_err = _resolve_specs_dir(config, workspace)
    spec_data, error = _load_spec_data(spec_id, specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    start = time.perf_counter()
    blocked_tasks = list_blocked_tasks(spec_data)
    blocked_tasks.sort(key=lambda entry: entry.get("task_id", ""))
    total_count = len(blocked_tasks)

    if start_after_id:
        try:
            start_index = next(i for i, entry in enumerate(blocked_tasks) if entry.get("task_id") == start_after_id)
            blocked_tasks = blocked_tasks[start_index + 1 :]
        except StopIteration:
            pass

    page_tasks = blocked_tasks[: page_size + 1]
    has_more = len(page_tasks) > page_size
    if has_more:
        page_tasks = page_tasks[:page_size]

    next_cursor = None
    if has_more and page_tasks:
        next_cursor = encode_cursor({"last_id": page_tasks[-1].get("task_id")})

    elapsed_ms = (time.perf_counter() - start) * 1000
    warnings = _pagination_warnings(total_count, has_more)
    response = paginated_response(
        data={
            "spec_id": spec_id,
            "count": len(page_tasks),
            "blocked_tasks": page_tasks,
        },
        cursor=next_cursor,
        has_more=has_more,
        page_size=page_size,
        total_count=total_count,
        warnings=warnings or None,
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.counter(_metric(action), labels={"status": "success"})
    return response
