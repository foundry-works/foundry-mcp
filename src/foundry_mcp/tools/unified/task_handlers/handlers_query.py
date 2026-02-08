"""Query/read action handlers: prepare, next, info, check-deps, progress, list, query, hierarchy, session-config."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.pagination import (
    CursorError,
    decode_cursor,
    encode_cursor,
    normalize_page_size,
    paginated_response,
)
from foundry_mcp.core.progress import (
    get_progress_summary,
    list_phases,
)
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
)
from foundry_mcp.core.task import (
    check_dependencies,
    get_next_task,
    prepare_task as core_prepare_task,
)
from foundry_mcp.cli.context import (
    AutonomousSession,
    get_context_tracker,
)
from foundry_mcp.core.llm_config import get_workflow_config, WorkflowMode

from foundry_mcp.tools.unified.task_handlers._helpers import (
    _ALLOWED_STATUS,
    _TASK_DEFAULT_PAGE_SIZE,
    _TASK_MAX_PAGE_SIZE,
    _attach_meta,
    _filter_hierarchy,
    _load_spec_data,
    _metric,
    _metrics,
    _pagination_warnings,
    _request_id,
    _resolve_specs_dir,
    _validation_error,
)


def _handle_prepare(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    action = "prepare"
    spec_id = payload.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec identifier",
            request_id=request_id,
        )
    task_id = payload.get("task_id")
    if task_id is not None and (not isinstance(task_id, str) or not task_id.strip()):
        return _validation_error(
            field="task_id",
            action=action,
            message="task_id must be a non-empty string",
            request_id=request_id,
        )

    workspace = payload.get("workspace")
    specs_dir, specs_err = _resolve_specs_dir(config, workspace)
    if specs_err:
        return specs_err

    start = time.perf_counter()
    result = core_prepare_task(
        spec_id=spec_id.strip(), specs_dir=specs_dir, task_id=task_id
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})
    return _attach_meta(result, request_id=request_id, duration_ms=elapsed_ms)


def _handle_next(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    action = "next"
    spec_id = payload.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec identifier",
            request_id=request_id,
        )
    workspace = payload.get("workspace")
    specs_dir, _specs_err = _resolve_specs_dir(config, workspace)
    spec_data, error = _load_spec_data(spec_id.strip(), specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    start = time.perf_counter()
    next_task = get_next_task(spec_data)
    elapsed_ms = (time.perf_counter() - start) * 1000
    telemetry = {"duration_ms": round(elapsed_ms, 2)}

    if next_task:
        task_id, task_data = next_task
        response = success_response(
            spec_id=spec_id.strip(),
            found=True,
            task_id=task_id,
            title=task_data.get("title", ""),
            type=task_data.get("type", "task"),
            status=task_data.get("status", "pending"),
            metadata=task_data.get("metadata", {}),
            request_id=request_id,
            telemetry=telemetry,
        )
    else:
        hierarchy = spec_data.get("hierarchy", {})
        all_tasks = [
            node
            for node in hierarchy.values()
            if node.get("type") in {"task", "subtask", "verify"}
        ]
        completed = sum(1 for node in all_tasks if node.get("status") == "completed")
        pending = sum(1 for node in all_tasks if node.get("status") == "pending")
        response = success_response(
            spec_id=spec_id.strip(),
            found=False,
            spec_complete=pending == 0 and completed > 0,
            message="All tasks completed"
            if pending == 0 and completed > 0
            else "No actionable tasks (tasks may be blocked)",
            request_id=request_id,
            telemetry=telemetry,
        )

    _metrics.counter(_metric(action), labels={"status": "success"})
    return asdict(response)


def _handle_info(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    action = "info"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")
    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec identifier",
            request_id=request_id,
        )
    if not isinstance(task_id, str) or not task_id.strip():
        return _validation_error(
            field="task_id",
            action=action,
            message="Provide a non-empty task identifier",
            request_id=request_id,
        )

    workspace = payload.get("workspace")
    specs_dir, _specs_err = _resolve_specs_dir(config, workspace)
    spec_data, error = _load_spec_data(spec_id.strip(), specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    task = spec_data.get("hierarchy", {}).get(task_id.strip())
    if task is None:
        return asdict(
            error_response(
                f"Task not found: {task_id.strip()}",
                error_code=ErrorCode.TASK_NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Verify the task ID exists in the hierarchy",
                request_id=request_id,
            )
        )

    response = success_response(
        spec_id=spec_id.strip(),
        task_id=task_id.strip(),
        task=task,
        request_id=request_id,
    )
    _metrics.counter(_metric(action), labels={"status": "success"})
    return asdict(response)


def _handle_check_deps(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    action = "check-deps"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")
    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec identifier",
            request_id=request_id,
        )
    if not isinstance(task_id, str) or not task_id.strip():
        return _validation_error(
            field="task_id",
            action=action,
            message="Provide a non-empty task identifier",
            request_id=request_id,
        )

    workspace = payload.get("workspace")
    specs_dir, _specs_err = _resolve_specs_dir(config, workspace)
    spec_data, error = _load_spec_data(spec_id.strip(), specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    start = time.perf_counter()
    deps = check_dependencies(spec_data, task_id.strip())
    elapsed_ms = (time.perf_counter() - start) * 1000
    response = success_response(
        **deps,
        spec_id=spec_id.strip(),
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})
    return asdict(response)


def _handle_progress(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    action = "progress"
    spec_id = payload.get("spec_id")
    node_id = payload.get("node_id", "spec-root")
    include_phases = payload.get("include_phases", True)

    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec identifier",
            request_id=request_id,
        )
    if not isinstance(node_id, str) or not node_id.strip():
        return _validation_error(
            field="node_id",
            action=action,
            message="Provide a non-empty node identifier",
            request_id=request_id,
        )
    if not isinstance(include_phases, bool):
        return _validation_error(
            field="include_phases",
            action=action,
            message="Expected a boolean value",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    workspace = payload.get("workspace")
    specs_dir, _specs_err = _resolve_specs_dir(config, workspace)
    spec_data, error = _load_spec_data(spec_id.strip(), specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    progress = get_progress_summary(spec_data, node_id.strip())
    if include_phases:
        progress["phases"] = list_phases(spec_data)

    response = success_response(
        **progress,
        request_id=request_id,
    )
    _metrics.counter(_metric(action), labels={"status": "success"})
    return asdict(response)


def _handle_list(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    action = "list"
    spec_id = payload.get("spec_id")
    status_filter = payload.get("status_filter")
    include_completed = payload.get("include_completed", True)
    limit = payload.get("limit")
    cursor = payload.get("cursor")

    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec identifier",
            request_id=request_id,
        )
    if status_filter is not None:
        if not isinstance(status_filter, str) or status_filter not in _ALLOWED_STATUS:
            return _validation_error(
                field="status_filter",
                action=action,
                message=f"Status must be one of: {sorted(_ALLOWED_STATUS)}",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )
    if not isinstance(include_completed, bool):
        return _validation_error(
            field="include_completed",
            action=action,
            message="Expected a boolean value",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

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
    spec_data, error = _load_spec_data(spec_id.strip(), specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    start = time.perf_counter()
    hierarchy = spec_data.get("hierarchy", {})
    tasks: List[Dict[str, Any]] = []
    for node_id, node in hierarchy.items():
        if node.get("type") not in {"task", "subtask", "verify"}:
            continue
        status = node.get("status", "pending")
        if status_filter and status != status_filter:
            continue
        if not include_completed and status == "completed":
            continue
        tasks.append(
            {
                "id": node_id,
                "title": node.get("title", "Untitled"),
                "type": node.get("type", "task"),
                "status": status,
                "icon": node.get("icon"),
                "file_path": node.get("metadata", {}).get("file_path"),
                "parent": node.get("parent"),
            }
        )

    tasks.sort(key=lambda item: item.get("id", ""))
    total_count = len(tasks)

    if start_after_id:
        try:
            start_index = next(
                i for i, task in enumerate(tasks) if task.get("id") == start_after_id
            )
            tasks = tasks[start_index + 1 :]
        except StopIteration:
            pass

    page_tasks = tasks[: page_size + 1]
    has_more = len(page_tasks) > page_size
    if has_more:
        page_tasks = page_tasks[:page_size]

    next_cursor = None
    if has_more and page_tasks:
        next_cursor = encode_cursor({"last_id": page_tasks[-1].get("id")})

    _ = (time.perf_counter() - start) * 1000  # timing placeholder
    warnings = _pagination_warnings(total_count, has_more)
    response = paginated_response(
        data={
            "spec_id": spec_id.strip(),
            "tasks": page_tasks,
            "count": len(page_tasks),
        },
        cursor=next_cursor,
        has_more=has_more,
        page_size=page_size,
        total_count=total_count,
        warnings=warnings or None,
        request_id=request_id,
    )
    _metrics.counter(_metric(action), labels={"status": "success"})
    return response


def _handle_query(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    action = "query"
    spec_id = payload.get("spec_id")
    status = payload.get("status")
    parent = payload.get("parent")
    limit = payload.get("limit")
    cursor = payload.get("cursor")

    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec identifier",
            request_id=request_id,
        )
    if status is not None:
        if not isinstance(status, str) or status not in _ALLOWED_STATUS:
            return _validation_error(
                field="status",
                action=action,
                message=f"Status must be one of: {sorted(_ALLOWED_STATUS)}",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )
    if parent is not None and (not isinstance(parent, str) or not parent.strip()):
        return _validation_error(
            field="parent",
            action=action,
            message="Parent must be a non-empty string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

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
    spec_data, error = _load_spec_data(spec_id.strip(), specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    start = time.perf_counter()
    hierarchy = spec_data.get("hierarchy", {})
    tasks: List[Dict[str, Any]] = []
    for task_id, task_data in hierarchy.items():
        if status and task_data.get("status") != status:
            continue
        if parent and task_data.get("parent") != parent:
            continue
        tasks.append(
            {
                "task_id": task_id,
                "title": task_data.get("title", task_id),
                "status": task_data.get("status", "pending"),
                "type": task_data.get("type", "task"),
                "parent": task_data.get("parent"),
            }
        )

    tasks.sort(key=lambda item: item.get("task_id", ""))
    total_count = len(tasks)

    if start_after_id:
        try:
            start_index = next(
                i
                for i, task in enumerate(tasks)
                if task.get("task_id") == start_after_id
            )
            tasks = tasks[start_index + 1 :]
        except StopIteration:
            pass

    page_tasks = tasks[: page_size + 1]
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
            "spec_id": spec_id.strip(),
            "tasks": page_tasks,
            "count": len(page_tasks),
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


def _handle_hierarchy(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    action = "hierarchy"
    spec_id = payload.get("spec_id")
    max_depth = payload.get("max_depth", 2)
    include_metadata = payload.get("include_metadata", False)
    limit = payload.get("limit")
    cursor = payload.get("cursor")

    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec identifier",
            request_id=request_id,
        )
    if not isinstance(max_depth, int) or max_depth < 0 or max_depth > 10:
        return _validation_error(
            field="max_depth",
            action=action,
            message="max_depth must be between 0 and 10",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )
    if not isinstance(include_metadata, bool):
        return _validation_error(
            field="include_metadata",
            action=action,
            message="Expected a boolean value",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

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
    spec_data, error = _load_spec_data(spec_id.strip(), specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    start = time.perf_counter()
    full_hierarchy = spec_data.get("hierarchy", {})
    filtered = _filter_hierarchy(full_hierarchy, max_depth, include_metadata)
    sorted_ids = sorted(filtered.keys())

    if start_after_id:
        try:
            start_index = sorted_ids.index(start_after_id) + 1
        except ValueError:
            start_index = 0
    else:
        start_index = 0

    page_ids = sorted_ids[start_index : start_index + page_size + 1]
    has_more = len(page_ids) > page_size
    if has_more:
        page_ids = page_ids[:page_size]

    hierarchy_page = {node_id: filtered[node_id] for node_id in page_ids}
    next_cursor = None
    if has_more and page_ids:
        next_cursor = encode_cursor({"last_id": page_ids[-1]})

    elapsed_ms = (time.perf_counter() - start) * 1000
    warnings = _pagination_warnings(len(filtered), has_more)
    response = paginated_response(
        data={
            "spec_id": spec_id.strip(),
            "hierarchy": hierarchy_page,
            "node_count": len(hierarchy_page),
            "total_nodes": len(filtered),
            "filters_applied": {
                "max_depth": max_depth,
                "include_metadata": include_metadata,
            },
        },
        cursor=next_cursor,
        has_more=has_more,
        page_size=page_size,
        total_count=len(filtered),
        warnings=warnings or None,
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.counter(_metric(action), labels={"status": "success"})
    return response


def _handle_session_config(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    """
    Handle session-config action: get/set autonomous mode preferences.

    This action manages the ephemeral autonomous session state, allowing
    agents to enable/disable autonomous mode and track task completion
    during autonomous execution.

    Parameters:
        get: If true, just return current session config without changes
        auto_mode: Set autonomous mode enabled (true) or disabled (false)

    Returns:
        Current session configuration including autonomous state
    """
    from datetime import datetime, timezone

    request_id = _request_id()
    action = "session-config"
    start = time.perf_counter()

    # Get parameters
    get_only = payload.get("get", False)
    auto_mode = payload.get("auto_mode")

    # Get the context tracker and session
    tracker = get_context_tracker()
    session = tracker.get_or_create_session()

    # Initialize autonomous if not present, applying defaults from WorkflowConfig
    if session.autonomous is None:
        workflow_config = get_workflow_config()
        # Determine initial state based on workflow mode
        initial_enabled = False
        initial_batch_mode = False
        initial_batch_size = workflow_config.batch_size

        if workflow_config.mode == WorkflowMode.AUTONOMOUS:
            initial_enabled = True
        elif workflow_config.mode == WorkflowMode.BATCH:
            initial_enabled = True
            initial_batch_mode = True

        session.autonomous = AutonomousSession(
            enabled=initial_enabled,
            batch_mode=initial_batch_mode,
            batch_size=initial_batch_size,
            batch_remaining=initial_batch_size if initial_batch_mode else None,
        )

    # If just getting, return current state
    if get_only:
        elapsed_ms = (time.perf_counter() - start) * 1000
        response = success_response(
            session_id=session.session_id,
            autonomous=session.autonomous.to_dict(),
            message="Current session configuration",
            request_id=request_id,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
        )
        _metrics.counter(_metric(action), labels={"status": "success", "operation": "get"})
        return asdict(response)

    # Handle auto_mode setting
    if auto_mode is not None:
        if not isinstance(auto_mode, bool):
            return _validation_error(
                field="auto_mode",
                action=action,
                message="auto_mode must be a boolean (true/false)",
                request_id=request_id,
            )

        previous_enabled = session.autonomous.enabled
        session.autonomous.enabled = auto_mode

        if auto_mode and not previous_enabled:
            # Starting autonomous mode - apply workflow config for batch settings
            workflow_config = get_workflow_config()
            session.autonomous.started_at = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
            session.autonomous.tasks_completed = 0
            session.autonomous.pause_reason = None

            # Configure batch mode based on workflow config
            if workflow_config.mode == WorkflowMode.BATCH:
                session.autonomous.batch_mode = True
                session.autonomous.batch_size = workflow_config.batch_size
                session.autonomous.batch_remaining = workflow_config.batch_size
            else:
                session.autonomous.batch_mode = False
                session.autonomous.batch_remaining = None
        elif not auto_mode and previous_enabled:
            # Stopping autonomous mode
            session.autonomous.pause_reason = "user"

    elapsed_ms = (time.perf_counter() - start) * 1000
    response = success_response(
        session_id=session.session_id,
        autonomous=session.autonomous.to_dict(),
        message="Autonomous mode enabled" if session.autonomous.enabled else "Autonomous mode disabled",
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.counter(_metric(action), labels={"status": "success", "operation": "set"})
    return asdict(response)
