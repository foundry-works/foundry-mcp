"""Unified task router with validation, pagination, and shared delegates."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.context import generate_correlation_id, get_correlation_id
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import get_metrics, mcp_tool
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
    update_parent_status,
)
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
)
from foundry_mcp.core.spec import find_specs_directory, load_spec, save_spec
from foundry_mcp.core.journal import (
    add_journal_entry,
    get_blocker_info,
    list_blocked_tasks,
    mark_blocked,
    unblock as unblock_task,
    update_task_status,
)
from foundry_mcp.core.task import (
    add_task,
    check_dependencies,
    get_next_task,
    prepare_task as core_prepare_task,
    remove_task,
    update_estimate,
    update_task_metadata,
)
from foundry_mcp.tools.unified.router import (
    ActionDefinition,
    ActionRouter,
    ActionRouterError,
)

logger = logging.getLogger(__name__)
_metrics = get_metrics()

_TASK_DEFAULT_PAGE_SIZE = 25
_TASK_MAX_PAGE_SIZE = 100
_TASK_WARNING_THRESHOLD = 75
_ALLOWED_STATUS = {"pending", "in_progress", "completed", "blocked"}


def _request_id() -> str:
    return get_correlation_id() or generate_correlation_id(prefix="task")


def _metric(action: str) -> str:
    return f"unified_tools.task.{action.replace('-', '_')}"


def _specs_dir_missing_error(request_id: str) -> dict:
    return asdict(
        error_response(
            "No specs directory found. Use --specs-dir or set SDD_SPECS_DIR.",
            error_code=ErrorCode.NOT_FOUND,
            error_type=ErrorType.NOT_FOUND,
            remediation="Set SDD_SPECS_DIR or invoke with --specs-dir",
            request_id=request_id,
        )
    )


def _validation_error(
    *,
    field: str,
    action: str,
    message: str,
    request_id: str,
    code: ErrorCode = ErrorCode.MISSING_REQUIRED,
    remediation: Optional[str] = None,
) -> dict:
    effective_remediation = remediation or f"Provide a valid '{field}' value"
    return asdict(
        error_response(
            f"Invalid field '{field}' for task.{action}: {message}",
            error_code=code,
            error_type=ErrorType.VALIDATION,
            remediation=effective_remediation,
            details={"field": field, "action": f"task.{action}"},
            request_id=request_id,
        )
    )


def _resolve_specs_dir(
    config: ServerConfig, workspace: Optional[str]
) -> Optional[Path]:
    try:
        if workspace:
            return find_specs_directory(workspace)

        candidate = getattr(config, "specs_dir", None)
        if isinstance(candidate, Path):
            return candidate
        if isinstance(candidate, str) and candidate.strip():
            return Path(candidate)

        return find_specs_directory()
    except Exception:  # pragma: no cover - defensive guard
        logger.exception(
            "Failed to resolve specs directory", extra={"workspace": workspace}
        )
        return None


def _load_spec_data(
    spec_id: str, specs_dir: Optional[Path], request_id: str
) -> Tuple[Optional[Dict[str, Any]], Optional[dict]]:
    if specs_dir is None:
        return None, _specs_dir_missing_error(request_id)

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, asdict(
            error_response(
                f"Spec not found: {spec_id}",
                error_code=ErrorCode.SPEC_NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation='Verify the spec ID via spec(action="list")',
                request_id=request_id,
            )
        )
    return spec_data, None


def _attach_meta(
    response: dict,
    *,
    request_id: str,
    duration_ms: Optional[float] = None,
    warnings: Optional[List[str]] = None,
) -> dict:
    meta = response.setdefault("meta", {"version": "response-v2"})
    meta["request_id"] = request_id
    if warnings:
        existing = list(meta.get("warnings") or [])
        existing.extend(warnings)
        meta["warnings"] = existing
    if duration_ms is not None:
        telemetry = dict(meta.get("telemetry") or {})
        telemetry["duration_ms"] = round(duration_ms, 2)
        meta["telemetry"] = telemetry
    return response


def _filter_hierarchy(
    hierarchy: Dict[str, Any],
    max_depth: int,
    include_metadata: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    for node_id, node_data in hierarchy.items():
        node_depth = node_id.count("-") if node_id != "spec-root" else 0
        if max_depth > 0 and node_depth > max_depth:
            continue

        filtered_node: Dict[str, Any] = {
            "type": node_data.get("type"),
            "title": node_data.get("title"),
            "status": node_data.get("status"),
        }
        if "children" in node_data:
            filtered_node["children"] = node_data["children"]
        if "parent" in node_data:
            filtered_node["parent"] = node_data["parent"]

        if include_metadata:
            if "metadata" in node_data:
                filtered_node["metadata"] = node_data["metadata"]
            if "dependencies" in node_data:
                filtered_node["dependencies"] = node_data["dependencies"]

        result[node_id] = filtered_node

    return result


def _pagination_warnings(total_count: int, has_more: bool) -> List[str]:
    warnings: List[str] = []
    if total_count > _TASK_WARNING_THRESHOLD:
        warnings.append(
            f"{total_count} results returned; consider using pagination to limit payload size."
        )
    if has_more:
        warnings.append("Additional results available. Follow the cursor to continue.")
    return warnings


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
    specs_dir = _resolve_specs_dir(config, workspace)
    if specs_dir is None:
        return _specs_dir_missing_error(request_id)

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
    specs_dir = _resolve_specs_dir(config, workspace)
    spec_data, error = _load_spec_data(spec_id.strip(), specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None  # narrow Optional

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
    specs_dir = _resolve_specs_dir(config, workspace)
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
    specs_dir = _resolve_specs_dir(config, workspace)
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
    specs_dir = _resolve_specs_dir(config, workspace)
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
    specs_dir = _resolve_specs_dir(config, workspace)
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

    (time.perf_counter() - start) * 1000
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
    specs_dir = _resolve_specs_dir(config, workspace)
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
    specs_dir = _resolve_specs_dir(config, workspace)
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


def _handle_update_status(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    action = "update-status"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")
    status = payload.get("status")
    note = payload.get("note")

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
    if not isinstance(status, str) or status not in _ALLOWED_STATUS:
        return _validation_error(
            field="status",
            action=action,
            message=f"Status must be one of: {sorted(_ALLOWED_STATUS)}",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )
    if note is not None and (not isinstance(note, str) or not note.strip()):
        return _validation_error(
            field="note",
            action=action,
            message="note must be a non-empty string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    workspace = payload.get("workspace")
    specs_dir = _resolve_specs_dir(config, workspace)
    spec_data, error = _load_spec_data(spec_id.strip(), specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    hierarchy = spec_data.get("hierarchy", {})
    task_key = task_id.strip()
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

    if specs_dir is None or not save_spec(spec_id.strip(), spec_data, specs_dir):
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
        spec_id=spec_id.strip(),
        task_id=task_key,
        new_status=status,
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})
    return asdict(response)


def _handle_start(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    action = "start"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")
    note = payload.get("note")

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
    if note is not None and (not isinstance(note, str) or not note.strip()):
        return _validation_error(
            field="note",
            action=action,
            message="note must be a non-empty string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    workspace = payload.get("workspace")
    specs_dir = _resolve_specs_dir(config, workspace)
    spec_data, error = _load_spec_data(spec_id.strip(), specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    start = time.perf_counter()
    deps = check_dependencies(spec_data, task_id.strip())
    if not deps.get("can_start", False):
        blockers = [
            b.get("title", b.get("id", ""))
            for b in (deps.get("blocked_by") or [])
            if isinstance(b, dict)
        ]
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

    updated = update_task_status(spec_data, task_id.strip(), "in_progress", note=None)
    if not updated:
        return asdict(
            error_response(
                f"Failed to start task: {task_id.strip()}",
                error_code=ErrorCode.CONFLICT,
                error_type=ErrorType.CONFLICT,
                remediation="Confirm the task exists and is not blocked",
                request_id=request_id,
            )
        )

    update_parent_status(spec_data, task_id.strip())

    if note:
        add_journal_entry(
            spec_data,
            title=f"Task Started: {task_id.strip()}",
            content=note,
            entry_type="status_change",
            task_id=task_id.strip(),
            author="foundry-mcp",
        )

    if specs_dir is None or not save_spec(spec_id.strip(), spec_data, specs_dir):
        return asdict(
            error_response(
                "Failed to save spec",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check filesystem permissions and retry",
                request_id=request_id,
            )
        )

    task_data = spec_data.get("hierarchy", {}).get(task_id.strip(), {})
    started_at = task_data.get("metadata", {}).get("started_at")
    elapsed_ms = (time.perf_counter() - start) * 1000
    response = success_response(
        spec_id=spec_id.strip(),
        task_id=task_id.strip(),
        started_at=started_at,
        title=task_data.get("title", ""),
        type=task_data.get("type", "task"),
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})
    return asdict(response)


def _handle_complete(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    action = "complete"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")
    completion_note = payload.get("completion_note")

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
    if not isinstance(completion_note, str) or not completion_note.strip():
        return _validation_error(
            field="completion_note",
            action=action,
            message="Provide a non-empty completion note",
            request_id=request_id,
        )

    workspace = payload.get("workspace")
    specs_dir = _resolve_specs_dir(config, workspace)
    spec_data, error = _load_spec_data(spec_id.strip(), specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    start = time.perf_counter()
    updated = update_task_status(spec_data, task_id.strip(), "completed", note=None)
    if not updated:
        return asdict(
            error_response(
                f"Failed to complete task: {task_id.strip()}",
                error_code=ErrorCode.CONFLICT,
                error_type=ErrorType.CONFLICT,
                remediation="Confirm the task exists and is not already completed",
                request_id=request_id,
            )
        )

    update_parent_status(spec_data, task_id.strip())

    task_data = spec_data.get("hierarchy", {}).get(task_id.strip(), {})
    add_journal_entry(
        spec_data,
        title=f"Task Completed: {task_data.get('title', task_id.strip())}",
        content=completion_note,
        entry_type="status_change",
        task_id=task_id.strip(),
        author="foundry-mcp",
    )

    if specs_dir is None or not save_spec(spec_id.strip(), spec_data, specs_dir):
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
    progress = get_progress_summary(spec_data)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response = success_response(
        spec_id=spec_id.strip(),
        task_id=task_id.strip(),
        completed_at=completed_at,
        progress={
            "completed_tasks": progress.get("completed_tasks", 0),
            "total_tasks": progress.get("total_tasks", 0),
            "percentage": progress.get("percentage", 0),
        },
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})
    return asdict(response)


def _handle_block(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    action = "block"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")
    reason = payload.get("reason")
    blocker_type = payload.get("blocker_type", "dependency")
    ticket = payload.get("ticket")

    valid_types = {"dependency", "technical", "resource", "decision"}

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
    if not isinstance(reason, str) or not reason.strip():
        return _validation_error(
            field="reason",
            action=action,
            message="Provide a non-empty blocker reason",
            request_id=request_id,
        )
    if not isinstance(blocker_type, str) or blocker_type not in valid_types:
        return _validation_error(
            field="blocker_type",
            action=action,
            message=f"blocker_type must be one of: {sorted(valid_types)}",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )
    if ticket is not None and not isinstance(ticket, str):
        return _validation_error(
            field="ticket",
            action=action,
            message="ticket must be a string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    workspace = payload.get("workspace")
    specs_dir = _resolve_specs_dir(config, workspace)
    spec_data, error = _load_spec_data(spec_id.strip(), specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    start = time.perf_counter()
    blocked = mark_blocked(
        spec_data,
        task_id.strip(),
        reason.strip(),
        blocker_type=blocker_type,
        ticket=ticket,
    )
    if not blocked:
        return asdict(
            error_response(
                f"Task not found: {task_id.strip()}",
                error_code=ErrorCode.TASK_NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Verify the task ID exists in the hierarchy",
                request_id=request_id,
            )
        )

    add_journal_entry(
        spec_data,
        title=f"Task Blocked: {task_id.strip()}",
        content=f"Blocker ({blocker_type}): {reason.strip()}"
        + (f" [Ticket: {ticket}]" if ticket else ""),
        entry_type="blocker",
        task_id=task_id.strip(),
        author="foundry-mcp",
    )

    if specs_dir is None or not save_spec(spec_id.strip(), spec_data, specs_dir):
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
        spec_id=spec_id.strip(),
        task_id=task_id.strip(),
        blocker_type=blocker_type,
        reason=reason.strip(),
        ticket=ticket,
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})
    return asdict(response)


def _handle_unblock(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    action = "unblock"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")
    resolution = payload.get("resolution")

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
    if resolution is not None and (
        not isinstance(resolution, str) or not resolution.strip()
    ):
        return _validation_error(
            field="resolution",
            action=action,
            message="resolution must be a non-empty string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    workspace = payload.get("workspace")
    specs_dir = _resolve_specs_dir(config, workspace)
    spec_data, error = _load_spec_data(spec_id.strip(), specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    start = time.perf_counter()
    blocker = get_blocker_info(spec_data, task_id.strip())
    if blocker is None:
        return asdict(
            error_response(
                f"Task {task_id.strip()} is not blocked",
                error_code=ErrorCode.CONFLICT,
                error_type=ErrorType.CONFLICT,
                remediation="Confirm the task is currently blocked before unblocking",
                request_id=request_id,
            )
        )

    unblocked = unblock_task(spec_data, task_id.strip(), resolution)
    if not unblocked:
        return asdict(
            error_response(
                f"Failed to unblock task: {task_id.strip()}",
                error_code=ErrorCode.CONFLICT,
                error_type=ErrorType.CONFLICT,
                remediation="Confirm the task exists and is currently blocked",
                request_id=request_id,
            )
        )

    add_journal_entry(
        spec_data,
        title=f"Task Unblocked: {task_id.strip()}",
        content=f"Resolved: {resolution.strip() if isinstance(resolution, str) else 'Blocker resolved'}",
        entry_type="note",
        task_id=task_id.strip(),
        author="foundry-mcp",
    )

    if specs_dir is None or not save_spec(spec_id.strip(), spec_data, specs_dir):
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
        spec_id=spec_id.strip(),
        task_id=task_id.strip(),
        previous_blocker={
            "type": blocker.blocker_type,
            "description": blocker.description,
        },
        resolution=(resolution.strip() if isinstance(resolution, str) else None)
        or "Blocker resolved",
        new_status="pending",
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})
    return asdict(response)


def _handle_list_blocked(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    action = "list-blocked"
    spec_id = payload.get("spec_id")
    cursor = payload.get("cursor")
    limit = payload.get("limit")

    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec identifier",
            request_id=request_id,
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
    specs_dir = _resolve_specs_dir(config, workspace)
    spec_data, error = _load_spec_data(spec_id.strip(), specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    start = time.perf_counter()
    blocked_tasks = list_blocked_tasks(spec_data)
    blocked_tasks.sort(key=lambda entry: entry.get("task_id", ""))
    total_count = len(blocked_tasks)

    if start_after_id:
        try:
            start_index = next(
                i
                for i, entry in enumerate(blocked_tasks)
                if entry.get("task_id") == start_after_id
            )
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
            "spec_id": spec_id.strip(),
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


def _handle_add(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    action = "add"
    spec_id = payload.get("spec_id")
    parent = payload.get("parent")
    title = payload.get("title")
    description = payload.get("description")
    task_type = payload.get("task_type", "task")
    estimated_hours = payload.get("estimated_hours")
    position = payload.get("position")

    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec identifier",
            request_id=request_id,
        )
    if not isinstance(parent, str) or not parent.strip():
        return _validation_error(
            field="parent",
            action=action,
            message="Provide a non-empty parent node identifier",
            request_id=request_id,
        )
    if not isinstance(title, str) or not title.strip():
        return _validation_error(
            field="title",
            action=action,
            message="Provide a non-empty task title",
            request_id=request_id,
        )
    if description is not None and not isinstance(description, str):
        return _validation_error(
            field="description",
            action=action,
            message="description must be a string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )
    if not isinstance(task_type, str):
        return _validation_error(
            field="task_type",
            action=action,
            message="task_type must be a string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )
    if estimated_hours is not None and not isinstance(estimated_hours, (int, float)):
        return _validation_error(
            field="estimated_hours",
            action=action,
            message="estimated_hours must be a number",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )
    if position is not None and (not isinstance(position, int) or position < 0):
        return _validation_error(
            field="position",
            action=action,
            message="position must be a non-negative integer",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    dry_run = payload.get("dry_run", False)
    if dry_run is not None and not isinstance(dry_run, bool):
        return _validation_error(
            field="dry_run",
            action=action,
            message="dry_run must be a boolean",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )
    dry_run_bool = bool(dry_run)

    workspace = payload.get("workspace")
    specs_dir = _resolve_specs_dir(config, workspace)
    if specs_dir is None:
        return _specs_dir_missing_error(request_id)

    start = time.perf_counter()
    if dry_run_bool:
        spec_data, spec_error = _load_spec_data(spec_id.strip(), specs_dir, request_id)
        if spec_error:
            return spec_error

        hierarchy = (spec_data or {}).get("hierarchy", {})
        parent_node = (
            hierarchy.get(parent.strip()) if isinstance(hierarchy, dict) else None
        )
        if not isinstance(parent_node, dict):
            elapsed_ms = (time.perf_counter() - start) * 1000
            return asdict(
                error_response(
                    f"Parent node '{parent.strip()}' not found",
                    error_code=ErrorCode.NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation="Verify the parent node ID exists in the specification",
                    request_id=request_id,
                    telemetry={"duration_ms": round(elapsed_ms, 2)},
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        response = success_response(
            data={
                "spec_id": spec_id.strip(),
                "parent": parent.strip(),
                "title": title.strip(),
                "task_type": task_type,
                "position": position,
                "dry_run": True,
            },
            request_id=request_id,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
        )
        _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
        _metrics.counter(
            _metric(action), labels={"status": "success", "dry_run": "true"}
        )
        return asdict(response)

    result, error = add_task(
        spec_id=spec_id.strip(),
        parent_id=parent.strip(),
        title=title.strip(),
        description=description,
        task_type=task_type,
        estimated_hours=float(estimated_hours) if estimated_hours is not None else None,
        position=position,
        specs_dir=specs_dir,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if error or result is None:
        code = (
            ErrorCode.NOT_FOUND
            if "not found" in (error or "").lower()
            else ErrorCode.VALIDATION_ERROR
        )
        err_type = (
            ErrorType.NOT_FOUND if code == ErrorCode.NOT_FOUND else ErrorType.VALIDATION
        )
        return asdict(
            error_response(
                error or "Failed to add task",
                error_code=code,
                error_type=err_type,
                remediation="Verify parent/task inputs and retry",
                request_id=request_id,
            )
        )

    response = success_response(
        **result,
        spec_id=spec_id.strip(),
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})
    return asdict(response)


def _handle_remove(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    action = "remove"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")
    cascade = payload.get("cascade", False)

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
    if not isinstance(cascade, bool):
        return _validation_error(
            field="cascade",
            action=action,
            message="cascade must be a boolean",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    dry_run = payload.get("dry_run", False)
    if dry_run is not None and not isinstance(dry_run, bool):
        return _validation_error(
            field="dry_run",
            action=action,
            message="dry_run must be a boolean",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )
    dry_run_bool = bool(dry_run)

    workspace = payload.get("workspace")
    specs_dir = _resolve_specs_dir(config, workspace)
    if specs_dir is None:
        return _specs_dir_missing_error(request_id)

    start = time.perf_counter()
    if dry_run_bool:
        spec_data, spec_error = _load_spec_data(spec_id.strip(), specs_dir, request_id)
        if spec_error:
            return spec_error

        hierarchy = (spec_data or {}).get("hierarchy", {})
        node = hierarchy.get(task_id.strip()) if isinstance(hierarchy, dict) else None
        if not isinstance(node, dict):
            elapsed_ms = (time.perf_counter() - start) * 1000
            return asdict(
                error_response(
                    f"Task '{task_id.strip()}' not found",
                    error_code=ErrorCode.TASK_NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation="Verify the task ID exists in the specification",
                    request_id=request_id,
                    telemetry={"duration_ms": round(elapsed_ms, 2)},
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        response = success_response(
            data={
                "spec_id": spec_id.strip(),
                "task_id": task_id.strip(),
                "cascade": cascade,
                "dry_run": True,
            },
            request_id=request_id,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
        )
        _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
        _metrics.counter(
            _metric(action), labels={"status": "success", "dry_run": "true"}
        )
        return asdict(response)

    result, error = remove_task(
        spec_id=spec_id.strip(),
        task_id=task_id.strip(),
        cascade=cascade,
        specs_dir=specs_dir,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if error or result is None:
        code = (
            ErrorCode.NOT_FOUND
            if "not found" in (error or "").lower()
            else ErrorCode.VALIDATION_ERROR
        )
        err_type = (
            ErrorType.NOT_FOUND if code == ErrorCode.NOT_FOUND else ErrorType.VALIDATION
        )
        return asdict(
            error_response(
                error or "Failed to remove task",
                error_code=code,
                error_type=err_type,
                remediation="Verify the task ID and cascade flag",
                request_id=request_id,
            )
        )

    response = success_response(
        **result,
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})
    return asdict(response)


def _handle_update_estimate(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    action = "update-estimate"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")
    estimated_hours = payload.get("estimated_hours")
    complexity = payload.get("complexity")

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
    if estimated_hours is not None and not isinstance(estimated_hours, (int, float)):
        return _validation_error(
            field="estimated_hours",
            action=action,
            message="estimated_hours must be a number",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )
    if complexity is not None and not isinstance(complexity, str):
        return _validation_error(
            field="complexity",
            action=action,
            message="complexity must be a string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    dry_run = payload.get("dry_run", False)
    if dry_run is not None and not isinstance(dry_run, bool):
        return _validation_error(
            field="dry_run",
            action=action,
            message="dry_run must be a boolean",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )
    dry_run_bool = bool(dry_run)

    normalized_complexity: Optional[str] = None
    if isinstance(complexity, str):
        normalized_complexity = complexity.strip().lower() or None

    if estimated_hours is None and normalized_complexity is None:
        return _validation_error(
            field="estimated_hours",
            action=action,
            message="Provide estimated_hours and/or complexity",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
            remediation="Provide hours and/or complexity to update",
        )

    workspace = payload.get("workspace")
    specs_dir = _resolve_specs_dir(config, workspace)
    if specs_dir is None:
        return _specs_dir_missing_error(request_id)

    start = time.perf_counter()
    if dry_run_bool:
        spec_data, spec_error = _load_spec_data(spec_id.strip(), specs_dir, request_id)
        if spec_error:
            return spec_error

        hierarchy = (spec_data or {}).get("hierarchy", {})
        task = hierarchy.get(task_id.strip()) if isinstance(hierarchy, dict) else None
        if not isinstance(task, dict):
            return asdict(
                error_response(
                    f"Task '{task_id.strip()}' not found",
                    error_code=ErrorCode.TASK_NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation="Verify the task ID exists in the specification",
                    request_id=request_id,
                )
            )

        metadata_candidate = task.get("metadata")
        if isinstance(metadata_candidate, dict):
            metadata: Dict[str, Any] = metadata_candidate
        else:
            metadata = {}
        data: Dict[str, Any] = {
            "spec_id": spec_id.strip(),
            "task_id": task_id.strip(),
            "dry_run": True,
            "previous_hours": metadata.get("estimated_hours"),
            "previous_complexity": metadata.get("complexity"),
        }
        if estimated_hours is not None:
            data["hours"] = float(estimated_hours)
        if normalized_complexity is not None:
            data["complexity"] = normalized_complexity

        elapsed_ms = (time.perf_counter() - start) * 1000
        response = success_response(
            data=data,
            request_id=request_id,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
        )
        _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
        _metrics.counter(
            _metric(action), labels={"status": "success", "dry_run": "true"}
        )
        return asdict(response)

    result, error = update_estimate(
        spec_id=spec_id.strip(),
        task_id=task_id.strip(),
        estimated_hours=float(estimated_hours) if estimated_hours is not None else None,
        complexity=normalized_complexity,
        specs_dir=specs_dir,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if error or result is None:
        code = (
            ErrorCode.NOT_FOUND
            if "not found" in (error or "").lower()
            else ErrorCode.VALIDATION_ERROR
        )
        err_type = (
            ErrorType.NOT_FOUND if code == ErrorCode.NOT_FOUND else ErrorType.VALIDATION
        )
        return asdict(
            error_response(
                error or "Failed to update estimate",
                error_code=code,
                error_type=err_type,
                remediation="Provide estimated_hours and/or a valid complexity",
                request_id=request_id,
            )
        )

    response = success_response(
        **result,
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})
    return asdict(response)


def _handle_update_metadata(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    action = "update-metadata"
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

    dry_run = payload.get("dry_run", False)
    if dry_run is not None and not isinstance(dry_run, bool):
        return _validation_error(
            field="dry_run",
            action=action,
            message="dry_run must be a boolean",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )
    dry_run_bool = bool(dry_run)

    custom_metadata = payload.get("custom_metadata")
    if custom_metadata is not None and not isinstance(custom_metadata, dict):
        return _validation_error(
            field="custom_metadata",
            action=action,
            message="custom_metadata must be an object",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
            remediation="Provide custom_metadata as a JSON object",
        )

    update_fields = [
        payload.get("file_path"),
        payload.get("description"),
        payload.get("task_category"),
        payload.get("actual_hours"),
        payload.get("status_note"),
        payload.get("verification_type"),
        payload.get("command"),
    ]
    has_update = any(field is not None for field in update_fields) or bool(
        custom_metadata
    )
    if not has_update:
        return _validation_error(
            field="file_path",
            action=action,
            message="Provide at least one metadata field",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
            remediation="Provide file_path, description, task_category, actual_hours, status_note, verification_type, command, and/or custom_metadata",
        )

    workspace = payload.get("workspace")
    specs_dir = _resolve_specs_dir(config, workspace)
    if specs_dir is None:
        return _specs_dir_missing_error(request_id)

    start = time.perf_counter()
    if dry_run_bool:
        spec_data, spec_error = _load_spec_data(spec_id.strip(), specs_dir, request_id)
        if spec_error:
            return spec_error

        hierarchy = (spec_data or {}).get("hierarchy", {})
        task = hierarchy.get(task_id.strip()) if isinstance(hierarchy, dict) else None
        if not isinstance(task, dict):
            return asdict(
                error_response(
                    f"Task '{task_id.strip()}' not found",
                    error_code=ErrorCode.TASK_NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation="Verify the task ID exists in the specification",
                    request_id=request_id,
                )
            )

        fields_updated: List[str] = []
        if payload.get("file_path") is not None:
            fields_updated.append("file_path")
        if payload.get("description") is not None:
            fields_updated.append("description")
        if payload.get("task_category") is not None:
            fields_updated.append("task_category")
        if payload.get("actual_hours") is not None:
            fields_updated.append("actual_hours")
        if payload.get("status_note") is not None:
            fields_updated.append("status_note")
        if payload.get("verification_type") is not None:
            fields_updated.append("verification_type")
        if payload.get("command") is not None:
            fields_updated.append("command")
        if custom_metadata:
            fields_updated.extend(sorted(custom_metadata.keys()))

        elapsed_ms = (time.perf_counter() - start) * 1000
        response = success_response(
            data={
                "spec_id": spec_id.strip(),
                "task_id": task_id.strip(),
                "fields_updated": fields_updated,
                "dry_run": True,
            },
            request_id=request_id,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
        )
        _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
        _metrics.counter(
            _metric(action), labels={"status": "success", "dry_run": "true"}
        )
        return asdict(response)

    result, error = update_task_metadata(
        spec_id=spec_id.strip(),
        task_id=task_id.strip(),
        file_path=payload.get("file_path"),
        description=payload.get("description"),
        task_category=payload.get("task_category"),
        actual_hours=payload.get("actual_hours"),
        status_note=payload.get("status_note"),
        verification_type=payload.get("verification_type"),
        command=payload.get("command"),
        custom_metadata=custom_metadata,
        specs_dir=specs_dir,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if error or result is None:
        code = (
            ErrorCode.NOT_FOUND
            if "not found" in (error or "").lower()
            else ErrorCode.VALIDATION_ERROR
        )
        err_type = (
            ErrorType.NOT_FOUND if code == ErrorCode.NOT_FOUND else ErrorType.VALIDATION
        )
        return asdict(
            error_response(
                error or "Failed to update metadata",
                error_code=code,
                error_type=err_type,
                remediation="Provide at least one metadata field to update",
                request_id=request_id,
            )
        )

    response = success_response(
        **result,
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})
    return asdict(response)


_ACTION_DEFINITIONS = [
    ActionDefinition(
        name="prepare",
        handler=_handle_prepare,
        summary="Prepare next actionable task context",
    ),
    ActionDefinition(
        name="next", handler=_handle_next, summary="Return the next actionable task"
    ),
    ActionDefinition(
        name="info", handler=_handle_info, summary="Fetch task metadata by ID"
    ),
    ActionDefinition(
        name="check-deps",
        handler=_handle_check_deps,
        summary="Analyze task dependencies and blockers",
    ),
    ActionDefinition(name="start", handler=_handle_start, summary="Start a task"),
    ActionDefinition(
        name="complete", handler=_handle_complete, summary="Complete a task"
    ),
    ActionDefinition(
        name="update-status",
        handler=_handle_update_status,
        summary="Update task status",
    ),
    ActionDefinition(name="block", handler=_handle_block, summary="Block a task"),
    ActionDefinition(name="unblock", handler=_handle_unblock, summary="Unblock a task"),
    ActionDefinition(
        name="list-blocked",
        handler=_handle_list_blocked,
        summary="List blocked tasks",
    ),
    ActionDefinition(name="add", handler=_handle_add, summary="Add a task"),
    ActionDefinition(name="remove", handler=_handle_remove, summary="Remove a task"),
    ActionDefinition(
        name="update-estimate",
        handler=_handle_update_estimate,
        summary="Update estimated effort",
    ),
    ActionDefinition(
        name="update-metadata",
        handler=_handle_update_metadata,
        summary="Update task metadata fields",
    ),
    ActionDefinition(
        name="progress",
        handler=_handle_progress,
        summary="Summarize completion metrics for a node",
    ),
    ActionDefinition(
        name="list",
        handler=_handle_list,
        summary="List tasks with pagination and optional filters",
    ),
    ActionDefinition(
        name="query",
        handler=_handle_query,
        summary="Query tasks by status or parent",
    ),
    ActionDefinition(
        name="hierarchy",
        handler=_handle_hierarchy,
        summary="Return paginated hierarchy slices",
    ),
]

_TASK_ROUTER = ActionRouter(tool_name="task", actions=_ACTION_DEFINITIONS)


def _dispatch_task_action(
    *, action: str, payload: Dict[str, Any], config: ServerConfig
) -> dict:
    try:
        return _TASK_ROUTER.dispatch(action=action, config=config, payload=payload)
    except ActionRouterError as exc:
        request_id = _request_id()
        allowed = ", ".join(exc.allowed_actions)
        return asdict(
            error_response(
                f"Unsupported task action '{action}'. Allowed actions: {allowed}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {allowed}",
                request_id=request_id,
            )
        )


def register_unified_task_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the consolidated task tool."""

    @canonical_tool(
        mcp,
        canonical_name="task",
    )
    @mcp_tool(tool_name="task", emit_metrics=True, audit=True)
    def task(
        action: str,
        spec_id: Optional[str] = None,
        task_id: Optional[str] = None,
        workspace: Optional[str] = None,
        status_filter: Optional[str] = None,
        include_completed: bool = True,
        node_id: str = "spec-root",
        include_phases: bool = True,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        parent: Optional[str] = None,
        status: Optional[str] = None,
        note: Optional[str] = None,
        completion_note: Optional[str] = None,
        reason: Optional[str] = None,
        blocker_type: str = "dependency",
        ticket: Optional[str] = None,
        resolution: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        task_type: str = "task",
        estimated_hours: Optional[float] = None,
        position: Optional[int] = None,
        cascade: bool = False,
        complexity: Optional[str] = None,
        file_path: Optional[str] = None,
        task_category: Optional[str] = None,
        actual_hours: Optional[float] = None,
        status_note: Optional[str] = None,
        verification_type: Optional[str] = None,
        command: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
        max_depth: int = 2,
        include_metadata: bool = False,
    ) -> dict:
        payload = {
            "spec_id": spec_id,
            "task_id": task_id,
            "workspace": workspace,
            "status_filter": status_filter,
            "include_completed": include_completed,
            "node_id": node_id,
            "include_phases": include_phases,
            "cursor": cursor,
            "limit": limit,
            "parent": parent,
            "status": status,
            "note": note,
            "completion_note": completion_note,
            "reason": reason,
            "blocker_type": blocker_type,
            "ticket": ticket,
            "resolution": resolution,
            "title": title,
            "description": description,
            "task_type": task_type,
            "estimated_hours": estimated_hours,
            "position": position,
            "cascade": cascade,
            "complexity": complexity,
            "file_path": file_path,
            "task_category": task_category,
            "actual_hours": actual_hours,
            "status_note": status_note,
            "verification_type": verification_type,
            "command": command,
            "custom_metadata": custom_metadata,
            "dry_run": dry_run,
            "max_depth": max_depth,
            "include_metadata": include_metadata,
        }
        return _dispatch_task_action(action=action, payload=payload, config=config)

    logger.debug("Registered unified task tool")


__all__ = [
    "register_unified_task_tool",
]
