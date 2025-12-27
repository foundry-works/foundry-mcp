"""Unified task router with validation, pagination, and shared delegates."""

from __future__ import annotations

import logging
import re
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
    sync_computed_fields,
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
    batch_update_tasks,
    check_dependencies,
    get_next_task,
    manage_task_dependency,
    move_task,
    prepare_task as core_prepare_task,
    remove_task,
    REQUIREMENT_TYPES,
    update_estimate,
    update_task_metadata,
    update_task_requirements,
)
from foundry_mcp.core.validation import (
    VALID_VERIFICATION_TYPES,
    VERIFICATION_TYPE_MAPPING,
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
    sync_computed_fields(spec_data)

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
    sync_computed_fields(spec_data)

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
    sync_computed_fields(spec_data)

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
    sync_computed_fields(spec_data)

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
    file_path = payload.get("file_path")

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
    if file_path is not None and not isinstance(file_path, str):
        return _validation_error(
            field="file_path",
            action=action,
            message="file_path must be a string",
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
                "file_path": file_path.strip() if file_path else None,
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
        file_path=file_path,
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

    acceptance_criteria = payload.get("acceptance_criteria")
    if acceptance_criteria is not None and not isinstance(acceptance_criteria, list):
        return _validation_error(
            field="acceptance_criteria",
            action=action,
            message="acceptance_criteria must be a list of strings",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    update_fields = [
        payload.get("title"),
        payload.get("file_path"),
        payload.get("description"),
        acceptance_criteria,
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
            field="title",
            action=action,
            message="Provide at least one field to update",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
            remediation="Provide title, file_path, description, acceptance_criteria, task_category, actual_hours, status_note, verification_type, command, and/or custom_metadata",
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
        if payload.get("title") is not None:
            fields_updated.append("title")
        if payload.get("file_path") is not None:
            fields_updated.append("file_path")
        if payload.get("description") is not None:
            fields_updated.append("description")
        if acceptance_criteria is not None:
            fields_updated.append("acceptance_criteria")
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
        title=payload.get("title"),
        file_path=payload.get("file_path"),
        description=payload.get("description"),
        acceptance_criteria=acceptance_criteria,
        task_category=payload.get("task_category"),
        actual_hours=payload.get("actual_hours"),
        status_note=payload.get("status_note"),
        verification_type=payload.get("verification_type"),
        command=payload.get("command"),
        custom_metadata=custom_metadata,
        dry_run=dry_run_bool,
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


def _handle_move(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    """Move a task to a new position or parent.

    Supports two modes:
    1. Reorder within parent: only specify position (new_parent=None)
    2. Reparent to different phase/task: specify new_parent, optionally position

    Updates task counts on affected parents. Prevents circular references.
    Emits warnings for cross-phase moves that might affect dependencies.
    """
    request_id = _request_id()
    action = "move"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")
    new_parent = payload.get("parent")  # Target parent (phase or task ID)
    position = payload.get("position")  # 1-based position in children list

    # Validate required fields
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

    # Validate optional new_parent
    if new_parent is not None and (
        not isinstance(new_parent, str) or not new_parent.strip()
    ):
        return _validation_error(
            field="parent",
            action=action,
            message="parent must be a non-empty string if provided",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    # Validate optional position (must be positive integer)
    if position is not None:
        if not isinstance(position, int) or position < 1:
            return _validation_error(
                field="position",
                action=action,
                message="position must be a positive integer (1-based)",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )

    # Validate dry_run
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

    # Call the core move_task function
    result, error, warnings = move_task(
        spec_id=spec_id.strip(),
        task_id=task_id.strip(),
        new_parent=new_parent.strip() if new_parent else None,
        position=position,
        dry_run=dry_run_bool,
        specs_dir=specs_dir,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if error or result is None:
        # Determine appropriate error code based on error message
        error_lower = (error or "").lower()
        if "not found" in error_lower:
            code = ErrorCode.TASK_NOT_FOUND
            err_type = ErrorType.NOT_FOUND
            remediation = "Verify the task ID and parent ID exist in the specification"
        elif "circular" in error_lower:
            code = ErrorCode.CIRCULAR_DEPENDENCY
            err_type = ErrorType.CONFLICT
            remediation = "Task cannot be moved under its own descendants"
        elif "invalid position" in error_lower:
            code = ErrorCode.INVALID_POSITION
            err_type = ErrorType.VALIDATION
            remediation = "Specify a valid position within the children list"
        elif "cannot move" in error_lower or "invalid" in error_lower:
            code = ErrorCode.INVALID_PARENT
            err_type = ErrorType.VALIDATION
            remediation = "Specify a valid phase, group, or task as the target parent"
        else:
            code = ErrorCode.VALIDATION_ERROR
            err_type = ErrorType.VALIDATION
            remediation = "Check task ID, parent, and position parameters"

        return asdict(
            error_response(
                error or "Failed to move task",
                error_code=code,
                error_type=err_type,
                remediation=remediation,
                request_id=request_id,
                telemetry={"duration_ms": round(elapsed_ms, 2)},
            )
        )

    # Build success response with warnings if any
    response = success_response(
        **result,
        request_id=request_id,
        warnings=warnings if warnings else None,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(
        _metric(action),
        labels={"status": "success", "dry_run": str(dry_run_bool).lower()},
    )
    return asdict(response)


def _handle_add_dependency(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    """Add a dependency relationship between two tasks.

    Manages blocks, blocked_by, and depends relationships.
    Updates both source and target tasks atomically.

    Dependency types:
    - blocks: Source task blocks target (target cannot start until source completes)
    - blocked_by: Source task is blocked by target (source cannot start until target completes)
    - depends: Soft dependency (informational, doesn't block)
    """
    request_id = _request_id()
    action = "add-dependency"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")  # Source task
    target_id = payload.get("target_id")  # Target task
    dependency_type = payload.get("dependency_type", "blocks")

    # Validate required fields
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
            message="Provide a non-empty source task identifier",
            request_id=request_id,
        )
    if not isinstance(target_id, str) or not target_id.strip():
        return _validation_error(
            field="target_id",
            action=action,
            message="Provide a non-empty target task identifier",
            request_id=request_id,
        )

    # Validate dependency_type
    valid_types = ("blocks", "blocked_by", "depends")
    if dependency_type not in valid_types:
        return _validation_error(
            field="dependency_type",
            action=action,
            message=f"Must be one of: {', '.join(valid_types)}",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    # Validate dry_run
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

    # Call the core function
    result, error = manage_task_dependency(
        spec_id=spec_id.strip(),
        source_task_id=task_id.strip(),
        target_task_id=target_id.strip(),
        dependency_type=dependency_type,
        action="add",
        dry_run=dry_run_bool,
        specs_dir=specs_dir,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if error or result is None:
        # Determine appropriate error code based on error message
        error_lower = (error or "").lower()
        if "not found" in error_lower:
            code = ErrorCode.TASK_NOT_FOUND
            err_type = ErrorType.NOT_FOUND
            remediation = "Verify both task IDs exist in the specification"
        elif "circular" in error_lower:
            code = ErrorCode.CIRCULAR_DEPENDENCY
            err_type = ErrorType.CONFLICT
            remediation = "This dependency would create a cycle"
        elif "itself" in error_lower:
            code = ErrorCode.SELF_REFERENCE
            err_type = ErrorType.VALIDATION
            remediation = "A task cannot depend on itself"
        elif "already exists" in error_lower:
            code = ErrorCode.DUPLICATE_ENTRY
            err_type = ErrorType.CONFLICT
            remediation = "This dependency already exists"
        else:
            code = ErrorCode.VALIDATION_ERROR
            err_type = ErrorType.VALIDATION
            remediation = "Check task IDs and dependency type"

        return asdict(
            error_response(
                error or "Failed to add dependency",
                error_code=code,
                error_type=err_type,
                remediation=remediation,
                request_id=request_id,
                telemetry={"duration_ms": round(elapsed_ms, 2)},
            )
        )

    # Build success response
    response = success_response(
        **result,
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(
        _metric(action),
        labels={"status": "success", "dry_run": str(dry_run_bool).lower()},
    )
    return asdict(response)


def _handle_remove_dependency(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    """Remove a dependency relationship between two tasks.

    Removes blocks, blocked_by, or depends relationships.
    Updates both source and target tasks atomically for reciprocal relationships.
    """
    request_id = _request_id()
    action = "remove-dependency"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")  # Source task
    target_id = payload.get("target_id")  # Target task
    dependency_type = payload.get("dependency_type", "blocks")

    # Validate required fields
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
            message="Provide a non-empty source task identifier",
            request_id=request_id,
        )
    if not isinstance(target_id, str) or not target_id.strip():
        return _validation_error(
            field="target_id",
            action=action,
            message="Provide a non-empty target task identifier",
            request_id=request_id,
        )

    # Validate dependency_type
    valid_types = ("blocks", "blocked_by", "depends")
    if dependency_type not in valid_types:
        return _validation_error(
            field="dependency_type",
            action=action,
            message=f"Must be one of: {', '.join(valid_types)}",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    # Validate dry_run
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

    # Call the core function
    result, error = manage_task_dependency(
        spec_id=spec_id.strip(),
        source_task_id=task_id.strip(),
        target_task_id=target_id.strip(),
        dependency_type=dependency_type,
        action="remove",
        dry_run=dry_run_bool,
        specs_dir=specs_dir,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if error or result is None:
        # Determine appropriate error code based on error message
        error_lower = (error or "").lower()
        if "does not exist" in error_lower:
            # Dependency relationship doesn't exist
            code = ErrorCode.DEPENDENCY_NOT_FOUND
            err_type = ErrorType.NOT_FOUND
            remediation = "This dependency does not exist"
        elif "not found" in error_lower:
            # Task or spec not found
            code = ErrorCode.TASK_NOT_FOUND
            err_type = ErrorType.NOT_FOUND
            remediation = "Verify both task IDs exist in the specification"
        else:
            code = ErrorCode.VALIDATION_ERROR
            err_type = ErrorType.VALIDATION
            remediation = "Check task IDs and dependency type"

        return asdict(
            error_response(
                error or "Failed to remove dependency",
                error_code=code,
                error_type=err_type,
                remediation=remediation,
                request_id=request_id,
                telemetry={"duration_ms": round(elapsed_ms, 2)},
            )
        )

    # Build success response
    response = success_response(
        **result,
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(
        _metric(action),
        labels={"status": "success", "dry_run": str(dry_run_bool).lower()},
    )
    return asdict(response)


def _handle_add_requirement(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    """Add a structured requirement to a task's metadata.

    Requirements are stored in metadata.requirements as a list of objects:
    [{"id": "req-1", "type": "acceptance", "text": "..."}, ...]

    Each requirement has:
    - id: Auto-generated unique ID (e.g., "req-1", "req-2")
    - type: Requirement type (acceptance, technical, constraint)
    - text: Requirement description text
    """
    request_id = _request_id()
    action = "add-requirement"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")
    requirement_type = payload.get("requirement_type")
    text = payload.get("text")

    # Validate required fields
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
    if not isinstance(requirement_type, str) or not requirement_type.strip():
        return _validation_error(
            field="requirement_type",
            action=action,
            message="Provide a requirement type",
            request_id=request_id,
        )

    # Validate requirement_type
    requirement_type_lower = requirement_type.lower().strip()
    if requirement_type_lower not in REQUIREMENT_TYPES:
        return _validation_error(
            field="requirement_type",
            action=action,
            message=f"Must be one of: {', '.join(REQUIREMENT_TYPES)}",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    # Validate text
    if not isinstance(text, str) or not text.strip():
        return _validation_error(
            field="text",
            action=action,
            message="Provide non-empty requirement text",
            request_id=request_id,
        )

    # Validate dry_run
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

    # Call the core function
    result, error = update_task_requirements(
        spec_id=spec_id.strip(),
        task_id=task_id.strip(),
        action="add",
        requirement_type=requirement_type_lower,
        text=text.strip(),
        dry_run=dry_run_bool,
        specs_dir=specs_dir,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if error or result is None:
        # Determine appropriate error code based on error message
        error_lower = (error or "").lower()
        if "not found" in error_lower:
            if "specification" in error_lower:
                code = ErrorCode.SPEC_NOT_FOUND
                err_type = ErrorType.NOT_FOUND
                remediation = "Verify the spec ID exists"
            else:
                code = ErrorCode.TASK_NOT_FOUND
                err_type = ErrorType.NOT_FOUND
                remediation = "Verify the task ID exists in the specification"
        elif "maximum" in error_lower or "limit" in error_lower:
            code = ErrorCode.LIMIT_EXCEEDED
            err_type = ErrorType.VALIDATION
            remediation = "Remove some requirements before adding new ones"
        elif "requirement_type" in error_lower:
            code = ErrorCode.INVALID_FORMAT
            err_type = ErrorType.VALIDATION
            remediation = f"Use one of: {', '.join(REQUIREMENT_TYPES)}"
        else:
            code = ErrorCode.VALIDATION_ERROR
            err_type = ErrorType.VALIDATION
            remediation = "Check task ID and requirement fields"

        return asdict(
            error_response(
                error or "Failed to add requirement",
                error_code=code,
                error_type=err_type,
                remediation=remediation,
                request_id=request_id,
                telemetry={"duration_ms": round(elapsed_ms, 2)},
            )
        )

    # Build success response
    response = success_response(
        **result,
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(
        _metric(action),
        labels={"status": "success", "dry_run": str(dry_run_bool).lower()},
    )
    return asdict(response)


_VALID_NODE_TYPES = {"task", "verify", "phase", "subtask"}
# Note: VALID_VERIFICATION_TYPES imported from foundry_mcp.core.validation


def _match_nodes_for_batch(
    hierarchy: Dict[str, Any],
    *,
    phase_id: Optional[str] = None,
    pattern: Optional[str] = None,
    node_type: Optional[str] = None,
) -> List[str]:
    """Filter nodes by phase_id, regex pattern on title/id, and/or node_type.

    All provided filters are combined with AND logic.
    Returns list of matching node IDs.
    """
    matched: List[str] = []
    compiled_pattern = None
    if pattern:
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return []  # Invalid regex returns empty

    for node_id, node_data in hierarchy.items():
        if node_id == "spec-root":
            continue

        # Filter by node_type if specified
        if node_type and node_data.get("type") != node_type:
            continue

        # Filter by phase_id if specified (must be under that phase)
        if phase_id:
            node_parent = node_data.get("parent")
            # Direct children of the phase
            if node_parent != phase_id:
                # Check if it's a nested child (e.g., subtask under task under phase)
                parent_node = hierarchy.get(node_parent, {})
                if parent_node.get("parent") != phase_id:
                    continue

        # Filter by regex pattern on title or node_id
        if compiled_pattern:
            title = node_data.get("title", "")
            if not (compiled_pattern.search(title) or compiled_pattern.search(node_id)):
                continue

        matched.append(node_id)

    return sorted(matched)


def _handle_metadata_batch(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    """Batch update metadata across multiple tasks matching specified criteria.

    Filters (combined with AND logic):
    - status_filter: Filter by task status (pending, in_progress, completed, blocked)
    - parent_filter: Filter by parent node ID (e.g., phase-1, task-2-1)
    - pattern: Regex pattern to match task titles/IDs

    Legacy filters (deprecated, use parent_filter instead):
    - phase_id: Alias for parent_filter

    Metadata fields supported:
    - description, file_path, estimated_hours, category, labels, owners
    - update_metadata: Dict for custom metadata fields (verification_type, command, etc.)
    """
    request_id = _request_id()
    action = "metadata-batch"
    start = time.perf_counter()

    # Required: spec_id
    spec_id = payload.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec identifier",
            request_id=request_id,
        )
    spec_id = spec_id.strip()

    # Extract filter parameters
    status_filter = payload.get("status_filter")
    parent_filter = payload.get("parent_filter")
    phase_id = payload.get("phase_id")  # Legacy alias for parent_filter
    pattern = payload.get("pattern")

    # Use phase_id as parent_filter if parent_filter not provided (backwards compat)
    if parent_filter is None and phase_id is not None:
        parent_filter = phase_id

    # Validate status_filter
    if status_filter is not None:
        if not isinstance(status_filter, str) or status_filter not in _ALLOWED_STATUS:
            return _validation_error(
                field="status_filter",
                action=action,
                message=f"status_filter must be one of: {sorted(_ALLOWED_STATUS)}",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )

    # Validate parent_filter
    if parent_filter is not None:
        if not isinstance(parent_filter, str) or not parent_filter.strip():
            return _validation_error(
                field="parent_filter",
                action=action,
                message="parent_filter must be a non-empty string",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )
        parent_filter = parent_filter.strip()

    # Validate pattern
    if pattern is not None:
        if not isinstance(pattern, str) or not pattern.strip():
            return _validation_error(
                field="pattern",
                action=action,
                message="pattern must be a non-empty string",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )
        try:
            re.compile(pattern)
        except re.error as exc:
            return _validation_error(
                field="pattern",
                action=action,
                message=f"Invalid regex pattern: {exc}",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )
        pattern = pattern.strip()

    # At least one filter must be provided
    if not any([status_filter, parent_filter, pattern]):
        return _validation_error(
            field="status_filter",
            action=action,
            message="Provide at least one filter: status_filter, parent_filter, or pattern",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
            remediation="Specify status_filter, parent_filter (or phase_id), and/or pattern to target tasks",
        )

    # Extract metadata fields
    description = payload.get("description")
    file_path = payload.get("file_path")
    estimated_hours = payload.get("estimated_hours")
    category = payload.get("category")
    labels = payload.get("labels")
    owners = payload.get("owners")
    update_metadata = payload.get("update_metadata")  # Dict for custom fields
    dry_run = payload.get("dry_run", False)

    # Validate metadata fields
    if description is not None and not isinstance(description, str):
        return _validation_error(
            field="description",
            action=action,
            message="description must be a string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    if file_path is not None and not isinstance(file_path, str):
        return _validation_error(
            field="file_path",
            action=action,
            message="file_path must be a string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    if estimated_hours is not None:
        if not isinstance(estimated_hours, (int, float)) or estimated_hours < 0:
            return _validation_error(
                field="estimated_hours",
                action=action,
                message="estimated_hours must be a non-negative number",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )

    if category is not None and not isinstance(category, str):
        return _validation_error(
            field="category",
            action=action,
            message="category must be a string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    if labels is not None:
        if not isinstance(labels, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in labels.items()
        ):
            return _validation_error(
                field="labels",
                action=action,
                message="labels must be a dict with string keys and values",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )

    if owners is not None:
        if not isinstance(owners, list) or not all(isinstance(o, str) for o in owners):
            return _validation_error(
                field="owners",
                action=action,
                message="owners must be a list of strings",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )

    if update_metadata is not None and not isinstance(update_metadata, dict):
        return _validation_error(
            field="update_metadata",
            action=action,
            message="update_metadata must be a dict",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    if dry_run is not None and not isinstance(dry_run, bool):
        return _validation_error(
            field="dry_run",
            action=action,
            message="dry_run must be a boolean",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    # At least one metadata field must be provided
    has_metadata = any([
        description is not None,
        file_path is not None,
        estimated_hours is not None,
        category is not None,
        labels is not None,
        owners is not None,
        update_metadata,
    ])
    if not has_metadata:
        return _validation_error(
            field="description",
            action=action,
            message="Provide at least one metadata field to update",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
            remediation="Specify description, file_path, estimated_hours, category, labels, owners, or update_metadata",
        )

    # Resolve specs directory
    workspace = payload.get("workspace")
    specs_dir = _resolve_specs_dir(config, workspace)
    if specs_dir is None:
        return _specs_dir_missing_error(request_id)

    # Delegate to core helper
    result, error = batch_update_tasks(
        spec_id,
        status_filter=status_filter,
        parent_filter=parent_filter,
        pattern=pattern,
        description=description,
        file_path=file_path,
        estimated_hours=float(estimated_hours) if estimated_hours is not None else None,
        category=category,
        labels=labels,
        owners=owners,
        custom_metadata=update_metadata,
        dry_run=bool(dry_run),
        specs_dir=specs_dir,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    if error:
        _metrics.counter(_metric(action), labels={"status": "error"})
        # Map helper errors to response-v2 format
        if "not found" in error.lower():
            return asdict(
                error_response(
                    error,
                    error_code=ErrorCode.NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation="Check spec_id and parent_filter values",
                    request_id=request_id,
                )
            )
        if "at least one" in error.lower() or "must be" in error.lower():
            return asdict(
                error_response(
                    error,
                    error_code=ErrorCode.VALIDATION_ERROR,
                    error_type=ErrorType.VALIDATION,
                    remediation="Check filter and metadata parameters",
                    request_id=request_id,
                )
            )
        return asdict(
            error_response(
                error,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check filesystem permissions and retry",
                request_id=request_id,
            )
        )

    assert result is not None

    # Build response with response-v2 envelope
    warnings: List[str] = result.get("warnings", [])
    if result["matched_count"] > _TASK_WARNING_THRESHOLD and not warnings:
        warnings.append(
            f"Updated {result['matched_count']} tasks; consider using more specific filters."
        )

    response = success_response(
        spec_id=result["spec_id"],
        matched_count=result["matched_count"],
        updated_count=result["updated_count"],
        skipped_count=result.get("skipped_count", 0),
        nodes=result["nodes"],
        filters=result["filters"],
        metadata_applied=result["metadata_applied"],
        dry_run=result["dry_run"],
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )

    response_dict = asdict(response)
    if warnings:
        meta = response_dict.setdefault("meta", {})
        meta["warnings"] = warnings
    if result.get("skipped_tasks"):
        response_dict["data"]["skipped_tasks"] = result["skipped_tasks"]

    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})
    return response_dict


def _handle_fix_verification_types(
    *, config: ServerConfig, payload: Dict[str, Any]
) -> dict:
    """Fix verification types across all verify nodes in a spec.

    This action:
    1. Finds all verify nodes with invalid or missing verification_type
    2. Maps legacy values (e.g., 'test' -> 'run-tests') using VERIFICATION_TYPE_MAPPING
    3. Sets missing types to 'run-tests' (default)
    4. Sets unknown types to 'manual' (fallback)

    Supports dry-run mode to preview changes without persisting.
    """
    request_id = _request_id()
    action = "fix-verification-types"

    # Required: spec_id
    spec_id = payload.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec identifier",
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

    # Load spec
    workspace = payload.get("workspace")
    specs_dir = _resolve_specs_dir(config, workspace)
    spec_data, error = _load_spec_data(spec_id.strip(), specs_dir, request_id)
    if error:
        return error
    assert spec_data is not None

    start = time.perf_counter()
    hierarchy = spec_data.get("hierarchy", {})

    # Find verify nodes and collect fixes
    fixes: List[Dict[str, Any]] = []
    for node_id, node_data in hierarchy.items():
        if node_data.get("type") != "verify":
            continue

        metadata = node_data.get("metadata", {})
        current_type = metadata.get("verification_type")

        # Determine the fix needed
        fix_info: Optional[Dict[str, Any]] = None

        if current_type is None:
            # Missing verification_type -> default to 'run-tests'
            fix_info = {
                "node_id": node_id,
                "title": node_data.get("title", ""),
                "issue": "missing",
                "old_value": None,
                "new_value": "run-tests",
            }
        elif current_type not in VALID_VERIFICATION_TYPES:
            # Invalid type -> check mapping or fallback to 'manual'
            mapped = VERIFICATION_TYPE_MAPPING.get(current_type)
            if mapped:
                fix_info = {
                    "node_id": node_id,
                    "title": node_data.get("title", ""),
                    "issue": "legacy",
                    "old_value": current_type,
                    "new_value": mapped,
                }
            else:
                fix_info = {
                    "node_id": node_id,
                    "title": node_data.get("title", ""),
                    "issue": "invalid",
                    "old_value": current_type,
                    "new_value": "manual",
                }

        if fix_info:
            fixes.append(fix_info)

            if not dry_run_bool:
                # Apply the fix
                if "metadata" not in node_data:
                    node_data["metadata"] = {}
                node_data["metadata"]["verification_type"] = fix_info["new_value"]

    # Save if not dry_run and there were fixes
    if not dry_run_bool and fixes:
        if specs_dir is None or not save_spec(spec_id.strip(), spec_data, specs_dir):
            return asdict(
                error_response(
                    "Failed to save spec after fixing verification types",
                    error_code=ErrorCode.INTERNAL_ERROR,
                    error_type=ErrorType.INTERNAL,
                    remediation="Check filesystem permissions and retry",
                    request_id=request_id,
                )
            )

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Count by issue type
    missing_count = sum(1 for f in fixes if f["issue"] == "missing")
    legacy_count = sum(1 for f in fixes if f["issue"] == "legacy")
    invalid_count = sum(1 for f in fixes if f["issue"] == "invalid")

    response = success_response(
        spec_id=spec_id.strip(),
        total_fixes=len(fixes),
        applied_count=len(fixes) if not dry_run_bool else 0,
        fixes=fixes,
        summary={
            "missing_set_to_run_tests": missing_count,
            "legacy_mapped": legacy_count,
            "invalid_set_to_manual": invalid_count,
        },
        valid_types=sorted(VALID_VERIFICATION_TYPES),
        legacy_mappings=VERIFICATION_TYPE_MAPPING,
        dry_run=dry_run_bool,
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
        name="move",
        handler=_handle_move,
        summary="Move task to new position or parent",
    ),
    ActionDefinition(
        name="add-dependency",
        handler=_handle_add_dependency,
        summary="Add a dependency between two tasks",
    ),
    ActionDefinition(
        name="remove-dependency",
        handler=_handle_remove_dependency,
        summary="Remove a dependency between two tasks",
    ),
    ActionDefinition(
        name="add-requirement",
        handler=_handle_add_requirement,
        summary="Add a structured requirement to a task",
    ),
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
        name="metadata-batch",
        handler=_handle_metadata_batch,
        summary="Batch update metadata across multiple nodes matching filters",
    ),
    ActionDefinition(
        name="fix-verification-types",
        handler=_handle_fix_verification_types,
        summary="Fix invalid/missing verification types across verify nodes",
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
        acceptance_criteria: Optional[List[str]] = None,
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
        # metadata-batch specific parameters
        phase_id: Optional[str] = None,
        pattern: Optional[str] = None,
        node_type: Optional[str] = None,
        owners: Optional[List[str]] = None,
        labels: Optional[Dict[str, str]] = None,
        category: Optional[str] = None,
        parent_filter: Optional[str] = None,
        update_metadata: Optional[Dict[str, Any]] = None,
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
            "acceptance_criteria": acceptance_criteria,
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
            # metadata-batch specific
            "phase_id": phase_id,
            "pattern": pattern,
            "node_type": node_type,
            "owners": owners,
            "labels": labels,
            "category": category,
            "parent_filter": parent_filter,
            "update_metadata": update_metadata,
        }
        return _dispatch_task_action(action=action, payload=payload, config=config)

    logger.debug("Registered unified task tool")


__all__ = [
    "register_unified_task_tool",
]
