"""Mutation action handlers: add, remove, move, add-dependency, remove-dependency, add-requirement, update-estimate, update-metadata."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
)
from foundry_mcp.core.task import (
    add_task,
    manage_task_dependency,
    move_task,
    remove_task,
    REQUIREMENT_TYPES,
    update_estimate,
    update_task_metadata,
    update_task_requirements,
)

from foundry_mcp.tools.unified.task_handlers._helpers import (
    _check_autonomy_write_lock,
    _load_spec_data,
    _metric,
    _metrics,
    _request_id,
    _resolve_specs_dir,
    _validation_error,
)


def _handle_add(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "add"
    spec_id = payload.get("spec_id")
    parent = payload.get("parent")
    bypass_autonomy_lock = payload.get("bypass_autonomy_lock", False)
    bypass_reason = payload.get("bypass_reason")

    title = payload.get("title")
    description = payload.get("description")
    task_type = payload.get("task_type", "task")
    estimated_hours = payload.get("estimated_hours")
    position = payload.get("position")
    file_path = payload.get("file_path")

    # Research-specific parameters
    research_type = payload.get("research_type")
    blocking_mode = payload.get("blocking_mode")
    query = payload.get("query")

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

    # Validate research-specific parameters when task_type is "research"
    if task_type == "research":
        from foundry_mcp.core.validation import VALID_RESEARCH_TYPES, RESEARCH_BLOCKING_MODES

        if research_type is not None and not isinstance(research_type, str):
            return _validation_error(
                field="research_type",
                action=action,
                message="research_type must be a string",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )
        if research_type and research_type not in VALID_RESEARCH_TYPES:
            return _validation_error(
                field="research_type",
                action=action,
                message=f"Must be one of: {', '.join(sorted(VALID_RESEARCH_TYPES))}",
                request_id=request_id,
            )
        if blocking_mode is not None and not isinstance(blocking_mode, str):
            return _validation_error(
                field="blocking_mode",
                action=action,
                message="blocking_mode must be a string",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )
        if blocking_mode and blocking_mode not in RESEARCH_BLOCKING_MODES:
            return _validation_error(
                field="blocking_mode",
                action=action,
                message=f"Must be one of: {', '.join(sorted(RESEARCH_BLOCKING_MODES))}",
                request_id=request_id,
            )
        if query is not None and not isinstance(query, str):
            return _validation_error(
                field="query",
                action=action,
                message="query must be a string",
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

    # Check autonomy write-lock before proceeding with protected mutation
    lock_error = _check_autonomy_write_lock(
        spec_id=spec_id.strip(),
        workspace=workspace,
        bypass_autonomy_lock=bool(bypass_autonomy_lock),
        bypass_reason=bypass_reason,
        request_id=request_id,
    )
    if lock_error:
        return lock_error

    specs_dir, specs_err = _resolve_specs_dir(config, workspace)
    if specs_err:
        return specs_err

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
        dry_run_data: Dict[str, Any] = {
            "spec_id": spec_id.strip(),
            "parent": parent.strip(),
            "title": title.strip(),
            "task_type": task_type,
            "position": position,
            "file_path": file_path.strip() if file_path else None,
            "dry_run": True,
        }
        # Include research parameters in dry_run response
        if task_type == "research":
            dry_run_data["research_type"] = research_type
            dry_run_data["blocking_mode"] = blocking_mode
            dry_run_data["query"] = query
        response = success_response(
            data=dry_run_data,
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
        # Research-specific parameters
        research_type=research_type,
        blocking_mode=blocking_mode,
        query=query,
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


def _handle_remove(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "remove"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")
    cascade = payload.get("cascade", False)
    bypass_autonomy_lock = payload.get("bypass_autonomy_lock", False)
    bypass_reason = payload.get("bypass_reason")

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

    # Check autonomy write-lock before proceeding with protected mutation
    lock_error = _check_autonomy_write_lock(
        spec_id=spec_id.strip(),
        workspace=workspace,
        bypass_autonomy_lock=bool(bypass_autonomy_lock),
        bypass_reason=bypass_reason,
        request_id=request_id,
    )
    if lock_error:
        return lock_error

    specs_dir, specs_err = _resolve_specs_dir(config, workspace)
    if specs_err:
        return specs_err

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


def _handle_update_estimate(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "update-estimate"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")
    estimated_hours = payload.get("estimated_hours")
    complexity = payload.get("complexity")
    bypass_autonomy_lock = payload.get("bypass_autonomy_lock", False)
    bypass_reason = payload.get("bypass_reason")

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

    # Check autonomy write-lock before proceeding with protected mutation
    lock_error = _check_autonomy_write_lock(
        spec_id=spec_id.strip(),
        workspace=workspace,
        bypass_autonomy_lock=bool(bypass_autonomy_lock),
        bypass_reason=bypass_reason,
        request_id=request_id,
    )
    if lock_error:
        return lock_error

    specs_dir, specs_err = _resolve_specs_dir(config, workspace)
    if specs_err:
        return specs_err

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


def _handle_update_metadata(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "update-metadata"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")
    bypass_autonomy_lock = payload.get("bypass_autonomy_lock", False)
    bypass_reason_param = payload.get("bypass_reason")

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

    # Check autonomy write-lock before proceeding with protected mutation
    lock_error = _check_autonomy_write_lock(
        spec_id=spec_id.strip(),
        workspace=workspace,
        bypass_autonomy_lock=bool(bypass_autonomy_lock),
        bypass_reason=bypass_reason_param,
        request_id=request_id,
    )
    if lock_error:
        return lock_error

    specs_dir, specs_err = _resolve_specs_dir(config, workspace)
    if specs_err:
        return specs_err

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


def _handle_move(*, config: ServerConfig, **payload: Any) -> dict:
    """Move a task to a new position or parent."""
    request_id = _request_id()
    action = "move"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")
    new_parent = payload.get("parent")
    position = payload.get("position")
    bypass_autonomy_lock = payload.get("bypass_autonomy_lock", False)
    bypass_reason = payload.get("bypass_reason")

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

    if position is not None:
        if not isinstance(position, int) or position < 1:
            return _validation_error(
                field="position",
                action=action,
                message="position must be a positive integer (1-based)",
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

    # Check autonomy write-lock before proceeding with protected mutation
    lock_error = _check_autonomy_write_lock(
        spec_id=spec_id.strip(),
        workspace=workspace,
        bypass_autonomy_lock=bool(bypass_autonomy_lock),
        bypass_reason=bypass_reason,
        request_id=request_id,
    )
    if lock_error:
        return lock_error

    specs_dir, specs_err = _resolve_specs_dir(config, workspace)
    if specs_err:
        return specs_err

    start = time.perf_counter()

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


def _handle_add_dependency(*, config: ServerConfig, **payload: Any) -> dict:
    """Add a dependency relationship between two tasks."""
    request_id = _request_id()
    action = "add-dependency"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")
    target_id = payload.get("target_id")
    dependency_type = payload.get("dependency_type", "blocks")
    bypass_autonomy_lock = payload.get("bypass_autonomy_lock", False)
    bypass_reason = payload.get("bypass_reason")

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

    valid_types = ("blocks", "blocked_by", "depends")
    if dependency_type not in valid_types:
        return _validation_error(
            field="dependency_type",
            action=action,
            message=f"Must be one of: {', '.join(valid_types)}",
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

    # Check autonomy write-lock before proceeding with protected mutation
    lock_error = _check_autonomy_write_lock(
        spec_id=spec_id.strip(),
        workspace=workspace,
        bypass_autonomy_lock=bool(bypass_autonomy_lock),
        bypass_reason=bypass_reason,
        request_id=request_id,
    )
    if lock_error:
        return lock_error

    specs_dir, specs_err = _resolve_specs_dir(config, workspace)
    if specs_err:
        return specs_err

    start = time.perf_counter()

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


def _handle_remove_dependency(*, config: ServerConfig, **payload: Any) -> dict:
    """Remove a dependency relationship between two tasks."""
    request_id = _request_id()
    action = "remove-dependency"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")
    target_id = payload.get("target_id")
    dependency_type = payload.get("dependency_type", "blocks")
    bypass_autonomy_lock = payload.get("bypass_autonomy_lock", False)
    bypass_reason = payload.get("bypass_reason")

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

    valid_types = ("blocks", "blocked_by", "depends")
    if dependency_type not in valid_types:
        return _validation_error(
            field="dependency_type",
            action=action,
            message=f"Must be one of: {', '.join(valid_types)}",
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

    # Check autonomy write-lock before proceeding with protected mutation
    lock_error = _check_autonomy_write_lock(
        spec_id=spec_id.strip(),
        workspace=workspace,
        bypass_autonomy_lock=bool(bypass_autonomy_lock),
        bypass_reason=bypass_reason,
        request_id=request_id,
    )
    if lock_error:
        return lock_error

    specs_dir, specs_err = _resolve_specs_dir(config, workspace)
    if specs_err:
        return specs_err

    start = time.perf_counter()

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
        error_lower = (error or "").lower()
        if "does not exist" in error_lower:
            code = ErrorCode.DEPENDENCY_NOT_FOUND
            err_type = ErrorType.NOT_FOUND
            remediation = "This dependency does not exist"
        elif "not found" in error_lower:
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


def _handle_add_requirement(*, config: ServerConfig, **payload: Any) -> dict:
    """Add a structured requirement to a task's metadata."""
    request_id = _request_id()
    action = "add-requirement"
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")
    requirement_type = payload.get("requirement_type")
    text = payload.get("text")
    bypass_autonomy_lock = payload.get("bypass_autonomy_lock", False)
    bypass_reason = payload.get("bypass_reason")

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

    requirement_type_lower = requirement_type.lower().strip()
    if requirement_type_lower not in REQUIREMENT_TYPES:
        return _validation_error(
            field="requirement_type",
            action=action,
            message=f"Must be one of: {', '.join(REQUIREMENT_TYPES)}",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    if not isinstance(text, str) or not text.strip():
        return _validation_error(
            field="text",
            action=action,
            message="Provide non-empty requirement text",
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

    workspace = payload.get("workspace")

    # Check autonomy write-lock before proceeding with protected mutation
    lock_error = _check_autonomy_write_lock(
        spec_id=spec_id.strip(),
        workspace=workspace,
        bypass_autonomy_lock=bool(bypass_autonomy_lock),
        bypass_reason=bypass_reason,
        request_id=request_id,
    )
    if lock_error:
        return lock_error

    specs_dir, specs_err = _resolve_specs_dir(config, workspace)
    if specs_err:
        return specs_err

    start = time.perf_counter()

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
