"""Batch action handlers: prepare-batch, start-batch, complete-batch, reset-batch, metadata-batch, fix-verification-types."""

from __future__ import annotations

import re
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.batch_operations import (
    prepare_batch_context,
    start_batch,
    complete_batch,
    reset_batch,
    DEFAULT_MAX_TASKS,
    DEFAULT_TOKEN_BUDGET,
    STALE_TASK_THRESHOLD_HOURS,
)
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
)
from foundry_mcp.core.spec import save_spec
from foundry_mcp.core.task import batch_update_tasks
from foundry_mcp.core.validation.constants import VALID_VERIFICATION_TYPES

from foundry_mcp.tools.unified.param_schema import Bool, Dict_, List_, Num, Str, validate_payload
from foundry_mcp.tools.unified.task_handlers._helpers import (
    _ALLOWED_STATUS,
    _TASK_WARNING_THRESHOLD,
    _VALID_NODE_TYPES,
    _attach_meta,
    _check_autonomy_write_lock,
    _load_spec_data,
    _match_nodes_for_batch,
    _metric,
    _metrics,
    _request_id,
    _resolve_specs_dir,
    _validation_error,
)

# ---------------------------------------------------------------------------
# Declarative validation schemas
# ---------------------------------------------------------------------------

_PREPARE_BATCH_SCHEMA = {
    "spec_id": Str(required=True),
    "max_tasks": Num(integer_only=True, min_val=1),
    "token_budget": Num(integer_only=True, min_val=1000),
}

_START_BATCH_SCHEMA = {
    "spec_id": Str(required=True),
    "task_ids": List_(required=True, min_items=1),
}

_COMPLETE_BATCH_SCHEMA = {
    "spec_id": Str(required=True),
    "completions": List_(required=True, min_items=1),
}

_RESET_BATCH_SCHEMA = {
    "spec_id": Str(required=True),
}

_METADATA_BATCH_SCHEMA = {
    "spec_id": Str(required=True),
    "status_filter": Str(choices=frozenset(_ALLOWED_STATUS)),
    "description": Str(),
    "file_path": Str(),
    "estimated_hours": Num(min_val=0),
    "category": Str(),
    "labels": Dict_(),
    "owners": List_(),
    "update_metadata": Dict_(),
    "dry_run": Bool(default=False),
}

_FIX_VERIFICATION_TYPES_SCHEMA = {
    "spec_id": Str(required=True),
    "dry_run": Bool(default=False),
}


def _handle_prepare_batch(*, config: ServerConfig, **payload: Any) -> dict:
    """
    Handle prepare-batch action for parallel task execution.

    Returns multiple independent tasks with context for parallel implementation.
    """
    request_id = _request_id()
    action = "prepare-batch"

    # Defaults before validation
    if payload.get("max_tasks") is None:
        payload["max_tasks"] = DEFAULT_MAX_TASKS
    if payload.get("token_budget") is None:
        payload["token_budget"] = DEFAULT_TOKEN_BUDGET

    err = validate_payload(payload, _PREPARE_BATCH_SCHEMA,
                           tool_name="task", action=action,
                           request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    max_tasks = payload["max_tasks"]
    token_budget = payload["token_budget"]

    workspace = payload.get("workspace")
    specs_dir, specs_err = _resolve_specs_dir(config, workspace)
    if specs_err:
        return specs_err

    start = time.perf_counter()
    result, error = prepare_batch_context(
        spec_id=spec_id,
        max_tasks=max_tasks,
        token_budget=token_budget,
        specs_dir=specs_dir,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if error:
        _metrics.counter(_metric(action), labels={"status": "error"})
        return asdict(
            error_response(
                error,
                error_code=ErrorCode.OPERATION_FAILED,
                error_type=ErrorType.VALIDATION,
                request_id=request_id,
            )
        )

    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})

    # Build response with batch context
    response = success_response(
        spec_id=spec_id,
        tasks=result.get("tasks", []),
        task_count=result.get("task_count", 0),
        spec_complete=result.get("spec_complete", False),
        all_blocked=result.get("all_blocked", False),
        stale_tasks=result.get("stale_tasks", []),
        dependency_graph=result.get("dependency_graph", {}),
        token_estimate=result.get("token_estimate", 0),
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )

    warnings = result.get("warnings", [])
    return _attach_meta(
        asdict(response),
        request_id=request_id,
        duration_ms=elapsed_ms,
        warnings=warnings if warnings else None,
    )


def _handle_start_batch(*, config: ServerConfig, **payload: Any) -> dict:
    """
    Handle start-batch action for atomically starting multiple tasks.

    Validates all tasks can be started before making any changes.
    """
    request_id = _request_id()
    action = "start-batch"

    err = validate_payload(payload, _START_BATCH_SCHEMA,
                           tool_name="task", action=action,
                           request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    task_ids = payload["task_ids"]
    bypass_autonomy_lock = payload.get("bypass_autonomy_lock", False)
    bypass_reason = payload.get("bypass_reason")

    # Per-element string validation (schema validates list, not elements)
    for i, tid in enumerate(task_ids):
        if not isinstance(tid, str) or not tid.strip():
            return _validation_error(
                field=f"task_ids[{i}]",
                action=action,
                message="Each task ID must be a non-empty string",
                request_id=request_id,
                code=ErrorCode.VALIDATION_ERROR,
            )

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

    specs_dir, specs_err = _resolve_specs_dir(config, workspace)
    if specs_err:
        return specs_err

    start = time.perf_counter()
    result, error = start_batch(
        spec_id=spec_id,
        task_ids=[tid.strip() for tid in task_ids],
        specs_dir=specs_dir,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if error:
        _metrics.counter(_metric(action), labels={"status": "error"})
        # Include partial results in error response
        return asdict(
            error_response(
                error,
                error_code=ErrorCode.OPERATION_FAILED,
                error_type=ErrorType.VALIDATION,
                request_id=request_id,
                details=result if result else None,
            )
        )

    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})

    response = success_response(
        spec_id=spec_id,
        started=result.get("started", []),
        started_count=result.get("started_count", 0),
        started_at=result.get("started_at"),
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    return _attach_meta(asdict(response), request_id=request_id, duration_ms=elapsed_ms)


def _handle_complete_batch(*, config: ServerConfig, **payload: Any) -> dict:
    """Handle complete-batch action for completing multiple tasks with partial failure support."""
    request_id = _request_id()
    action = "complete-batch"

    err = validate_payload(payload, _COMPLETE_BATCH_SCHEMA,
                           tool_name="task", action=action,
                           request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    completions = payload["completions"]
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

    specs_dir, specs_err = _resolve_specs_dir(config, workspace)
    if specs_err:
        return specs_err

    start = time.perf_counter()
    result, error = complete_batch(spec_id=spec_id, completions=completions, specs_dir=specs_dir)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if error:
        _metrics.counter(_metric(action), labels={"status": "error"})
        return asdict(error_response(error, error_code=ErrorCode.OPERATION_FAILED, error_type=ErrorType.VALIDATION, request_id=request_id, details=result if result else None))

    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})

    response = success_response(
        spec_id=spec_id,
        results=result.get("results", {}),
        completed_count=result.get("completed_count", 0),
        failed_count=result.get("failed_count", 0),
        total_processed=result.get("total_processed", 0),
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    return _attach_meta(asdict(response), request_id=request_id, duration_ms=elapsed_ms)


def _handle_reset_batch(*, config: ServerConfig, **payload: Any) -> dict:
    """
    Handle reset-batch action for resetting stale or specified in_progress tasks.

    Resets tasks back to pending status and clears started_at timestamp.
    If task_ids not provided, finds stale tasks automatically based on threshold.
    """
    request_id = _request_id()
    action = "reset-batch"

    err = validate_payload(payload, _RESET_BATCH_SCHEMA,
                           tool_name="task", action=action,
                           request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]

    # Optional: specific task IDs to reset
    task_ids = payload.get("task_ids")
    if task_ids is not None:
        if not isinstance(task_ids, list):
            return _validation_error(
                field="task_ids",
                action=action,
                message="task_ids must be a list of strings",
                request_id=request_id,
            )
        # Validate all task_ids are strings
        for i, tid in enumerate(task_ids):
            if not isinstance(tid, str) or not tid.strip():
                return _validation_error(
                    field=f"task_ids[{i}]",
                    action=action,
                    message="Each task ID must be a non-empty string",
                    request_id=request_id,
                    code=ErrorCode.VALIDATION_ERROR,
                )
        task_ids = [tid.strip() for tid in task_ids]

    # Optional: threshold in hours for stale detection
    threshold_hours = payload.get("threshold_hours", STALE_TASK_THRESHOLD_HOURS)
    if not isinstance(threshold_hours, (int, float)) or threshold_hours <= 0:
        return _validation_error(
            field="threshold_hours",
            action=action,
            message="threshold_hours must be a positive number",
            request_id=request_id,
        )

    workspace = payload.get("workspace")
    specs_dir, specs_err = _resolve_specs_dir(config, workspace)
    if specs_err:
        return specs_err

    start = time.perf_counter()
    result, error = reset_batch(
        spec_id=spec_id,
        task_ids=task_ids,
        threshold_hours=float(threshold_hours),
        specs_dir=specs_dir,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if error:
        _metrics.counter(_metric(action), labels={"status": "error"})
        return asdict(
            error_response(
                error,
                error_code=ErrorCode.OPERATION_FAILED,
                error_type=ErrorType.VALIDATION,
                request_id=request_id,
                details=result if result else None,
            )
        )

    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})

    response = success_response(
        spec_id=spec_id,
        reset=result.get("reset", []),
        reset_count=result.get("reset_count", 0),
        errors=result.get("errors"),
        message=result.get("message"),
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )
    return _attach_meta(asdict(response), request_id=request_id, duration_ms=elapsed_ms)


def _handle_metadata_batch(*, config: ServerConfig, **payload: Any) -> dict:
    """Batch update metadata across multiple tasks matching specified criteria.

    Filters (combined with AND logic):
    - status_filter: Filter by task status (pending, in_progress, completed, blocked)
    - parent_filter: Filter by parent node ID (e.g., phase-1, task-2-1)
    - pattern: Regex pattern to match task titles/IDs

    Metadata fields supported:
    - description, file_path, estimated_hours, category, labels, owners
    - update_metadata: Dict for custom metadata fields (verification_type, command, etc.)
    """
    request_id = _request_id()
    action = "metadata-batch"
    start = time.perf_counter()

    err = validate_payload(payload, _METADATA_BATCH_SCHEMA,
                           tool_name="task", action=action,
                           request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    status_filter = payload.get("status_filter")
    parent_filter = payload.get("parent_filter")
    pattern = payload.get("pattern")

    # Validate parent_filter (schema can't reject empty optional strings)
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

    # Validate pattern (non-empty + regex compilation)
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
            remediation="Specify status_filter, parent_filter, and/or pattern to target tasks",
        )

    # Extract metadata fields (type-checked by schema)
    description = payload.get("description")
    file_path = payload.get("file_path")
    estimated_hours = payload.get("estimated_hours")
    category = payload.get("category")
    labels = payload.get("labels")
    owners = payload.get("owners")
    update_metadata = payload.get("update_metadata")
    dry_run = payload["dry_run"]

    # Content validation for labels and owners (schema checks type only)
    if labels is not None and not all(
        isinstance(k, str) and isinstance(v, str) for k, v in labels.items()
    ):
        return _validation_error(
            field="labels",
            action=action,
            message="labels must be a dict with string keys and values",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    if owners is not None and not all(isinstance(o, str) for o in owners):
        return _validation_error(
            field="owners",
            action=action,
            message="owners must be a list of strings",
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
    specs_dir, specs_err = _resolve_specs_dir(config, workspace)
    if specs_err:
        return specs_err

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
    *, config: ServerConfig, **payload: Any
) -> dict:
    """Fix verification types across all verify nodes in a spec.

    This action:
    1. Finds all verify nodes with invalid or missing verification_type
    2. Sets missing types to 'run-tests' (default)
    3. Sets unknown types to 'manual' (fallback)

    Supports dry-run mode to preview changes without persisting.
    """
    request_id = _request_id()
    action = "fix-verification-types"

    err = validate_payload(payload, _FIX_VERIFICATION_TYPES_SCHEMA,
                           tool_name="task", action=action,
                           request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    dry_run_bool = payload["dry_run"]

    # Load spec
    workspace = payload.get("workspace")
    specs_dir, _specs_err = _resolve_specs_dir(config, workspace)
    spec_data, error = _load_spec_data(spec_id, specs_dir, request_id)
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
            # Invalid type -> fallback to 'manual'
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
        if specs_dir is None or not save_spec(spec_id, spec_data, specs_dir):
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
    invalid_count = sum(1 for f in fixes if f["issue"] == "invalid")

    response = success_response(
        spec_id=spec_id,
        total_fixes=len(fixes),
        applied_count=len(fixes) if not dry_run_bool else 0,
        fixes=fixes,
        summary={
            "missing_set_to_run_tests": missing_count,
            "invalid_set_to_manual": invalid_count,
        },
        valid_types=sorted(VALID_VERIFICATION_TYPES),
        dry_run=dry_run_bool,
        request_id=request_id,
        telemetry={"duration_ms": round(elapsed_ms, 2)},
    )

    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
    _metrics.counter(_metric(action), labels={"status": "success"})
    return asdict(response)
