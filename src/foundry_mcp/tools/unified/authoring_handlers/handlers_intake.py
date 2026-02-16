"""Intake queue action handlers: intake-add, intake-list, intake-dismiss."""

from __future__ import annotations

import re
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.errors.storage import LockAcquisitionError
from foundry_mcp.core.intake import IntakeStore, INTAKE_ID_PATTERN
from foundry_mcp.core.observability import audit_log
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    sanitize_error_message,
    success_response,
)

from foundry_mcp.tools.unified.authoring_handlers._helpers import (
    _metric_name,
    _metrics,
    _request_id,
    _resolve_specs_dir,
    _validation_error,
    logger,
)
from foundry_mcp.tools.unified.param_schema import Bool, Num, Str, validate_payload

# Validation constants for intake
_INTAKE_TITLE_MAX_LEN = 140
_INTAKE_DESC_MAX_LEN = 2000
_INTAKE_TAG_MAX_LEN = 32
_INTAKE_TAG_MAX_COUNT = 20
_INTAKE_SOURCE_MAX_LEN = 100
_INTAKE_REQUESTER_MAX_LEN = 100
_INTAKE_IDEMPOTENCY_KEY_MAX_LEN = 64
_INTAKE_PRIORITY_VALUES = ("p0", "p1", "p2", "p3", "p4")
_INTAKE_PRIORITY_ALIASES = {
    "critical": "p0",
    "highest": "p0",
    "high": "p1",
    "medium": "p2",
    "normal": "p2",
    "low": "p3",
    "lowest": "p4",
}
_INTAKE_TAG_PATTERN = "^[a-z0-9_-]+$"
_TAG_REGEX = re.compile(_INTAKE_TAG_PATTERN)

# Intake list constants (from intake.py)
_INTAKE_LIST_DEFAULT_LIMIT = 50
_INTAKE_LIST_MAX_LIMIT = 200

# Intake dismiss constants
_INTAKE_DISMISS_REASON_MAX_LEN = 200

_INTAKE_ADD_SCHEMA = {
    "title": Str(required=True, max_length=_INTAKE_TITLE_MAX_LEN,
                 error_code=ErrorCode.MISSING_REQUIRED),
    # description, source, requester, idempotency_key: imperative (strip-or-None + dual error codes)
    # priority: imperative (alias normalization)
    # tags: imperative (per-element regex validation)
    "dry_run": Bool(default=False, error_code=ErrorCode.INVALID_FORMAT),
    "path": Str(error_code=ErrorCode.INVALID_FORMAT),
}

_INTAKE_LIST_SCHEMA = {
    # limit: imperative (default from constant + bool-passthrough behavior)
    "cursor": Str(error_code=ErrorCode.INVALID_FORMAT),
    "path": Str(error_code=ErrorCode.INVALID_FORMAT),
}

_INTAKE_DISMISS_SCHEMA = {
    # intake_id: imperative (regex pattern + MISSING_REQUIRED vs INVALID_FORMAT)
    "reason": Str(max_length=_INTAKE_DISMISS_REASON_MAX_LEN,
                  error_code=ErrorCode.INVALID_FORMAT),
    "dry_run": Bool(default=False, error_code=ErrorCode.INVALID_FORMAT),
    "path": Str(error_code=ErrorCode.INVALID_FORMAT),
}


def _handle_intake_add(*, config: ServerConfig, **payload: Any) -> dict:
    """Add a new intake item to the notes queue."""
    request_id = _request_id()
    action = "intake-add"

    err = validate_payload(payload, _INTAKE_ADD_SCHEMA,
                           tool_name="authoring", action=action,
                           request_id=request_id)
    if err:
        return err

    title = payload["title"]
    dry_run = payload["dry_run"]
    path = payload.get("path")

    # Validate description (optional, max 2000 chars)
    description = payload.get("description")
    if description is not None:
        if not isinstance(description, str):
            return _validation_error(
                field="description",
                action=action,
                message="Description must be a string",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )
        description = description.strip() or None
        if description and len(description) > _INTAKE_DESC_MAX_LEN:
            return _validation_error(
                field="description",
                action=action,
                message=f"Description exceeds maximum length of {_INTAKE_DESC_MAX_LEN} characters",
                request_id=request_id,
                code=ErrorCode.VALIDATION_ERROR,
                remediation=f"Shorten description to {_INTAKE_DESC_MAX_LEN} characters or less",
            )

    # Validate priority (optional, enum p0-p4, default p2)
    # Handle both missing key AND explicit null from JSON
    priority = payload.get("priority")
    if priority is None:
        priority = "p2"  # Default for both missing and explicit null
    elif not isinstance(priority, str):
        return _validation_error(
            field="priority",
            action=action,
            message=f"Priority must be a string. Valid values: {', '.join(_INTAKE_PRIORITY_VALUES)}",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
            remediation=f"Use {', '.join(_INTAKE_PRIORITY_VALUES)} or aliases: {', '.join(_INTAKE_PRIORITY_ALIASES.keys())}",
        )

    priority = priority.strip().lower()

    # Map human-readable aliases to canonical values
    if priority in _INTAKE_PRIORITY_ALIASES:
        priority = _INTAKE_PRIORITY_ALIASES[priority]

    if priority not in _INTAKE_PRIORITY_VALUES:
        return _validation_error(
            field="priority",
            action=action,
            message=f"Priority must be one of: {', '.join(_INTAKE_PRIORITY_VALUES)}",
            request_id=request_id,
            code=ErrorCode.VALIDATION_ERROR,
            remediation=f"Use p0-p4 or aliases like 'high', 'medium', 'low'. Default is p2 (medium).",
        )

    # Validate tags (optional, max 20 items, each 1-32 chars, lowercase pattern)
    tags = payload.get("tags", [])
    if tags is None:
        tags = []
    if not isinstance(tags, list):
        return _validation_error(
            field="tags",
            action=action,
            message="Tags must be a list of strings",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )
    if len(tags) > _INTAKE_TAG_MAX_COUNT:
        return _validation_error(
            field="tags",
            action=action,
            message=f"Maximum {_INTAKE_TAG_MAX_COUNT} tags allowed",
            request_id=request_id,
            code=ErrorCode.VALIDATION_ERROR,
        )
    validated_tags = []
    for i, tag in enumerate(tags):
        if not isinstance(tag, str):
            return _validation_error(
                field=f"tags[{i}]",
                action=action,
                message="Each tag must be a string",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )
        tag = tag.strip().lower()
        if not tag:
            continue
        if len(tag) > _INTAKE_TAG_MAX_LEN:
            return _validation_error(
                field=f"tags[{i}]",
                action=action,
                message=f"Tag exceeds maximum length of {_INTAKE_TAG_MAX_LEN} characters",
                request_id=request_id,
                code=ErrorCode.VALIDATION_ERROR,
            )
        if not _TAG_REGEX.match(tag):
            return _validation_error(
                field=f"tags[{i}]",
                action=action,
                message=f"Tag must match pattern {_INTAKE_TAG_PATTERN} (lowercase alphanumeric, hyphens, underscores)",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )
        validated_tags.append(tag)
    tags = validated_tags

    # Validate source (optional, max 100 chars)
    source = payload.get("source")
    if source is not None:
        if not isinstance(source, str):
            return _validation_error(
                field="source",
                action=action,
                message="Source must be a string",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )
        source = source.strip() or None
        if source and len(source) > _INTAKE_SOURCE_MAX_LEN:
            return _validation_error(
                field="source",
                action=action,
                message=f"Source exceeds maximum length of {_INTAKE_SOURCE_MAX_LEN} characters",
                request_id=request_id,
                code=ErrorCode.VALIDATION_ERROR,
            )

    # Validate requester (optional, max 100 chars)
    requester = payload.get("requester")
    if requester is not None:
        if not isinstance(requester, str):
            return _validation_error(
                field="requester",
                action=action,
                message="Requester must be a string",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )
        requester = requester.strip() or None
        if requester and len(requester) > _INTAKE_REQUESTER_MAX_LEN:
            return _validation_error(
                field="requester",
                action=action,
                message=f"Requester exceeds maximum length of {_INTAKE_REQUESTER_MAX_LEN} characters",
                request_id=request_id,
                code=ErrorCode.VALIDATION_ERROR,
            )

    # Validate idempotency_key (optional, max 64 chars)
    idempotency_key = payload.get("idempotency_key")
    if idempotency_key is not None:
        if not isinstance(idempotency_key, str):
            return _validation_error(
                field="idempotency_key",
                action=action,
                message="Idempotency key must be a string",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )
        idempotency_key = idempotency_key.strip() or None
        if idempotency_key and len(idempotency_key) > _INTAKE_IDEMPOTENCY_KEY_MAX_LEN:
            return _validation_error(
                field="idempotency_key",
                action=action,
                message=f"Idempotency key exceeds maximum length of {_INTAKE_IDEMPOTENCY_KEY_MAX_LEN} characters",
                request_id=request_id,
                code=ErrorCode.VALIDATION_ERROR,
            )

    # Resolve specs directory
    specs_dir, specs_err = _resolve_specs_dir(config, path)
    if specs_err:
        return specs_err

    # Audit log
    audit_log(
        "tool_invocation",
        tool="authoring",
        action=action,
        title=title[:100],  # Truncate for logging
        dry_run=dry_run,
    )

    metric_key = _metric_name(action)
    start_time = time.perf_counter()

    try:
        # Get notes_dir from config (allows customization via TOML or env var)
        notes_dir = config.get_notes_dir(specs_dir)
        store = IntakeStore(specs_dir, notes_dir=notes_dir)
        item, was_duplicate, lock_wait_ms = store.add(
            title=title,
            description=description,
            priority=priority,
            tags=tags,
            source=source,
            requester=requester,
            idempotency_key=idempotency_key,
            dry_run=dry_run,
        )
    except LockAcquisitionError:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _metrics.counter(metric_key, labels={"status": "error"})
        return asdict(
            error_response(
                "Failed to acquire file lock within timeout. Resource is busy.",
                error_code=ErrorCode.RESOURCE_BUSY,
                error_type=ErrorType.UNAVAILABLE,
                remediation="Retry after a moment",
                request_id=request_id,
                telemetry={"duration_ms": round(elapsed_ms, 2)},
            )
        )
    except Exception as exc:
        logger.exception("Unexpected error adding intake item")
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _metrics.counter(metric_key, labels={"status": "error"})
        return asdict(
            error_response(
                sanitize_error_message(exc, context="authoring.intake-add"),
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check logs for details",
                request_id=request_id,
                telemetry={"duration_ms": round(elapsed_ms, 2)},
            )
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    _metrics.timer(metric_key + ".duration_ms", elapsed_ms)
    _metrics.counter(metric_key, labels={"status": "success", "dry_run": str(dry_run).lower()})

    data = {
        "item": item.to_dict(),
        "intake_path": store.intake_path,
        "was_duplicate": was_duplicate,
    }

    meta_extra = {}
    if dry_run:
        meta_extra["dry_run"] = True

    return asdict(
        success_response(
            data=data,
            telemetry={"duration_ms": round(elapsed_ms, 2), "lock_wait_ms": round(lock_wait_ms, 2)},
            request_id=request_id,
            meta=meta_extra,
        )
    )


def _handle_intake_list(*, config: ServerConfig, **payload: Any) -> dict:
    """List intake items with status='new' in FIFO order with pagination."""
    request_id = _request_id()
    action = "intake-list"

    err = validate_payload(payload, _INTAKE_LIST_SCHEMA,
                           tool_name="authoring", action=action,
                           request_id=request_id)
    if err:
        return err

    path = payload.get("path")
    # Strip-or-None for cursor (schema strips but doesn't convert empty to None)
    cursor = payload.get("cursor")
    if isinstance(cursor, str):
        cursor = cursor.strip() or None

    # Validate limit (optional, default 50, range 1-200) — imperative due to
    # pre-applied default from constant + dual error codes
    limit = payload.get("limit", _INTAKE_LIST_DEFAULT_LIMIT)
    if limit is not None:
        if not isinstance(limit, int):
            return _validation_error(
                field="limit",
                action=action,
                message="limit must be an integer",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )
        if limit < 1 or limit > _INTAKE_LIST_MAX_LIMIT:
            return _validation_error(
                field="limit",
                action=action,
                message=f"limit must be between 1 and {_INTAKE_LIST_MAX_LIMIT}",
                request_id=request_id,
                code=ErrorCode.VALIDATION_ERROR,
                remediation=f"Use a value between 1 and {_INTAKE_LIST_MAX_LIMIT} (default: {_INTAKE_LIST_DEFAULT_LIMIT})",
            )

    # Resolve specs directory
    specs_dir, specs_err = _resolve_specs_dir(config, path)
    if specs_err:
        return specs_err

    # Audit log
    audit_log(
        "tool_invocation",
        tool="authoring",
        action=action,
        limit=limit,
        has_cursor=cursor is not None,
    )

    metric_key = _metric_name(action)
    start_time = time.perf_counter()

    try:
        # Get notes_dir from config (allows customization via TOML or env var)
        notes_dir = config.get_notes_dir(specs_dir)
        store = IntakeStore(specs_dir, notes_dir=notes_dir)
        items, total_count, next_cursor, has_more, lock_wait_ms = store.list_new(
            cursor=cursor,
            limit=limit,
        )
    except LockAcquisitionError:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _metrics.counter(metric_key, labels={"status": "error"})
        return asdict(
            error_response(
                "Failed to acquire file lock within timeout. Resource is busy.",
                error_code=ErrorCode.RESOURCE_BUSY,
                error_type=ErrorType.UNAVAILABLE,
                remediation="Retry after a moment",
                request_id=request_id,
                telemetry={"duration_ms": round(elapsed_ms, 2)},
            )
        )
    except Exception as exc:
        logger.exception("Unexpected error listing intake items")
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _metrics.counter(metric_key, labels={"status": "error"})
        return asdict(
            error_response(
                sanitize_error_message(exc, context="authoring.intake-list"),
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check logs for details",
                request_id=request_id,
                telemetry={"duration_ms": round(elapsed_ms, 2)},
            )
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    _metrics.timer(metric_key + ".duration_ms", elapsed_ms)
    _metrics.counter(metric_key, labels={"status": "success"})

    data = {
        "items": [item.to_dict() for item in items],
        "total_count": total_count,
        "intake_path": store.intake_path,
    }

    # Build pagination metadata
    pagination = None
    if has_more or cursor is not None:
        pagination = {
            "cursor": next_cursor,
            "has_more": has_more,
            "page_size": limit,
        }

    return asdict(
        success_response(
            data=data,
            pagination=pagination,
            telemetry={
                "duration_ms": round(elapsed_ms, 2),
                "lock_wait_ms": round(lock_wait_ms, 2),
            },
            request_id=request_id,
        )
    )


def _handle_intake_dismiss(*, config: ServerConfig, **payload: Any) -> dict:
    """Dismiss an intake item by changing its status to 'dismissed'."""
    request_id = _request_id()
    action = "intake-dismiss"

    err = validate_payload(payload, _INTAKE_DISMISS_SCHEMA,
                           tool_name="authoring", action=action,
                           request_id=request_id)
    if err:
        return err

    dry_run = payload["dry_run"]
    path = payload.get("path")

    # Strip-or-None for reason (schema strips but doesn't convert empty to None)
    reason = payload.get("reason")
    if isinstance(reason, str):
        reason = reason.strip() or None

    # intake_id validated imperatively — regex pattern + MISSING_REQUIRED vs INVALID_FORMAT
    intake_id = payload.get("intake_id")
    if not isinstance(intake_id, str) or not intake_id.strip():
        return _validation_error(
            field="intake_id",
            action=action,
            message="Provide a valid intake_id",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    intake_id = intake_id.strip()
    if not INTAKE_ID_PATTERN.match(intake_id):
        return _validation_error(
            field="intake_id",
            action=action,
            message="intake_id must match pattern intake-<uuid>",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
            remediation="Use format: intake-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        )

    # Resolve specs directory
    specs_dir, specs_err = _resolve_specs_dir(config, path)
    if specs_err:
        return specs_err

    # Audit log
    audit_log(
        "tool_invocation",
        tool="authoring",
        action=action,
        intake_id=intake_id,
        dry_run=dry_run,
    )

    metric_key = _metric_name(action)
    start_time = time.perf_counter()

    try:
        # Get notes_dir from config (allows customization via TOML or env var)
        notes_dir = config.get_notes_dir(specs_dir)
        store = IntakeStore(specs_dir, notes_dir=notes_dir)
        item, lock_wait_ms = store.dismiss(
            intake_id=intake_id,
            reason=reason,
            dry_run=dry_run,
        )
    except LockAcquisitionError:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _metrics.counter(metric_key, labels={"status": "error"})
        return asdict(
            error_response(
                "Failed to acquire file lock within timeout. Resource is busy.",
                error_code=ErrorCode.RESOURCE_BUSY,
                error_type=ErrorType.UNAVAILABLE,
                remediation="Retry after a moment",
                request_id=request_id,
                telemetry={"duration_ms": round(elapsed_ms, 2)},
            )
        )
    except Exception as exc:
        logger.exception("Unexpected error dismissing intake item")
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _metrics.counter(metric_key, labels={"status": "error"})
        return asdict(
            error_response(
                sanitize_error_message(exc, context="authoring.intake-dismiss"),
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check logs for details",
                request_id=request_id,
                telemetry={"duration_ms": round(elapsed_ms, 2)},
            )
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Handle not found case
    if item is None:
        _metrics.counter(metric_key, labels={"status": "not_found"})
        return asdict(
            error_response(
                f"Intake item not found: {intake_id}",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Verify the intake_id exists using intake-list action",
                request_id=request_id,
                telemetry={"duration_ms": round(elapsed_ms, 2), "lock_wait_ms": round(lock_wait_ms, 2)},
            )
        )

    _metrics.timer(metric_key + ".duration_ms", elapsed_ms)
    _metrics.counter(metric_key, labels={"status": "success", "dry_run": str(dry_run).lower()})

    data = {
        "item": item.to_dict(),
        "intake_path": store.intake_path,
    }

    meta_extra = {}
    if dry_run:
        meta_extra["dry_run"] = True

    return asdict(
        success_response(
            data=data,
            telemetry={
                "duration_ms": round(elapsed_ms, 2),
                "lock_wait_ms": round(lock_wait_ms, 2),
            },
            request_id=request_id,
            meta=meta_extra,
        )
    )
