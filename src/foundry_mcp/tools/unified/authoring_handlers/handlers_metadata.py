"""Metadata action handlers: assumption-add, assumption-list, revision-add."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.observability import audit_log
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    sanitize_error_message,
    success_response,
)
from foundry_mcp.core.spec import (
    add_assumption,
    add_revision,
    list_assumptions,
)

from foundry_mcp.tools.unified.authoring_handlers._helpers import (
    _assumption_exists,
    _metric_name,
    _metrics,
    _request_id,
    _resolve_specs_dir,
    _validation_error,
    logger,
)
from foundry_mcp.tools.unified.param_schema import Bool, Str, validate_payload

_ASSUMPTION_ADD_SCHEMA = {
    "spec_id": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
    "text": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
    "assumption_type": Str(),
    "author": Str(),
    "dry_run": Bool(default=False),
    "path": Str(),
}

_ASSUMPTION_LIST_SCHEMA = {
    "spec_id": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
    "assumption_type": Str(),
    "path": Str(),
}

_REVISION_ADD_SCHEMA = {
    "spec_id": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
    "version": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
    "changes": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
    "author": Str(),
    "dry_run": Bool(default=False),
    "path": Str(),
}


def _handle_assumption_add(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "assumption-add"

    err = validate_payload(payload, _ASSUMPTION_ADD_SCHEMA,
                           tool_name="authoring", action=action,
                           request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    text = payload["text"]
    assumption_type = payload.get("assumption_type")
    author = payload.get("author")
    dry_run = payload["dry_run"]
    path = payload.get("path")

    specs_dir, specs_err = _resolve_specs_dir(config, path)
    if specs_err:
        return specs_err

    warnings: List[str] = []
    if _assumption_exists(spec_id, specs_dir, text):
        warnings.append(
            "An assumption with identical text already exists; another entry will be appended"
        )

    audit_log(
        "tool_invocation",
        tool="authoring",
        action=action,
        spec_id=spec_id,
        assumption_type=assumption_type,
        dry_run=dry_run,
    )

    metric_key = _metric_name(action)

    if dry_run:
        _metrics.counter(metric_key, labels={"status": "success", "dry_run": "true"})
        data = {
            "spec_id": spec_id,
            "assumption_id": "(preview)",
            "text": text,
            "type": assumption_type,
            "dry_run": True,
            "note": "Dry run - no changes made",
        }
        if author:
            data["author"] = author
        return asdict(
            success_response(
                data=data,
                warnings=warnings or None,
                request_id=request_id,
            )
        )

    start_time = time.perf_counter()
    try:
        result, error = add_assumption(
            spec_id=spec_id,
            text=text,
            assumption_type=assumption_type,
            author=author,
            specs_dir=specs_dir,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected error adding assumption")
        _metrics.counter(metric_key, labels={"status": "error"})
        return asdict(
            error_response(
                sanitize_error_message(exc, context="authoring"),
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check logs for details",
                request_id=request_id,
            )
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    _metrics.timer(metric_key + ".duration_ms", elapsed_ms)

    if error:
        _metrics.counter(metric_key, labels={"status": "error"})
        if "not found" in error.lower():
            return asdict(
                error_response(
                    f"Specification '{spec_id}' not found",
                    error_code=ErrorCode.SPEC_NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation='Verify the spec ID via spec(action="list")',
                    request_id=request_id,
                )
            )
        return asdict(
            error_response(
                f"Failed to add assumption: {error}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check that the spec exists",
                request_id=request_id,
            )
        )

    data = {
        "spec_id": spec_id,
        "assumption_id": result.get("assumption_id") if result else None,
        "text": text,
        "type": assumption_type,
        "dry_run": False,
    }
    if author:
        data["author"] = author

    _metrics.counter(metric_key, labels={"status": "success"})
    return asdict(
        success_response(
            data=data,
            warnings=warnings or None,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )


def _handle_assumption_list(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "assumption-list"

    err = validate_payload(payload, _ASSUMPTION_LIST_SCHEMA,
                           tool_name="authoring", action=action,
                           request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    assumption_type = payload.get("assumption_type")
    path = payload.get("path")

    specs_dir, specs_err = _resolve_specs_dir(config, path)
    if specs_err:
        return specs_err

    audit_log(
        "tool_invocation",
        tool="authoring",
        action=action,
        spec_id=spec_id,
        assumption_type=assumption_type,
    )

    metric_key = _metric_name(action)
    start_time = time.perf_counter()
    try:
        result, error = list_assumptions(
            spec_id=spec_id,
            assumption_type=assumption_type,
            specs_dir=specs_dir,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected error listing assumptions")
        _metrics.counter(metric_key, labels={"status": "error"})
        return asdict(
            error_response(
                sanitize_error_message(exc, context="authoring"),
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check logs for details",
                request_id=request_id,
            )
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    _metrics.timer(metric_key + ".duration_ms", elapsed_ms)

    if error:
        _metrics.counter(metric_key, labels={"status": "error"})
        if "not found" in error.lower():
            return asdict(
                error_response(
                    f"Specification '{spec_id}' not found",
                    error_code=ErrorCode.SPEC_NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation='Verify the spec ID via spec(action="list")',
                    request_id=request_id,
                )
            )
        return asdict(
            error_response(
                f"Failed to list assumptions: {error}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check that the spec exists",
                request_id=request_id,
            )
        )

    warnings: List[str] = []
    if assumption_type:
        warnings.append(
            "assumption_type filter is advisory only; all assumptions are returned"
        )

    _metrics.counter(metric_key, labels={"status": "success"})
    return asdict(
        success_response(
            data=result or {"spec_id": spec_id, "assumptions": [], "total_count": 0},
            warnings=warnings or None,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )


def _handle_revision_add(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "revision-add"

    err = validate_payload(payload, _REVISION_ADD_SCHEMA,
                           tool_name="authoring", action=action,
                           request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    version = payload["version"]
    changes = payload["changes"]
    author = payload.get("author")
    dry_run = payload["dry_run"]
    path = payload.get("path")

    specs_dir, specs_err = _resolve_specs_dir(config, path)
    if specs_err:
        return specs_err

    audit_log(
        "tool_invocation",
        tool="authoring",
        action=action,
        spec_id=spec_id,
        version=version,
        dry_run=dry_run,
    )

    metric_key = _metric_name(action)
    if dry_run:
        _metrics.counter(metric_key, labels={"status": "success", "dry_run": "true"})
        data = {
            "spec_id": spec_id,
            "version": version,
            "changes": changes,
            "dry_run": True,
            "note": "Dry run - no changes made",
        }
        if author:
            data["author"] = author
        return asdict(
            success_response(
                data=data,
                request_id=request_id,
            )
        )

    start_time = time.perf_counter()
    try:
        result, error = add_revision(
            spec_id=spec_id,
            version=version,
            changelog=changes,
            author=author,
            specs_dir=specs_dir,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected error adding revision")
        _metrics.counter(metric_key, labels={"status": "error"})
        return asdict(
            error_response(
                sanitize_error_message(exc, context="authoring"),
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check logs for details",
                request_id=request_id,
            )
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    _metrics.timer(metric_key + ".duration_ms", elapsed_ms)

    if error:
        _metrics.counter(metric_key, labels={"status": "error"})
        if "not found" in error.lower():
            return asdict(
                error_response(
                    f"Specification '{spec_id}' not found",
                    error_code=ErrorCode.SPEC_NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation='Verify the spec ID via spec(action="list")',
                    request_id=request_id,
                )
            )
        return asdict(
            error_response(
                f"Failed to add revision: {error}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check that the spec exists",
                request_id=request_id,
            )
        )

    data = {
        "spec_id": spec_id,
        "version": version,
        "changes": changes,
        "dry_run": False,
    }
    if author:
        data["author"] = author
    if result and result.get("date"):
        data["date"] = result["date"]

    _metrics.counter(metric_key, labels={"status": "success"})
    return asdict(
        success_response(
            data=data,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )
