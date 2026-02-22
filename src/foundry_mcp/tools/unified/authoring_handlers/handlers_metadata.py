"""Metadata action handlers: assumption, constraint, risk, question, success-criterion, revision."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from foundry_mcp.config.server import ServerConfig
from foundry_mcp.core.observability import audit_log
from foundry_mcp.core.responses.builders import (
    error_response,
    success_response,
)
from foundry_mcp.core.responses.sanitization import sanitize_error_message
from foundry_mcp.core.responses.types import (
    ErrorCode,
    ErrorType,
)
from foundry_mcp.core.spec import (
    add_assumption,
    add_constraint,
    add_question,
    add_revision,
    add_risk,
    add_success_criterion,
    list_assumptions,
    list_constraints,
    list_questions,
    list_risks,
    list_success_criteria,
)
from foundry_mcp.tools.unified.authoring_handlers._helpers import (
    _assumption_exists,
    _metric_name,
    _metrics,
    _request_id,
    _resolve_specs_dir,
    logger,
)
from foundry_mcp.tools.unified.param_schema import Bool, Str, validate_payload

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

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

_TEXT_ADD_SCHEMA = {
    "spec_id": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
    "text": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
    "dry_run": Bool(default=False),
    "path": Str(),
}

_TEXT_LIST_SCHEMA = {
    "spec_id": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
    "path": Str(),
}

_RISK_ADD_SCHEMA = {
    "spec_id": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
    "description": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
    "text": Str(),  # alias for description
    "dry_run": Bool(default=False),
    "path": Str(),
    # likelihood, impact, mitigation are optional strings
}


# ---------------------------------------------------------------------------
# Shared add/list helpers for simple string-array metadata
# ---------------------------------------------------------------------------


def _handle_text_add(
    action: str,
    core_fn: Callable[..., Tuple[Optional[Dict[str, Any]], Optional[str]]],
    entity_name: str,
    *,
    config: ServerConfig,
    **payload: Any,
) -> dict:
    """Generic handler for adding a text entry to a metadata array."""
    request_id = _request_id()

    err = validate_payload(payload, _TEXT_ADD_SCHEMA, tool_name="authoring", action=action, request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    text = payload["text"]
    dry_run = payload["dry_run"]
    path = payload.get("path")

    specs_dir, specs_err = _resolve_specs_dir(config, path)
    if specs_err:
        return specs_err

    audit_log("tool_invocation", tool="authoring", action=action, spec_id=spec_id, dry_run=dry_run)

    metric_key = _metric_name(action)

    if dry_run:
        _metrics.counter(metric_key, labels={"status": "success", "dry_run": "true"})
        return asdict(
            success_response(
                data={"spec_id": spec_id, "text": text, "dry_run": True, "note": "Dry run - no changes made"},
                request_id=request_id,
            )
        )

    start_time = time.perf_counter()
    try:
        result, error = core_fn(spec_id=spec_id, text=text, specs_dir=specs_dir)
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error in %s", action)
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
                f"Failed to add {entity_name}: {error}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check that the spec exists",
                request_id=request_id,
            )
        )

    _metrics.counter(metric_key, labels={"status": "success"})
    return asdict(
        success_response(
            data={"spec_id": spec_id, "text": text, "dry_run": False, **(result or {})},
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )


def _handle_text_list(
    action: str,
    core_fn: Callable[..., Tuple[Optional[Dict[str, Any]], Optional[str]]],
    entity_name: str,
    *,
    config: ServerConfig,
    **payload: Any,
) -> dict:
    """Generic handler for listing text entries from a metadata array."""
    request_id = _request_id()

    err = validate_payload(payload, _TEXT_LIST_SCHEMA, tool_name="authoring", action=action, request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    path = payload.get("path")

    specs_dir, specs_err = _resolve_specs_dir(config, path)
    if specs_err:
        return specs_err

    audit_log("tool_invocation", tool="authoring", action=action, spec_id=spec_id)

    metric_key = _metric_name(action)
    start_time = time.perf_counter()
    try:
        result, error = core_fn(spec_id=spec_id, specs_dir=specs_dir)
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error in %s", action)
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
                f"Failed to list {entity_name}: {error}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check that the spec exists",
                request_id=request_id,
            )
        )

    _metrics.counter(metric_key, labels={"status": "success"})
    return asdict(
        success_response(
            data=result or {"spec_id": spec_id, entity_name: [], "total_count": 0},
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )


# ---------------------------------------------------------------------------
# Assumption handlers (existing — kept with original signature)
# ---------------------------------------------------------------------------


def _handle_assumption_add(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "assumption-add"

    err = validate_payload(payload, _ASSUMPTION_ADD_SCHEMA, tool_name="authoring", action=action, request_id=request_id)
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
    assert specs_dir is not None

    warnings: List[str] = []
    if _assumption_exists(spec_id, specs_dir, text):
        warnings.append("An assumption with identical text already exists; another entry will be appended")

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

    err = validate_payload(
        payload, _ASSUMPTION_LIST_SCHEMA, tool_name="authoring", action=action, request_id=request_id
    )
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
        warnings.append("assumption_type filter is advisory only; all assumptions are returned")

    _metrics.counter(metric_key, labels={"status": "success"})
    return asdict(
        success_response(
            data=result or {"spec_id": spec_id, "assumptions": [], "total_count": 0},
            warnings=warnings or None,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )


# ---------------------------------------------------------------------------
# Revision handler (existing)
# ---------------------------------------------------------------------------


def _handle_revision_add(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "revision-add"

    err = validate_payload(payload, _REVISION_ADD_SCHEMA, tool_name="authoring", action=action, request_id=request_id)
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


# ---------------------------------------------------------------------------
# Constraint handlers
# ---------------------------------------------------------------------------


def _handle_constraint_add(*, config: ServerConfig, **payload: Any) -> dict:
    return _handle_text_add("constraint-add", add_constraint, "constraint", config=config, **payload)


def _handle_constraint_list(*, config: ServerConfig, **payload: Any) -> dict:
    return _handle_text_list("constraint-list", list_constraints, "constraints", config=config, **payload)


# ---------------------------------------------------------------------------
# Question handlers
# ---------------------------------------------------------------------------


def _handle_question_add(*, config: ServerConfig, **payload: Any) -> dict:
    return _handle_text_add("question-add", add_question, "question", config=config, **payload)


def _handle_question_list(*, config: ServerConfig, **payload: Any) -> dict:
    return _handle_text_list("question-list", list_questions, "questions", config=config, **payload)


# ---------------------------------------------------------------------------
# Success criterion handlers
# ---------------------------------------------------------------------------


def _handle_success_criterion_add(*, config: ServerConfig, **payload: Any) -> dict:
    return _handle_text_add(
        "success-criterion-add", add_success_criterion, "success criterion", config=config, **payload
    )


def _handle_success_criteria_list(*, config: ServerConfig, **payload: Any) -> dict:
    return _handle_text_list(
        "success-criteria-list", list_success_criteria, "success_criteria", config=config, **payload
    )


# ---------------------------------------------------------------------------
# Risk handler (structured — not a simple text array)
# ---------------------------------------------------------------------------


def _handle_risk_add(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "risk-add"

    err = validate_payload(payload, _RISK_ADD_SCHEMA, tool_name="authoring", action=action, request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    # Accept 'description' or 'text' as the description field
    description = payload.get("description") or payload.get("text", "")
    dry_run = payload.get("dry_run", False)
    path = payload.get("path")

    specs_dir, specs_err = _resolve_specs_dir(config, path)
    if specs_err:
        return specs_err

    audit_log("tool_invocation", tool="authoring", action=action, spec_id=spec_id, dry_run=dry_run)

    metric_key = _metric_name(action)

    if dry_run:
        _metrics.counter(metric_key, labels={"status": "success", "dry_run": "true"})
        return asdict(
            success_response(
                data={
                    "spec_id": spec_id,
                    "description": description,
                    "dry_run": True,
                    "note": "Dry run - no changes made",
                },
                request_id=request_id,
            )
        )

    start_time = time.perf_counter()
    try:
        result, error = add_risk(
            spec_id=spec_id,
            description=description,
            likelihood=payload.get("likelihood") if isinstance(payload.get("likelihood"), str) else None,
            impact=payload.get("impact") if isinstance(payload.get("impact"), str) else None,
            mitigation=payload.get("mitigation") if isinstance(payload.get("mitigation"), str) else None,
            specs_dir=specs_dir,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error adding risk")
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
                f"Failed to add risk: {error}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check that the spec exists and description is provided",
                request_id=request_id,
            )
        )

    _metrics.counter(metric_key, labels={"status": "success"})
    return asdict(
        success_response(
            data={"spec_id": spec_id, "dry_run": False, **(result or {})},
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )


def _handle_risk_list(*, config: ServerConfig, **payload: Any) -> dict:
    return _handle_text_list("risk-list", list_risks, "risks", config=config, **payload)
