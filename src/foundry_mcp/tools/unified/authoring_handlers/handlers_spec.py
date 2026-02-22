"""Spec CRUD action handlers: spec-create, spec-template, spec-update-frontmatter, spec-find-replace, spec-rollback."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict

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
    CATEGORIES,
    TEMPLATES,
    create_spec,
    find_replace_in_spec,
    generate_spec_data,
    rollback_spec,
    update_frontmatter,
)
from foundry_mcp.core.validation.rules import validate_spec
from foundry_mcp.tools.unified.authoring_handlers._helpers import (
    _metric_name,
    _metrics,
    _request_id,
    _resolve_specs_dir,
    _validation_error,
    logger,
)
from foundry_mcp.tools.unified.param_schema import Bool, Str, validate_payload

# Valid scopes for find-replace
_FIND_REPLACE_SCOPES = {"all", "titles", "descriptions"}

_SPEC_UPDATE_FRONTMATTER_SCHEMA = {
    "spec_id": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
    "key": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
    # value validated imperatively — any type, just required
    "dry_run": Bool(default=False, error_code=ErrorCode.INVALID_FORMAT),
    "path": Str(error_code=ErrorCode.INVALID_FORMAT),
}

_SPEC_FIND_REPLACE_SCHEMA = {
    "spec_id": Str(
        required=True, error_code=ErrorCode.MISSING_REQUIRED, remediation="Pass the spec identifier to authoring"
    ),
    "find": Str(
        required=True, error_code=ErrorCode.MISSING_REQUIRED, remediation="Specify the text or regex pattern to find"
    ),
    # replace validated imperatively — needs MISSING_REQUIRED vs INVALID_FORMAT
    "scope": Str(
        choices=frozenset(_FIND_REPLACE_SCOPES),
        error_code=ErrorCode.INVALID_FORMAT,
        remediation=f"Use one of: {sorted(_FIND_REPLACE_SCOPES)}",
    ),
    "use_regex": Bool(default=False, error_code=ErrorCode.INVALID_FORMAT),
    "case_sensitive": Bool(default=True, error_code=ErrorCode.INVALID_FORMAT),
    "dry_run": Bool(default=False, error_code=ErrorCode.INVALID_FORMAT),
    "path": Str(error_code=ErrorCode.INVALID_FORMAT),
}

_SPEC_ROLLBACK_SCHEMA = {
    "spec_id": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
    "version": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
    "dry_run": Bool(default=False),
    "path": Str(),
}

_SPEC_CREATE_SCHEMA = {
    "name": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
    "template": Str(
        choices=frozenset(TEMPLATES),
        remediation="Use template='empty' and add phases via phase-add-bulk",
    ),
    "category": Str(choices=frozenset(CATEGORIES), remediation=f"Use one of: {', '.join(CATEGORIES)}"),
    "mission": Str(),
    "plan_path": Str(
        required=True, error_code=ErrorCode.MISSING_REQUIRED, remediation="Provide the path to the markdown plan file"
    ),
    "plan_review_path": Str(
        required=True,
        error_code=ErrorCode.MISSING_REQUIRED,
        remediation="Provide the path to the synthesized plan review file",
    ),
    "dry_run": Bool(default=False),
    "path": Str(),
}


def _handle_spec_create(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "spec-create"

    # Apply defaults for template and category before validation
    payload["template"] = payload.get("template") or "empty"
    payload["category"] = payload.get("category") or "implementation"

    err = validate_payload(payload, _SPEC_CREATE_SCHEMA, tool_name="authoring", action=action, request_id=request_id)
    if err:
        return err

    name = payload["name"]
    template = payload["template"]
    category = payload["category"]
    mission = payload.get("mission")
    plan_path = payload["plan_path"]
    plan_review_path = payload["plan_review_path"]
    dry_run = payload["dry_run"]
    path = payload.get("path")

    specs_dir, specs_err = _resolve_specs_dir(config, path)
    if specs_err:
        return specs_err
    assert specs_dir is not None

    if dry_run:
        # Generate spec data for preflight validation
        spec_data, gen_error = generate_spec_data(
            name=name.strip(),
            template=template,
            category=category,
            mission=mission,
            plan_path=plan_path,
            plan_review_path=plan_review_path,
        )
        if gen_error:
            return _validation_error(
                field="spec",
                action=action,
                message=gen_error,
                request_id=request_id,
                code=ErrorCode.VALIDATION_ERROR,
            )

        assert spec_data is not None
        # Run full validation on generated spec
        validation_result = validate_spec(spec_data)
        diagnostics = [
            {
                "code": d.code,
                "message": d.message,
                "severity": d.severity,
                "location": d.location,
                "suggested_fix": d.suggested_fix,
            }
            for d in validation_result.diagnostics
        ]

        return asdict(
            success_response(
                data={
                    "name": name.strip(),
                    "spec_id": spec_data["spec_id"],
                    "template": template,
                    "category": category,
                    "mission": mission.strip() if isinstance(mission, str) else None,
                    "dry_run": True,
                    "is_valid": validation_result.is_valid,
                    "error_count": validation_result.error_count,
                    "warning_count": validation_result.warning_count,
                    "diagnostics": diagnostics,
                    "note": "Preflight validation complete - no changes made",
                },
                request_id=request_id,
            )
        )

    start_time = time.perf_counter()
    audit_log(
        "tool_invocation",
        tool="authoring",
        action="spec_create",
        name=name.strip(),
        template=template,
        category=category,
    )

    result, error = create_spec(
        name=name.strip(),
        template=template,
        category=category,
        mission=mission,
        plan_path=plan_path,
        plan_review_path=plan_review_path,
        specs_dir=specs_dir,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    metric_key = _metric_name(action)
    _metrics.timer(metric_key + ".duration_ms", elapsed_ms)

    if error:
        _metrics.counter(metric_key, labels={"status": "error"})
        lowered = error.lower()
        if "already exists" in lowered:
            return asdict(
                error_response(
                    f"A specification with name '{name.strip()}' already exists",
                    error_code=ErrorCode.DUPLICATE_ENTRY,
                    error_type=ErrorType.CONFLICT,
                    remediation="Use a different name or update the existing spec",
                    request_id=request_id,
                    telemetry={"duration_ms": round(elapsed_ms, 2)},
                )
            )
        return asdict(
            error_response(
                f"Failed to create specification: {error}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check that the specs directory is writable",
                request_id=request_id,
                telemetry={"duration_ms": round(elapsed_ms, 2)},
            )
        )

    data: Dict[str, Any] = {
        "spec_id": (result or {}).get("spec_id"),
        "spec_path": (result or {}).get("spec_path"),
        "template": template,
        "category": category,
        "name": name.strip(),
    }
    if result and result.get("structure"):
        data["structure"] = result["structure"]

    _metrics.counter(metric_key, labels={"status": "success"})
    return asdict(
        success_response(
            data=data,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )


def _handle_spec_template(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "spec-template"

    template_action = payload.get("template_action")
    if not isinstance(template_action, str) or not template_action.strip():
        return _validation_error(
            field="template_action",
            action=action,
            message="Provide one of: list, show, apply",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    template_action = template_action.strip().lower()
    if template_action not in ("list", "show", "apply"):
        return _validation_error(
            field="template_action",
            action=action,
            message="template_action must be one of: list, show, apply",
            request_id=request_id,
            remediation="Use list, show, or apply",
        )

    template_name = payload.get("template_name")
    if template_action in ("show", "apply"):
        if not isinstance(template_name, str) or not template_name.strip():
            return _validation_error(
                field="template_name",
                action=action,
                message="Provide a template name",
                request_id=request_id,
                code=ErrorCode.MISSING_REQUIRED,
            )
        template_name = template_name.strip()
        if template_name not in TEMPLATES:
            return asdict(
                error_response(
                    f"Template '{template_name}' not found",
                    error_code=ErrorCode.NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation=f"Use template_action='list' to see available templates. Valid: {', '.join(TEMPLATES)}",
                    request_id=request_id,
                )
            )

    data: Dict[str, Any] = {"action": template_action}
    if template_action == "list":
        data["templates"] = [
            {
                "name": "empty",
                "description": "Blank spec with no phases - use phase-add-bulk to add structure",
            },
        ]
        data["total_count"] = 1
        data["message"] = "Use 'empty' template, then add phases via phase-add-bulk"
    elif template_action == "show":
        data["template_name"] = template_name
        data["content"] = {
            "name": template_name,
            "description": "Blank spec with no phases",
            "usage": "Use authoring(action='spec-create', name='your-spec') to create, then add phases via phase-add-bulk",
        }
    else:
        data["template_name"] = template_name
        data["generated"] = {
            "template": template_name,
            "message": "Use spec-create to create an empty spec, then add phases",
        }
        data["instructions"] = (
            "1. Create spec: authoring(action='spec-create', name='your-spec-name')\n"
            "2. Add phases: authoring(action='phase-add-bulk', spec_id='...', title='Phase 1', tasks=[...])"
        )

    return asdict(success_response(data=data, request_id=request_id))


def _handle_spec_update_frontmatter(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "spec-update-frontmatter"

    err = validate_payload(
        payload, _SPEC_UPDATE_FRONTMATTER_SCHEMA, tool_name="authoring", action=action, request_id=request_id
    )
    if err:
        return err

    spec_id = payload["spec_id"]
    key = payload["key"]
    dry_run = payload["dry_run"]
    path = payload.get("path")

    # value is required but can be any type — keep imperative
    value = payload.get("value")
    if value is None:
        return _validation_error(
            field="value",
            action=action,
            message="Provide a value",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )

    specs_dir, specs_err = _resolve_specs_dir(config, path)
    if specs_err:
        return specs_err

    if dry_run:
        return asdict(
            success_response(
                data={
                    "spec_id": spec_id.strip(),
                    "key": key.strip(),
                    "value": value,
                    "dry_run": True,
                    "note": "Dry run - no changes made",
                },
                request_id=request_id,
            )
        )

    start_time = time.perf_counter()
    result, error = update_frontmatter(
        spec_id=spec_id.strip(),
        key=key.strip(),
        value=value,
        specs_dir=specs_dir,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    metric_key = _metric_name(action)
    _metrics.timer(metric_key + ".duration_ms", elapsed_ms)

    if error or not result:
        _metrics.counter(metric_key, labels={"status": "error"})
        lowered = (error or "").lower()
        if "not found" in lowered and "spec" in lowered:
            return asdict(
                error_response(
                    f"Specification '{spec_id.strip()}' not found",
                    error_code=ErrorCode.SPEC_NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation='Verify the spec ID exists using spec(action="list")',
                    request_id=request_id,
                    telemetry={"duration_ms": round(elapsed_ms, 2)},
                )
            )
        if "use dedicated" in lowered:
            return asdict(
                error_response(
                    error or "Invalid metadata key",
                    error_code=ErrorCode.VALIDATION_ERROR,
                    error_type=ErrorType.VALIDATION,
                    remediation="Use authoring(action='assumption-add') or authoring(action='revision-add') for list fields",
                    request_id=request_id,
                    telemetry={"duration_ms": round(elapsed_ms, 2)},
                )
            )
        return asdict(
            error_response(
                error or "Failed to update frontmatter",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation="Provide a valid key and value",
                request_id=request_id,
                telemetry={"duration_ms": round(elapsed_ms, 2)},
            )
        )

    _metrics.counter(metric_key, labels={"status": "success"})
    return asdict(
        success_response(
            data=result,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )


def _handle_spec_find_replace(*, config: ServerConfig, **payload: Any) -> dict:
    """Find and replace text across spec hierarchy nodes.

    Supports literal or regex find/replace across titles and/or descriptions.
    Returns a preview in dry_run mode, or applies changes and returns a summary.
    """
    request_id = _request_id()
    action = "spec-find-replace"

    # Pre-apply scope default before schema validation
    payload["scope"] = payload.get("scope") or "all"

    err = validate_payload(
        payload, _SPEC_FIND_REPLACE_SCHEMA, tool_name="authoring", action=action, request_id=request_id
    )
    if err:
        return err

    spec_id = payload["spec_id"]
    find = payload["find"]
    scope = payload["scope"]
    use_regex = payload["use_regex"]
    case_sensitive = payload["case_sensitive"]
    dry_run = payload["dry_run"]
    path = payload.get("path")

    # replace needs MISSING_REQUIRED vs INVALID_FORMAT — keep imperative
    replace = payload.get("replace")
    if replace is None:
        return _validation_error(
            field="replace",
            action=action,
            message="Provide a replace value (use empty string to delete matches)",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
            remediation="Provide a replacement string (use empty string to delete)",
        )
    if not isinstance(replace, str):
        return _validation_error(
            field="replace",
            action=action,
            message="replace must be a string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
            remediation="Provide a string value for replace parameter",
        )

    specs_dir, specs_err = _resolve_specs_dir(config, path)
    if specs_err:
        return specs_err

    audit_log(
        "tool_invocation",
        tool="authoring",
        action=action,
        spec_id=spec_id,
        find=find[:50] + "..." if len(find) > 50 else find,
        use_regex=use_regex,
        dry_run=dry_run,
    )

    metric_key = _metric_name(action)
    start_time = time.perf_counter()

    try:
        result, error = find_replace_in_spec(
            spec_id,
            find,
            replace,
            scope=scope,
            use_regex=use_regex,
            case_sensitive=case_sensitive,
            dry_run=dry_run,
            specs_dir=specs_dir,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected error in spec find-replace")
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
        # Map error types
        if "not found" in error.lower():
            return asdict(
                error_response(
                    error,
                    error_code=ErrorCode.NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation="Check spec_id value",
                    request_id=request_id,
                    telemetry={"duration_ms": round(elapsed_ms, 2)},
                )
            )
        if "invalid regex" in error.lower():
            return asdict(
                error_response(
                    error,
                    error_code=ErrorCode.INVALID_FORMAT,
                    error_type=ErrorType.VALIDATION,
                    remediation="Check regex syntax",
                    request_id=request_id,
                    telemetry={"duration_ms": round(elapsed_ms, 2)},
                )
            )
        return asdict(
            error_response(
                error,
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation="Check find and replace parameters",
                request_id=request_id,
                telemetry={"duration_ms": round(elapsed_ms, 2)},
            )
        )

    _metrics.counter(metric_key, labels={"status": "success", "dry_run": str(dry_run).lower()})
    return asdict(
        success_response(
            data=result,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )


def _handle_spec_rollback(*, config: ServerConfig, **payload: Any) -> dict:
    """Restore a spec from a backup timestamp."""
    request_id = _request_id()
    action = "spec-rollback"

    err = validate_payload(payload, _SPEC_ROLLBACK_SCHEMA, tool_name="authoring", action=action, request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    timestamp = payload["version"]
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
        timestamp=timestamp,
        dry_run=dry_run,
    )

    metric_key = _metric_name(action)
    start_time = time.perf_counter()

    result = rollback_spec(
        spec_id=spec_id,
        timestamp=timestamp,
        specs_dir=specs_dir,
        dry_run=dry_run,
        create_backup=True,
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    if not result.get("success"):
        _metrics.counter(metric_key, labels={"status": "error"})
        error_msg = result.get("error", "Unknown error during rollback")

        # Determine error code based on error message
        if "not found" in error_msg.lower():
            error_code = ErrorCode.NOT_FOUND
            error_type = ErrorType.NOT_FOUND
            remediation = "Use spec(action='history') to list available backups"
        else:
            error_code = ErrorCode.INTERNAL_ERROR
            error_type = ErrorType.INTERNAL
            remediation = "Check spec and backup file permissions"

        return asdict(
            error_response(
                error_msg,
                error_code=error_code,
                error_type=error_type,
                remediation=remediation,
                request_id=request_id,
                telemetry={"duration_ms": round(elapsed_ms, 2)},
            )
        )

    _metrics.counter(metric_key, labels={"status": "success", "dry_run": str(dry_run).lower()})
    return asdict(
        success_response(
            spec_id=spec_id,
            timestamp=timestamp,
            dry_run=dry_run,
            restored_from=result.get("restored_from"),
            backup_created=result.get("backup_created"),
            request_id=request_id,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
        )
    )
