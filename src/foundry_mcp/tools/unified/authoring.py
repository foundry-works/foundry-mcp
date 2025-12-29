"""Unified authoring tool backed by ActionRouter and shared validation."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.context import generate_correlation_id, get_correlation_id
from foundry_mcp.core.feature_flags import FeatureFlag, FlagState, get_flag_service
from foundry_mcp.core.intake import IntakeStore, LockAcquisitionError, INTAKE_ID_PATTERN
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import audit_log, get_metrics, mcp_tool
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    sanitize_error_message,
    success_response,
)
from foundry_mcp.core.spec import (
    ASSUMPTION_TYPES,
    CATEGORIES,
    PHASE_TEMPLATES,
    TEMPLATES,
    add_assumption,
    add_phase,
    add_phase_bulk,
    add_revision,
    apply_phase_template,
    create_spec,
    find_replace_in_spec,
    find_specs_directory,
    generate_spec_data,
    get_phase_template_structure,
    list_assumptions,
    load_spec,
    move_phase,
    remove_phase,
    rollback_spec,
    update_frontmatter,
    update_phase_metadata,
)
from foundry_mcp.core.validation import validate_spec
from foundry_mcp.tools.unified.router import (
    ActionDefinition,
    ActionRouter,
    ActionRouterError,
)

logger = logging.getLogger(__name__)
_metrics = get_metrics()

# Register intake_tools feature flag
_flag_service = get_flag_service()
try:
    _flag_service.register(
        FeatureFlag(
            name="intake_tools",
            description="Bikelane intake queue tools (add, list, dismiss)",
            state=FlagState.EXPERIMENTAL,
            default_enabled=False,
        )
    )
except ValueError:
    pass  # Flag already registered


def _intake_feature_flag_blocked(request_id: str) -> Optional[dict]:
    """Check if intake tools are blocked by feature flag."""
    if _flag_service.is_enabled("intake_tools"):
        return None

    return asdict(
        error_response(
            "Intake tools are disabled by feature flag",
            error_code=ErrorCode.FEATURE_DISABLED,
            error_type=ErrorType.FEATURE_FLAG,
            data={"feature": "intake_tools"},
            remediation="Enable the 'intake_tools' feature flag to use intake actions.",
            request_id=request_id,
        )
    )


_ACTION_SUMMARY = {
    "spec-create": "Scaffold a new SDD specification",
    "spec-template": "List/show/apply spec templates",
    "spec-update-frontmatter": "Update a top-level metadata field",
    "spec-find-replace": "Find and replace text across spec titles and descriptions",
    "spec-rollback": "Restore a spec from a backup timestamp",
    "phase-add": "Add a new phase under spec-root with verification scaffolding",
    "phase-add-bulk": "Add a phase with pre-defined tasks in a single atomic operation",
    "phase-template": "List/show/apply phase templates to add pre-configured phases",
    "phase-move": "Reorder a phase within spec-root children",
    "phase-update-metadata": "Update metadata fields of an existing phase",
    "phase-remove": "Remove an existing phase (and optionally dependents)",
    "assumption-add": "Append an assumption entry to spec metadata",
    "assumption-list": "List recorded assumptions for a spec",
    "revision-add": "Record a revision entry in the spec history",
    "intake-add": "Capture a new work idea in the bikelane intake queue",
    "intake-list": "List new intake items awaiting triage in FIFO order",
    "intake-dismiss": "Dismiss an intake item from the triage queue",
}


def _metric_name(action: str) -> str:
    return f"authoring.{action.replace('-', '_')}"


def _request_id() -> str:
    return get_correlation_id() or generate_correlation_id(prefix="authoring")


def _validation_error(
    *,
    field: str,
    action: str,
    message: str,
    request_id: str,
    code: ErrorCode = ErrorCode.VALIDATION_ERROR,
    remediation: Optional[str] = None,
) -> dict:
    return asdict(
        error_response(
            f"Invalid field '{field}' for authoring.{action}: {message}",
            error_code=code,
            error_type=ErrorType.VALIDATION,
            remediation=remediation,
            details={"field": field, "action": f"authoring.{action}"},
            request_id=request_id,
        )
    )


def _specs_directory_missing_error(request_id: str) -> dict:
    return asdict(
        error_response(
            "No specs directory found. Use specs_dir parameter or set SDD_SPECS_DIR.",
            error_code=ErrorCode.NOT_FOUND,
            error_type=ErrorType.NOT_FOUND,
            remediation="Use --specs-dir or set SDD_SPECS_DIR",
            request_id=request_id,
        )
    )


def _resolve_specs_dir(config: ServerConfig, path: Optional[str]) -> Optional[Path]:
    try:
        if path:
            return find_specs_directory(path)
        return config.specs_dir or find_specs_directory()
    except Exception:  # pragma: no cover - defensive guard
        logger.exception("Failed to resolve specs directory", extra={"path": path})
        return None


def _phase_exists(spec_id: str, specs_dir: Path, title: str) -> bool:
    try:
        spec_data = load_spec(spec_id, specs_dir)
    except Exception:  # pragma: no cover - defensive guard
        logger.exception(
            "Failed to inspect spec for duplicate phases", extra={"spec_id": spec_id}
        )
        return False

    if not spec_data:
        return False

    hierarchy = spec_data.get("hierarchy", {})
    if not isinstance(hierarchy, dict):
        return False

    normalized = title.strip().casefold()
    for node in hierarchy.values():
        if isinstance(node, dict) and node.get("type") == "phase":
            node_title = str(node.get("title", "")).strip().casefold()
            if node_title and node_title == normalized:
                return True
    return False


def _assumption_exists(spec_id: str, specs_dir: Path, text: str) -> bool:
    result, error = list_assumptions(spec_id=spec_id, specs_dir=specs_dir)
    if error or not result:
        return False

    normalized = text.strip().casefold()
    for entry in result.get("assumptions", []):
        entry_text = str(entry.get("text", "")).strip().casefold()
        if entry_text and entry_text == normalized:
            return True
    return False


def _handle_spec_create(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "spec-create"

    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        return _validation_error(
            field="name",
            action=action,
            message="Provide a non-empty specification name",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )

    template = payload.get("template") or "medium"
    if not isinstance(template, str):
        return _validation_error(
            field="template",
            action=action,
            message="template must be a string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )
    template = template.strip() or "medium"
    if template not in TEMPLATES:
        return _validation_error(
            field="template",
            action=action,
            message=f"Template must be one of: {', '.join(TEMPLATES)}",
            request_id=request_id,
            remediation=f"Use one of: {', '.join(TEMPLATES)}",
        )

    category = payload.get("category") or "implementation"
    if not isinstance(category, str):
        return _validation_error(
            field="category",
            action=action,
            message="category must be a string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )
    category = category.strip() or "implementation"
    if category not in CATEGORIES:
        return _validation_error(
            field="category",
            action=action,
            message=f"Category must be one of: {', '.join(CATEGORIES)}",
            request_id=request_id,
            remediation=f"Use one of: {', '.join(CATEGORIES)}",
        )

    mission = payload.get("mission")
    if mission is not None and not isinstance(mission, str):
        return _validation_error(
            field="mission",
            action=action,
            message="mission must be a string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    if template in ("medium", "complex"):
        if not isinstance(mission, str) or not mission.strip():
            return _validation_error(
                field="mission",
                action=action,
                message="mission is required for medium/complex specifications",
                request_id=request_id,
                code=ErrorCode.MISSING_REQUIRED,
                remediation="Provide a concise mission statement",
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

    path = payload.get("path")
    if path is not None and not isinstance(path, str):
        return _validation_error(
            field="path",
            action=action,
            message="path must be a string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    specs_dir = _resolve_specs_dir(config, path)
    if specs_dir is None:
        return _specs_directory_missing_error(request_id)

    if dry_run:
        # Generate spec data for preflight validation
        spec_data, gen_error = generate_spec_data(
            name=name.strip(),
            template=template,
            category=category,
            mission=mission,
        )
        if gen_error:
            return _validation_error(
                field="spec",
                action=action,
                message=gen_error,
                request_id=request_id,
                code=ErrorCode.VALIDATION_ERROR,
            )

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
                "name": "simple",
                "description": "Minimal spec with 1 phase and basic tasks",
            },
            {
                "name": "medium",
                "description": "Standard spec with 2-3 phases (default)",
            },
            {
                "name": "complex",
                "description": "Multi-phase spec with groups and subtasks",
            },
            {
                "name": "security",
                "description": "Security-focused spec with audit tasks",
            },
        ]
        data["total_count"] = len(data["templates"])
    elif template_action == "show":
        data["template_name"] = template_name
        data["content"] = {
            "name": template_name,
            "description": f"Template structure for '{template_name}' specs",
            "usage": f"Use authoring(action='spec-create', template='{template_name}') to create a spec",
        }
    else:
        data["template_name"] = template_name
        data["generated"] = {
            "template": template_name,
            "message": f"Use authoring(action='spec-create', template='{template_name}') to create a new spec",
        }
        data["instructions"] = (
            f"Call authoring(action='spec-create', name='your-spec-name', template='{template_name}')"
        )

    return asdict(success_response(data=data, request_id=request_id))


def _handle_spec_update_frontmatter(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "spec-update-frontmatter"

    spec_id = payload.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec identifier",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )

    key = payload.get("key")
    if not isinstance(key, str) or not key.strip():
        return _validation_error(
            field="key",
            action=action,
            message="Provide a non-empty metadata key",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )

    value = payload.get("value")
    if value is None:
        return _validation_error(
            field="value",
            action=action,
            message="Provide a value",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
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

    path = payload.get("path")
    if path is not None and not isinstance(path, str):
        return _validation_error(
            field="path",
            action=action,
            message="path must be a string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    specs_dir = _resolve_specs_dir(config, path)
    if specs_dir is None:
        return _specs_directory_missing_error(request_id)

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


# Valid scopes for find-replace
_FIND_REPLACE_SCOPES = {"all", "titles", "descriptions"}


def _handle_spec_find_replace(*, config: ServerConfig, **payload: Any) -> dict:
    """Find and replace text across spec hierarchy nodes.

    Supports literal or regex find/replace across titles and/or descriptions.
    Returns a preview in dry_run mode, or applies changes and returns a summary.
    """
    request_id = _request_id()
    action = "spec-find-replace"

    # Required: spec_id
    spec_id = payload.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec_id parameter",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
            remediation="Pass the spec identifier to authoring",
        )
    spec_id = spec_id.strip()

    # Required: find
    find = payload.get("find")
    if not isinstance(find, str) or not find:
        return _validation_error(
            field="find",
            action=action,
            message="Provide a non-empty find pattern",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
            remediation="Specify the text or regex pattern to find",
        )

    # Required: replace (can be empty string to delete matches)
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

    # Optional: scope (default: "all")
    scope = payload.get("scope", "all")
    if not isinstance(scope, str) or scope not in _FIND_REPLACE_SCOPES:
        return _validation_error(
            field="scope",
            action=action,
            message=f"scope must be one of: {sorted(_FIND_REPLACE_SCOPES)}",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
            remediation=f"Use one of: {sorted(_FIND_REPLACE_SCOPES)}",
        )

    # Optional: use_regex (default: False)
    use_regex = payload.get("use_regex", False)
    if not isinstance(use_regex, bool):
        return _validation_error(
            field="use_regex",
            action=action,
            message="use_regex must be a boolean",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
            remediation="Set use_regex to true or false",
        )

    # Optional: case_sensitive (default: True)
    case_sensitive = payload.get("case_sensitive", True)
    if not isinstance(case_sensitive, bool):
        return _validation_error(
            field="case_sensitive",
            action=action,
            message="case_sensitive must be a boolean",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
            remediation="Set case_sensitive to true or false",
        )

    # Optional: dry_run (default: False)
    dry_run = payload.get("dry_run", False)
    if not isinstance(dry_run, bool):
        return _validation_error(
            field="dry_run",
            action=action,
            message="dry_run must be a boolean",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
            remediation="Set dry_run to true or false",
        )

    # Optional: path (workspace)
    path = payload.get("path")
    if path is not None and not isinstance(path, str):
        return _validation_error(
            field="path",
            action=action,
            message="Workspace path must be a string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    specs_dir = _resolve_specs_dir(config, path)
    if specs_dir is None:
        return _specs_directory_missing_error(request_id)

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

    spec_id = payload.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec_id parameter",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    spec_id = spec_id.strip()

    timestamp = payload.get("version")  # Use 'version' parameter for timestamp
    if not isinstance(timestamp, str) or not timestamp.strip():
        return _validation_error(
            field="version",
            action=action,
            message="Provide the backup timestamp to restore (use spec history to list)",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    timestamp = timestamp.strip()

    dry_run = payload.get("dry_run", False)
    if not isinstance(dry_run, bool):
        return _validation_error(
            field="dry_run",
            action=action,
            message="Expected a boolean value",
            request_id=request_id,
        )

    path = payload.get("path")
    if path is not None and not isinstance(path, str):
        return _validation_error(
            field="path",
            action=action,
            message="Workspace path must be a string",
            request_id=request_id,
        )

    specs_dir = _resolve_specs_dir(config, path)
    if specs_dir is None:
        return _specs_directory_missing_error(request_id)

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


def _handle_phase_add(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "phase-add"

    spec_id = payload.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec_id parameter",
            remediation="Pass the spec identifier to authoring",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    spec_id = spec_id.strip()

    title = payload.get("title")
    if not isinstance(title, str) or not title.strip():
        return _validation_error(
            field="title",
            action=action,
            message="Provide a non-empty phase title",
            remediation="Include a descriptive title for the new phase",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    title = title.strip()

    description = payload.get("description")
    if description is not None and not isinstance(description, str):
        return _validation_error(
            field="description",
            action=action,
            message="Description must be a string",
            request_id=request_id,
        )
    purpose = payload.get("purpose")
    if purpose is not None and not isinstance(purpose, str):
        return _validation_error(
            field="purpose",
            action=action,
            message="Purpose must be a string",
            request_id=request_id,
        )

    estimated_hours = payload.get("estimated_hours")
    if estimated_hours is not None:
        if isinstance(estimated_hours, bool) or not isinstance(
            estimated_hours, (int, float)
        ):
            return _validation_error(
                field="estimated_hours",
                action=action,
                message="Provide a numeric value",
                request_id=request_id,
            )
        if estimated_hours < 0:
            return _validation_error(
                field="estimated_hours",
                action=action,
                message="Value must be non-negative",
                remediation="Set hours to zero or greater",
                request_id=request_id,
            )
        estimated_hours = float(estimated_hours)

    position = payload.get("position")
    if position is not None:
        if isinstance(position, bool) or not isinstance(position, int):
            return _validation_error(
                field="position",
                action=action,
                message="Position must be an integer",
                request_id=request_id,
            )
        if position < 0:
            return _validation_error(
                field="position",
                action=action,
                message="Position must be >= 0",
                request_id=request_id,
            )

    link_previous = payload.get("link_previous", True)
    if not isinstance(link_previous, bool):
        return _validation_error(
            field="link_previous",
            action=action,
            message="Expected a boolean value",
            request_id=request_id,
        )

    dry_run = payload.get("dry_run", False)
    if not isinstance(dry_run, bool):
        return _validation_error(
            field="dry_run",
            action=action,
            message="Expected a boolean value",
            request_id=request_id,
        )

    path = payload.get("path")
    if path is not None and not isinstance(path, str):
        return _validation_error(
            field="path",
            action=action,
            message="Workspace path must be a string",
            request_id=request_id,
        )

    specs_dir = _resolve_specs_dir(config, path)
    if specs_dir is None:
        return _specs_directory_missing_error(request_id)

    warnings: List[str] = []
    if _phase_exists(spec_id, specs_dir, title):
        warnings.append(
            f"Phase titled '{title}' already exists; the new phase will still be added"
        )

    audit_log(
        "tool_invocation",
        tool="authoring",
        action=action,
        spec_id=spec_id,
        title=title,
        dry_run=dry_run,
        link_previous=link_previous,
    )

    metric_key = _metric_name(action)

    if dry_run:
        _metrics.counter(metric_key, labels={"status": "success", "dry_run": "true"})
        return asdict(
            success_response(
                data={
                    "spec_id": spec_id,
                    "phase_id": "(preview)",
                    "title": title,
                    "dry_run": True,
                    "note": "Dry run - no changes made",
                },
                warnings=warnings or None,
                request_id=request_id,
            )
        )

    start_time = time.perf_counter()
    try:
        result, error = add_phase(
            spec_id=spec_id,
            title=title,
            description=description,
            purpose=purpose,
            estimated_hours=estimated_hours,
            position=position,
            link_previous=link_previous,
            specs_dir=specs_dir,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected error adding phase")
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
        lowered = error.lower()
        if "specification" in lowered and "not found" in lowered:
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
                f"Failed to add phase: {error}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check input values and retry",
                request_id=request_id,
            )
        )

    _metrics.counter(metric_key, labels={"status": "success"})
    return asdict(
        success_response(
            data={"spec_id": spec_id, "dry_run": False, **(result or {})},
            warnings=warnings or None,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )


def _handle_phase_update_metadata(*, config: ServerConfig, **payload: Any) -> dict:
    """Update metadata fields of an existing phase."""
    request_id = _request_id()
    action = "phase-update-metadata"

    spec_id = payload.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec_id parameter",
            remediation="Pass the spec identifier to authoring",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    spec_id = spec_id.strip()

    phase_id = payload.get("phase_id")
    if not isinstance(phase_id, str) or not phase_id.strip():
        return _validation_error(
            field="phase_id",
            action=action,
            message="Provide a non-empty phase_id parameter",
            remediation="Pass the phase identifier (e.g., 'phase-1')",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    phase_id = phase_id.strip()

    # Extract optional metadata fields
    estimated_hours = payload.get("estimated_hours")
    description = payload.get("description")
    purpose = payload.get("purpose")

    # Validate at least one field is provided
    has_update = any(v is not None for v in [estimated_hours, description, purpose])
    if not has_update:
        return _validation_error(
            field="metadata",
            action=action,
            message="At least one metadata field must be provided",
            remediation="Include estimated_hours, description, or purpose",
            request_id=request_id,
            code=ErrorCode.VALIDATION_FAILED,
        )

    # Validate estimated_hours if provided
    if estimated_hours is not None:
        if isinstance(estimated_hours, bool) or not isinstance(
            estimated_hours, (int, float)
        ):
            return _validation_error(
                field="estimated_hours",
                action=action,
                message="Provide a numeric value",
                remediation="Set estimated_hours to a number >= 0",
                request_id=request_id,
            )
        if estimated_hours < 0:
            return _validation_error(
                field="estimated_hours",
                action=action,
                message="Value must be non-negative",
                remediation="Set hours to zero or greater",
                request_id=request_id,
            )
        estimated_hours = float(estimated_hours)

    # Validate description if provided
    if description is not None and not isinstance(description, str):
        return _validation_error(
            field="description",
            action=action,
            message="Description must be a string",
            remediation="Provide a text description",
            request_id=request_id,
        )

    # Validate purpose if provided
    if purpose is not None and not isinstance(purpose, str):
        return _validation_error(
            field="purpose",
            action=action,
            message="Purpose must be a string",
            remediation="Provide a text purpose",
            request_id=request_id,
        )

    dry_run = payload.get("dry_run", False)
    if not isinstance(dry_run, bool):
        return _validation_error(
            field="dry_run",
            action=action,
            message="Expected a boolean value",
            remediation="Set dry_run to true or false",
            request_id=request_id,
        )

    path = payload.get("path")
    if path is not None and not isinstance(path, str):
        return _validation_error(
            field="path",
            action=action,
            message="Workspace path must be a string",
            remediation="Provide a valid workspace path",
            request_id=request_id,
        )

    specs_dir = _resolve_specs_dir(config, path)
    if specs_dir is None:
        return _specs_directory_missing_error(request_id)

    audit_log(
        "tool_invocation",
        tool="authoring",
        action=action,
        spec_id=spec_id,
        phase_id=phase_id,
        dry_run=dry_run,
    )

    metric_key = _metric_name(action)
    start_time = time.perf_counter()

    try:
        result, error = update_phase_metadata(
            spec_id=spec_id,
            phase_id=phase_id,
            estimated_hours=estimated_hours,
            description=description,
            purpose=purpose,
            dry_run=dry_run,
            specs_dir=specs_dir,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected error updating phase metadata")
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
        lowered = error.lower()
        if "specification" in lowered and "not found" in lowered:
            return asdict(
                error_response(
                    f"Specification '{spec_id}' not found",
                    error_code=ErrorCode.SPEC_NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation='Verify the spec ID via spec(action="list")',
                    request_id=request_id,
                )
            )
        if "phase" in lowered and "not found" in lowered:
            return asdict(
                error_response(
                    f"Phase '{phase_id}' not found in spec '{spec_id}'",
                    error_code=ErrorCode.TASK_NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation='Verify the phase ID via task(action="query")',
                    request_id=request_id,
                )
            )
        if "not a phase" in lowered:
            return asdict(
                error_response(
                    f"Node '{phase_id}' is not a phase",
                    error_code=ErrorCode.VALIDATION_FAILED,
                    error_type=ErrorType.VALIDATION,
                    remediation="Provide a valid phase ID (e.g., 'phase-1')",
                    request_id=request_id,
                )
            )
        return asdict(
            error_response(
                f"Failed to update phase metadata: {error}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check input values and retry",
                request_id=request_id,
            )
        )

    _metrics.counter(metric_key, labels={"status": "success"})
    return asdict(
        success_response(
            data={"spec_id": spec_id, "phase_id": phase_id, **(result or {})},
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )


def _handle_phase_add_bulk(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "phase-add-bulk"

    # Validate spec_id
    spec_id = payload.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec_id parameter",
            remediation="Pass the spec identifier to authoring",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    spec_id = spec_id.strip()

    # Require macro format: {phase: {...}, tasks: [...]}
    phase_obj = payload.get("phase")
    if not isinstance(phase_obj, dict):
        return _validation_error(
            field="phase",
            action=action,
            message="Provide a phase object with metadata",
            remediation="Use macro format: {phase: {title: '...', description: '...'}, tasks: [...]}",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )

    # Extract phase metadata from nested object
    title = phase_obj.get("title")
    if not isinstance(title, str) or not title.strip():
        return _validation_error(
            field="phase.title",
            action=action,
            message="Provide a non-empty phase title",
            remediation="Include phase.title in the phase object",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    title = title.strip()

    # Validate tasks array
    tasks = payload.get("tasks")
    if not tasks or not isinstance(tasks, list) or len(tasks) == 0:
        return _validation_error(
            field="tasks",
            action=action,
            message="Provide at least one task definition",
            remediation="Include a tasks array with type and title for each task",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )

    # Validate each task in the array
    valid_task_types = {"task", "verify"}
    for idx, task_def in enumerate(tasks):
        if not isinstance(task_def, dict):
            return _validation_error(
                field=f"tasks[{idx}]",
                action=action,
                message="Each task must be a dictionary",
                request_id=request_id,
            )

        task_type = task_def.get("type")
        if not task_type or task_type not in valid_task_types:
            return _validation_error(
                field=f"tasks[{idx}].type",
                action=action,
                message="Task type must be 'task' or 'verify'",
                remediation="Set type to 'task' or 'verify'",
                request_id=request_id,
            )

        task_title = task_def.get("title")
        if not task_title or not isinstance(task_title, str) or not task_title.strip():
            return _validation_error(
                field=f"tasks[{idx}].title",
                action=action,
                message="Each task must have a non-empty title",
                request_id=request_id,
                code=ErrorCode.MISSING_REQUIRED,
            )

        est_hours = task_def.get("estimated_hours")
        if est_hours is not None:
            if isinstance(est_hours, bool) or not isinstance(est_hours, (int, float)):
                return _validation_error(
                    field=f"tasks[{idx}].estimated_hours",
                    action=action,
                    message="estimated_hours must be a number",
                    request_id=request_id,
                )
            if est_hours < 0:
                return _validation_error(
                    field=f"tasks[{idx}].estimated_hours",
                    action=action,
                    message="estimated_hours must be non-negative",
                    request_id=request_id,
                )

    # Validate optional phase metadata (from phase object)
    description = phase_obj.get("description")
    if description is not None and not isinstance(description, str):
        return _validation_error(
            field="phase.description",
            action=action,
            message="Description must be a string",
            request_id=request_id,
        )

    purpose = phase_obj.get("purpose")
    if purpose is not None and not isinstance(purpose, str):
        return _validation_error(
            field="phase.purpose",
            action=action,
            message="Purpose must be a string",
            request_id=request_id,
        )

    estimated_hours = phase_obj.get("estimated_hours")
    if estimated_hours is not None:
        if isinstance(estimated_hours, bool) or not isinstance(
            estimated_hours, (int, float)
        ):
            return _validation_error(
                field="phase.estimated_hours",
                action=action,
                message="Provide a numeric value",
                request_id=request_id,
            )
        if estimated_hours < 0:
            return _validation_error(
                field="phase.estimated_hours",
                action=action,
                message="Value must be non-negative",
                remediation="Set hours to zero or greater",
                request_id=request_id,
            )
        estimated_hours = float(estimated_hours)

    # Handle metadata_defaults from both top-level and phase object
    # Top-level serves as base, phase-level overrides
    top_level_defaults = payload.get("metadata_defaults")
    if top_level_defaults is not None and not isinstance(top_level_defaults, dict):
        return _validation_error(
            field="metadata_defaults",
            action=action,
            message="metadata_defaults must be a dictionary",
            request_id=request_id,
        )

    phase_level_defaults = phase_obj.get("metadata_defaults")
    if phase_level_defaults is not None and not isinstance(phase_level_defaults, dict):
        return _validation_error(
            field="phase.metadata_defaults",
            action=action,
            message="metadata_defaults must be a dictionary",
            request_id=request_id,
        )

    # Merge: top-level as base, phase-level overrides
    metadata_defaults = None
    if top_level_defaults or phase_level_defaults:
        metadata_defaults = {**(top_level_defaults or {}), **(phase_level_defaults or {})}

    position = payload.get("position")
    if position is not None:
        if isinstance(position, bool) or not isinstance(position, int):
            return _validation_error(
                field="position",
                action=action,
                message="Position must be an integer",
                request_id=request_id,
            )
        if position < 0:
            return _validation_error(
                field="position",
                action=action,
                message="Position must be >= 0",
                request_id=request_id,
            )

    link_previous = payload.get("link_previous", True)
    if not isinstance(link_previous, bool):
        return _validation_error(
            field="link_previous",
            action=action,
            message="Expected a boolean value",
            request_id=request_id,
        )

    dry_run = payload.get("dry_run", False)
    if not isinstance(dry_run, bool):
        return _validation_error(
            field="dry_run",
            action=action,
            message="Expected a boolean value",
            request_id=request_id,
        )

    path = payload.get("path")
    if path is not None and not isinstance(path, str):
        return _validation_error(
            field="path",
            action=action,
            message="Workspace path must be a string",
            request_id=request_id,
        )

    specs_dir = _resolve_specs_dir(config, path)
    if specs_dir is None:
        return _specs_directory_missing_error(request_id)

    # Check for duplicate phase title (warning only)
    warnings: List[str] = []
    if _phase_exists(spec_id, specs_dir, title):
        warnings.append(
            f"Phase titled '{title}' already exists; the new phase will still be added"
        )

    audit_log(
        "tool_invocation",
        tool="authoring",
        action=action,
        spec_id=spec_id,
        title=title,
        task_count=len(tasks),
        dry_run=dry_run,
        link_previous=link_previous,
    )

    metric_key = _metric_name(action)

    if dry_run:
        _metrics.counter(metric_key, labels={"status": "success", "dry_run": "true"})
        preview_tasks = [
            {"task_id": "(preview)", "title": t.get("title", ""), "type": t.get("type", "")}
            for t in tasks
        ]
        return asdict(
            success_response(
                data={
                    "spec_id": spec_id,
                    "phase_id": "(preview)",
                    "title": title,
                    "tasks_created": preview_tasks,
                    "total_tasks": len(tasks),
                    "dry_run": True,
                    "note": "Dry run - no changes made",
                },
                warnings=warnings or None,
                request_id=request_id,
            )
        )

    start_time = time.perf_counter()
    try:
        result, error = add_phase_bulk(
            spec_id=spec_id,
            phase_title=title,
            tasks=tasks,
            phase_description=description,
            phase_purpose=purpose,
            phase_estimated_hours=estimated_hours,
            metadata_defaults=metadata_defaults,
            position=position,
            link_previous=link_previous,
            specs_dir=specs_dir,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected error in phase-add-bulk")
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
        lowered = error.lower()
        if "specification" in lowered and "not found" in lowered:
            return asdict(
                error_response(
                    f"Specification '{spec_id}' not found",
                    error_code=ErrorCode.SPEC_NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation='Verify the spec ID via spec(action="list")',
                    request_id=request_id,
                )
            )
        if "task at index" in lowered:
            return asdict(
                error_response(
                    error,
                    error_code=ErrorCode.VALIDATION_ERROR,
                    error_type=ErrorType.VALIDATION,
                    remediation="Check each task has valid type and title",
                    request_id=request_id,
                )
            )
        return asdict(
            error_response(
                f"Failed to add phase with tasks: {error}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check input values and retry",
                request_id=request_id,
            )
        )

    _metrics.counter(metric_key, labels={"status": "success"})
    return asdict(
        success_response(
            data={"spec_id": spec_id, "dry_run": False, **(result or {})},
            warnings=warnings or None,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )


def _handle_phase_template(*, config: ServerConfig, **payload: Any) -> dict:
    """Handle phase-template action: list/show/apply phase templates."""
    request_id = _request_id()
    action = "phase-template"

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
        if template_name not in PHASE_TEMPLATES:
            return asdict(
                error_response(
                    f"Phase template '{template_name}' not found",
                    error_code=ErrorCode.NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation=f"Use template_action='list' to see available templates. Valid: {', '.join(PHASE_TEMPLATES)}",
                    request_id=request_id,
                )
            )

    data: Dict[str, Any] = {"action": template_action}

    if template_action == "list":
        data["templates"] = [
            {
                "name": "planning",
                "description": "Requirements gathering and initial planning phase",
                "tasks": 2,
                "estimated_hours": 4,
            },
            {
                "name": "implementation",
                "description": "Core development and feature implementation phase",
                "tasks": 2,
                "estimated_hours": 8,
            },
            {
                "name": "testing",
                "description": "Comprehensive testing and quality assurance phase",
                "tasks": 2,
                "estimated_hours": 6,
            },
            {
                "name": "security",
                "description": "Security audit and hardening phase",
                "tasks": 2,
                "estimated_hours": 6,
            },
            {
                "name": "documentation",
                "description": "Technical documentation and knowledge capture phase",
                "tasks": 2,
                "estimated_hours": 4,
            },
        ]
        data["total_count"] = len(data["templates"])
        data["note"] = "All templates include automatic verification scaffolding (run-tests + fidelity)"
        return asdict(success_response(data=data, request_id=request_id))

    elif template_action == "show":
        try:
            template_struct = get_phase_template_structure(template_name)
            data["template_name"] = template_name
            data["content"] = {
                "name": template_name,
                "title": template_struct["title"],
                "description": template_struct["description"],
                "purpose": template_struct["purpose"],
                "estimated_hours": template_struct["estimated_hours"],
                "tasks": template_struct["tasks"],
                "includes_verification": template_struct["includes_verification"],
            }
            data["usage"] = (
                f"Use authoring(action='phase-template', template_action='apply', "
                f"template_name='{template_name}', spec_id='your-spec-id') to apply this template"
            )
            return asdict(success_response(data=data, request_id=request_id))
        except ValueError as exc:
            return asdict(
                error_response(
                    str(exc),
                    error_code=ErrorCode.NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    request_id=request_id,
                )
            )

    else:  # apply
        spec_id = payload.get("spec_id")
        if not isinstance(spec_id, str) or not spec_id.strip():
            return _validation_error(
                field="spec_id",
                action=action,
                message="Provide the target spec_id to apply the template to",
                request_id=request_id,
                code=ErrorCode.MISSING_REQUIRED,
            )
        spec_id = spec_id.strip()

        # Optional parameters for apply
        category = payload.get("category", "implementation")
        if not isinstance(category, str):
            return _validation_error(
                field="category",
                action=action,
                message="Category must be a string",
                request_id=request_id,
            )
        category = category.strip()
        if category and category not in CATEGORIES:
            return _validation_error(
                field="category",
                action=action,
                message=f"Category must be one of: {', '.join(CATEGORIES)}",
                request_id=request_id,
            )

        position = payload.get("position")
        if position is not None:
            if isinstance(position, bool) or not isinstance(position, int):
                return _validation_error(
                    field="position",
                    action=action,
                    message="Position must be an integer",
                    request_id=request_id,
                )
            if position < 0:
                return _validation_error(
                    field="position",
                    action=action,
                    message="Position must be >= 0",
                    request_id=request_id,
                )

        link_previous = payload.get("link_previous", True)
        if not isinstance(link_previous, bool):
            return _validation_error(
                field="link_previous",
                action=action,
                message="Expected a boolean value",
                request_id=request_id,
            )

        dry_run = payload.get("dry_run", False)
        if not isinstance(dry_run, bool):
            return _validation_error(
                field="dry_run",
                action=action,
                message="Expected a boolean value",
                request_id=request_id,
            )

        path = payload.get("path")
        if path is not None and not isinstance(path, str):
            return _validation_error(
                field="path",
                action=action,
                message="Workspace path must be a string",
                request_id=request_id,
            )

        specs_dir = _resolve_specs_dir(config, path)
        if specs_dir is None:
            return _specs_directory_missing_error(request_id)

        audit_log(
            "tool_invocation",
            tool="authoring",
            action=action,
            spec_id=spec_id,
            template_name=template_name,
            dry_run=dry_run,
            link_previous=link_previous,
        )

        metric_key = _metric_name(action)

        if dry_run:
            _metrics.counter(metric_key, labels={"status": "success", "dry_run": "true"})
            template_struct = get_phase_template_structure(template_name, category)
            return asdict(
                success_response(
                    data={
                        "spec_id": spec_id,
                        "template_applied": template_name,
                        "phase_id": "(preview)",
                        "title": template_struct["title"],
                        "tasks_created": [
                            {"task_id": "(preview)", "title": t["title"], "type": "task"}
                            for t in template_struct["tasks"]
                        ],
                        "total_tasks": len(template_struct["tasks"]),
                        "dry_run": True,
                        "note": "Dry run - no changes made. Verification scaffolding will be auto-added.",
                    },
                    request_id=request_id,
                )
            )

        start_time = time.perf_counter()
        try:
            result, error = apply_phase_template(
                spec_id=spec_id,
                template=template_name,
                specs_dir=specs_dir,
                category=category,
                position=position,
                link_previous=link_previous,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Unexpected error in phase-template apply")
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
            lowered = error.lower()
            if "specification" in lowered and "not found" in lowered:
                return asdict(
                    error_response(
                        f"Specification '{spec_id}' not found",
                        error_code=ErrorCode.SPEC_NOT_FOUND,
                        error_type=ErrorType.NOT_FOUND,
                        remediation='Verify the spec ID via spec(action="list")',
                        request_id=request_id,
                    )
                )
            if "invalid phase template" in lowered:
                return asdict(
                    error_response(
                        error,
                        error_code=ErrorCode.VALIDATION_ERROR,
                        error_type=ErrorType.VALIDATION,
                        remediation=f"Valid templates: {', '.join(PHASE_TEMPLATES)}",
                        request_id=request_id,
                    )
                )
            return asdict(
                error_response(
                    f"Failed to apply phase template: {error}",
                    error_code=ErrorCode.INTERNAL_ERROR,
                    error_type=ErrorType.INTERNAL,
                    remediation="Check input values and retry",
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


def _handle_phase_move(*, config: ServerConfig, **payload: Any) -> dict:
    """Handle phase-move action: reorder a phase within spec-root children."""
    request_id = _request_id()
    action = "phase-move"

    spec_id = payload.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec_id parameter",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
            remediation='Use spec(action="list") to find available spec IDs',
        )
    spec_id = spec_id.strip()

    phase_id = payload.get("phase_id")
    if not isinstance(phase_id, str) or not phase_id.strip():
        return _validation_error(
            field="phase_id",
            action=action,
            message="Provide the phase identifier (e.g., phase-1)",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
            remediation="Specify a phase ID like phase-1 or phase-2",
        )
    phase_id = phase_id.strip()

    position = payload.get("position")
    if position is None:
        return _validation_error(
            field="position",
            action=action,
            message="Provide the target position (1-based index)",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
            remediation="Specify position as a positive integer (1 = first)",
        )
    if isinstance(position, bool) or not isinstance(position, int):
        return _validation_error(
            field="position",
            action=action,
            message="Position must be an integer",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
            remediation="Provide position as an integer, e.g. position=2",
        )
    if position < 1:
        return _validation_error(
            field="position",
            action=action,
            message="Position must be a positive integer (1-based)",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
            remediation="Use 1 for first position, 2 for second, etc.",
        )

    link_previous = payload.get("link_previous", True)
    if not isinstance(link_previous, bool):
        return _validation_error(
            field="link_previous",
            action=action,
            message="Expected a boolean value",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
            remediation="Use true or false for link_previous",
        )

    dry_run = payload.get("dry_run", False)
    if not isinstance(dry_run, bool):
        return _validation_error(
            field="dry_run",
            action=action,
            message="Expected a boolean value",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
            remediation="Use true or false for dry_run",
        )

    path = payload.get("path")
    if path is not None and not isinstance(path, str):
        return _validation_error(
            field="path",
            action=action,
            message="Workspace path must be a string",
            request_id=request_id,
            remediation="Provide a valid filesystem path string",
            code=ErrorCode.INVALID_FORMAT,
        )

    specs_dir = _resolve_specs_dir(config, path)
    if specs_dir is None:
        return _specs_directory_missing_error(request_id)

    audit_log(
        "tool_invocation",
        tool="authoring",
        action=action,
        spec_id=spec_id,
        phase_id=phase_id,
        position=position,
        link_previous=link_previous,
        dry_run=dry_run,
    )

    metric_key = _metric_name(action)
    start_time = time.perf_counter()

    try:
        result, error = move_phase(
            spec_id=spec_id,
            phase_id=phase_id,
            position=position,
            link_previous=link_previous,
            dry_run=dry_run,
            specs_dir=specs_dir,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected error moving phase")
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
        lowered = error.lower()
        if "specification" in lowered and "not found" in lowered:
            return asdict(
                error_response(
                    f"Specification '{spec_id}' not found",
                    error_code=ErrorCode.SPEC_NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation='Verify the spec ID via spec(action="list")',
                    request_id=request_id,
                    telemetry={"duration_ms": round(elapsed_ms, 2)},
                )
            )
        if "phase" in lowered and "not found" in lowered:
            return asdict(
                error_response(
                    f"Phase '{phase_id}' not found in spec",
                    error_code=ErrorCode.PHASE_NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation="Confirm the phase exists in the hierarchy",
                    request_id=request_id,
                    telemetry={"duration_ms": round(elapsed_ms, 2)},
                )
            )
        if "not a phase" in lowered:
            return asdict(
                error_response(
                    f"Node '{phase_id}' is not a phase",
                    error_code=ErrorCode.VALIDATION_ERROR,
                    error_type=ErrorType.VALIDATION,
                    remediation="Provide a valid phase ID (e.g., phase-1)",
                    request_id=request_id,
                    telemetry={"duration_ms": round(elapsed_ms, 2)},
                )
            )
        if "invalid position" in lowered or "must be" in lowered:
            return asdict(
                error_response(
                    error,
                    error_code=ErrorCode.VALIDATION_ERROR,
                    error_type=ErrorType.VALIDATION,
                    remediation="Provide a valid 1-based position within range",
                    request_id=request_id,
                    telemetry={"duration_ms": round(elapsed_ms, 2)},
                )
            )
        return asdict(
            error_response(
                f"Failed to move phase: {error}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check input values and retry",
                request_id=request_id,
                telemetry={"duration_ms": round(elapsed_ms, 2)},
            )
        )

    _metrics.counter(metric_key, labels={"status": "success"})
    return asdict(
        success_response(
            data=result or {},
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )


def _handle_phase_remove(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "phase-remove"

    spec_id = payload.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec_id parameter",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    spec_id = spec_id.strip()

    phase_id = payload.get("phase_id")
    if not isinstance(phase_id, str) or not phase_id.strip():
        return _validation_error(
            field="phase_id",
            action=action,
            message="Provide the phase identifier (e.g., phase-1)",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    phase_id = phase_id.strip()

    force = payload.get("force", False)
    if not isinstance(force, bool):
        return _validation_error(
            field="force",
            action=action,
            message="Expected a boolean value",
            request_id=request_id,
        )

    dry_run = payload.get("dry_run", False)
    if not isinstance(dry_run, bool):
        return _validation_error(
            field="dry_run",
            action=action,
            message="Expected a boolean value",
            request_id=request_id,
        )

    path = payload.get("path")
    if path is not None and not isinstance(path, str):
        return _validation_error(
            field="path",
            action=action,
            message="Workspace path must be a string",
            request_id=request_id,
        )

    specs_dir = _resolve_specs_dir(config, path)
    if specs_dir is None:
        return _specs_directory_missing_error(request_id)

    audit_log(
        "tool_invocation",
        tool="authoring",
        action=action,
        spec_id=spec_id,
        phase_id=phase_id,
        force=force,
        dry_run=dry_run,
    )

    metric_key = _metric_name(action)
    if dry_run:
        _metrics.counter(
            metric_key, labels={"status": "success", "force": str(force).lower()}
        )
        return asdict(
            success_response(
                data={
                    "spec_id": spec_id,
                    "phase_id": phase_id,
                    "force": force,
                    "dry_run": True,
                    "note": "Dry run - no changes made",
                },
                request_id=request_id,
            )
        )

    start_time = time.perf_counter()
    try:
        result, error = remove_phase(
            spec_id=spec_id,
            phase_id=phase_id,
            force=force,
            specs_dir=specs_dir,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected error removing phase")
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
        lowered = error.lower()
        if "spec" in lowered and "not found" in lowered:
            return asdict(
                error_response(
                    f"Specification '{spec_id}' not found",
                    error_code=ErrorCode.SPEC_NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation='Verify the spec ID via spec(action="list")',
                    request_id=request_id,
                )
            )
        if "phase" in lowered and "not found" in lowered:
            return asdict(
                error_response(
                    f"Phase '{phase_id}' not found in spec",
                    error_code=ErrorCode.PHASE_NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation="Confirm the phase exists in the hierarchy",
                    request_id=request_id,
                )
            )
        if "not a phase" in lowered:
            return asdict(
                error_response(
                    f"Node '{phase_id}' is not a phase",
                    error_code=ErrorCode.VALIDATION_ERROR,
                    error_type=ErrorType.VALIDATION,
                    remediation="Use task-remove for non-phase nodes",
                    request_id=request_id,
                )
            )
        if "non-completed" in lowered or "has" in lowered and "task" in lowered:
            return asdict(
                error_response(
                    f"Phase '{phase_id}' has non-completed tasks. Use force=True to remove anyway",
                    error_code=ErrorCode.CONFLICT,
                    error_type=ErrorType.CONFLICT,
                    remediation="Set force=True to remove active phases",
                    request_id=request_id,
                )
            )
        return asdict(
            error_response(
                f"Failed to remove phase: {error}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check input values and retry",
                request_id=request_id,
            )
        )

    _metrics.counter(
        metric_key, labels={"status": "success", "force": str(force).lower()}
    )
    return asdict(
        success_response(
            data={"spec_id": spec_id, "dry_run": False, **(result or {})},
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )


def _handle_assumption_add(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "assumption-add"

    spec_id = payload.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec_id parameter",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    spec_id = spec_id.strip()

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return _validation_error(
            field="text",
            action=action,
            message="Provide the assumption text",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    text = text.strip()

    assumption_type = payload.get("assumption_type") or "constraint"
    if assumption_type not in ASSUMPTION_TYPES:
        return _validation_error(
            field="assumption_type",
            action=action,
            message=f"Must be one of: {', '.join(ASSUMPTION_TYPES)}",
            request_id=request_id,
        )

    author = payload.get("author")
    if author is not None and not isinstance(author, str):
        return _validation_error(
            field="author",
            action=action,
            message="Author must be a string",
            request_id=request_id,
        )

    dry_run = payload.get("dry_run", False)
    if not isinstance(dry_run, bool):
        return _validation_error(
            field="dry_run",
            action=action,
            message="Expected a boolean value",
            request_id=request_id,
        )

    path = payload.get("path")
    if path is not None and not isinstance(path, str):
        return _validation_error(
            field="path",
            action=action,
            message="Workspace path must be a string",
            request_id=request_id,
        )

    specs_dir = _resolve_specs_dir(config, path)
    if specs_dir is None:
        return _specs_directory_missing_error(request_id)

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

    spec_id = payload.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec_id parameter",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    spec_id = spec_id.strip()

    assumption_type = payload.get("assumption_type")
    if assumption_type is not None and assumption_type not in ASSUMPTION_TYPES:
        return _validation_error(
            field="assumption_type",
            action=action,
            message=f"Must be one of: {', '.join(ASSUMPTION_TYPES)}",
            request_id=request_id,
        )

    path = payload.get("path")
    if path is not None and not isinstance(path, str):
        return _validation_error(
            field="path",
            action=action,
            message="Workspace path must be a string",
            request_id=request_id,
        )

    specs_dir = _resolve_specs_dir(config, path)
    if specs_dir is None:
        return _specs_directory_missing_error(request_id)

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

    spec_id = payload.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            field="spec_id",
            action=action,
            message="Provide a non-empty spec_id parameter",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    spec_id = spec_id.strip()

    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        return _validation_error(
            field="version",
            action=action,
            message="Provide the revision version (e.g., 1.1)",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    version = version.strip()

    changes = payload.get("changes")
    if not isinstance(changes, str) or not changes.strip():
        return _validation_error(
            field="changes",
            action=action,
            message="Provide a summary of changes",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    changes = changes.strip()

    author = payload.get("author")
    if author is not None and not isinstance(author, str):
        return _validation_error(
            field="author",
            action=action,
            message="Author must be a string",
            request_id=request_id,
        )

    dry_run = payload.get("dry_run", False)
    if not isinstance(dry_run, bool):
        return _validation_error(
            field="dry_run",
            action=action,
            message="Expected a boolean value",
            request_id=request_id,
        )

    path = payload.get("path")
    if path is not None and not isinstance(path, str):
        return _validation_error(
            field="path",
            action=action,
            message="Workspace path must be a string",
            request_id=request_id,
        )

    specs_dir = _resolve_specs_dir(config, path)
    if specs_dir is None:
        return _specs_directory_missing_error(request_id)

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


# Validation constants for intake
_INTAKE_TITLE_MAX_LEN = 140
_INTAKE_DESC_MAX_LEN = 2000
_INTAKE_TAG_MAX_LEN = 32
_INTAKE_TAG_MAX_COUNT = 20
_INTAKE_SOURCE_MAX_LEN = 100
_INTAKE_REQUESTER_MAX_LEN = 100
_INTAKE_IDEMPOTENCY_KEY_MAX_LEN = 64
_INTAKE_PRIORITY_VALUES = ("p0", "p1", "p2", "p3", "p4")
_INTAKE_TAG_PATTERN = "^[a-z0-9_-]+$"
_TAG_REGEX = re.compile(_INTAKE_TAG_PATTERN)


def _handle_intake_add(*, config: ServerConfig, **payload: Any) -> dict:
    """Add a new intake item to the bikelane queue."""
    request_id = _request_id()
    action = "intake-add"

    # Check feature flag
    blocked = _intake_feature_flag_blocked(request_id)
    if blocked:
        return blocked

    # Validate title (required, 1-140 chars)
    title = payload.get("title")
    if not isinstance(title, str) or not title.strip():
        return _validation_error(
            field="title",
            action=action,
            message="Provide a non-empty title (1-140 characters)",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    title = title.strip()
    if len(title) > _INTAKE_TITLE_MAX_LEN:
        return _validation_error(
            field="title",
            action=action,
            message=f"Title exceeds maximum length of {_INTAKE_TITLE_MAX_LEN} characters",
            request_id=request_id,
            code=ErrorCode.VALIDATION_ERROR,
            remediation=f"Shorten title to {_INTAKE_TITLE_MAX_LEN} characters or less",
        )

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
    priority = payload.get("priority", "p2")
    if not isinstance(priority, str):
        return _validation_error(
            field="priority",
            action=action,
            message="Priority must be a string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )
    priority = priority.strip().lower()
    if priority not in _INTAKE_PRIORITY_VALUES:
        return _validation_error(
            field="priority",
            action=action,
            message=f"Priority must be one of: {', '.join(_INTAKE_PRIORITY_VALUES)}",
            request_id=request_id,
            code=ErrorCode.VALIDATION_ERROR,
            remediation="Use p0 (highest) through p4 (lowest), default is p2",
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

    # Validate dry_run
    dry_run = payload.get("dry_run", False)
    if not isinstance(dry_run, bool):
        return _validation_error(
            field="dry_run",
            action=action,
            message="dry_run must be a boolean",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    # Validate path
    path = payload.get("path")
    if path is not None and not isinstance(path, str):
        return _validation_error(
            field="path",
            action=action,
            message="path must be a string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    # Resolve specs directory
    specs_dir = _resolve_specs_dir(config, path)
    if specs_dir is None:
        return _specs_directory_missing_error(request_id)

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
        # Get bikelane_dir from config (allows customization via TOML or env var)
        bikelane_dir = config.get_bikelane_dir(specs_dir)
        store = IntakeStore(specs_dir, bikelane_dir=bikelane_dir)
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


# Intake list constants (from intake.py)
_INTAKE_LIST_DEFAULT_LIMIT = 50
_INTAKE_LIST_MAX_LIMIT = 200


def _handle_intake_list(*, config: ServerConfig, **payload: Any) -> dict:
    """List intake items with status='new' in FIFO order with pagination."""
    request_id = _request_id()
    action = "intake-list"

    # Check feature flag
    blocked = _intake_feature_flag_blocked(request_id)
    if blocked:
        return blocked

    # Validate limit (optional, default 50, range 1-200)
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

    # Validate cursor (optional string)
    cursor = payload.get("cursor")
    if cursor is not None:
        if not isinstance(cursor, str):
            return _validation_error(
                field="cursor",
                action=action,
                message="cursor must be a string",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )
        cursor = cursor.strip() or None

    # Validate path (optional workspace override)
    path = payload.get("path")
    if path is not None and not isinstance(path, str):
        return _validation_error(
            field="path",
            action=action,
            message="path must be a string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    # Resolve specs directory
    specs_dir = _resolve_specs_dir(config, path)
    if specs_dir is None:
        return _specs_directory_missing_error(request_id)

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
        # Get bikelane_dir from config (allows customization via TOML or env var)
        bikelane_dir = config.get_bikelane_dir(specs_dir)
        store = IntakeStore(specs_dir, bikelane_dir=bikelane_dir)
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


# Intake dismiss constants
_INTAKE_DISMISS_REASON_MAX_LEN = 200


def _handle_intake_dismiss(*, config: ServerConfig, **payload: Any) -> dict:
    """Dismiss an intake item by changing its status to 'dismissed'."""
    request_id = _request_id()
    action = "intake-dismiss"

    # Check feature flag
    blocked = _intake_feature_flag_blocked(request_id)
    if blocked:
        return blocked

    # Validate intake_id (required, must match pattern)
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

    # Validate reason (optional, max 200 chars)
    reason = payload.get("reason")
    if reason is not None:
        if not isinstance(reason, str):
            return _validation_error(
                field="reason",
                action=action,
                message="reason must be a string",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )
        reason = reason.strip() or None
        if reason and len(reason) > _INTAKE_DISMISS_REASON_MAX_LEN:
            return _validation_error(
                field="reason",
                action=action,
                message=f"reason exceeds maximum length of {_INTAKE_DISMISS_REASON_MAX_LEN} characters",
                request_id=request_id,
                code=ErrorCode.VALIDATION_ERROR,
                remediation=f"Shorten reason to {_INTAKE_DISMISS_REASON_MAX_LEN} characters or less",
            )

    # Validate dry_run
    dry_run = payload.get("dry_run", False)
    if not isinstance(dry_run, bool):
        return _validation_error(
            field="dry_run",
            action=action,
            message="dry_run must be a boolean",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    # Validate path
    path = payload.get("path")
    if path is not None and not isinstance(path, str):
        return _validation_error(
            field="path",
            action=action,
            message="path must be a string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    # Resolve specs directory
    specs_dir = _resolve_specs_dir(config, path)
    if specs_dir is None:
        return _specs_directory_missing_error(request_id)

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
        # Get bikelane_dir from config (allows customization via TOML or env var)
        bikelane_dir = config.get_bikelane_dir(specs_dir)
        store = IntakeStore(specs_dir, bikelane_dir=bikelane_dir)
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


_AUTHORING_ROUTER = ActionRouter(
    tool_name="authoring",
    actions=[
        ActionDefinition(
            name="spec-create",
            handler=_handle_spec_create,
            summary=_ACTION_SUMMARY["spec-create"],
            aliases=("spec_create",),
        ),
        ActionDefinition(
            name="spec-template",
            handler=_handle_spec_template,
            summary=_ACTION_SUMMARY["spec-template"],
            aliases=("spec_template",),
        ),
        ActionDefinition(
            name="spec-update-frontmatter",
            handler=_handle_spec_update_frontmatter,
            summary=_ACTION_SUMMARY["spec-update-frontmatter"],
            aliases=("spec_update_frontmatter",),
        ),
        ActionDefinition(
            name="spec-find-replace",
            handler=_handle_spec_find_replace,
            summary=_ACTION_SUMMARY["spec-find-replace"],
            aliases=("spec_find_replace",),
        ),
        ActionDefinition(
            name="spec-rollback",
            handler=_handle_spec_rollback,
            summary=_ACTION_SUMMARY["spec-rollback"],
            aliases=("spec_rollback",),
        ),
        ActionDefinition(
            name="phase-add",
            handler=_handle_phase_add,
            summary=_ACTION_SUMMARY["phase-add"],
            aliases=("phase_add",),
        ),
        ActionDefinition(
            name="phase-add-bulk",
            handler=_handle_phase_add_bulk,
            summary=_ACTION_SUMMARY["phase-add-bulk"],
            aliases=("phase_add_bulk",),
        ),
        ActionDefinition(
            name="phase-template",
            handler=_handle_phase_template,
            summary=_ACTION_SUMMARY["phase-template"],
            aliases=("phase_template",),
        ),
        ActionDefinition(
            name="phase-move",
            handler=_handle_phase_move,
            summary=_ACTION_SUMMARY["phase-move"],
            aliases=("phase_move",),
        ),
        ActionDefinition(
            name="phase-update-metadata",
            handler=_handle_phase_update_metadata,
            summary=_ACTION_SUMMARY["phase-update-metadata"],
            aliases=("phase_update_metadata",),
        ),
        ActionDefinition(
            name="phase-remove",
            handler=_handle_phase_remove,
            summary=_ACTION_SUMMARY["phase-remove"],
            aliases=("phase_remove",),
        ),
        ActionDefinition(
            name="assumption-add",
            handler=_handle_assumption_add,
            summary=_ACTION_SUMMARY["assumption-add"],
            aliases=("assumption_add",),
        ),
        ActionDefinition(
            name="assumption-list",
            handler=_handle_assumption_list,
            summary=_ACTION_SUMMARY["assumption-list"],
            aliases=("assumption_list",),
        ),
        ActionDefinition(
            name="revision-add",
            handler=_handle_revision_add,
            summary=_ACTION_SUMMARY["revision-add"],
            aliases=("revision_add",),
        ),
        ActionDefinition(
            name="intake-add",
            handler=_handle_intake_add,
            summary=_ACTION_SUMMARY["intake-add"],
            aliases=("intake_add",),
        ),
        ActionDefinition(
            name="intake-list",
            handler=_handle_intake_list,
            summary=_ACTION_SUMMARY["intake-list"],
            aliases=("intake_list",),
        ),
        ActionDefinition(
            name="intake-dismiss",
            handler=_handle_intake_dismiss,
            summary=_ACTION_SUMMARY["intake-dismiss"],
            aliases=("intake_dismiss",),
        ),
    ],
)


def _dispatch_authoring_action(
    *, action: str, payload: Dict[str, Any], config: ServerConfig
) -> dict:
    try:
        return _AUTHORING_ROUTER.dispatch(action=action, config=config, **payload)
    except ActionRouterError as exc:
        request_id = _request_id()
        allowed = ", ".join(exc.allowed_actions)
        return asdict(
            error_response(
                f"Unsupported authoring action '{action}'. Allowed actions: {allowed}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {allowed}",
                request_id=request_id,
            )
        )


def register_unified_authoring_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the consolidated authoring tool."""

    @canonical_tool(
        mcp,
        canonical_name="authoring",
    )
    @mcp_tool(tool_name="authoring", emit_metrics=True, audit=True)
    def authoring(
        action: str,
        spec_id: Optional[str] = None,
        name: Optional[str] = None,
        template: Optional[str] = None,
        category: Optional[str] = None,
        mission: Optional[str] = None,
        template_action: Optional[str] = None,
        template_name: Optional[str] = None,
        key: Optional[str] = None,
        value: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        purpose: Optional[str] = None,
        estimated_hours: Optional[float] = None,
        position: Optional[int] = None,
        link_previous: bool = True,
        phase_id: Optional[str] = None,
        force: bool = False,
        text: Optional[str] = None,
        assumption_type: Optional[str] = None,
        author: Optional[str] = None,
        version: Optional[str] = None,
        changes: Optional[str] = None,
        tasks: Optional[List[Dict[str, Any]]] = None,
        phase: Optional[Dict[str, Any]] = None,
        metadata_defaults: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
        path: Optional[str] = None,
        # spec-find-replace parameters
        find: Optional[str] = None,
        replace: Optional[str] = None,
        scope: Optional[str] = None,
        use_regex: bool = False,
        case_sensitive: bool = True,
        # intake parameters
        priority: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        requester: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> dict:
        """Execute authoring workflows via the action router."""

        payload = {
            "spec_id": spec_id,
            "name": name,
            "template": template,
            "category": category,
            "mission": mission,
            "template_action": template_action,
            "template_name": template_name,
            "key": key,
            "value": value,
            "title": title,
            "description": description,
            "purpose": purpose,
            "estimated_hours": estimated_hours,
            "position": position,
            "link_previous": link_previous,
            "phase_id": phase_id,
            "force": force,
            "text": text,
            "assumption_type": assumption_type,
            "author": author,
            "version": version,
            "changes": changes,
            "tasks": tasks,
            "phase": phase,
            "metadata_defaults": metadata_defaults,
            "dry_run": dry_run,
            "path": path,
            # spec-find-replace parameters
            "find": find,
            "replace": replace,
            "scope": scope,
            "use_regex": use_regex,
            "case_sensitive": case_sensitive,
            # intake parameters
            "priority": priority,
            "tags": tags,
            "source": source,
            "requester": requester,
            "idempotency_key": idempotency_key,
        }
        return _dispatch_authoring_action(action=action, payload=payload, config=config)

    logger.debug("Registered unified authoring tool")


__all__ = [
    "register_unified_authoring_tool",
]
