"""Unified authoring tool backed by ActionRouter and shared validation."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.context import generate_correlation_id, get_correlation_id
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
    TEMPLATES,
    add_assumption,
    add_phase,
    add_revision,
    create_spec,
    find_specs_directory,
    list_assumptions,
    load_spec,
    remove_phase,
    update_frontmatter,
)
from foundry_mcp.tools.unified.router import (
    ActionDefinition,
    ActionRouter,
    ActionRouterError,
)

logger = logging.getLogger(__name__)
_metrics = get_metrics()

_ACTION_SUMMARY = {
    "spec-create": "Scaffold a new SDD specification",
    "spec-template": "List/show/apply spec templates",
    "spec-update-frontmatter": "Update a top-level metadata field",
    "phase-add": "Add a new phase under spec-root with verification scaffolding",
    "phase-remove": "Remove an existing phase (and optionally dependents)",
    "assumption-add": "Append an assumption entry to spec metadata",
    "assumption-list": "List recorded assumptions for a spec",
    "revision-add": "Record a revision entry in the spec history",
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
                    "name": name.strip(),
                    "template": template,
                    "category": category,
                    "dry_run": True,
                    "note": "Dry run - no changes made",
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
            name="phase-add",
            handler=_handle_phase_add,
            summary=_ACTION_SUMMARY["phase-add"],
            aliases=("phase_add",),
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
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """Execute authoring workflows via the action router."""

        payload = {
            "spec_id": spec_id,
            "name": name,
            "template": template,
            "category": category,
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
            "dry_run": dry_run,
            "path": path,
        }
        return _dispatch_authoring_action(action=action, payload=payload, config=config)

    logger.debug("Registered unified authoring tool")


__all__ = [
    "register_unified_authoring_tool",
]
