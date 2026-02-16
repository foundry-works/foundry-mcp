"""Phase management action handlers: phase-add, phase-update-metadata, phase-add-bulk, phase-template, phase-move, phase-remove."""

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
    CATEGORIES,
    PHASE_TEMPLATES,
    add_phase,
    add_phase_bulk,
    apply_phase_template,
    get_phase_template_structure,
    move_phase,
    remove_phase,
    update_phase_metadata,
)
from foundry_mcp.core.task import TASK_TYPES

from foundry_mcp.tools.unified.authoring_handlers._helpers import (
    _metric_name,
    _metrics,
    _phase_exists,
    _request_id,
    _resolve_specs_dir,
    _validation_error,
    logger,
)
from foundry_mcp.tools.unified.param_schema import AtLeastOne, Bool, Num, Str, validate_payload

_PHASE_ADD_SCHEMA = {
    "spec_id": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED,
                   remediation="Pass the spec identifier to authoring"),
    "title": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED,
                 remediation="Include a descriptive title for the new phase"),
    "description": Str(),
    "purpose": Str(),
    "estimated_hours": Num(min_val=0, remediation="Set hours to zero or greater"),
    "position": Num(integer_only=True, min_val=0),
    "link_previous": Bool(default=True),
    "dry_run": Bool(default=False),
    "path": Str(),
}

_PHASE_UPDATE_METADATA_SCHEMA = {
    "spec_id": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED,
                   remediation="Pass the spec identifier to authoring"),
    "phase_id": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED,
                    remediation="Pass the phase identifier (e.g., 'phase-1')"),
    "estimated_hours": Num(min_val=0, remediation="Set hours to zero or greater"),
    "description": Str(),
    "purpose": Str(),
    "dry_run": Bool(default=False),
    "path": Str(),
}

_PHASE_ADD_BULK_SCHEMA = {
    "spec_id": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED,
                   remediation="Pass the spec identifier to authoring"),
    "position": Num(integer_only=True, min_val=0),
    "link_previous": Bool(default=True),
    "dry_run": Bool(default=False),
    "path": Str(),
}

_PHASE_MOVE_SCHEMA = {
    "spec_id": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED,
                   remediation='Use spec(action="list") to find available spec IDs'),
    "phase_id": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED,
                    remediation="Specify a phase ID like phase-1 or phase-2"),
    # position validated imperatively — needs MISSING_REQUIRED vs INVALID_FORMAT
    "link_previous": Bool(default=True, error_code=ErrorCode.INVALID_FORMAT),
    "dry_run": Bool(default=False, error_code=ErrorCode.INVALID_FORMAT),
    "path": Str(error_code=ErrorCode.INVALID_FORMAT),
}

_PHASE_REMOVE_SCHEMA = {
    "spec_id": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
    "phase_id": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
    "force": Bool(default=False),
    "dry_run": Bool(default=False),
    "path": Str(),
}


def _handle_phase_add(*, config: ServerConfig, **payload: Any) -> dict:
    request_id = _request_id()
    action = "phase-add"

    err = validate_payload(payload, _PHASE_ADD_SCHEMA,
                           tool_name="authoring", action=action,
                           request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    title = payload["title"]
    description = payload.get("description")
    purpose = payload.get("purpose")
    estimated_hours = payload.get("estimated_hours")
    position = payload.get("position")
    link_previous = payload["link_previous"]
    dry_run = payload["dry_run"]
    path = payload.get("path")

    specs_dir, specs_err = _resolve_specs_dir(config, path)
    if specs_err:
        return specs_err

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

    err = validate_payload(payload, _PHASE_UPDATE_METADATA_SCHEMA,
                           tool_name="authoring", action=action,
                           request_id=request_id,
                           cross_field_rules=[
                               AtLeastOne(
                                   fields=("estimated_hours", "description", "purpose"),
                                   error_code=ErrorCode.VALIDATION_ERROR,
                                   remediation="Include estimated_hours, description, or purpose",
                               ),
                           ])
    if err:
        return err

    spec_id = payload["spec_id"]
    phase_id = payload["phase_id"]
    estimated_hours = payload.get("estimated_hours")
    description = payload.get("description")
    purpose = payload.get("purpose")
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

    err = validate_payload(payload, _PHASE_ADD_BULK_SCHEMA,
                           tool_name="authoring", action=action,
                           request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    position = payload.get("position")
    link_previous = payload["link_previous"]
    dry_run = payload["dry_run"]
    path = payload.get("path")

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
    valid_task_types = set(TASK_TYPES)  # task, subtask, verify, research
    valid_blocking_modes = {"none", "soft", "hard"}
    valid_research_types = {"chat", "consensus", "thinkdeep", "ideate", "deep-research"}
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
                message=f"Task type must be one of: {', '.join(sorted(valid_task_types))}",
                remediation=f"Set type to one of: {', '.join(sorted(valid_task_types))}",
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

        # Validate research-specific parameters when type is "research"
        if task_type == "research":
            blocking_mode = task_def.get("blocking_mode")
            if blocking_mode is not None and blocking_mode not in valid_blocking_modes:
                return _validation_error(
                    field=f"tasks[{idx}].blocking_mode",
                    action=action,
                    message=f"blocking_mode must be one of: {', '.join(sorted(valid_blocking_modes))}",
                    remediation="Set blocking_mode to 'none', 'soft', or 'hard'",
                    request_id=request_id,
                )

            research_type = task_def.get("research_type")
            if research_type is not None and research_type not in valid_research_types:
                return _validation_error(
                    field=f"tasks[{idx}].research_type",
                    action=action,
                    message=f"research_type must be one of: {', '.join(sorted(valid_research_types))}",
                    remediation="Set research_type to 'chat', 'consensus', 'thinkdeep', 'ideate', or 'deep-research'",
                    request_id=request_id,
                )

            query = task_def.get("query")
            if query is not None and not isinstance(query, str):
                return _validation_error(
                    field=f"tasks[{idx}].query",
                    action=action,
                    message="query must be a string",
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

    specs_dir, specs_err = _resolve_specs_dir(config, path)
    if specs_err:
        return specs_err

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

        specs_dir, specs_err = _resolve_specs_dir(config, path)
        if specs_err:
            return specs_err

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

    err = validate_payload(payload, _PHASE_MOVE_SCHEMA,
                           tool_name="authoring", action=action,
                           request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    phase_id = payload["phase_id"]
    link_previous = payload["link_previous"]
    dry_run = payload["dry_run"]
    path = payload.get("path")

    # Position requires MISSING_REQUIRED vs INVALID_FORMAT distinction
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

    specs_dir, specs_err = _resolve_specs_dir(config, path)
    if specs_err:
        return specs_err

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

    err = validate_payload(payload, _PHASE_REMOVE_SCHEMA,
                           tool_name="authoring", action=action,
                           request_id=request_id)
    if err:
        return err

    spec_id = payload["spec_id"]
    phase_id = payload["phase_id"]
    force = payload["force"]
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
