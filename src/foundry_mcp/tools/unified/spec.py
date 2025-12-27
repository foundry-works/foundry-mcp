"""Unified spec tooling with action routing.

This router consolidates the high-volume spec-* tool family behind a single
`spec(action=...)` surface.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import audit_log, get_metrics, mcp_tool
from foundry_mcp.core.pagination import (
    CursorError,
    decode_cursor,
    encode_cursor,
    normalize_page_size,
)
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
)
from foundry_mcp.core.spec import (
    TEMPLATES,
    TEMPLATE_DESCRIPTIONS,
    check_spec_completeness,
    detect_duplicate_tasks,
    diff_specs,
    find_spec_file,
    find_specs_directory,
    list_spec_backups,
    list_specs,
    load_spec,
)
from foundry_mcp.core.validation import (
    VALID_NODE_TYPES,
    VALID_STATUSES,
    VALID_TASK_CATEGORIES,
    VALID_VERIFICATION_TYPES,
    apply_fixes,
    calculate_stats,
    get_fix_actions,
    validate_spec,
)
from foundry_mcp.core.journal import (
    VALID_BLOCKER_TYPES,
    VALID_ENTRY_TYPES,
)
from foundry_mcp.core.lifecycle import VALID_FOLDERS
from foundry_mcp.tools.unified.router import (
    ActionDefinition,
    ActionRouter,
    ActionRouterError,
)

logger = logging.getLogger(__name__)
_metrics = get_metrics()

_DEFAULT_PAGE_SIZE = 100
_MAX_PAGE_SIZE = 1000


def _resolve_specs_dir(
    config: ServerConfig, workspace: Optional[str]
) -> Optional[Path]:
    if workspace:
        return find_specs_directory(workspace)
    return config.specs_dir or find_specs_directory()


def _handle_find(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    spec_id = payload.get("spec_id")
    workspace = payload.get("workspace")

    if not isinstance(spec_id, str) or not spec_id.strip():
        return asdict(
            error_response(
                "spec_id is required",
                error_code=ErrorCode.MISSING_REQUIRED,
                error_type=ErrorType.VALIDATION,
                remediation="Provide a spec_id parameter",
            )
        )

    specs_dir = _resolve_specs_dir(config, workspace)
    if not specs_dir:
        return asdict(
            error_response(
                "No specs directory found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Ensure you're in a project with a specs/ directory or pass workspace.",
                details={"workspace": workspace},
            )
        )

    spec_file = find_spec_file(spec_id, specs_dir)
    if spec_file:
        return asdict(
            success_response(
                found=True,
                spec_id=spec_id,
                path=str(spec_file),
                status_folder=spec_file.parent.name,
            )
        )

    return asdict(success_response(found=False, spec_id=spec_id))


def _handle_get(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    """Return raw spec JSON content in minified form."""
    import json as _json

    spec_id = payload.get("spec_id")
    workspace = payload.get("workspace")

    if not isinstance(spec_id, str) or not spec_id.strip():
        return asdict(
            error_response(
                "spec_id is required",
                error_code=ErrorCode.MISSING_REQUIRED,
                error_type=ErrorType.VALIDATION,
                remediation="Provide a spec_id parameter",
            )
        )

    specs_dir = _resolve_specs_dir(config, workspace)
    if not specs_dir:
        return asdict(
            error_response(
                "No specs directory found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Ensure you're in a project with a specs/ directory or pass workspace.",
                details={"workspace": workspace},
            )
        )

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return asdict(
            error_response(
                f"Spec not found: {spec_id}",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation=f"Verify the spec_id exists. Use spec(action='list') to see available specs.",
                details={"spec_id": spec_id},
            )
        )

    # Return minified JSON string to minimize token usage
    minified_spec = _json.dumps(spec_data, separators=(",", ":"))
    return asdict(success_response(spec=minified_spec))


def _handle_list(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    status = payload.get("status", "all")
    include_progress = payload.get("include_progress", True)
    cursor = payload.get("cursor")
    limit = payload.get("limit")
    workspace = payload.get("workspace")

    specs_dir = _resolve_specs_dir(config, workspace)
    if not specs_dir:
        return asdict(
            error_response(
                "No specs directory found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Ensure you're in a project with a specs/ directory or pass workspace.",
                details={"workspace": workspace},
            )
        )

    page_size = normalize_page_size(
        limit, default=_DEFAULT_PAGE_SIZE, maximum=_MAX_PAGE_SIZE
    )

    start_after_id = None
    if cursor:
        try:
            cursor_data = decode_cursor(cursor)
            start_after_id = cursor_data.get("last_id")
        except CursorError as exc:
            return asdict(
                error_response(
                    f"Invalid pagination cursor: {exc}",
                    error_code=ErrorCode.INVALID_FORMAT,
                    error_type=ErrorType.VALIDATION,
                    remediation="Use the cursor value returned by the previous spec(action=list) call.",
                )
            )

    filter_status = None if status == "all" else status
    all_specs = list_specs(specs_dir=specs_dir, status=filter_status)
    all_specs.sort(key=lambda entry: entry.get("spec_id", ""))

    if not include_progress:
        for entry in all_specs:
            entry.pop("total_tasks", None)
            entry.pop("completed_tasks", None)
            entry.pop("progress_percentage", None)

    if start_after_id:
        start_index = 0
        for idx, entry in enumerate(all_specs):
            if entry.get("spec_id") == start_after_id:
                start_index = idx + 1
                break
        all_specs = all_specs[start_index:]

    page_specs = all_specs[: page_size + 1]
    has_more = len(page_specs) > page_size
    if has_more:
        page_specs = page_specs[:page_size]

    next_cursor = None
    if has_more and page_specs:
        next_cursor = encode_cursor({"last_id": page_specs[-1].get("spec_id")})

    return asdict(
        success_response(
            specs=page_specs,
            count=len(page_specs),
            pagination={
                "cursor": next_cursor,
                "has_more": has_more,
                "page_size": page_size,
            },
        )
    )


def _handle_validate(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    spec_id = payload.get("spec_id")
    workspace = payload.get("workspace")

    if not isinstance(spec_id, str) or not spec_id.strip():
        return asdict(
            error_response(
                "spec_id is required",
                error_code=ErrorCode.MISSING_REQUIRED,
                error_type=ErrorType.VALIDATION,
            )
        )

    specs_dir = _resolve_specs_dir(config, workspace)
    if not specs_dir:
        return asdict(
            error_response(
                "No specs directory found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Ensure you're in a project with a specs/ directory or pass workspace.",
                details={"workspace": workspace},
            )
        )

    spec_data = load_spec(spec_id, specs_dir)
    if not spec_data:
        return asdict(
            error_response(
                f"Spec not found: {spec_id}",
                error_code=ErrorCode.SPEC_NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation='Verify the spec ID exists using spec(action="list").',
                details={"spec_id": spec_id},
            )
        )

    result = validate_spec(spec_data)
    diagnostics = [
        {
            "code": diag.code,
            "message": diag.message,
            "severity": diag.severity,
            "category": diag.category,
            "location": diag.location,
            "suggested_fix": diag.suggested_fix,
            "auto_fixable": diag.auto_fixable,
        }
        for diag in result.diagnostics
    ]

    return asdict(
        success_response(
            spec_id=result.spec_id,
            is_valid=result.is_valid,
            error_count=result.error_count,
            warning_count=result.warning_count,
            info_count=result.info_count,
            diagnostics=diagnostics,
        )
    )


def _handle_fix(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    spec_id = payload.get("spec_id")

    dry_run_value = payload.get("dry_run", False)
    if dry_run_value is not None and not isinstance(dry_run_value, bool):
        return asdict(
            error_response(
                "dry_run must be a boolean",
                error_code=ErrorCode.INVALID_FORMAT,
                error_type=ErrorType.VALIDATION,
                remediation="Provide dry_run=true|false",
                details={"field": "dry_run"},
            )
        )
    dry_run = dry_run_value if isinstance(dry_run_value, bool) else False

    create_backup_value = payload.get("create_backup", True)
    if create_backup_value is not None and not isinstance(create_backup_value, bool):
        return asdict(
            error_response(
                "create_backup must be a boolean",
                error_code=ErrorCode.INVALID_FORMAT,
                error_type=ErrorType.VALIDATION,
                remediation="Provide create_backup=true|false",
                details={"field": "create_backup"},
            )
        )
    create_backup = (
        create_backup_value if isinstance(create_backup_value, bool) else True
    )

    workspace = payload.get("workspace")

    if not isinstance(spec_id, str) or not spec_id.strip():
        return asdict(
            error_response(
                "spec_id is required",
                error_code=ErrorCode.MISSING_REQUIRED,
                error_type=ErrorType.VALIDATION,
            )
        )

    specs_dir = _resolve_specs_dir(config, workspace)
    if not specs_dir:
        return asdict(
            error_response(
                "No specs directory found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Ensure you're in a project with a specs/ directory or pass workspace.",
                details={"workspace": workspace},
            )
        )

    spec_path = find_spec_file(spec_id, specs_dir)
    if not spec_path:
        return asdict(
            error_response(
                f"Spec not found: {spec_id}",
                error_code=ErrorCode.SPEC_NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation='Verify the spec ID exists using spec(action="list").',
                details={"spec_id": spec_id},
            )
        )

    spec_data = load_spec(spec_id, specs_dir)
    if not spec_data:
        return asdict(
            error_response(
                f"Failed to load spec: {spec_id}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check spec JSON validity and retry.",
                details={"spec_id": spec_id},
            )
        )

    validation_result = validate_spec(spec_data)
    actions = get_fix_actions(validation_result, spec_data)

    if not actions:
        return asdict(
            success_response(
                spec_id=spec_id,
                applied_count=0,
                skipped_count=0,
                message="No auto-fixable issues found",
            )
        )

    report = apply_fixes(
        actions, str(spec_path), dry_run=dry_run, create_backup=create_backup
    )

    applied_actions = [
        {
            "id": action.id,
            "description": action.description,
            "category": action.category,
        }
        for action in report.applied_actions
    ]
    skipped_actions = [
        {
            "id": action.id,
            "description": action.description,
            "category": action.category,
        }
        for action in report.skipped_actions
    ]

    return asdict(
        success_response(
            spec_id=spec_id,
            dry_run=dry_run,
            applied_count=len(report.applied_actions),
            skipped_count=len(report.skipped_actions),
            applied_actions=applied_actions,
            skipped_actions=skipped_actions,
            backup_path=report.backup_path,
        )
    )


def _handle_stats(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    spec_id = payload.get("spec_id")
    workspace = payload.get("workspace")

    if not isinstance(spec_id, str) or not spec_id.strip():
        return asdict(
            error_response(
                "spec_id is required",
                error_code=ErrorCode.MISSING_REQUIRED,
                error_type=ErrorType.VALIDATION,
            )
        )

    specs_dir = _resolve_specs_dir(config, workspace)
    if not specs_dir:
        return asdict(
            error_response(
                "No specs directory found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Ensure you're in a project with a specs/ directory or pass workspace.",
                details={"workspace": workspace},
            )
        )

    spec_path = find_spec_file(spec_id, specs_dir)
    if not spec_path:
        return asdict(
            error_response(
                f"Spec not found: {spec_id}",
                error_code=ErrorCode.SPEC_NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation='Verify the spec ID exists using spec(action="list").',
                details={"spec_id": spec_id},
            )
        )

    spec_data = load_spec(spec_id, specs_dir)
    if not spec_data:
        return asdict(
            error_response(
                f"Failed to load spec: {spec_id}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check spec JSON validity and retry.",
                details={"spec_id": spec_id},
            )
        )

    stats = calculate_stats(spec_data, str(spec_path))
    return asdict(
        success_response(
            spec_id=stats.spec_id,
            title=stats.title,
            version=stats.version,
            status=stats.status,
            totals=stats.totals,
            status_counts=stats.status_counts,
            max_depth=stats.max_depth,
            avg_tasks_per_phase=stats.avg_tasks_per_phase,
            verification_coverage=stats.verification_coverage,
            progress=stats.progress,
            file_size_kb=stats.file_size_kb,
        )
    )


def _handle_validate_fix(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    spec_id = payload.get("spec_id")

    auto_fix_value = payload.get("auto_fix", True)
    if auto_fix_value is not None and not isinstance(auto_fix_value, bool):
        return asdict(
            error_response(
                "auto_fix must be a boolean",
                error_code=ErrorCode.INVALID_FORMAT,
                error_type=ErrorType.VALIDATION,
                remediation="Provide auto_fix=true|false",
                details={"field": "auto_fix"},
            )
        )
    auto_fix = auto_fix_value if isinstance(auto_fix_value, bool) else True

    workspace = payload.get("workspace")

    if not isinstance(spec_id, str) or not spec_id.strip():
        return asdict(
            error_response(
                "spec_id is required",
                error_code=ErrorCode.MISSING_REQUIRED,
                error_type=ErrorType.VALIDATION,
            )
        )

    specs_dir = _resolve_specs_dir(config, workspace)
    if not specs_dir:
        return asdict(
            error_response(
                "No specs directory found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Ensure you're in a project with a specs/ directory or pass workspace.",
                details={"workspace": workspace},
            )
        )

    spec_path = find_spec_file(spec_id, specs_dir)
    if not spec_path:
        return asdict(
            error_response(
                f"Spec not found: {spec_id}",
                error_code=ErrorCode.SPEC_NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation='Verify the spec ID exists using spec(action="list").',
                details={"spec_id": spec_id},
            )
        )

    spec_data = load_spec(spec_id, specs_dir)
    if not spec_data:
        return asdict(
            error_response(
                f"Failed to load spec: {spec_id}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check spec JSON validity and retry.",
                details={"spec_id": spec_id},
            )
        )

    result = validate_spec(spec_data)
    response_data: Dict[str, Any] = {
        "spec_id": result.spec_id,
        "is_valid": result.is_valid,
        "error_count": result.error_count,
        "warning_count": result.warning_count,
    }

    if auto_fix and not result.is_valid:
        actions = get_fix_actions(result, spec_data)
        if actions:
            report = apply_fixes(
                actions, str(spec_path), dry_run=False, create_backup=True
            )
            response_data["fixes_applied"] = len(report.applied_actions)
            response_data["backup_path"] = report.backup_path

            post_spec = load_spec(spec_id, specs_dir)
            if post_spec:
                post_result = validate_spec(post_spec)
                response_data["post_fix_is_valid"] = post_result.is_valid
                response_data["post_fix_error_count"] = post_result.error_count
        else:
            response_data["fixes_applied"] = 0
            response_data["message"] = "No auto-fixable issues found"
    else:
        response_data["fixes_applied"] = 0

    response_data["diagnostics"] = [
        {
            "code": diag.code,
            "message": diag.message,
            "severity": diag.severity,
            "category": diag.category,
            "location": diag.location,
            "auto_fixable": diag.auto_fixable,
        }
        for diag in result.diagnostics
    ]

    return asdict(success_response(**response_data))


def _handle_analyze(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    tool_name = "spec_analyze"
    start_time = time.perf_counter()

    directory = payload.get("directory")
    path = payload.get("path")
    ws_path = Path(directory or path or ".").resolve()

    audit_log(
        "tool_invocation",
        tool="spec-analyze",
        action="analyze_specs",
        directory=str(ws_path),
    )

    specs_dir = find_specs_directory(str(ws_path))
    has_specs = specs_dir is not None

    analysis_data: Dict[str, Any] = {
        "directory": str(ws_path),
        "has_specs": has_specs,
        "specs_dir": str(specs_dir) if specs_dir else None,
    }

    if has_specs and specs_dir:
        folder_counts: Dict[str, int] = {}
        for folder in ["active", "pending", "completed", "archived"]:
            folder_path = specs_dir / folder
            if folder_path.exists():
                folder_counts[folder] = len(list(folder_path.glob("*.json")))
            else:
                folder_counts[folder] = 0

        analysis_data["spec_counts"] = folder_counts
        analysis_data["total_specs"] = sum(folder_counts.values())

        docs_dir = specs_dir / ".human-readable"
        analysis_data["documentation_available"] = docs_dir.exists() and any(
            docs_dir.glob("*.md")
        )

        codebase_json = ws_path / "docs" / "codebase.json"
        analysis_data["codebase_docs_available"] = codebase_json.exists()

    duration_ms = (time.perf_counter() - start_time) * 1000
    _metrics.counter(f"analysis.{tool_name}", labels={"status": "success"})
    _metrics.timer(f"analysis.{tool_name}.duration_ms", duration_ms)

    return asdict(
        success_response(
            **analysis_data,
            telemetry={"duration_ms": round(duration_ms, 2)},
        )
    )


def _handle_analyze_deps(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    tool_name = "spec_analyze_deps"
    start_time = time.perf_counter()

    spec_id = payload.get("spec_id")
    threshold = payload.get("bottleneck_threshold")
    path = payload.get("path")

    if not isinstance(spec_id, str) or not spec_id:
        return asdict(
            error_response(
                "spec_id is required",
                error_code=ErrorCode.MISSING_REQUIRED,
                error_type=ErrorType.VALIDATION,
                remediation="Provide a spec_id parameter (e.g., my-feature-spec)",
            )
        )

    bottleneck_threshold = int(threshold) if isinstance(threshold, int) else 3

    ws_path = Path(path) if isinstance(path, str) and path else Path.cwd()

    audit_log(
        "tool_invocation",
        tool="spec-analyze-deps",
        action="analyze_dependencies",
        spec_id=spec_id,
    )

    specs_dir = find_specs_directory(str(ws_path))
    if not specs_dir:
        return asdict(
            error_response(
                f"Specs directory not found in {ws_path}",
                data={"spec_id": spec_id, "workspace": str(ws_path)},
            )
        )

    spec_file = find_spec_file(spec_id, specs_dir)
    if not spec_file:
        return asdict(
            error_response(
                f"Spec '{spec_id}' not found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                data={"spec_id": spec_id, "specs_dir": str(specs_dir)},
                remediation="Ensure the spec exists in specs/active or specs/pending",
            )
        )

    spec_data = load_spec(spec_id, specs_dir)
    if not spec_data:
        return asdict(
            error_response(
                f"Failed to load spec '{spec_id}'",
                data={"spec_id": spec_id, "spec_file": str(spec_file)},
            )
        )

    hierarchy = spec_data.get("hierarchy", {})

    dependency_count = 0
    blocks_count: Dict[str, int] = {}
    bottlenecks: List[Dict[str, Any]] = []

    for node in hierarchy.values():
        deps = node.get("dependencies", {})
        blocked_by = deps.get("blocked_by", [])
        dependency_count += len(blocked_by)
        for blocker_id in blocked_by:
            blocks_count[blocker_id] = blocks_count.get(blocker_id, 0) + 1

    for task_id, count in blocks_count.items():
        if count >= bottleneck_threshold:
            task = hierarchy.get(task_id, {})
            bottlenecks.append(
                {
                    "task_id": task_id,
                    "title": task.get("title", ""),
                    "status": task.get("status", ""),
                    "blocks_count": count,
                }
            )

    bottlenecks.sort(key=lambda item: item["blocks_count"], reverse=True)

    visited: set[str] = set()
    rec_stack: set[str] = set()
    circular_deps: List[str] = []

    def detect_cycle(node_id: str, path: List[str]) -> bool:
        visited.add(node_id)
        rec_stack.add(node_id)

        node = hierarchy.get(node_id, {})
        for child_id in node.get("children", []):
            if child_id not in visited:
                if detect_cycle(child_id, path + [child_id]):
                    return True
            elif child_id in rec_stack:
                circular_deps.append(" -> ".join(path + [child_id]))
                return True

        rec_stack.remove(node_id)
        return False

    if "spec-root" in hierarchy:
        detect_cycle("spec-root", ["spec-root"])

    duration_ms = (time.perf_counter() - start_time) * 1000
    _metrics.counter(f"analysis.{tool_name}", labels={"status": "success"})
    _metrics.timer(f"analysis.{tool_name}.duration_ms", duration_ms)

    return asdict(
        success_response(
            spec_id=spec_id,
            dependency_count=dependency_count,
            bottlenecks=bottlenecks,
            bottleneck_threshold=bottleneck_threshold,
            circular_deps=circular_deps,
            has_cycles=len(circular_deps) > 0,
            telemetry={"duration_ms": round(duration_ms, 2)},
        )
    )


def _handle_schema(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    """Return schema information for all valid values in the spec system."""
    # Build templates with descriptions
    templates_with_desc = [
        {"name": t, "description": TEMPLATE_DESCRIPTIONS.get(t, "")}
        for t in TEMPLATES
    ]
    return asdict(
        success_response(
            templates=templates_with_desc,
            node_types=sorted(VALID_NODE_TYPES),
            statuses=sorted(VALID_STATUSES),
            task_categories=sorted(VALID_TASK_CATEGORIES),
            verification_types=sorted(VALID_VERIFICATION_TYPES),
            journal_entry_types=sorted(VALID_ENTRY_TYPES),
            blocker_types=sorted(VALID_BLOCKER_TYPES),
            status_folders=sorted(VALID_FOLDERS),
        )
    )


def _handle_diff(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    """Compare two specs and return categorized changes."""
    spec_id = payload.get("spec_id")
    if not spec_id:
        return asdict(
            error_response(
                "spec_id is required for diff action",
                error_code=ErrorCode.MISSING_REQUIRED,
                error_type=ErrorType.VALIDATION,
                remediation="Provide the spec_id of the current spec to compare",
            )
        )

    # Target can be a backup timestamp or another spec_id
    target = payload.get("target")
    workspace = payload.get("workspace")
    max_results = payload.get("limit")

    specs_dir = _resolve_specs_dir(config, workspace)
    if not specs_dir:
        return asdict(
            error_response(
                "No specs directory found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Ensure you're in a project with a specs/ directory",
            )
        )

    # If no target specified, diff against latest backup
    if not target:
        backups = list_spec_backups(spec_id, specs_dir=specs_dir)
        if backups["count"] == 0:
            return asdict(
                error_response(
                    f"No backups found for spec '{spec_id}'",
                    error_code=ErrorCode.NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation="Create a backup first using spec save operations",
                )
            )
        # Use latest backup as source (older state)
        source_path = backups["backups"][0]["file_path"]
    else:
        # Check if target is a timestamp (backup) or spec_id
        backup_file = specs_dir / ".backups" / spec_id / f"{target}.json"
        if backup_file.is_file():
            source_path = str(backup_file)
        else:
            # Treat as another spec_id
            source_path = target

    result = diff_specs(
        source=source_path,
        target=spec_id,
        specs_dir=specs_dir,
        max_results=max_results,
    )

    if "error" in result and not result.get("success", True):
        return asdict(
            error_response(
                result["error"],
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Verify both specs exist and are accessible",
            )
        )

    return asdict(
        success_response(
            spec_id=spec_id,
            compared_to=source_path if not target else target,
            summary=result["summary"],
            changes=result["changes"],
            partial=result["partial"],
        )
    )


def _handle_history(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    """List spec history including backups and revision history."""
    spec_id = payload.get("spec_id")
    if not spec_id:
        return asdict(
            error_response(
                "spec_id is required for history action",
                error_code=ErrorCode.MISSING_REQUIRED,
                error_type=ErrorType.VALIDATION,
                remediation="Provide the spec_id to view history",
            )
        )

    workspace = payload.get("workspace")
    cursor = payload.get("cursor")
    limit = payload.get("limit")

    specs_dir = _resolve_specs_dir(config, workspace)
    if not specs_dir:
        return asdict(
            error_response(
                "No specs directory found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Ensure you're in a project with a specs/ directory",
            )
        )

    # Get backups with pagination
    backups_result = list_spec_backups(
        spec_id, specs_dir=specs_dir, cursor=cursor, limit=limit
    )

    # Get revision history from spec metadata
    spec_data = load_spec(spec_id, specs_dir)
    revision_history = []
    if spec_data:
        metadata = spec_data.get("metadata", {})
        revision_history = metadata.get("revision_history", [])

    # Merge and sort entries (backups and revisions)
    history_entries = []

    # Add backups as history entries
    for backup in backups_result["backups"]:
        history_entries.append({
            "type": "backup",
            "timestamp": backup["timestamp"],
            "file_path": backup["file_path"],
            "file_size_bytes": backup["file_size_bytes"],
        })

    # Add revision history entries
    for rev in revision_history:
        history_entries.append({
            "type": "revision",
            "timestamp": rev.get("date"),
            "version": rev.get("version"),
            "changes": rev.get("changes"),
            "author": rev.get("author"),
        })

    return asdict(
        success_response(
            spec_id=spec_id,
            entries=history_entries,
            backup_count=backups_result["count"],
            revision_count=len(revision_history),
            pagination=backups_result["pagination"],
        )
    )


def _handle_completeness_check(
    *, config: ServerConfig, payload: Dict[str, Any]
) -> dict:
    """Check spec completeness and return a score (0-100)."""
    spec_id = payload.get("spec_id")
    if not spec_id or not isinstance(spec_id, str) or not spec_id.strip():
        return asdict(
            error_response(
                "spec_id is required for completeness-check action",
                error_code=ErrorCode.MISSING_REQUIRED,
                error_type=ErrorType.VALIDATION,
                remediation="Provide the spec_id to check completeness",
            )
        )

    workspace = payload.get("workspace")
    specs_dir = _resolve_specs_dir(config, workspace)
    if not specs_dir:
        return asdict(
            error_response(
                "No specs directory found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Ensure you're in a project with a specs/ directory",
            )
        )

    result, error = check_spec_completeness(spec_id, specs_dir=specs_dir)
    if error:
        return asdict(
            error_response(
                error,
                error_code=ErrorCode.SPEC_NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation='Verify the spec ID exists using spec(action="list").',
                details={"spec_id": spec_id},
            )
        )

    return asdict(success_response(**result))


def _handle_duplicate_detection(
    *, config: ServerConfig, payload: Dict[str, Any]
) -> dict:
    """Detect duplicate or near-duplicate tasks in a spec."""
    spec_id = payload.get("spec_id")
    if not spec_id or not isinstance(spec_id, str) or not spec_id.strip():
        return asdict(
            error_response(
                "spec_id is required for duplicate-detection action",
                error_code=ErrorCode.MISSING_REQUIRED,
                error_type=ErrorType.VALIDATION,
                remediation="Provide the spec_id to check for duplicates",
            )
        )

    workspace = payload.get("workspace")
    scope = payload.get("scope", "titles")
    threshold = payload.get("threshold", 0.8)
    max_pairs = payload.get("max_pairs", 100)

    # Validate threshold
    if not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
        return asdict(
            error_response(
                "threshold must be a number between 0.0 and 1.0",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
            )
        )

    specs_dir = _resolve_specs_dir(config, workspace)
    if not specs_dir:
        return asdict(
            error_response(
                "No specs directory found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Ensure you're in a project with a specs/ directory",
            )
        )

    result, error = detect_duplicate_tasks(
        spec_id,
        scope=scope,
        threshold=threshold,
        max_pairs=max_pairs,
        specs_dir=specs_dir,
    )
    if error:
        return asdict(
            error_response(
                error,
                error_code=ErrorCode.SPEC_NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation='Verify the spec ID exists using spec(action="list").',
                details={"spec_id": spec_id},
            )
        )

    return asdict(success_response(**result))


_ACTIONS = [
    ActionDefinition(name="find", handler=_handle_find, summary="Find a spec by ID"),
    ActionDefinition(name="get", handler=_handle_get, summary="Get raw spec JSON (minified)"),
    ActionDefinition(name="list", handler=_handle_list, summary="List specs"),
    ActionDefinition(
        name="validate", handler=_handle_validate, summary="Validate a spec"
    ),
    ActionDefinition(name="fix", handler=_handle_fix, summary="Auto-fix a spec"),
    ActionDefinition(name="stats", handler=_handle_stats, summary="Get spec stats"),
    ActionDefinition(
        name="validate-fix",
        handler=_handle_validate_fix,
        summary="Validate and optionally auto-fix",
    ),
    ActionDefinition(
        name="analyze", handler=_handle_analyze, summary="Analyze spec directory"
    ),
    ActionDefinition(
        name="analyze-deps",
        handler=_handle_analyze_deps,
        summary="Analyze spec dependency graph",
    ),
    ActionDefinition(
        name="schema",
        handler=_handle_schema,
        summary="Get valid values for spec fields",
    ),
    ActionDefinition(
        name="diff",
        handler=_handle_diff,
        summary="Compare spec against backup or another spec",
    ),
    ActionDefinition(
        name="history",
        handler=_handle_history,
        summary="List spec backups and revision history",
    ),
    ActionDefinition(
        name="completeness-check",
        handler=_handle_completeness_check,
        summary="Check spec completeness and return a score (0-100)",
    ),
    ActionDefinition(
        name="duplicate-detection",
        handler=_handle_duplicate_detection,
        summary="Detect duplicate or near-duplicate tasks",
    ),
]

_SPEC_ROUTER = ActionRouter(tool_name="spec", actions=_ACTIONS)


def _dispatch_spec_action(
    *, action: str, payload: Dict[str, Any], config: ServerConfig
) -> dict:
    try:
        return _SPEC_ROUTER.dispatch(action=action, payload=payload, config=config)
    except ActionRouterError as exc:
        allowed = ", ".join(exc.allowed_actions)
        return asdict(
            error_response(
                f"Unsupported spec action '{action}'. Allowed actions: {allowed}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {allowed}",
            )
        )


def register_unified_spec_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the consolidated spec tool."""

    @canonical_tool(mcp, canonical_name="spec")
    @mcp_tool(tool_name="spec", emit_metrics=True, audit=True)
    def spec(
        action: str,
        spec_id: Optional[str] = None,
        workspace: Optional[str] = None,
        status: str = "all",
        include_progress: bool = True,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        dry_run: bool = False,
        create_backup: bool = True,
        auto_fix: bool = True,
        directory: Optional[str] = None,
        path: Optional[str] = None,
        bottleneck_threshold: Optional[int] = None,
        target: Optional[str] = None,
    ) -> dict:
        payload = {
            "spec_id": spec_id,
            "workspace": workspace,
            "status": status,
            "include_progress": include_progress,
            "cursor": cursor,
            "limit": limit,
            "dry_run": dry_run,
            "create_backup": create_backup,
            "auto_fix": auto_fix,
            "directory": directory,
            "path": path,
            "bottleneck_threshold": bottleneck_threshold,
            "target": target,
        }
        return _dispatch_spec_action(action=action, payload=payload, config=config)


__all__ = [
    "register_unified_spec_tool",
]
