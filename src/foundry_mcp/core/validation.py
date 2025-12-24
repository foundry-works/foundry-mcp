"""
Validation operations for SDD spec files.
Provides spec validation, auto-fix capabilities, and statistics.

Security Note:
    This module uses size limits from foundry_mcp.core.security to protect
    against resource exhaustion attacks. See docs/mcp_best_practices/04-validation-input-hygiene.md
"""

from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import json
import re
import copy
from datetime import datetime, timezone

from foundry_mcp.core.security import (
    MAX_INPUT_SIZE,
    MAX_ARRAY_LENGTH,
    MAX_STRING_LENGTH,
    MAX_NESTED_DEPTH,
)


# Validation result data structures


@dataclass
class Diagnostic:
    """
    Structured diagnostic for MCP consumption.

    Provides a machine-readable format for validation findings
    that can be easily processed by MCP tools.
    """

    code: str  # Diagnostic code (e.g., "MISSING_FILE_PATH", "INVALID_STATUS")
    message: str  # Human-readable description
    severity: str  # "error", "warning", "info"
    category: str  # Category for grouping (e.g., "metadata", "structure", "counts")
    location: Optional[str] = None  # Node ID or path where issue occurred
    suggested_fix: Optional[str] = None  # Suggested fix description
    auto_fixable: bool = False  # Whether this can be auto-fixed


@dataclass
class ValidationResult:
    """
    Complete validation result for a spec file.
    """

    spec_id: str
    is_valid: bool
    diagnostics: List[Diagnostic] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0


@dataclass
class FixAction:
    """
    Represents a candidate auto-fix operation.
    """

    id: str
    description: str
    category: str
    severity: str
    auto_apply: bool
    preview: str
    apply: Callable[[Dict[str, Any]], None]


@dataclass
class FixReport:
    """
    Outcome of applying a set of fix actions.
    """

    spec_path: Optional[str] = None
    backup_path: Optional[str] = None
    applied_actions: List[FixAction] = field(default_factory=list)
    skipped_actions: List[FixAction] = field(default_factory=list)
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None


@dataclass
class SpecStats:
    """
    Statistics for a spec file.
    """

    spec_id: str
    title: str
    version: str
    status: str
    totals: Dict[str, int] = field(default_factory=dict)
    status_counts: Dict[str, int] = field(default_factory=dict)
    max_depth: int = 0
    avg_tasks_per_phase: float = 0.0
    verification_coverage: float = 0.0
    progress: float = 0.0
    file_size_kb: float = 0.0


# Constants

STATUS_FIELDS = {"pending", "in_progress", "completed", "blocked"}
VALID_NODE_TYPES = {"spec", "phase", "group", "task", "subtask", "verify"}
VALID_STATUSES = {"pending", "in_progress", "completed", "blocked"}
VALID_TASK_CATEGORIES = {
    "investigation",
    "implementation",
    "refactoring",
    "decision",
    "research",
}
VALID_VERIFICATION_TYPES = {"run-tests", "fidelity", "manual"}

# Legacy to canonical verification type mapping
VERIFICATION_TYPE_MAPPING = {
    "test": "run-tests",
    "auto": "run-tests",
}

# Common field name typos/alternatives
FIELD_NAME_SUGGESTIONS = {
    "category": "task_category",
    "type": "node type or verification_type",
    "desc": "description",
    "details": "description",
}


def _suggest_value(value: str, valid_values: set, n: int = 1) -> Optional[str]:
    """
    Suggest a close match for an invalid value.

    Args:
        value: The invalid value provided
        valid_values: Set of valid values to match against
        n: Number of suggestions to return (default 1)

    Returns:
        Suggestion string like "did you mean 'X'?" or None if no close match
    """
    if not value:
        return None
    matches = get_close_matches(value.lower(), [v.lower() for v in valid_values], n=n, cutoff=0.6)
    if matches:
        # Find the original-case version of the match
        for v in valid_values:
            if v.lower() == matches[0]:
                return f"did you mean '{v}'?"
        return f"did you mean '{matches[0]}'?"
    return None


# Validation functions


def _requires_rich_task_fields(spec_data: Dict[str, Any]) -> bool:
    metadata = spec_data.get("metadata", {})
    if not isinstance(metadata, dict):
        return False

    template = metadata.get("template")
    if isinstance(template, str) and template.strip().lower() in {"medium", "complex"}:
        return True

    complexity = metadata.get("complexity")
    if isinstance(complexity, str) and complexity.strip().lower() in {
        "medium",
        "complex",
        "high",
    }:
        return True

    return False


def validate_spec_input(
    raw_input: str | bytes,
    *,
    max_size: Optional[int] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[ValidationResult]]:
    """
    Validate and parse raw spec input with size checks.

    Performs size validation before JSON parsing to prevent resource
    exhaustion attacks from oversized payloads.

    Args:
        raw_input: Raw JSON string or bytes to validate
        max_size: Maximum allowed size in bytes (default: MAX_INPUT_SIZE)

    Returns:
        Tuple of (parsed_data, error_result):
        - On success: (dict, None)
        - On failure: (None, ValidationResult with error)

    Example:
        >>> spec_data, error = validate_spec_input(json_string)
        >>> if error:
        ...     return error_response(error.diagnostics[0].message)
        >>> result = validate_spec(spec_data)
    """
    effective_max_size = max_size if max_size is not None else MAX_INPUT_SIZE

    # Convert to bytes if string for consistent size checking
    if isinstance(raw_input, str):
        input_bytes = raw_input.encode("utf-8")
    else:
        input_bytes = raw_input

    # Check input size
    if len(input_bytes) > effective_max_size:
        error_result = ValidationResult(
            spec_id="unknown",
            is_valid=False,
            error_count=1,
        )
        error_result.diagnostics.append(
            Diagnostic(
                code="INPUT_TOO_LARGE",
                message=f"Input size ({len(input_bytes):,} bytes) exceeds maximum allowed ({effective_max_size:,} bytes)",
                severity="error",
                category="security",
                suggested_fix=f"Reduce input size to under {effective_max_size:,} bytes",
            )
        )
        return None, error_result

    # Try to parse JSON
    try:
        if isinstance(raw_input, bytes):
            spec_data = json.loads(raw_input.decode("utf-8"))
        else:
            spec_data = json.loads(raw_input)
    except json.JSONDecodeError as e:
        error_result = ValidationResult(
            spec_id="unknown",
            is_valid=False,
            error_count=1,
        )
        error_result.diagnostics.append(
            Diagnostic(
                code="INVALID_JSON",
                message=f"Failed to parse JSON: {e}",
                severity="error",
                category="structure",
            )
        )
        return None, error_result

    # Spec data must be a dict
    if not isinstance(spec_data, dict):
        error_result = ValidationResult(
            spec_id="unknown",
            is_valid=False,
            error_count=1,
        )
        error_result.diagnostics.append(
            Diagnostic(
                code="INVALID_SPEC_TYPE",
                message=f"Spec must be a JSON object, got {type(spec_data).__name__}",
                severity="error",
                category="structure",
            )
        )
        return None, error_result

    return spec_data, None


def validate_spec(spec_data: Dict[str, Any]) -> ValidationResult:
    """
    Validate a spec file and return structured diagnostics.

    Args:
        spec_data: Parsed JSON spec data

    Returns:
        ValidationResult with all diagnostics

    Note:
        For raw JSON input, use validate_spec_input() first to perform
        size validation before parsing.
    """
    spec_id = spec_data.get("spec_id", "unknown")
    result = ValidationResult(spec_id=spec_id, is_valid=True)

    # Check overall structure size (defense in depth)
    _validate_size_limits(spec_data, result)

    # Run all validation checks
    _validate_structure(spec_data, result)

    hierarchy = spec_data.get("hierarchy", {})
    if hierarchy:
        _validate_hierarchy(hierarchy, result)
        _validate_nodes(hierarchy, result)
        _validate_task_counts(hierarchy, result)
        _validate_dependencies(hierarchy, result)
        _validate_metadata(spec_data, hierarchy, result)

    # Count diagnostics by severity
    for diag in result.diagnostics:
        if diag.severity == "error":
            result.error_count += 1
        elif diag.severity == "warning":
            result.warning_count += 1
        else:
            result.info_count += 1

    result.is_valid = result.error_count == 0
    return result


def _iter_valid_nodes(
    hierarchy: Dict[str, Any],
    result: ValidationResult,
    report_invalid: bool = True,
):
    """
    Iterate over hierarchy yielding only valid (dict) nodes.

    Args:
        hierarchy: The hierarchy dict to iterate
        result: ValidationResult to append errors to
        report_invalid: Whether to report invalid nodes as errors (default True,
                        set False if already reported by another function)

    Yields:
        Tuples of (node_id, node) where node is a valid dict
    """
    for node_id, node in hierarchy.items():
        if not isinstance(node, dict):
            if report_invalid:
                result.diagnostics.append(
                    Diagnostic(
                        code="INVALID_NODE_STRUCTURE",
                        message=f"Node '{node_id}' is not a valid object (got {type(node).__name__})",
                        severity="error",
                        category="node",
                        location=str(node_id),
                        suggested_fix="Ensure all hierarchy values are valid node objects",
                    )
                )
            continue
        yield node_id, node


def _validate_size_limits(spec_data: Dict[str, Any], result: ValidationResult) -> None:
    """Validate size limits on spec data structures (defense in depth)."""

    def count_items(obj: Any, depth: int = 0) -> tuple[int, int]:
        """Count total items and max depth in nested structure."""
        if depth > MAX_NESTED_DEPTH:
            return 0, depth

        if isinstance(obj, dict):
            total = len(obj)
            max_d = depth
            for v in obj.values():
                sub_count, sub_depth = count_items(v, depth + 1)
                total += sub_count
                max_d = max(max_d, sub_depth)
            return total, max_d
        elif isinstance(obj, list):
            total = len(obj)
            max_d = depth
            for item in obj:
                sub_count, sub_depth = count_items(item, depth + 1)
                total += sub_count
                max_d = max(max_d, sub_depth)
            return total, max_d
        else:
            return 1, depth

    # Check hierarchy nesting depth
    hierarchy = spec_data.get("hierarchy", {})
    if hierarchy:
        _, max_depth = count_items(hierarchy)
        if max_depth > MAX_NESTED_DEPTH:
            result.diagnostics.append(
                Diagnostic(
                    code="EXCESSIVE_NESTING",
                    message=f"Hierarchy nesting depth ({max_depth}) exceeds maximum ({MAX_NESTED_DEPTH})",
                    severity="warning",
                    category="security",
                    suggested_fix="Flatten hierarchy structure to reduce nesting depth",
                )
            )

    # Check array lengths in common locations
    children = hierarchy.get("children", [])
    if len(children) > MAX_ARRAY_LENGTH:
        result.diagnostics.append(
            Diagnostic(
                code="EXCESSIVE_ARRAY_LENGTH",
                message=f"Root children array ({len(children)} items) exceeds maximum ({MAX_ARRAY_LENGTH})",
                severity="warning",
                category="security",
                location="hierarchy.children",
                suggested_fix="Split large phase/task lists into smaller groups",
            )
        )

    # Check journal array length
    journal = spec_data.get("journal", [])
    if len(journal) > MAX_ARRAY_LENGTH:
        result.diagnostics.append(
            Diagnostic(
                code="EXCESSIVE_JOURNAL_LENGTH",
                message=f"Journal array ({len(journal)} entries) exceeds maximum ({MAX_ARRAY_LENGTH})",
                severity="warning",
                category="security",
                location="journal",
                suggested_fix="Archive old journal entries or split into separate files",
            )
        )


def _validate_structure(spec_data: Dict[str, Any], result: ValidationResult) -> None:
    """Validate top-level structure and required fields."""
    required_fields = ["spec_id", "generated", "last_updated", "hierarchy"]

    for field_name in required_fields:
        if field_name not in spec_data:
            result.diagnostics.append(
                Diagnostic(
                    code="MISSING_REQUIRED_FIELD",
                    message=f"Missing required field '{field_name}'",
                    severity="error",
                    category="structure",
                    suggested_fix=f"Add required field '{field_name}' to spec",
                    auto_fixable=False,
                )
            )

    # Validate spec_id format
    spec_id = spec_data.get("spec_id")
    if spec_id and not _is_valid_spec_id(spec_id):
        result.diagnostics.append(
            Diagnostic(
                code="INVALID_SPEC_ID_FORMAT",
                message=f"spec_id '{spec_id}' doesn't follow format: {{feature}}-{{YYYY-MM-DD}}-{{nnn}}",
                severity="warning",
                category="structure",
                location="spec_id",
            )
        )

    # Validate date fields
    for field_name in ["generated", "last_updated"]:
        value = spec_data.get(field_name)
        if value and not _is_valid_iso8601(value):
            result.diagnostics.append(
                Diagnostic(
                    code="INVALID_DATE_FORMAT",
                    message=f"'{field_name}' should be in ISO 8601 format",
                    severity="warning",
                    category="structure",
                    location=field_name,
                    suggested_fix="Normalize timestamp to ISO 8601 format",
                    auto_fixable=True,
                )
            )

    if _requires_rich_task_fields(spec_data):
        metadata = spec_data.get("metadata", {})
        mission = metadata.get("mission") if isinstance(metadata, dict) else None
        if not isinstance(mission, str) or not mission.strip():
            result.diagnostics.append(
                Diagnostic(
                    code="MISSING_MISSION",
                    message="Spec metadata.mission is required for medium/complex specifications",
                    severity="error",
                    category="metadata",
                    location="metadata.mission",
                    suggested_fix="Set metadata.mission to a concise goal statement",
                    auto_fixable=False,
                )
            )

    # Check hierarchy is dict
    hierarchy = spec_data.get("hierarchy")
    if hierarchy is not None and not isinstance(hierarchy, dict):
        result.diagnostics.append(
            Diagnostic(
                code="INVALID_HIERARCHY_TYPE",
                message="'hierarchy' must be a dictionary",
                severity="error",
                category="structure",
            )
        )
    elif hierarchy is not None and len(hierarchy) == 0:
        result.diagnostics.append(
            Diagnostic(
                code="EMPTY_HIERARCHY",
                message="'hierarchy' is empty",
                severity="error",
                category="structure",
            )
        )


def _validate_hierarchy(hierarchy: Dict[str, Any], result: ValidationResult) -> None:
    """Validate hierarchy integrity: parent/child references, no orphans, no cycles."""
    # Check spec-root exists
    if "spec-root" not in hierarchy:
        result.diagnostics.append(
            Diagnostic(
                code="MISSING_SPEC_ROOT",
                message="Missing 'spec-root' node in hierarchy",
                severity="error",
                category="hierarchy",
            )
        )
        return

    root = hierarchy["spec-root"]
    if root.get("parent") is not None:
        result.diagnostics.append(
            Diagnostic(
                code="INVALID_ROOT_PARENT",
                message="'spec-root' must have parent: null",
                severity="error",
                category="hierarchy",
                location="spec-root",
                suggested_fix="Set spec-root parent to null",
                auto_fixable=True,
            )
        )

    # Validate parent references
    for node_id, node in _iter_valid_nodes(hierarchy, result):
        parent_id = node.get("parent")

        if node_id != "spec-root" and parent_id is None:
            result.diagnostics.append(
                Diagnostic(
                    code="NULL_PARENT",
                    message=f"Node '{node_id}' has null parent (only spec-root should)",
                    severity="error",
                    category="hierarchy",
                    location=node_id,
                )
            )

        if parent_id and parent_id not in hierarchy:
            result.diagnostics.append(
                Diagnostic(
                    code="MISSING_PARENT",
                    message=f"Node '{node_id}' references non-existent parent '{parent_id}'",
                    severity="error",
                    category="hierarchy",
                    location=node_id,
                )
            )

    # Validate child references
    for node_id, node in _iter_valid_nodes(hierarchy, result, report_invalid=False):
        children = node.get("children", [])

        if not isinstance(children, list):
            result.diagnostics.append(
                Diagnostic(
                    code="INVALID_CHILDREN_TYPE",
                    message=f"Node '{node_id}' children field must be a list",
                    severity="error",
                    category="hierarchy",
                    location=node_id,
                )
            )
            continue

        for child_id in children:
            if child_id not in hierarchy:
                result.diagnostics.append(
                    Diagnostic(
                        code="MISSING_CHILD",
                        message=f"Node '{node_id}' references non-existent child '{child_id}'",
                        severity="error",
                        category="hierarchy",
                        location=node_id,
                    )
                )
            else:
                child_node = hierarchy[child_id]
                if child_node.get("parent") != node_id:
                    result.diagnostics.append(
                        Diagnostic(
                            code="PARENT_CHILD_MISMATCH",
                            message=f"'{node_id}' lists '{child_id}' as child, but '{child_id}' has parent='{child_node.get('parent')}'",
                            severity="error",
                            category="hierarchy",
                            location=node_id,
                            suggested_fix="Align parent references with children list",
                            auto_fixable=True,
                        )
                    )

    # Check for orphaned nodes
    reachable = set()

    def traverse(node_id: str) -> None:
        if node_id in reachable:
            return
        reachable.add(node_id)
        node = hierarchy.get(node_id, {})
        for child_id in node.get("children", []):
            if child_id in hierarchy:
                traverse(child_id)

    traverse("spec-root")

    orphaned = set(hierarchy.keys()) - reachable
    if orphaned:
        orphan_list = ", ".join(sorted(orphaned))
        result.diagnostics.append(
            Diagnostic(
                code="ORPHANED_NODES",
                message=f"Found {len(orphaned)} orphaned node(s) not reachable from spec-root: {orphan_list}",
                severity="error",
                category="hierarchy",
                suggested_fix="Attach orphaned nodes to spec-root or remove them",
                auto_fixable=True,
            )
        )

    # Check for cycles
    visited = set()
    rec_stack = set()

    def has_cycle(node_id: str) -> bool:
        visited.add(node_id)
        rec_stack.add(node_id)

        node = hierarchy.get(node_id, {})
        for child_id in node.get("children", []):
            if child_id not in visited:
                if has_cycle(child_id):
                    return True
            elif child_id in rec_stack:
                return True

        rec_stack.remove(node_id)
        return False

    if has_cycle("spec-root"):
        result.diagnostics.append(
            Diagnostic(
                code="CYCLE_DETECTED",
                message="Cycle detected in hierarchy tree",
                severity="error",
                category="hierarchy",
            )
        )


def _validate_nodes(hierarchy: Dict[str, Any], result: ValidationResult) -> None:
    """Validate node structure and required fields."""
    required_fields = [
        "type",
        "title",
        "status",
        "parent",
        "children",
        "total_tasks",
        "completed_tasks",
        "metadata",
    ]

    for node_id, node in _iter_valid_nodes(hierarchy, result, report_invalid=False):
        # Check required fields
        for field_name in required_fields:
            if field_name not in node:
                result.diagnostics.append(
                    Diagnostic(
                        code="MISSING_NODE_FIELD",
                        message=f"Node '{node_id}' missing required field '{field_name}'",
                        severity="error",
                        category="node",
                        location=node_id,
                        suggested_fix="Add missing required fields with sensible defaults",
                        auto_fixable=True,
                    )
                )

        # Validate type
        node_type = node.get("type")
        if node_type and node_type not in VALID_NODE_TYPES:
            hint = _suggest_value(node_type, VALID_NODE_TYPES)
            msg = f"Node '{node_id}' has invalid type '{node_type}'"
            if hint:
                msg += f"; {hint}"
            result.diagnostics.append(
                Diagnostic(
                    code="INVALID_NODE_TYPE",
                    message=msg,
                    severity="error",
                    category="node",
                    location=node_id,
                    suggested_fix=f"Valid types: {', '.join(sorted(VALID_NODE_TYPES))}",
                    auto_fixable=True,
                )
            )

        # Validate status
        status = node.get("status")
        if status and status not in VALID_STATUSES:
            hint = _suggest_value(status, VALID_STATUSES)
            msg = f"Node '{node_id}' has invalid status '{status}'"
            if hint:
                msg += f"; {hint}"
            result.diagnostics.append(
                Diagnostic(
                    code="INVALID_STATUS",
                    message=msg,
                    severity="error",
                    category="node",
                    location=node_id,
                    suggested_fix=f"Valid statuses: {', '.join(sorted(VALID_STATUSES))}",
                    auto_fixable=True,
                )
            )

        # Check title is not empty
        title = node.get("title")
        if title is not None and not str(title).strip():
            result.diagnostics.append(
                Diagnostic(
                    code="EMPTY_TITLE",
                    message=f"Node '{node_id}' has empty title",
                    severity="warning",
                    category="node",
                    location=node_id,
                    suggested_fix="Generate title from node ID",
                    auto_fixable=True,
                )
            )

        # Validate dependencies structure
        if "dependencies" in node:
            deps = node["dependencies"]
            if not isinstance(deps, dict):
                result.diagnostics.append(
                    Diagnostic(
                        code="INVALID_DEPENDENCIES_TYPE",
                        message=f"Node '{node_id}' dependencies must be a dictionary",
                        severity="error",
                        category="dependency",
                        location=node_id,
                        suggested_fix="Create dependencies dict with blocks/blocked_by/depends arrays",
                        auto_fixable=True,
                    )
                )
            else:
                for dep_key in ["blocks", "blocked_by", "depends"]:
                    if dep_key in deps and not isinstance(deps[dep_key], list):
                        result.diagnostics.append(
                            Diagnostic(
                                code="INVALID_DEPENDENCY_FIELD",
                                message=f"Node '{node_id}' dependencies.{dep_key} must be a list",
                                severity="error",
                                category="dependency",
                                location=node_id,
                            )
                        )


def _validate_task_counts(hierarchy: Dict[str, Any], result: ValidationResult) -> None:
    """Validate task count accuracy and propagation."""
    for node_id, node in _iter_valid_nodes(hierarchy, result, report_invalid=False):
        total_tasks = node.get("total_tasks", 0)
        completed_tasks = node.get("completed_tasks", 0)
        children = node.get("children", [])

        # Completed can't exceed total
        if completed_tasks > total_tasks:
            result.diagnostics.append(
                Diagnostic(
                    code="COMPLETED_EXCEEDS_TOTAL",
                    message=f"Node '{node_id}' has completed_tasks ({completed_tasks}) > total_tasks ({total_tasks})",
                    severity="error",
                    category="counts",
                    location=node_id,
                    suggested_fix="Recalculate total/completed task rollups for parent nodes",
                    auto_fixable=True,
                )
            )

        # If node has children, verify counts match sum
        if children:
            child_total = 0
            child_completed = 0

            for child_id in children:
                if child_id in hierarchy:
                    child_node = hierarchy[child_id]
                    child_total += child_node.get("total_tasks", 0)
                    child_completed += child_node.get("completed_tasks", 0)

            if total_tasks != child_total:
                result.diagnostics.append(
                    Diagnostic(
                        code="TOTAL_TASKS_MISMATCH",
                        message=f"Node '{node_id}' total_tasks ({total_tasks}) doesn't match sum of children ({child_total})",
                        severity="error",
                        category="counts",
                        location=node_id,
                        suggested_fix="Recalculate total/completed task rollups",
                        auto_fixable=True,
                    )
                )

            if completed_tasks != child_completed:
                result.diagnostics.append(
                    Diagnostic(
                        code="COMPLETED_TASKS_MISMATCH",
                        message=f"Node '{node_id}' completed_tasks ({completed_tasks}) doesn't match sum of children ({child_completed})",
                        severity="error",
                        category="counts",
                        location=node_id,
                        suggested_fix="Recalculate total/completed task rollups",
                        auto_fixable=True,
                    )
                )
        else:
            # Leaf nodes should have total_tasks = 1
            node_type = node.get("type")
            if node_type in ["task", "subtask", "verify"]:
                if total_tasks != 1:
                    result.diagnostics.append(
                        Diagnostic(
                            code="INVALID_LEAF_COUNT",
                            message=f"Leaf node '{node_id}' (type={node_type}) should have total_tasks=1, has {total_tasks}",
                            severity="warning",
                            category="counts",
                            location=node_id,
                            suggested_fix="Set leaf node total_tasks to 1",
                            auto_fixable=True,
                        )
                    )


def _validate_dependencies(hierarchy: Dict[str, Any], result: ValidationResult) -> None:
    """Validate dependency graph and bidirectional consistency."""
    for node_id, node in _iter_valid_nodes(hierarchy, result, report_invalid=False):
        if "dependencies" not in node:
            continue

        deps = node["dependencies"]
        if not isinstance(deps, dict):
            continue

        # Check dependency references exist
        for dep_type in ["blocks", "blocked_by", "depends"]:
            if dep_type not in deps:
                continue

            for dep_id in deps[dep_type]:
                if dep_id not in hierarchy:
                    result.diagnostics.append(
                        Diagnostic(
                            code="MISSING_DEPENDENCY_TARGET",
                            message=f"Node '{node_id}' {dep_type} references non-existent node '{dep_id}'",
                            severity="error",
                            category="dependency",
                            location=node_id,
                        )
                    )

        # Check bidirectional consistency for blocks/blocked_by
        for blocked_id in deps.get("blocks", []):
            if blocked_id in hierarchy:
                blocked_node = hierarchy[blocked_id]
                blocked_deps = blocked_node.get("dependencies", {})
                if isinstance(blocked_deps, dict):
                    if node_id not in blocked_deps.get("blocked_by", []):
                        result.diagnostics.append(
                            Diagnostic(
                                code="BIDIRECTIONAL_INCONSISTENCY",
                                message=f"'{node_id}' blocks '{blocked_id}', but '{blocked_id}' doesn't list '{node_id}' in blocked_by",
                                severity="error",
                                category="dependency",
                                location=node_id,
                                suggested_fix="Synchronize bidirectional dependency relationships",
                                auto_fixable=True,
                            )
                        )

        for blocker_id in deps.get("blocked_by", []):
            if blocker_id in hierarchy:
                blocker_node = hierarchy[blocker_id]
                blocker_deps = blocker_node.get("dependencies", {})
                if isinstance(blocker_deps, dict):
                    if node_id not in blocker_deps.get("blocks", []):
                        result.diagnostics.append(
                            Diagnostic(
                                code="BIDIRECTIONAL_INCONSISTENCY",
                                message=f"'{node_id}' blocked_by '{blocker_id}', but '{blocker_id}' doesn't list '{node_id}' in blocks",
                                severity="error",
                                category="dependency",
                                location=node_id,
                                suggested_fix="Synchronize bidirectional dependency relationships",
                                auto_fixable=True,
                            )
                        )


def _validate_metadata(
    spec_data: Dict[str, Any],
    hierarchy: Dict[str, Any],
    result: ValidationResult,
) -> None:
    """Validate type-specific metadata requirements."""
    requires_rich_tasks = _requires_rich_task_fields(spec_data)

    def _nonempty_string(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def _has_description(metadata: Dict[str, Any]) -> bool:
        if _nonempty_string(metadata.get("description")):
            return True
        details = metadata.get("details")
        if _nonempty_string(details):
            return True
        if isinstance(details, list):
            return any(_nonempty_string(item) for item in details)
        return False

    for node_id, node in _iter_valid_nodes(hierarchy, result, report_invalid=False):
        node_type = node.get("type")
        metadata = node.get("metadata", {})

        if not isinstance(metadata, dict):
            result.diagnostics.append(
                Diagnostic(
                    code="INVALID_METADATA_TYPE",
                    message=f"Node '{node_id}' metadata must be a dictionary",
                    severity="error",
                    category="metadata",
                    location=node_id,
                )
            )
            continue

        # Verify nodes
        if node_type == "verify":
            verification_type = metadata.get("verification_type")

            if not verification_type:
                result.diagnostics.append(
                    Diagnostic(
                        code="MISSING_VERIFICATION_TYPE",
                        message=f"Verify node '{node_id}' missing metadata.verification_type",
                        severity="error",
                        category="metadata",
                        location=node_id,
                        suggested_fix="Set verification_type to 'run-tests', 'fidelity', or 'manual'",
                        auto_fixable=True,
                    )
                )
            elif verification_type not in VALID_VERIFICATION_TYPES:
                hint = _suggest_value(verification_type, VALID_VERIFICATION_TYPES)
                msg = f"Verify node '{node_id}' has invalid verification_type '{verification_type}'"
                if hint:
                    msg += f"; {hint}"
                result.diagnostics.append(
                    Diagnostic(
                        code="INVALID_VERIFICATION_TYPE",
                        message=msg,
                        severity="error",
                        category="metadata",
                        location=node_id,
                        suggested_fix=f"Valid types: {', '.join(sorted(VALID_VERIFICATION_TYPES))}",
                        auto_fixable=True,
                    )
                )

        # Task nodes
        if node_type == "task":
            raw_task_category = metadata.get("task_category")
            task_category = None
            if isinstance(raw_task_category, str) and raw_task_category.strip():
                task_category = raw_task_category.strip().lower()

            # Check for common field name typo: 'category' instead of 'task_category'
            if task_category is None and "category" in metadata and "task_category" not in metadata:
                result.diagnostics.append(
                    Diagnostic(
                        code="UNKNOWN_FIELD",
                        message=f"Task node '{node_id}' has unknown field 'category'; did you mean 'task_category'?",
                        severity="warning",
                        category="metadata",
                        location=node_id,
                        suggested_fix="Rename 'category' to 'task_category'",
                        auto_fixable=False,
                    )
                )

            if task_category is not None and task_category not in VALID_TASK_CATEGORIES:
                hint = _suggest_value(task_category, VALID_TASK_CATEGORIES)
                msg = f"Task node '{node_id}' has invalid task_category '{task_category}'"
                if hint:
                    msg += f"; {hint}"
                result.diagnostics.append(
                    Diagnostic(
                        code="INVALID_TASK_CATEGORY",
                        message=msg,
                        severity="error",
                        category="metadata",
                        location=node_id,
                        suggested_fix=f"Valid categories: {', '.join(sorted(VALID_TASK_CATEGORIES))}",
                        auto_fixable=False,  # Disabled: manual fix required
                    )
                )

            if requires_rich_tasks and task_category is None:
                result.diagnostics.append(
                    Diagnostic(
                        code="MISSING_TASK_CATEGORY",
                        message=f"Task node '{node_id}' missing metadata.task_category",
                        severity="error",
                        category="metadata",
                        location=node_id,
                        suggested_fix="Set metadata.task_category to a valid category",
                        auto_fixable=False,
                    )
                )

            if requires_rich_tasks and not _has_description(metadata):
                result.diagnostics.append(
                    Diagnostic(
                        code="MISSING_TASK_DESCRIPTION",
                        message=f"Task node '{node_id}' missing metadata.description",
                        severity="error",
                        category="metadata",
                        location=node_id,
                        suggested_fix="Provide metadata.description (or details) for the task",
                        auto_fixable=False,
                    )
                )

            if requires_rich_tasks:
                acceptance_criteria = metadata.get("acceptance_criteria")
                if acceptance_criteria is None:
                    result.diagnostics.append(
                        Diagnostic(
                            code="MISSING_ACCEPTANCE_CRITERIA",
                            message=f"Task node '{node_id}' missing metadata.acceptance_criteria",
                            severity="error",
                            category="metadata",
                            location=node_id,
                            suggested_fix="Provide a non-empty acceptance_criteria list",
                            auto_fixable=False,
                        )
                    )
                elif not isinstance(acceptance_criteria, list):
                    result.diagnostics.append(
                        Diagnostic(
                            code="INVALID_ACCEPTANCE_CRITERIA",
                            message=(
                                f"Task node '{node_id}' metadata.acceptance_criteria must be a list of strings"
                            ),
                            severity="error",
                            category="metadata",
                            location=node_id,
                            suggested_fix="Provide acceptance_criteria as an array of strings",
                            auto_fixable=False,
                        )
                    )
                elif not acceptance_criteria:
                    result.diagnostics.append(
                        Diagnostic(
                            code="MISSING_ACCEPTANCE_CRITERIA",
                            message=f"Task node '{node_id}' must include at least one acceptance criterion",
                            severity="error",
                            category="metadata",
                            location=node_id,
                            suggested_fix="Add at least one acceptance criterion",
                            auto_fixable=False,
                        )
                    )
                else:
                    invalid_items = [
                        idx
                        for idx, item in enumerate(acceptance_criteria)
                        if not _nonempty_string(item)
                    ]
                    if invalid_items:
                        result.diagnostics.append(
                            Diagnostic(
                                code="INVALID_ACCEPTANCE_CRITERIA",
                                message=(
                                    f"Task node '{node_id}' has invalid acceptance_criteria entries"
                                ),
                                severity="error",
                                category="metadata",
                                location=node_id,
                                suggested_fix="Ensure acceptance_criteria contains non-empty strings",
                                auto_fixable=False,
                            )
                        )

            category_for_file_path = task_category
            if category_for_file_path is None:
                legacy_category = metadata.get("category")
                if isinstance(legacy_category, str) and legacy_category.strip():
                    category_for_file_path = legacy_category.strip().lower()

            # file_path required for implementation and refactoring.
            # Do not auto-generate placeholder paths; the authoring agent/user must
            # provide a real path in the target codebase.
            if category_for_file_path in ["implementation", "refactoring"]:
                file_path = metadata.get("file_path")
                if not _nonempty_string(file_path):
                    result.diagnostics.append(
                        Diagnostic(
                            code="MISSING_FILE_PATH",
                            message=f"Task node '{node_id}' with category '{category_for_file_path}' missing metadata.file_path",
                            severity="error",
                            category="metadata",
                            location=node_id,
                            suggested_fix=(
                                "Set metadata.file_path to the real repo-relative path of the primary file impacted"
                            ),
                            auto_fixable=False,
                        )
                    )


# Fix action functions


def get_fix_actions(
    result: ValidationResult, spec_data: Dict[str, Any]
) -> List[FixAction]:
    """
    Generate fix actions from validation diagnostics.

    Args:
        result: ValidationResult with diagnostics
        spec_data: Original spec data

    Returns:
        List of FixAction objects that can be applied
    """
    actions: List[FixAction] = []
    seen_ids = set()
    hierarchy = spec_data.get("hierarchy", {})

    for diag in result.diagnostics:
        if not diag.auto_fixable:
            continue

        action = _build_fix_action(diag, spec_data, hierarchy)
        if action and action.id not in seen_ids:
            actions.append(action)
            seen_ids.add(action.id)

    return actions


def _build_fix_action(
    diag: Diagnostic, spec_data: Dict[str, Any], hierarchy: Dict[str, Any]
) -> Optional[FixAction]:
    """Build a fix action for a diagnostic."""
    code = diag.code

    if code == "INVALID_DATE_FORMAT":
        return _build_date_fix(diag, spec_data)

    if code == "PARENT_CHILD_MISMATCH":
        return _build_hierarchy_align_fix(diag, hierarchy)

    if code == "ORPHANED_NODES":
        return _build_orphan_fix(diag, hierarchy)

    if code == "INVALID_ROOT_PARENT":
        return _build_root_parent_fix(diag, hierarchy)

    if code == "MISSING_NODE_FIELD":
        return _build_missing_fields_fix(diag, hierarchy)

    if code == "INVALID_NODE_TYPE":
        return _build_type_normalize_fix(diag, hierarchy)

    if code == "INVALID_STATUS":
        return _build_status_normalize_fix(diag, hierarchy)

    if code == "EMPTY_TITLE":
        return _build_title_generate_fix(diag, hierarchy)

    if code in [
        "TOTAL_TASKS_MISMATCH",
        "COMPLETED_TASKS_MISMATCH",
        "COMPLETED_EXCEEDS_TOTAL",
        "INVALID_LEAF_COUNT",
    ]:
        return _build_counts_fix(diag, spec_data)

    if code == "BIDIRECTIONAL_INCONSISTENCY":
        return _build_bidirectional_fix(diag, hierarchy)

    if code == "INVALID_DEPENDENCIES_TYPE":
        return _build_deps_structure_fix(diag, hierarchy)

    if code == "MISSING_VERIFICATION_TYPE":
        return _build_verification_type_fix(diag, hierarchy)

    if code == "INVALID_VERIFICATION_TYPE":
        return _build_invalid_verification_type_fix(diag, hierarchy)

    # INVALID_TASK_CATEGORY auto-fix disabled - manual correction required
    # if code == "INVALID_TASK_CATEGORY":
    #     return _build_task_category_fix(diag, hierarchy)

    return None


def _build_date_fix(diag: Diagnostic, spec_data: Dict[str, Any]) -> Optional[FixAction]:
    """Build fix for date normalization."""
    field_name = diag.location
    if not field_name:
        return None

    def apply(data: Dict[str, Any]) -> None:
        value = data.get(field_name)
        normalized = _normalize_timestamp(value)
        if normalized:
            data[field_name] = normalized

    return FixAction(
        id=f"date.normalize:{field_name}",
        description=f"Normalize {field_name} to ISO 8601",
        category="structure",
        severity=diag.severity,
        auto_apply=True,
        preview=f"Normalize timestamp field: {field_name}",
        apply=apply,
    )


def _build_hierarchy_align_fix(
    diag: Diagnostic, hierarchy: Dict[str, Any]
) -> Optional[FixAction]:
    """Build fix for parent/child alignment."""
    # Parse node IDs from message
    match = re.search(r"'([^']+)' lists '([^']+)' as child", diag.message)
    if not match:
        return None

    parent_id = match.group(1)
    child_id = match.group(2)

    def apply(data: Dict[str, Any]) -> None:
        hier = data.get("hierarchy", {})
        parent = hier.get(parent_id)
        child = hier.get(child_id)
        if parent and child:
            children = parent.setdefault("children", [])
            if child_id not in children:
                children.append(child_id)
            child["parent"] = parent_id

    return FixAction(
        id=f"hierarchy.align:{parent_id}->{child_id}",
        description=f"Align {child_id} parent reference with {parent_id}",
        category="hierarchy",
        severity=diag.severity,
        auto_apply=True,
        preview=f"Align {child_id} parent reference with {parent_id}",
        apply=apply,
    )


def _build_orphan_fix(
    diag: Diagnostic, hierarchy: Dict[str, Any]
) -> Optional[FixAction]:
    """Build fix for orphaned nodes."""
    match = re.search(r"not reachable from spec-root:\s*(.+)$", diag.message)
    if not match:
        return None

    orphan_list_str = match.group(1)
    orphan_ids = [nid.strip() for nid in orphan_list_str.split(",")]

    def apply(data: Dict[str, Any]) -> None:
        hier = data.get("hierarchy", {})
        spec_root = hier.get("spec-root")
        if not spec_root:
            return

        root_children = spec_root.setdefault("children", [])
        for orphan_id in orphan_ids:
            if orphan_id in hier:
                hier[orphan_id]["parent"] = "spec-root"
                if orphan_id not in root_children:
                    root_children.append(orphan_id)

    return FixAction(
        id=f"hierarchy.attach_orphans:{len(orphan_ids)}",
        description=f"Attach {len(orphan_ids)} orphaned node(s) to spec-root",
        category="hierarchy",
        severity=diag.severity,
        auto_apply=True,
        preview=f"Attach {len(orphan_ids)} orphaned node(s) to spec-root",
        apply=apply,
    )


def _build_root_parent_fix(
    diag: Diagnostic, hierarchy: Dict[str, Any]
) -> Optional[FixAction]:
    """Build fix for spec-root having non-null parent."""

    def apply(data: Dict[str, Any]) -> None:
        hier = data.get("hierarchy", {})
        spec_root = hier.get("spec-root")
        if spec_root:
            spec_root["parent"] = None

    return FixAction(
        id="hierarchy.fix_root_parent",
        description="Set spec-root parent to null",
        category="hierarchy",
        severity=diag.severity,
        auto_apply=True,
        preview="Set spec-root parent to null",
        apply=apply,
    )


def _build_missing_fields_fix(
    diag: Diagnostic, hierarchy: Dict[str, Any]
) -> Optional[FixAction]:
    """Build fix for missing node fields."""
    node_id = diag.location
    if not node_id:
        return None

    def apply(data: Dict[str, Any]) -> None:
        hier = data.get("hierarchy", {})
        node = hier.get(node_id)
        if not node:
            return

        if "type" not in node:
            node["type"] = "task"
        if "title" not in node:
            node["title"] = node_id.replace("-", " ").title()
        if "status" not in node:
            node["status"] = "pending"
        if "parent" not in node:
            # Find actual parent by checking which node lists this node as a child
            # This prevents regression where we set parent="spec-root" but the node
            # is actually in another node's children list (causing PARENT_CHILD_MISMATCH)
            actual_parent = "spec-root"  # fallback if not found in any children list
            for other_id, other_node in hier.items():
                if not isinstance(other_node, dict):
                    continue
                children = other_node.get("children", [])
                if isinstance(children, list) and node_id in children:
                    actual_parent = other_id
                    break
            node["parent"] = actual_parent
        if "children" not in node:
            node["children"] = []
        if "total_tasks" not in node:
            node["total_tasks"] = (
                1 if node.get("type") in {"task", "subtask", "verify"} else 0
            )
        if "completed_tasks" not in node:
            node["completed_tasks"] = 0
        if "metadata" not in node:
            node["metadata"] = {}

    return FixAction(
        id=f"node.add_missing_fields:{node_id}",
        description=f"Add missing fields to {node_id}",
        category="node",
        severity=diag.severity,
        auto_apply=True,
        preview=f"Add missing required fields to {node_id}",
        apply=apply,
    )


def _build_type_normalize_fix(
    diag: Diagnostic, hierarchy: Dict[str, Any]
) -> Optional[FixAction]:
    """Build fix for invalid node type."""
    node_id = diag.location
    if not node_id:
        return None

    def apply(data: Dict[str, Any]) -> None:
        hier = data.get("hierarchy", {})
        node = hier.get(node_id)
        if not node:
            return
        node["type"] = _normalize_node_type(node.get("type", ""))

    return FixAction(
        id=f"node.normalize_type:{node_id}",
        description=f"Normalize type for {node_id}",
        category="node",
        severity=diag.severity,
        auto_apply=True,
        preview=f"Normalize node type for {node_id}",
        apply=apply,
    )


def _build_status_normalize_fix(
    diag: Diagnostic, hierarchy: Dict[str, Any]
) -> Optional[FixAction]:
    """Build fix for invalid status."""
    node_id = diag.location
    if not node_id:
        return None

    def apply(data: Dict[str, Any]) -> None:
        hier = data.get("hierarchy", {})
        node = hier.get(node_id)
        if not node:
            return
        node["status"] = _normalize_status(node.get("status"))

    return FixAction(
        id=f"status.normalize:{node_id}",
        description=f"Normalize status for {node_id}",
        category="node",
        severity=diag.severity,
        auto_apply=True,
        preview=f"Normalize status for {node_id}",
        apply=apply,
    )


def _build_title_generate_fix(
    diag: Diagnostic, hierarchy: Dict[str, Any]
) -> Optional[FixAction]:
    """Build fix for empty title."""
    node_id = diag.location
    if not node_id:
        return None

    def apply(data: Dict[str, Any]) -> None:
        hier = data.get("hierarchy", {})
        node = hier.get(node_id)
        if not node:
            return
        node["title"] = node_id.replace("-", " ").replace("_", " ").title()

    return FixAction(
        id=f"node.generate_title:{node_id}",
        description=f"Generate title for {node_id}",
        category="node",
        severity=diag.severity,
        auto_apply=True,
        preview=f"Generate title from node ID for {node_id}",
        apply=apply,
    )


def _build_counts_fix(
    diag: Diagnostic, spec_data: Dict[str, Any]
) -> Optional[FixAction]:
    """Build fix for task count issues."""

    def apply(data: Dict[str, Any]) -> None:
        _recalculate_counts(data)

    return FixAction(
        id="counts.recalculate",
        description="Recalculate task count rollups",
        category="counts",
        severity=diag.severity,
        auto_apply=True,
        preview="Recalculate total/completed task rollups across the hierarchy",
        apply=apply,
    )


def _build_bidirectional_fix(
    diag: Diagnostic, hierarchy: Dict[str, Any]
) -> Optional[FixAction]:
    """Build fix for bidirectional dependency inconsistency."""
    # Parse node IDs from message
    blocks_match = re.search(r"'([^']+)' blocks '([^']+)'", diag.message)
    blocked_by_match = re.search(r"'([^']+)' blocked_by '([^']+)'", diag.message)

    if blocks_match:
        blocker_id = blocks_match.group(1)
        blocked_id = blocks_match.group(2)
    elif blocked_by_match:
        blocked_id = blocked_by_match.group(1)
        blocker_id = blocked_by_match.group(2)
    else:
        return None

    def apply(data: Dict[str, Any]) -> None:
        hier = data.get("hierarchy", {})
        blocker = hier.get(blocker_id)
        blocked = hier.get(blocked_id)
        if not blocker or not blocked:
            return

        # Ensure dependencies structure
        if not isinstance(blocker.get("dependencies"), dict):
            blocker["dependencies"] = {"blocks": [], "blocked_by": [], "depends": []}
        if not isinstance(blocked.get("dependencies"), dict):
            blocked["dependencies"] = {"blocks": [], "blocked_by": [], "depends": []}

        blocker_deps = blocker["dependencies"]
        blocked_deps = blocked["dependencies"]

        # Ensure all fields exist
        for dep_key in ["blocks", "blocked_by", "depends"]:
            blocker_deps.setdefault(dep_key, [])
            blocked_deps.setdefault(dep_key, [])

        # Sync relationship
        if blocked_id not in blocker_deps["blocks"]:
            blocker_deps["blocks"].append(blocked_id)
        if blocker_id not in blocked_deps["blocked_by"]:
            blocked_deps["blocked_by"].append(blocker_id)

    return FixAction(
        id=f"dependency.sync_bidirectional:{blocker_id}-{blocked_id}",
        description=f"Sync bidirectional dependency: {blocker_id} blocks {blocked_id}",
        category="dependency",
        severity=diag.severity,
        auto_apply=True,
        preview=f"Sync bidirectional dependency: {blocker_id} blocks {blocked_id}",
        apply=apply,
    )


def _build_deps_structure_fix(
    diag: Diagnostic, hierarchy: Dict[str, Any]
) -> Optional[FixAction]:
    """Build fix for missing dependencies structure."""
    node_id = diag.location
    if not node_id:
        return None

    def apply(data: Dict[str, Any]) -> None:
        hier = data.get("hierarchy", {})
        node = hier.get(node_id)
        if not node:
            return
        if not isinstance(node.get("dependencies"), dict):
            node["dependencies"] = {"blocks": [], "blocked_by": [], "depends": []}

    return FixAction(
        id=f"dependency.create_structure:{node_id}",
        description=f"Create dependencies structure for {node_id}",
        category="dependency",
        severity=diag.severity,
        auto_apply=True,
        preview=f"Create dependencies structure for {node_id}",
        apply=apply,
    )


def _build_verification_type_fix(
    diag: Diagnostic, hierarchy: Dict[str, Any]
) -> Optional[FixAction]:
    """Build fix for missing verification type."""
    node_id = diag.location
    if not node_id:
        return None

    def apply(data: Dict[str, Any]) -> None:
        hier = data.get("hierarchy", {})
        node = hier.get(node_id)
        if not node:
            return
        metadata = node.setdefault("metadata", {})
        if "verification_type" not in metadata:
            metadata["verification_type"] = "run-tests"

    return FixAction(
        id=f"metadata.fix_verification_type:{node_id}",
        description=f"Set verification_type to 'run-tests' for {node_id}",
        category="metadata",
        severity=diag.severity,
        auto_apply=True,
        preview=f"Set verification_type to 'run-tests' for {node_id}",
        apply=apply,
    )


def _build_invalid_verification_type_fix(
    diag: Diagnostic, hierarchy: Dict[str, Any]
) -> Optional[FixAction]:
    """Build fix for invalid verification type by mapping to canonical value."""
    node_id = diag.location
    if not node_id:
        return None

    def apply(data: Dict[str, Any]) -> None:
        hier = data.get("hierarchy", {})
        node = hier.get(node_id)
        if not node:
            return
        metadata = node.get("metadata", {})
        current_type = metadata.get("verification_type", "")

        # Map legacy values to canonical
        mapped_type = VERIFICATION_TYPE_MAPPING.get(current_type)
        if mapped_type:
            metadata["verification_type"] = mapped_type
        elif current_type not in VALID_VERIFICATION_TYPES:
            metadata["verification_type"] = "manual"  # safe fallback for unknown values

    return FixAction(
        id=f"metadata.fix_invalid_verification_type:{node_id}",
        description=f"Map verification_type to canonical value for {node_id}",
        category="metadata",
        severity=diag.severity,
        auto_apply=True,
        preview=f"Map legacy verification_type to canonical value for {node_id}",
        apply=apply,
    )


# NOTE: We intentionally do not auto-fix missing `metadata.file_path`.
# It must be a real repo-relative path in the target workspace.


def _build_task_category_fix(
    diag: Diagnostic, hierarchy: Dict[str, Any]
) -> Optional[FixAction]:
    """Build fix for invalid task category."""
    node_id = diag.location
    if not node_id:
        return None

    def apply(data: Dict[str, Any]) -> None:
        hier = data.get("hierarchy", {})
        node = hier.get(node_id)
        if not node:
            return
        metadata = node.setdefault("metadata", {})
        # Default to implementation
        metadata["task_category"] = "implementation"

    return FixAction(
        id=f"metadata.fix_task_category:{node_id}",
        description=f"Set task_category to 'implementation' for {node_id}",
        category="metadata",
        severity=diag.severity,
        auto_apply=True,
        preview=f"Set task_category to 'implementation' for {node_id}",
        apply=apply,
    )


def apply_fixes(
    actions: List[FixAction],
    spec_path: str,
    *,
    dry_run: bool = False,
    create_backup: bool = True,
    capture_diff: bool = False,
) -> FixReport:
    """
    Apply fix actions to a spec file.

    Args:
        actions: List of FixAction objects to apply
        spec_path: Path to spec file
        dry_run: If True, don't actually save changes
        create_backup: If True, create backup before modifying
        capture_diff: If True, capture before/after state

    Returns:
        FixReport with results
    """
    report = FixReport(spec_path=spec_path)

    if dry_run:
        report.skipped_actions.extend(actions)
        return report

    try:
        with open(spec_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return report

    # Capture before state
    if capture_diff:
        report.before_state = copy.deepcopy(data)

    # Create backup
    if create_backup:
        backup_path = Path(spec_path).with_suffix(".json.backup")
        try:
            with open(backup_path, "w") as f:
                json.dump(data, f, indent=2)
            report.backup_path = str(backup_path)
        except OSError:
            pass

    # Apply each action
    for action in actions:
        try:
            action.apply(data)
            report.applied_actions.append(action)
        except Exception:
            report.skipped_actions.append(action)

    # Recalculate counts after all fixes
    if report.applied_actions:
        _recalculate_counts(data)

    # Capture after state
    if capture_diff:
        report.after_state = copy.deepcopy(data)

    # Save changes
    try:
        with open(spec_path, "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass

    return report


# Statistics functions


def calculate_stats(
    spec_data: Dict[str, Any], file_path: Optional[str] = None
) -> SpecStats:
    """
    Calculate statistics for a spec file.

    Args:
        spec_data: Parsed JSON spec data
        file_path: Optional path to spec file for size calculation

    Returns:
        SpecStats with calculated metrics
    """
    hierarchy = spec_data.get("hierarchy", {}) or {}

    totals = {
        "nodes": len(hierarchy),
        "tasks": 0,
        "phases": 0,
        "verifications": 0,
    }

    status_counts = {status: 0 for status in STATUS_FIELDS}
    max_depth = 0

    def traverse(node_id: str, depth: int) -> None:
        nonlocal max_depth
        node = hierarchy.get(node_id, {})
        node_type = node.get("type")

        max_depth = max(max_depth, depth)

        if node_type in {"task", "subtask"}:
            totals["tasks"] += 1
            status = node.get("status", "").lower().replace(" ", "_").replace("-", "_")
            if status in status_counts:
                status_counts[status] += 1
        elif node_type == "phase":
            totals["phases"] += 1
        elif node_type == "verify":
            totals["verifications"] += 1

        for child_id in node.get("children", []) or []:
            if child_id in hierarchy:
                traverse(child_id, depth + 1)

    if "spec-root" in hierarchy:
        traverse("spec-root", 0)

    total_tasks = totals["tasks"]
    phase_count = totals["phases"] or 1
    avg_tasks_per_phase = round(total_tasks / phase_count, 2)

    root = hierarchy.get("spec-root", {})
    root_total_tasks = root.get("total_tasks", total_tasks)
    root_completed = root.get("completed_tasks", 0)

    verification_count = totals["verifications"]
    verification_coverage = (verification_count / total_tasks) if total_tasks else 0.0
    progress = (root_completed / root_total_tasks) if root_total_tasks else 0.0

    file_size = 0.0
    if file_path:
        try:
            file_size = Path(file_path).stat().st_size / 1024
        except OSError:
            file_size = 0.0

    return SpecStats(
        spec_id=spec_data.get("spec_id", "unknown"),
        title=spec_data.get("title", ""),
        version=spec_data.get("version", ""),
        status=root.get("status", "unknown"),
        totals=totals,
        status_counts=status_counts,
        max_depth=max_depth,
        avg_tasks_per_phase=avg_tasks_per_phase,
        verification_coverage=verification_coverage,
        progress=progress,
        file_size_kb=file_size,
    )


# Helper functions


def _is_valid_spec_id(spec_id: str) -> bool:
    """Check if spec_id follows the recommended format."""
    pattern = r"^[a-z0-9-]+-\d{4}-\d{2}-\d{2}-\d{3}$"
    return bool(re.match(pattern, spec_id))


def _is_valid_iso8601(value: str) -> bool:
    """Check if value is valid ISO 8601 date."""
    try:
        # Try parsing with Z suffix
        if value.endswith("Z"):
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            datetime.fromisoformat(value)
        return True
    except ValueError:
        return False


def _normalize_timestamp(value: Any) -> Optional[str]:
    """Normalize timestamp to ISO 8601 format."""
    if not value:
        return None

    text = str(value).strip()
    candidate = text.replace("Z", "")

    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            dt = datetime.strptime(candidate, fmt)
            return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            continue

    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        return None


def _normalize_status(value: Any) -> str:
    """Normalize status value."""
    if not value:
        return "pending"

    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "inprogress": "in_progress",
        "in__progress": "in_progress",
        "todo": "pending",
        "to_do": "pending",
        "complete": "completed",
        "done": "completed",
    }
    text = mapping.get(text, text)

    if text in VALID_STATUSES:
        return text

    return "pending"


def _normalize_node_type(value: Any) -> str:
    """Normalize node type value."""
    if not value:
        return "task"

    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    mapping = {
        "tasks": "task",
        "sub_task": "subtask",
        "verification": "verify",
        "validate": "verify",
    }
    text = mapping.get(text, text)

    if text in VALID_NODE_TYPES:
        return text

    return "task"


def _recalculate_counts(spec_data: Dict[str, Any]) -> None:
    """Recalculate task counts for all nodes in hierarchy."""
    hierarchy = spec_data.get("hierarchy", {})
    if not hierarchy:
        return

    # Process bottom-up: leaves first, then parents
    def calculate_node(node_id: str) -> tuple:
        """Return (total_tasks, completed_tasks) for a node."""
        node = hierarchy.get(node_id, {})
        children = node.get("children", [])
        node_type = node.get("type", "")
        status = node.get("status", "")

        if not children:
            # Leaf node
            if node_type in {"task", "subtask", "verify"}:
                total = 1
                completed = 1 if status == "completed" else 0
            else:
                total = 0
                completed = 0
        else:
            # Parent node: sum children
            total = 0
            completed = 0
            for child_id in children:
                if child_id in hierarchy:
                    child_total, child_completed = calculate_node(child_id)
                    total += child_total
                    completed += child_completed

        node["total_tasks"] = total
        node["completed_tasks"] = completed
        return total, completed

    if "spec-root" in hierarchy:
        calculate_node("spec-root")


# Verification management functions

# Valid verification results
VERIFICATION_RESULTS = ("PASSED", "FAILED", "PARTIAL")


def add_verification(
    spec_data: Dict[str, Any],
    verify_id: str,
    result: str,
    command: Optional[str] = None,
    output: Optional[str] = None,
    issues: Optional[str] = None,
    notes: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    Add verification result to a verify node.

    Records verification results including test outcomes, command output,
    and issues found during verification.

    Args:
        spec_data: The loaded spec data dict (modified in place).
        verify_id: Verification node ID (e.g., verify-1-1).
        result: Verification result (PASSED, FAILED, PARTIAL).
        command: Optional command that was run for verification.
        output: Optional command output or test results.
        issues: Optional issues found during verification.
        notes: Optional additional notes about the verification.

    Returns:
        Tuple of (success, error_message).
        On success: (True, None)
        On failure: (False, "error message")
    """
    # Validate result
    result_upper = result.upper().strip()
    if result_upper not in VERIFICATION_RESULTS:
        return (
            False,
            f"Invalid result '{result}'. Must be one of: {', '.join(VERIFICATION_RESULTS)}",
        )

    # Get hierarchy
    hierarchy = spec_data.get("hierarchy")
    if not hierarchy or not isinstance(hierarchy, dict):
        return False, "Invalid spec data: missing or invalid hierarchy"

    # Find the verify node
    node = hierarchy.get(verify_id)
    if node is None:
        return False, f"Verification node '{verify_id}' not found"

    # Validate node type
    node_type = node.get("type")
    if node_type != "verify":
        return False, f"Node '{verify_id}' is type '{node_type}', expected 'verify'"

    # Get or create metadata
    metadata = node.get("metadata")
    if metadata is None:
        metadata = {}
        node["metadata"] = metadata

    # Build verification result entry
    verification_entry: Dict[str, Any] = {
        "result": result_upper,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    if command:
        verification_entry["command"] = command.strip()

    if output:
        # Truncate output if very long
        max_output_len = MAX_STRING_LENGTH
        output_text = output.strip()
        if len(output_text) > max_output_len:
            output_text = output_text[:max_output_len] + "\n... (truncated)"
        verification_entry["output"] = output_text

    if issues:
        verification_entry["issues"] = issues.strip()

    if notes:
        verification_entry["notes"] = notes.strip()

    # Add to verification history (keep last N entries)
    verification_history = metadata.get("verification_history", [])
    if not isinstance(verification_history, list):
        verification_history = []

    verification_history.append(verification_entry)

    # Keep only last 10 entries
    if len(verification_history) > 10:
        verification_history = verification_history[-10:]

    metadata["verification_history"] = verification_history

    # Update latest result fields for quick access
    metadata["last_result"] = result_upper
    metadata["last_verified_at"] = verification_entry["timestamp"]

    return True, None


def execute_verification(
    spec_data: Dict[str, Any],
    verify_id: str,
    record: bool = False,
    timeout: int = 300,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute verification command and capture results.

    Runs the verification command defined in a verify node's metadata
    and captures output, exit code, and result status.

    Args:
        spec_data: The loaded spec data dict.
        verify_id: Verification node ID (e.g., verify-1-1).
        record: If True, automatically record result to spec using add_verification().
        timeout: Command timeout in seconds (default: 300).
        cwd: Working directory for command execution (default: current directory).

    Returns:
        Dict with execution results:
        - success: Whether execution completed (not result status)
        - spec_id: The specification ID
        - verify_id: The verification ID
        - result: Execution result (PASSED, FAILED, PARTIAL)
        - command: Command that was executed
        - output: Combined stdout/stderr output
        - exit_code: Command exit code
        - recorded: Whether result was recorded to spec
        - error: Error message if execution failed

    Example:
        >>> result = execute_verification(spec_data, "verify-1-1", record=True)
        >>> if result["success"]:
        ...     print(f"Verification {result['result']}: {result['exit_code']}")
    """
    import subprocess

    response: Dict[str, Any] = {
        "success": False,
        "spec_id": spec_data.get("spec_id", "unknown"),
        "verify_id": verify_id,
        "result": None,
        "command": None,
        "output": None,
        "exit_code": None,
        "recorded": False,
        "error": None,
    }

    # Get hierarchy
    hierarchy = spec_data.get("hierarchy")
    if not hierarchy or not isinstance(hierarchy, dict):
        response["error"] = "Invalid spec data: missing or invalid hierarchy"
        return response

    # Find the verify node
    node = hierarchy.get(verify_id)
    if node is None:
        response["error"] = f"Verification node '{verify_id}' not found"
        return response

    # Validate node type
    node_type = node.get("type")
    if node_type != "verify":
        response["error"] = (
            f"Node '{verify_id}' is type '{node_type}', expected 'verify'"
        )
        return response

    # Get command from metadata
    metadata = node.get("metadata", {})
    command = metadata.get("command")

    if not command:
        response["error"] = f"No command defined in verify node '{verify_id}' metadata"
        return response

    response["command"] = command

    # Execute the command
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )

        exit_code = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        # Combine output
        output_parts = []
        if stdout.strip():
            output_parts.append(stdout.strip())
        if stderr.strip():
            output_parts.append(f"[stderr]\n{stderr.strip()}")
        output = "\n".join(output_parts) if output_parts else "(no output)"

        # Truncate if too long
        if len(output) > MAX_STRING_LENGTH:
            output = output[:MAX_STRING_LENGTH] + "\n... (truncated)"

        response["exit_code"] = exit_code
        response["output"] = output

        # Determine result based on exit code
        if exit_code == 0:
            result = "PASSED"
        else:
            result = "FAILED"

        response["result"] = result
        response["success"] = True

        # Optionally record result to spec
        if record:
            record_success, record_error = add_verification(
                spec_data=spec_data,
                verify_id=verify_id,
                result=result,
                command=command,
                output=output,
            )
            if record_success:
                response["recorded"] = True
            else:
                response["recorded"] = False
                # Don't fail the whole operation, just note the recording failed
                if response.get("error"):
                    response["error"] += f"; Recording failed: {record_error}"
                else:
                    response["error"] = f"Recording failed: {record_error}"

    except subprocess.TimeoutExpired:
        response["error"] = f"Command timed out after {timeout} seconds"
        response["result"] = "FAILED"
        response["exit_code"] = -1
        response["output"] = f"Command timed out after {timeout} seconds"

    except subprocess.SubprocessError as e:
        response["error"] = f"Command execution failed: {e}"
        response["result"] = "FAILED"

    except Exception as e:
        response["error"] = f"Unexpected error: {e}"
        response["result"] = "FAILED"

    return response


def format_verification_summary(
    verification_data: Dict[str, Any] | List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Format verification results into a human-readable summary.

    Processes verification results (from execute_verification or JSON input)
    and produces a structured summary with counts and formatted text.

    Args:
        verification_data: Either:
            - A single verification result dict (from execute_verification)
            - A list of verification result dicts
            - A dict with "verifications" key containing a list

    Returns:
        Dict with formatted summary:
        - summary: Human-readable summary text
        - total_verifications: Total number of verifications
        - passed: Number of passed verifications
        - failed: Number of failed verifications
        - partial: Number of partial verifications
        - results: List of individual result summaries

    Example:
        >>> results = [
        ...     execute_verification(spec_data, "verify-1"),
        ...     execute_verification(spec_data, "verify-2"),
        ... ]
        >>> summary = format_verification_summary(results)
        >>> print(summary["summary"])
    """
    # Normalize input to a list of verification results
    verifications: List[Dict[str, Any]] = []

    if isinstance(verification_data, list):
        verifications = verification_data
    elif isinstance(verification_data, dict):
        if "verifications" in verification_data:
            verifications = verification_data.get("verifications", [])
        else:
            # Single verification result
            verifications = [verification_data]

    # Count results by type
    passed = 0
    failed = 0
    partial = 0
    results: List[Dict[str, Any]] = []

    for v in verifications:
        if not isinstance(v, dict):
            continue

        result = (v.get("result") or "").upper()
        verify_id = v.get("verify_id", "unknown")
        command = v.get("command", "")
        output = v.get("output", "")
        error = v.get("error")

        # Count by result type
        if result == "PASSED":
            passed += 1
            status_icon = "✓"
        elif result == "FAILED":
            failed += 1
            status_icon = "✗"
        elif result == "PARTIAL":
            partial += 1
            status_icon = "◐"
        else:
            status_icon = "?"

        # Build individual result summary
        result_entry: Dict[str, Any] = {
            "verify_id": verify_id,
            "result": result or "UNKNOWN",
            "status_icon": status_icon,
            "command": command,
        }

        if error:
            result_entry["error"] = error

        # Truncate output for summary
        if output:
            output_preview = output[:200].strip()
            if len(output) > 200:
                output_preview += "..."
            result_entry["output_preview"] = output_preview

        results.append(result_entry)

    # Calculate totals
    total = len(results)

    # Build summary text
    summary_lines = []
    summary_lines.append(f"Verification Summary: {total} total")
    summary_lines.append(f"  ✓ Passed:  {passed}")
    summary_lines.append(f"  ✗ Failed:  {failed}")
    if partial > 0:
        summary_lines.append(f"  ◐ Partial: {partial}")
    summary_lines.append("")

    # Add individual results
    if results:
        summary_lines.append("Results:")
        for r in results:
            icon = r["status_icon"]
            vid = r["verify_id"]
            res = r["result"]
            cmd = r.get("command", "")

            line = f"  {icon} {vid}: {res}"
            if cmd:
                # Truncate command for display
                cmd_display = cmd[:50]
                if len(cmd) > 50:
                    cmd_display += "..."
                line += f" ({cmd_display})"

            summary_lines.append(line)

            if r.get("error"):
                summary_lines.append(f"      Error: {r['error']}")

    summary_text = "\n".join(summary_lines)

    return {
        "summary": summary_text,
        "total_verifications": total,
        "passed": passed,
        "failed": failed,
        "partial": partial,
        "results": results,
    }
