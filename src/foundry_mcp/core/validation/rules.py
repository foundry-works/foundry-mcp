"""
Spec validation rules and checks.

Security Note:
    This module uses size limits from foundry_mcp.core.security to protect
    against resource exhaustion attacks. See dev_docs/mcp_best_practices/04-validation-input-hygiene.md
"""

from typing import Any, Dict

from foundry_mcp.core.security import (
    MAX_ARRAY_LENGTH,
    MAX_NESTED_DEPTH,
)
from foundry_mcp.core.validation.constants import (
    VALID_NODE_TYPES,
    VALID_STATUSES,
    VALID_TASK_CATEGORIES,
    VALID_VERIFICATION_TYPES,
)
from foundry_mcp.core.validation.models import Diagnostic, ValidationResult
from foundry_mcp.core.validation.normalization import (
    _is_valid_iso8601,
    _is_valid_spec_id,
    _suggest_value,
)


def _requires_rich_task_fields(spec_data: Dict[str, Any]) -> bool:
    """Check if spec requires rich task fields based on explicit complexity metadata."""
    metadata = spec_data.get("metadata", {})
    if not isinstance(metadata, dict):
        return False

    # Only check explicit complexity metadata (template no longer indicates complexity)
    complexity = metadata.get("complexity")
    if isinstance(complexity, str) and complexity.strip().lower() in {
        "medium",
        "complex",
        "high",
    }:
        return True

    return False


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
                    message="Spec metadata.mission is required when complexity is medium/complex/high",
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
