"""Fix action builders for validation diagnostics."""

import re
from typing import Any, Dict, List, Optional

from foundry_mcp.core.validation.constants import VALID_VERIFICATION_TYPES
from foundry_mcp.core.validation.models import Diagnostic, FixAction, ValidationResult
from foundry_mcp.core.validation.normalization import (
    _normalize_node_type,
    _normalize_status,
    _normalize_timestamp,
)
from foundry_mcp.core.validation.stats import _recalculate_counts


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

        if current_type not in VALID_VERIFICATION_TYPES:
            metadata["verification_type"] = "manual"  # safe fallback for unknown values

    return FixAction(
        id=f"metadata.fix_invalid_verification_type:{node_id}",
        description=f"Set verification_type to a canonical value for {node_id}",
        category="metadata",
        severity=diag.severity,
        auto_apply=True,
        preview=f"Set verification_type to canonical value for {node_id}",
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
