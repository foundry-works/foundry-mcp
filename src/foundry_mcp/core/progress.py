"""
Progress calculation utilities for SDD JSON specs.
Provides hierarchical progress recalculation and status updates.
"""

from datetime import datetime, timezone
from typing import Dict, List, Any


# Status icons for task visualization
STATUS_ICONS = {
    "pending": "⏳",
    "in_progress": "🔄",
    "completed": "✅",
    "blocked": "🚫",
    "failed": "❌",
}


def get_status_icon(status: str) -> str:
    """
    Get icon for a task status.

    Args:
        status: Task status string

    Returns:
        Status icon character
    """
    return STATUS_ICONS.get(status, "❓")


def recalculate_progress(spec_data: Dict[str, Any], node_id: str = "spec-root") -> Dict[str, Any]:
    """
    Recursively recalculate progress for a node and all its parents.

    Modifies spec_data in-place by updating completed_tasks, total_tasks,
    and status fields for the node and all ancestors.

    Args:
        spec_data: JSON spec file data dictionary
        node_id: Node to start recalculation from (default: spec-root)

    Returns:
        The modified spec_data dictionary (for convenience/chaining)
    """
    if not spec_data:
        return {}

    hierarchy = spec_data.get("hierarchy", {})

    if node_id not in hierarchy:
        return spec_data

    node = hierarchy[node_id]
    children = node.get("children", [])

    if not children:
        # Leaf node - set based on own status
        node["completed_tasks"] = 1 if node.get("status") == "completed" else 0
        node["total_tasks"] = 1
    else:
        # Non-leaf node - recursively calculate from children
        total_completed = 0
        total_tasks = 0

        for child_id in children:
            # Recursively recalculate child first
            recalculate_progress(spec_data, child_id)

            child = hierarchy.get(child_id, {})
            total_completed += child.get("completed_tasks", 0)
            total_tasks += child.get("total_tasks", 0)

        node["completed_tasks"] = total_completed
        node["total_tasks"] = total_tasks

    # Update node status based on progress
    update_node_status(node, hierarchy)

    return spec_data


def update_node_status(node: Dict[str, Any], hierarchy: Dict[str, Any] = None) -> None:
    """
    Update a node's status based on its children's progress.

    Modifies node in-place. Does not affect manually set statuses
    for leaf nodes (tasks).

    Args:
        node: Node dictionary from hierarchy
        hierarchy: Full hierarchy dictionary (needed to check child statuses)
    """
    # Don't auto-update status for leaf tasks (they're set manually)
    if node.get("type") == "task" and not node.get("children"):
        return

    # Track if node is blocked (we'll skip status changes but allow parent updates)
    is_blocked = node.get("status") == "blocked"

    # Handle manually-completed tasks with children
    if node.get("metadata", {}).get("completed_at") and node.get("children"):
        # Check if actual children progress matches the "completed" state
        actual_completed = node.get("completed_tasks", 0)
        total = node.get("total_tasks", 0)

        if actual_completed < total:
            # Inconsistent state: parent marked complete but children aren't.
            # Remove completed_at to allow normal status calculation to take over below.
            if "metadata" in node and "completed_at" in node["metadata"]:
                del node["metadata"]["completed_at"]
        else:
            # Consistent state, enforce completion
            node["status"] = "completed"

    # If blocked, don't change status but continue to allow count updates
    if is_blocked:
        return

    # Check if any children are in_progress (takes priority over count-based logic)
    if hierarchy and node.get("children"):
        for child_id in node.get("children", []):
            child = hierarchy.get(child_id, {})
            if child.get("status") == "in_progress":
                node["status"] = "in_progress"
                return

    completed = node.get("completed_tasks", 0)
    total = node.get("total_tasks", 0)

    if total == 0:
        node["status"] = "pending"
    elif completed == 0:
        node["status"] = "pending"
    elif completed == total:
        # Check if status is changing to completed (auto-completion)
        was_completed = node.get("status") == "completed"
        node["status"] = "completed"

        # Set needs_journaling flag for parent nodes (groups, phases)
        # when they auto-complete (not manually set)
        if not was_completed and node.get("type") in ["group", "phase"]:
            if "metadata" not in node:
                node["metadata"] = {}
            node["metadata"]["needs_journaling"] = True
            node["metadata"]["completed_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    else:
        node["status"] = "in_progress"


def update_parent_status(spec_data: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    """
    Update status and progress for a node's parent chain.

    Use this after updating a task status to propagate changes up the hierarchy.

    Args:
        spec_data: JSON spec file data dictionary
        node_id: Node whose parents should be updated

    Returns:
        The modified spec_data dictionary (for convenience/chaining)
    """
    if not spec_data:
        return {}

    hierarchy = spec_data.get("hierarchy", {})

    if node_id not in hierarchy:
        return spec_data

    node = hierarchy[node_id]
    parent_id = node.get("parent")

    # Walk up the parent chain
    while parent_id and parent_id in hierarchy:
        # Recalculate progress for parent
        recalculate_progress(spec_data, parent_id)

        # Move to next parent
        parent = hierarchy[parent_id]
        parent_id = parent.get("parent")

    return spec_data


def get_progress_summary(spec_data: Dict[str, Any], node_id: str = "spec-root") -> Dict[str, Any]:
    """
    Get progress summary for a node.

    Args:
        spec_data: JSON spec file data
        node_id: Node to get progress for (default: spec-root)

    Returns:
        Dictionary with progress information
    """
    if not spec_data:
        return {"error": "No state data provided"}

    # Recalculate progress to ensure counts are up-to-date
    recalculate_progress(spec_data, node_id)

    hierarchy = spec_data.get("hierarchy", {})
    node = hierarchy.get(node_id)

    if not node:
        return {"error": f"Node {node_id} not found"}

    total = node.get("total_tasks", 0)
    completed = node.get("completed_tasks", 0)
    percentage = int((completed / total * 100)) if total > 0 else 0

    # Extract spec_id from spec_data
    spec_id = spec_data.get("spec_id", "")

    # Find current phase (first in_progress, or first pending if none)
    current_phase = None
    for key, value in hierarchy.items():
        if value.get("type") == "phase":
            if value.get("status") == "in_progress":
                current_phase = {
                    "id": key,
                    "title": value.get("title", ""),
                    "completed": value.get("completed_tasks", 0),
                    "total": value.get("total_tasks", 0)
                }
                break
            elif current_phase is None and value.get("status") == "pending":
                current_phase = {
                    "id": key,
                    "title": value.get("title", ""),
                    "completed": value.get("completed_tasks", 0),
                    "total": value.get("total_tasks", 0)
                }

    return {
        "node_id": node_id,
        "spec_id": spec_id,
        "title": node.get("title", ""),
        "type": node.get("type", ""),
        "status": node.get("status", ""),
        "total_tasks": total,
        "completed_tasks": completed,
        "percentage": percentage,
        "remaining_tasks": total - completed,
        "current_phase": current_phase
    }


def list_phases(spec_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    List all phases with their status and progress.

    Args:
        spec_data: JSON spec file data

    Returns:
        List of phase dictionaries
    """
    if not spec_data:
        return []

    hierarchy = spec_data.get("hierarchy", {})

    phases = []
    for key, value in hierarchy.items():
        if value.get("type") == "phase":
            total = value.get("total_tasks", 0)
            completed = value.get("completed_tasks", 0)
            percentage = int((completed / total * 100)) if total > 0 else 0

            phases.append({
                "id": key,
                "title": value.get("title", ""),
                "status": value.get("status", ""),
                "completed_tasks": completed,
                "total_tasks": total,
                "percentage": percentage
            })

    # Sort by phase ID (phase-1, phase-2, etc.)
    phases.sort(key=lambda p: p["id"])

    return phases


def get_task_counts_by_status(spec_data: Dict[str, Any]) -> Dict[str, int]:
    """
    Count tasks by their status.

    Args:
        spec_data: JSON spec file data

    Returns:
        Dictionary mapping status to count
    """
    if not spec_data:
        return {"pending": 0, "in_progress": 0, "completed": 0, "blocked": 0}

    hierarchy = spec_data.get("hierarchy", {})

    counts = {
        "pending": 0,
        "in_progress": 0,
        "completed": 0,
        "blocked": 0
    }

    for node in hierarchy.values():
        if node.get("type") == "task":
            status = node.get("status", "pending")
            if status in counts:
                counts[status] += 1

    return counts


def sync_computed_fields(spec_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronize computed fields to their canonical top-level locations.

    This function should be called after any task status change to ensure
    progress_percentage, current_phase, and status are persisted to the spec.

    Updates (in-place):
    - progress_percentage: calculated from hierarchy counts
    - current_phase: first in_progress phase, or first pending if none
    - status: based on overall progress

    Args:
        spec_data: Spec data dictionary (modified in place)

    Returns:
        Dict with computed values for confirmation
    """
    if not spec_data:
        return {}

    hierarchy = spec_data.get("hierarchy", {})
    root = hierarchy.get("spec-root", {})

    # Calculate progress percentage
    total = root.get("total_tasks", 0)
    completed = root.get("completed_tasks", 0)
    progress_pct = int((completed / total * 100)) if total > 0 else 0

    # Determine current phase (first in_progress, or first pending if none)
    current_phase = None
    for key, node in hierarchy.items():
        if node.get("type") == "phase":
            if node.get("status") == "in_progress":
                current_phase = key
                break
            elif current_phase is None and node.get("status") == "pending":
                current_phase = key

    # Determine overall status based on progress
    if total == 0:
        status = "pending"
    elif completed == total:
        status = "completed"
    elif completed > 0:
        status = "in_progress"
    else:
        status = "pending"

    # Check if any task is blocked - if so, spec is blocked
    for node in hierarchy.values():
        if node.get("status") == "blocked":
            status = "blocked"
            break

    # Update top-level fields (canonical location)
    spec_data["progress_percentage"] = progress_pct
    spec_data["current_phase"] = current_phase
    spec_data["status"] = status

    # Update last_updated timestamp
    spec_data["last_updated"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    return {
        "progress_percentage": progress_pct,
        "current_phase": current_phase,
        "status": status
    }
