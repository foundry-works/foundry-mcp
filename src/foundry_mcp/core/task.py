"""
Task discovery and dependency operations for SDD workflows.
Provides finding next tasks, dependency checking, and task preparation.
"""

import re
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from foundry_mcp.core.spec import (
    CATEGORIES,
    load_spec,
    save_spec,
    find_spec_file,
    find_specs_directory,
    get_node,
)
from foundry_mcp.core.responses import success_response, error_response

# Valid task types for add_task
TASK_TYPES = ("task", "subtask", "verify")


def is_unblocked(spec_data: Dict[str, Any], task_id: str, task_data: Dict[str, Any]) -> bool:
    """
    Check if all blocking dependencies are completed.

    This checks both task-level dependencies and phase-level dependencies.
    A task is blocked if:
    1. Any of its direct task dependencies are not completed, OR
    2. Its parent phase is blocked by an incomplete phase

    Args:
        spec_data: JSON spec file data
        task_id: Task identifier
        task_data: Task data dictionary

    Returns:
        True if task has no blockers or all blockers are completed
    """
    hierarchy = spec_data.get("hierarchy", {})

    # Check task-level dependencies
    blocked_by = task_data.get("dependencies", {}).get("blocked_by", [])
    for blocker_id in blocked_by:
        blocker = hierarchy.get(blocker_id)
        if not blocker or blocker.get("status") != "completed":
            return False

    # Check phase-level dependencies
    # Walk up to find the parent phase
    parent_phase_id = None
    current = task_data
    while current:
        parent_id = current.get("parent")
        if not parent_id:
            break
        parent = hierarchy.get(parent_id)
        if not parent:
            break
        if parent.get("type") == "phase":
            parent_phase_id = parent_id
            break
        current = parent

    # If task belongs to a phase, check if that phase is blocked
    if parent_phase_id:
        parent_phase = hierarchy.get(parent_phase_id)
        if parent_phase:
            phase_blocked_by = parent_phase.get("dependencies", {}).get("blocked_by", [])
            for blocker_id in phase_blocked_by:
                blocker = hierarchy.get(blocker_id)
                if not blocker or blocker.get("status") != "completed":
                    return False

    return True


def is_in_current_phase(spec_data: Dict[str, Any], task_id: str, phase_id: str) -> bool:
    """
    Check if task belongs to current phase (including nested groups).

    Args:
        spec_data: JSON spec file data
        task_id: Task identifier
        phase_id: Phase identifier to check against

    Returns:
        True if task is within the phase hierarchy
    """
    hierarchy = spec_data.get("hierarchy", {})
    task = hierarchy.get(task_id)
    if not task:
        return False

    # Walk up parent chain to find phase
    current = task
    while current:
        parent_id = current.get("parent")
        if parent_id == phase_id:
            return True
        if not parent_id:
            return False
        current = hierarchy.get(parent_id)
    return False


def get_next_task(spec_data: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Find the next actionable task.

    Searches phases in order (in_progress first, then pending).
    Within each phase, finds leaf tasks (no children) before parent tasks.
    Only returns unblocked tasks with pending status.

    Args:
        spec_data: JSON spec file data

    Returns:
        Tuple of (task_id, task_data) or None if no task available
    """
    hierarchy = spec_data.get("hierarchy", {})

    # Get all phases in order
    spec_root = hierarchy.get("spec-root", {})
    phase_order = spec_root.get("children", [])

    # Build list of phases to check: in_progress first, then pending
    phases_to_check = []

    # First, add any in_progress phases
    for phase_id in phase_order:
        phase = hierarchy.get(phase_id, {})
        if phase.get("type") == "phase" and phase.get("status") == "in_progress":
            phases_to_check.append(phase_id)

    # Then add pending phases
    for phase_id in phase_order:
        phase = hierarchy.get(phase_id, {})
        if phase.get("type") == "phase" and phase.get("status") == "pending":
            phases_to_check.append(phase_id)

    if not phases_to_check:
        return None

    # Try each phase until we find actionable tasks
    for current_phase in phases_to_check:
        # Find first available task or subtask in current phase
        # Prefer leaf tasks (no children) over parent tasks
        candidates = []
        for key, value in hierarchy.items():
            if (value.get("type") in ["task", "subtask", "verify"] and
                value.get("status") == "pending" and
                is_unblocked(spec_data, key, value) and
                is_in_current_phase(spec_data, key, current_phase)):
                has_children = len(value.get("children", [])) > 0
                candidates.append((key, value, has_children))

        if candidates:
            # Sort: leaf tasks first (has_children=False), then by ID
            candidates.sort(key=lambda x: (x[2], x[0]))
            return (candidates[0][0], candidates[0][1])

    # No actionable tasks found in any phase
    return None


def check_dependencies(spec_data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """
    Check dependency status for a task.

    Args:
        spec_data: JSON spec file data
        task_id: Task identifier

    Returns:
        Dictionary with dependency analysis including:
        - task_id: The task being checked
        - can_start: Whether the task is unblocked
        - blocked_by: List of blocking task info
        - soft_depends: List of soft dependency info
        - blocks: List of tasks this blocks
    """
    hierarchy = spec_data.get("hierarchy", {})
    task = hierarchy.get(task_id)

    if not task:
        return {"error": f"Task {task_id} not found"}

    deps = task.get("dependencies", {})
    blocked_by = deps.get("blocked_by", [])
    depends = deps.get("depends", [])
    blocks = deps.get("blocks", [])

    result = {
        "task_id": task_id,
        "can_start": is_unblocked(spec_data, task_id, task),
        "blocked_by": [],
        "soft_depends": [],
        "blocks": []
    }

    # Get info for blocking tasks
    for dep_id in blocked_by:
        dep_task = hierarchy.get(dep_id)
        if dep_task:
            result["blocked_by"].append({
                "id": dep_id,
                "title": dep_task.get("title", ""),
                "status": dep_task.get("status", ""),
                "file": dep_task.get("metadata", {}).get("file_path", "")
            })

    # Get info for soft dependencies
    for dep_id in depends:
        dep_task = hierarchy.get(dep_id)
        if dep_task:
            result["soft_depends"].append({
                "id": dep_id,
                "title": dep_task.get("title", ""),
                "status": dep_task.get("status", ""),
                "file": dep_task.get("metadata", {}).get("file_path", "")
            })

    # Get info for tasks this blocks
    for dep_id in blocks:
        dep_task = hierarchy.get(dep_id)
        if dep_task:
            result["blocks"].append({
                "id": dep_id,
                "title": dep_task.get("title", ""),
                "status": dep_task.get("status", ""),
                "file": dep_task.get("metadata", {}).get("file_path", "")
            })

    return result


def _get_sibling_ids(
    hierarchy: Dict[str, Dict[str, Any]],
    parent_id: str,
    parent_node: Dict[str, Any],
) -> List[str]:
    """Return sibling IDs for a parent, falling back to scanning the hierarchy."""
    children = parent_node.get("children", [])
    if isinstance(children, list) and children:
        return [child_id for child_id in children if child_id in hierarchy]

    return [
        node_id
        for node_id, node in hierarchy.items()
        if node.get("parent") == parent_id
    ]


def _get_latest_journal_excerpt(
    journal_entries: List[Dict[str, Any]],
    task_id: str,
) -> Optional[Dict[str, Any]]:
    """Return the most recent journal entry for the given task."""
    if not journal_entries:
        return None

    filtered = [
        entry for entry in journal_entries if entry.get("task_id") == task_id
    ]
    if not filtered:
        return None

    filtered.sort(key=lambda entry: entry.get("timestamp") or "", reverse=True)
    latest = filtered[0]
    summary = (latest.get("content") or "").strip()

    return {
        "timestamp": latest.get("timestamp"),
        "entry_type": latest.get("entry_type"),
        "summary": summary,
    }


def _find_phase_node(hierarchy: Dict[str, Dict[str, Any]], task_node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Walk ancestor chain to find the nearest phase node."""
    current = task_node
    while current:
        parent_id = current.get("parent")
        if not parent_id:
            return None
        parent = hierarchy.get(parent_id)
        if not parent:
            return None
        if parent.get("type") == "phase":
            return parent
        current = parent
    return None


def get_previous_sibling(spec_data: Dict[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
    """
    Return metadata about the previous sibling for the given task.

    Args:
        spec_data: Loaded JSON spec dictionary.
        task_id: ID of the current task.

    Returns:
        Dictionary describing the previous sibling or None when the task is
        first in its group / has no siblings.
    """
    if not spec_data:
        return None

    hierarchy = spec_data.get("hierarchy", {})
    task = hierarchy.get(task_id)
    if not task:
        return None

    parent_id = task.get("parent")
    if not parent_id:
        return None

    parent = hierarchy.get(parent_id, {})
    sibling_ids = _get_sibling_ids(hierarchy, parent_id, parent)
    if not sibling_ids:
        return None

    try:
        task_index = sibling_ids.index(task_id)
    except ValueError:
        return None

    if task_index == 0:
        return None

    previous_id = sibling_ids[task_index - 1]
    previous_task = hierarchy.get(previous_id)
    if not previous_task:
        return None

    metadata = previous_task.get("metadata", {}) or {}
    journal_excerpt = _get_latest_journal_excerpt(
        spec_data.get("journal", []),
        previous_id,
    )

    return {
        "id": previous_id,
        "title": previous_task.get("title", ""),
        "status": previous_task.get("status", ""),
        "type": previous_task.get("type", ""),
        "file_path": metadata.get("file_path"),
        "completed_at": metadata.get("completed_at"),
        "journal_excerpt": journal_excerpt,
    }


def get_parent_context(spec_data: Dict[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
    """
    Return contextual information about the parent node for a task.

    Args:
        spec_data: Loaded JSON spec dictionary.
        task_id: ID of the current task.

    Returns:
        Dictionary with parent metadata or None if the task has no parent.
    """
    if not spec_data:
        return None

    hierarchy = spec_data.get("hierarchy", {})
    task = hierarchy.get(task_id)
    if not task:
        return None

    parent_id = task.get("parent")
    if not parent_id:
        return None

    parent = hierarchy.get(parent_id)
    if not parent:
        return None

    parent_metadata = parent.get("metadata", {}) or {}
    description = (
        parent_metadata.get("description")
        or parent_metadata.get("note")
        or parent.get("description")
    )

    children_ids = _get_sibling_ids(hierarchy, parent_id, parent)
    children_entries = [
        {
            "id": child_id,
            "title": hierarchy.get(child_id, {}).get("title", ""),
            "status": hierarchy.get(child_id, {}).get("status", ""),
        }
        for child_id in children_ids
    ]

    position_label = None
    if children_ids and task_id in children_ids:
        index = children_ids.index(task_id) + 1
        total = len(children_ids)
        label = "subtasks" if parent.get("type") == "task" else "children"
        position_label = f"{index} of {total} {label}"

    remaining_tasks = None
    completed_tasks = parent.get("completed_tasks")
    total_tasks = parent.get("total_tasks")
    if isinstance(completed_tasks, int) and isinstance(total_tasks, int):
        remaining_tasks = max(total_tasks - completed_tasks, 0)

    return {
        "id": parent_id,
        "title": parent.get("title", ""),
        "type": parent.get("type", ""),
        "status": parent.get("status", ""),
        "description": description,
        "completed_tasks": completed_tasks,
        "total_tasks": total_tasks,
        "remaining_tasks": remaining_tasks,
        "position_label": position_label,
        "children": children_entries,
    }


def get_phase_context(spec_data: Dict[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
    """
    Return phase-level context for a task, including progress metrics.

    Args:
        spec_data: Loaded JSON spec dictionary.
        task_id: ID of the current task.

    Returns:
        Dictionary with phase data or None if the task does not belong to a phase.
    """
    if not spec_data:
        return None

    hierarchy = spec_data.get("hierarchy", {})
    task = hierarchy.get(task_id)
    if not task:
        return None

    phase_node = _find_phase_node(hierarchy, task)
    if not phase_node:
        return None

    phase_id = None
    for node_id, node in hierarchy.items():
        if node is phase_node:
            phase_id = node_id
            break

    phase_metadata = phase_node.get("metadata", {}) or {}
    summary = (
        phase_metadata.get("description")
        or phase_metadata.get("note")
        or phase_node.get("description")
    )
    blockers = phase_node.get("dependencies", {}).get("blocked_by", []) or []

    completed = phase_node.get("completed_tasks")
    total = phase_node.get("total_tasks")
    percentage = None
    if isinstance(completed, int) and isinstance(total, int) and total > 0:
        percentage = int((completed / total) * 100)

    spec_root = hierarchy.get("spec-root", {})
    sequence_index = None
    phase_list = spec_root.get("children", [])
    if isinstance(phase_list, list) and phase_id in phase_list:
        sequence_index = phase_list.index(phase_id) + 1

    return {
        "id": phase_id,
        "title": phase_node.get("title", ""),
        "status": phase_node.get("status", ""),
        "sequence_index": sequence_index,
        "completed_tasks": completed,
        "total_tasks": total,
        "percentage": percentage,
        "summary": summary,
        "blockers": blockers,
    }


def get_task_journal_summary(
    spec_data: Dict[str, Any],
    task_id: str,
    max_entries: int = 3,
) -> Dict[str, Any]:
    """
    Return a compact summary of journal entries for a task.

    Args:
        spec_data: Loaded JSON spec dictionary.
        task_id: Task identifier.
        max_entries: Maximum entries to include in summary.

    Returns:
        Dictionary with entry_count and entries[]
    """
    if not spec_data or not task_id:
        return {"entry_count": 0, "entries": []}

    journal = spec_data.get("journal", []) or []
    filtered = [
        entry for entry in journal
        if entry.get("task_id") == task_id
    ]

    if not filtered:
        return {"entry_count": 0, "entries": []}

    filtered.sort(key=lambda entry: entry.get("timestamp") or "", reverse=True)
    entries = []
    for entry in filtered[:max_entries]:
        summary = (entry.get("content") or "").strip()
        entries.append({
            "timestamp": entry.get("timestamp"),
            "entry_type": entry.get("entry_type"),
            "title": entry.get("title"),
            "summary": summary,
            "author": entry.get("author"),
        })

    return {
        "entry_count": len(filtered),
        "entries": entries,
    }


def prepare_task(
    spec_id: str,
    specs_dir: Path,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Prepare complete context for task implementation.

    Combines task discovery, dependency checking, and context gathering.

    Args:
        spec_id: Specification ID
        specs_dir: Path to specs directory
        task_id: Optional task ID (auto-discovers if not provided)

    Returns:
        Complete task preparation data with context.
    """
    # Find and load spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if not spec_path:
        return asdict(error_response(f"Spec file not found for {spec_id}"))

    spec_data = load_spec(spec_id, specs_dir)
    if not spec_data:
        return asdict(error_response("Failed to load spec"))

    # Get task ID if not provided
    if not task_id:
        next_task = get_next_task(spec_data)
        if not next_task:
            # Check if spec is complete
            hierarchy = spec_data.get("hierarchy", {})
            all_tasks = [
                node for node in hierarchy.values()
                if node.get("type") in ["task", "subtask", "verify"]
            ]
            completed = sum(1 for t in all_tasks if t.get("status") == "completed")
            pending = sum(1 for t in all_tasks if t.get("status") == "pending")

            if pending == 0 and completed > 0:
                return asdict(success_response(
                    task_id=None,
                    spec_complete=True
                ))
            else:
                return asdict(error_response("No actionable tasks found"))

        task_id, _ = next_task

    # Get task info
    task_data = get_node(spec_data, task_id)
    if not task_data:
        return asdict(error_response(f"Task {task_id} not found"))

    # Check dependencies
    deps = check_dependencies(spec_data, task_id)

    # Gather context
    context = {
        "previous_sibling": get_previous_sibling(spec_data, task_id),
        "parent_task": get_parent_context(spec_data, task_id),
        "phase": get_phase_context(spec_data, task_id),
        "task_journal": get_task_journal_summary(spec_data, task_id),
    }

    return asdict(success_response(
        task_id=task_id,
        task_data=task_data,
        dependencies=deps,
        spec_complete=False,
        context=context
    ))


def _generate_task_id(parent_id: str, existing_children: List[str], task_type: str) -> str:
    """
    Generate a new task ID based on parent and existing siblings.

    For task IDs:
    - If parent is phase-N, generate task-N-M where M is next available
    - If parent is task-N-M, generate task-N-M-P where P is next available

    For verify IDs:
    - Same pattern but with "verify-" prefix

    Args:
        parent_id: Parent node ID
        existing_children: List of existing child IDs
        task_type: Type of task (task, subtask, verify)

    Returns:
        New task ID string
    """
    prefix = "verify" if task_type == "verify" else "task"

    # Extract numeric parts from parent
    if parent_id.startswith("phase-"):
        # Parent is phase-N, new task is task-N-1, task-N-2, etc.
        phase_num = parent_id.replace("phase-", "")
        base = f"{prefix}-{phase_num}"
    elif parent_id.startswith("task-") or parent_id.startswith("verify-"):
        # Parent is task-N-M or verify-N-M, new task appends next number
        # Remove the prefix (task- or verify-) to get the numeric path
        if parent_id.startswith("task-"):
            base = f"{prefix}-{parent_id[5:]}"  # len("task-") = 5
        else:
            base = f"{prefix}-{parent_id[7:]}"  # len("verify-") = 7
    else:
        # Unknown parent type, generate based on existing children count
        base = f"{prefix}-1"

    # Find the next available index
    pattern = re.compile(rf"^{re.escape(base)}-(\d+)$")
    max_index = 0
    for child_id in existing_children:
        match = pattern.match(child_id)
        if match:
            index = int(match.group(1))
            max_index = max(max_index, index)

    return f"{base}-{max_index + 1}"


def _update_ancestor_counts(hierarchy: Dict[str, Any], node_id: str, delta: int = 1) -> None:
    """
    Walk up the hierarchy and increment total_tasks for all ancestors.

    Args:
        hierarchy: The spec hierarchy dict
        node_id: Starting node ID
        delta: Amount to add to total_tasks (default 1)
    """
    current_id = node_id
    visited = set()

    while current_id:
        if current_id in visited:
            break
        visited.add(current_id)

        node = hierarchy.get(current_id)
        if not node:
            break

        # Increment total_tasks
        current_total = node.get("total_tasks", 0)
        node["total_tasks"] = current_total + delta

        # Move to parent
        current_id = node.get("parent")


def add_task(
    spec_id: str,
    parent_id: str,
    title: str,
    description: Optional[str] = None,
    task_type: str = "task",
    estimated_hours: Optional[float] = None,
    position: Optional[int] = None,
    file_path: Optional[str] = None,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Add a new task to a specification's hierarchy.

    Creates a new task, subtask, or verify node under the specified parent.
    Automatically generates the task ID and updates ancestor task counts.

    Args:
        spec_id: Specification ID to add task to.
        parent_id: Parent node ID (phase or task).
        title: Task title.
        description: Optional task description.
        task_type: Type of task (task, subtask, verify). Default: task.
        estimated_hours: Optional estimated hours.
        position: Optional position in parent's children list (0-based).
        file_path: Optional file path associated with this task.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"task_id": ..., "parent": ..., ...}, None)
        On failure: (None, "error message")
    """
    # Validate task_type
    if task_type not in TASK_TYPES:
        return None, f"Invalid task_type '{task_type}'. Must be one of: {', '.join(TASK_TYPES)}"

    # Validate title
    if not title or not title.strip():
        return None, "Title is required"

    title = title.strip()

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return None, "No specs directory found. Use specs_dir parameter or set SDD_SPECS_DIR."

    # Find and load the spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    hierarchy = spec_data.get("hierarchy", {})

    # Validate parent exists
    parent = hierarchy.get(parent_id)
    if parent is None:
        return None, f"Parent node '{parent_id}' not found"

    # Validate parent type (can add tasks to phases, groups, or tasks)
    parent_type = parent.get("type")
    if parent_type not in ("phase", "group", "task"):
        return None, f"Cannot add tasks to node type '{parent_type}'. Parent must be a phase, group, or task."

    # Get existing children
    existing_children = parent.get("children", [])
    if not isinstance(existing_children, list):
        existing_children = []

    # Generate task ID
    task_id = _generate_task_id(parent_id, existing_children, task_type)

    # Build metadata
    metadata: Dict[str, Any] = {}
    if description:
        metadata["description"] = description.strip()
    if estimated_hours is not None:
        metadata["estimated_hours"] = estimated_hours
    if file_path:
        metadata["file_path"] = file_path.strip()

    # Create the task node
    task_node = {
        "type": task_type,
        "title": title,
        "status": "pending",
        "parent": parent_id,
        "children": [],
        "total_tasks": 1,  # Counts itself
        "completed_tasks": 0,
        "metadata": metadata,
        "dependencies": {
            "blocks": [],
            "blocked_by": [],
            "depends": [],
        },
    }

    # Add to hierarchy
    hierarchy[task_id] = task_node

    # Update parent's children list
    if position is not None and 0 <= position <= len(existing_children):
        existing_children.insert(position, task_id)
    else:
        existing_children.append(task_id)
    parent["children"] = existing_children

    # Update ancestor task counts
    _update_ancestor_counts(hierarchy, parent_id, delta=1)

    # Save the spec
    success = save_spec(spec_id, spec_data, specs_dir)
    if not success:
        return None, "Failed to save specification"

    return {
        "task_id": task_id,
        "parent": parent_id,
        "title": title,
        "type": task_type,
        "position": position if position is not None else len(existing_children) - 1,
        "file_path": file_path.strip() if file_path else None,
    }, None


def _collect_descendants(hierarchy: Dict[str, Any], node_id: str) -> List[str]:
    """
    Recursively collect all descendant node IDs for a given node.

    Args:
        hierarchy: The spec hierarchy dict
        node_id: Starting node ID

    Returns:
        List of all descendant node IDs (not including the starting node)
    """
    descendants = []
    node = hierarchy.get(node_id)
    if not node:
        return descendants

    children = node.get("children", [])
    if not isinstance(children, list):
        return descendants

    for child_id in children:
        descendants.append(child_id)
        descendants.extend(_collect_descendants(hierarchy, child_id))

    return descendants


def _count_tasks_in_subtree(hierarchy: Dict[str, Any], node_ids: List[str]) -> Tuple[int, int]:
    """
    Count total and completed tasks in a list of nodes.

    Args:
        hierarchy: The spec hierarchy dict
        node_ids: List of node IDs to count

    Returns:
        Tuple of (total_count, completed_count)
    """
    total = 0
    completed = 0

    for node_id in node_ids:
        node = hierarchy.get(node_id)
        if not node:
            continue
        node_type = node.get("type")
        if node_type in ("task", "subtask", "verify"):
            total += 1
            if node.get("status") == "completed":
                completed += 1

    return total, completed


def _decrement_ancestor_counts(
    hierarchy: Dict[str, Any],
    node_id: str,
    total_delta: int,
    completed_delta: int,
) -> None:
    """
    Walk up the hierarchy and decrement task counts for all ancestors.

    Args:
        hierarchy: The spec hierarchy dict
        node_id: Starting node ID (the parent of the removed node)
        total_delta: Amount to subtract from total_tasks
        completed_delta: Amount to subtract from completed_tasks
    """
    current_id = node_id
    visited = set()

    while current_id:
        if current_id in visited:
            break
        visited.add(current_id)

        node = hierarchy.get(current_id)
        if not node:
            break

        # Decrement counts
        current_total = node.get("total_tasks", 0)
        current_completed = node.get("completed_tasks", 0)
        node["total_tasks"] = max(0, current_total - total_delta)
        node["completed_tasks"] = max(0, current_completed - completed_delta)

        # Move to parent
        current_id = node.get("parent")


def _remove_dependency_references(hierarchy: Dict[str, Any], removed_ids: List[str]) -> None:
    """
    Remove references to deleted nodes from all dependency lists.

    Args:
        hierarchy: The spec hierarchy dict
        removed_ids: List of node IDs being removed
    """
    removed_set = set(removed_ids)

    for node_id, node in hierarchy.items():
        deps = node.get("dependencies")
        if not deps or not isinstance(deps, dict):
            continue

        for key in ("blocks", "blocked_by", "depends"):
            dep_list = deps.get(key)
            if isinstance(dep_list, list):
                deps[key] = [d for d in dep_list if d not in removed_set]


def remove_task(
    spec_id: str,
    task_id: str,
    cascade: bool = False,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Remove a task from a specification's hierarchy.

    Removes the specified task and optionally all its descendants.
    Updates ancestor task counts and cleans up dependency references.

    Args:
        spec_id: Specification ID containing the task.
        task_id: Task ID to remove.
        cascade: If True, also remove all child tasks recursively.
                 If False and task has children, returns an error.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"task_id": ..., "children_removed": ..., ...}, None)
        On failure: (None, "error message")
    """
    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return None, "No specs directory found. Use specs_dir parameter or set SDD_SPECS_DIR."

    # Find and load the spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    hierarchy = spec_data.get("hierarchy", {})

    # Validate task exists
    task = hierarchy.get(task_id)
    if task is None:
        return None, f"Task '{task_id}' not found"

    # Validate task type (can only remove task, subtask, verify)
    task_type = task.get("type")
    if task_type not in ("task", "subtask", "verify"):
        return None, f"Cannot remove node type '{task_type}'. Only task, subtask, or verify nodes can be removed."

    # Check for children
    children = task.get("children", [])
    if isinstance(children, list) and len(children) > 0 and not cascade:
        return None, f"Task '{task_id}' has {len(children)} children. Use cascade=True to remove them."

    # Collect all nodes to remove
    nodes_to_remove = [task_id]
    if cascade:
        nodes_to_remove.extend(_collect_descendants(hierarchy, task_id))

    # Count tasks being removed (including the target node itself)
    total_removed, completed_removed = _count_tasks_in_subtree(hierarchy, nodes_to_remove)
    # The target node itself
    if task_type in ("task", "subtask", "verify"):
        total_removed += 1
        if task.get("status") == "completed":
            completed_removed += 1

    # Get parent before removing
    parent_id = task.get("parent")

    # Remove nodes from hierarchy
    for node_id in nodes_to_remove:
        if node_id in hierarchy:
            del hierarchy[node_id]

    # Update parent's children list
    if parent_id:
        parent = hierarchy.get(parent_id)
        if parent:
            parent_children = parent.get("children", [])
            if isinstance(parent_children, list) and task_id in parent_children:
                parent_children.remove(task_id)
                parent["children"] = parent_children

            # Update ancestor task counts
            _decrement_ancestor_counts(hierarchy, parent_id, total_removed, completed_removed)

    # Clean up dependency references
    _remove_dependency_references(hierarchy, nodes_to_remove)

    # Save the spec
    success = save_spec(spec_id, spec_data, specs_dir)
    if not success:
        return None, "Failed to save specification"

    return {
        "task_id": task_id,
        "spec_id": spec_id,
        "cascade": cascade,
        "children_removed": len(nodes_to_remove) - 1,  # Exclude the target itself
        "total_tasks_removed": total_removed,
    }, None


# Valid complexity levels for update_estimate
COMPLEXITY_LEVELS = ("low", "medium", "high")


def update_estimate(
    spec_id: str,
    task_id: str,
    estimated_hours: Optional[float] = None,
    complexity: Optional[str] = None,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Update effort/time estimates for a task.

    Updates the estimated_hours and/or complexity metadata for a task.
    At least one of estimated_hours or complexity must be provided.

    Args:
        spec_id: Specification ID containing the task.
        task_id: Task ID to update.
        estimated_hours: Optional estimated hours (float, must be >= 0).
        complexity: Optional complexity level (low, medium, high).
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"task_id": ..., "hours": ..., "complexity": ..., ...}, None)
        On failure: (None, "error message")
    """
    # Validate at least one field is provided
    if estimated_hours is None and complexity is None:
        return None, "At least one of estimated_hours or complexity must be provided"

    # Validate estimated_hours
    if estimated_hours is not None:
        if not isinstance(estimated_hours, (int, float)):
            return None, "estimated_hours must be a number"
        if estimated_hours < 0:
            return None, "estimated_hours must be >= 0"

    # Validate complexity
    if complexity is not None:
        complexity = complexity.lower().strip()
        if complexity not in COMPLEXITY_LEVELS:
            return None, f"Invalid complexity '{complexity}'. Must be one of: {', '.join(COMPLEXITY_LEVELS)}"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return None, "No specs directory found. Use specs_dir parameter or set SDD_SPECS_DIR."

    # Find and load the spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    hierarchy = spec_data.get("hierarchy", {})

    # Validate task exists
    task = hierarchy.get(task_id)
    if task is None:
        return None, f"Task '{task_id}' not found"

    # Validate task type (can only update task, subtask, verify)
    task_type = task.get("type")
    if task_type not in ("task", "subtask", "verify"):
        return None, f"Cannot update estimates for node type '{task_type}'. Only task, subtask, or verify nodes can be updated."

    # Get or create metadata
    metadata = task.get("metadata")
    if metadata is None:
        metadata = {}
        task["metadata"] = metadata

    # Track previous values for response
    previous_hours = metadata.get("estimated_hours")
    previous_complexity = metadata.get("complexity")

    # Update fields
    if estimated_hours is not None:
        metadata["estimated_hours"] = float(estimated_hours)

    if complexity is not None:
        metadata["complexity"] = complexity

    # Save the spec
    success = save_spec(spec_id, spec_data, specs_dir)
    if not success:
        return None, "Failed to save specification"

    result: Dict[str, Any] = {
        "spec_id": spec_id,
        "task_id": task_id,
    }

    if estimated_hours is not None:
        result["hours"] = float(estimated_hours)
        result["previous_hours"] = previous_hours

    if complexity is not None:
        result["complexity"] = complexity
        result["previous_complexity"] = previous_complexity

    return result, None


# Valid verification types for update_task_metadata
VERIFICATION_TYPES = ("run-tests", "fidelity", "manual")

# Valid task categories
TASK_CATEGORIES = CATEGORIES


def update_task_metadata(
    spec_id: str,
    task_id: str,
    file_path: Optional[str] = None,
    description: Optional[str] = None,
    acceptance_criteria: Optional[List[str]] = None,
    task_category: Optional[str] = None,
    actual_hours: Optional[float] = None,
    status_note: Optional[str] = None,
    verification_type: Optional[str] = None,
    command: Optional[str] = None,
    custom_metadata: Optional[Dict[str, Any]] = None,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Update arbitrary metadata fields on a task.

    Updates various metadata fields on a task including file path, description,
    category, hours, notes, verification type, and custom fields.
    At least one field must be provided.

    Args:
        spec_id: Specification ID containing the task.
        task_id: Task ID to update.
        file_path: Optional file path associated with the task.
        description: Optional task description.
        acceptance_criteria: Optional acceptance criteria list.
        task_category: Optional task category (implementation, refactoring, investigation, decision, research).
        actual_hours: Optional actual hours spent on task (must be >= 0).
        status_note: Optional status note or completion note.
        verification_type: Optional verification type (run-tests, fidelity, manual).
        command: Optional command executed for the task.
        custom_metadata: Optional dict of custom metadata fields to merge.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"task_id": ..., "fields_updated": [...], ...}, None)
        On failure: (None, "error message")
    """
    # Collect all provided fields
    updates: Dict[str, Any] = {}
    if file_path is not None:
        updates["file_path"] = file_path.strip() if file_path else None
    if description is not None:
        updates["description"] = description.strip() if description else None
    if acceptance_criteria is not None:
        updates["acceptance_criteria"] = acceptance_criteria
    if task_category is not None:
        updates["task_category"] = task_category
    if actual_hours is not None:
        updates["actual_hours"] = actual_hours
    if status_note is not None:
        updates["status_note"] = status_note.strip() if status_note else None
    if verification_type is not None:
        updates["verification_type"] = verification_type
    if command is not None:
        updates["command"] = command.strip() if command else None

    # Validate at least one field is provided
    if not updates and not custom_metadata:
        return None, "At least one metadata field must be provided"

    # Validate actual_hours
    if actual_hours is not None:
        if not isinstance(actual_hours, (int, float)):
            return None, "actual_hours must be a number"
        if actual_hours < 0:
            return None, "actual_hours must be >= 0"

    if acceptance_criteria is not None:
        if not isinstance(acceptance_criteria, list):
            return None, "acceptance_criteria must be a list of strings"
        cleaned_criteria = []
        for item in acceptance_criteria:
            if not isinstance(item, str) or not item.strip():
                return None, "acceptance_criteria must be a list of non-empty strings"
            cleaned_criteria.append(item.strip())
        updates["acceptance_criteria"] = cleaned_criteria

    # Validate task_category
    if task_category is not None:
        task_category_lower = task_category.lower().strip()
        if task_category_lower not in TASK_CATEGORIES:
            return None, f"Invalid task_category '{task_category}'. Must be one of: {', '.join(TASK_CATEGORIES)}"
        updates["task_category"] = task_category_lower

    # Validate verification_type
    if verification_type is not None:
        verification_type_lower = verification_type.lower().strip()
        if verification_type_lower not in VERIFICATION_TYPES:
            return None, f"Invalid verification_type '{verification_type}'. Must be one of: {', '.join(VERIFICATION_TYPES)}"
        updates["verification_type"] = verification_type_lower

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return None, "No specs directory found. Use specs_dir parameter or set SDD_SPECS_DIR."

    # Find and load the spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    hierarchy = spec_data.get("hierarchy", {})

    # Validate task exists
    task = hierarchy.get(task_id)
    if task is None:
        return None, f"Task '{task_id}' not found"

    # Validate task type (can only update task, subtask, verify)
    task_type = task.get("type")
    if task_type not in ("task", "subtask", "verify"):
        return None, f"Cannot update metadata for node type '{task_type}'. Only task, subtask, or verify nodes can be updated."

    # Get or create metadata
    metadata = task.get("metadata")
    if metadata is None:
        metadata = {}
        task["metadata"] = metadata

    # Track which fields were updated
    fields_updated = []

    # Apply updates
    for key, value in updates.items():
        if value is not None or key in metadata:
            metadata[key] = value
            fields_updated.append(key)

    # Apply custom metadata
    if custom_metadata and isinstance(custom_metadata, dict):
        for key, value in custom_metadata.items():
            # Don't allow overwriting core fields via custom_metadata
            if key not in ("type", "title", "status", "parent", "children", "dependencies"):
                metadata[key] = value
                if key not in fields_updated:
                    fields_updated.append(key)

    # Save the spec
    success = save_spec(spec_id, spec_data, specs_dir)
    if not success:
        return None, "Failed to save specification"

    return {
        "spec_id": spec_id,
        "task_id": task_id,
        "fields_updated": fields_updated,
    }, None
