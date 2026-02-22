"""
Query functions for task discovery, dependency checking, and preparation.

All functions in this module are read-only — they inspect spec data but never
mutate it.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from foundry_mcp.core.responses.builders import (
    error_response,
    success_response,
)
from foundry_mcp.core.spec import (
    find_spec_file,
    get_node,
    load_spec,
)


def is_unblocked(spec_data: Dict[str, Any], task_id: str, task_data: Dict[str, Any]) -> bool:
    """
    Check if all blocking dependencies are completed.

    This checks both task-level dependencies and phase-level dependencies.
    A task is blocked if:
    1. Any of its direct task dependencies are not completed, OR
    2. Its parent phase is blocked by an incomplete phase

    Research nodes have special blocking behavior based on blocking_mode:
    - "none": Research doesn't block dependents
    - "soft": Research is informational, doesn't block (default)
    - "hard": Research must complete before dependents can start

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
        if not blocker:
            continue

        # Special handling for research nodes based on blocking_mode
        if blocker.get("type") == "research":
            blocking_mode = blocker.get("metadata", {}).get("blocking_mode", "soft")
            if blocking_mode in ("none", "soft"):
                # Research with "none" or "soft" blocking mode doesn't block
                continue
            # "hard" mode falls through to standard completion check

        if blocker.get("status") != "completed":
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
            if (
                value.get("type") in ["task", "subtask", "verify"]
                and value.get("status") == "pending"
                and is_unblocked(spec_data, key, value)
                and is_in_current_phase(spec_data, key, current_phase)
            ):
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
        "blocks": [],
    }

    # Get info for blocking tasks
    for dep_id in blocked_by:
        dep_task = hierarchy.get(dep_id)
        if dep_task:
            result["blocked_by"].append(
                {
                    "id": dep_id,
                    "title": dep_task.get("title", ""),
                    "status": dep_task.get("status", ""),
                    "file": dep_task.get("metadata", {}).get("file_path", ""),
                }
            )

    # Get info for soft dependencies
    for dep_id in depends:
        dep_task = hierarchy.get(dep_id)
        if dep_task:
            result["soft_depends"].append(
                {
                    "id": dep_id,
                    "title": dep_task.get("title", ""),
                    "status": dep_task.get("status", ""),
                    "file": dep_task.get("metadata", {}).get("file_path", ""),
                }
            )

    # Get info for tasks this blocks
    for dep_id in blocks:
        dep_task = hierarchy.get(dep_id)
        if dep_task:
            result["blocks"].append(
                {
                    "id": dep_id,
                    "title": dep_task.get("title", ""),
                    "status": dep_task.get("status", ""),
                    "file": dep_task.get("metadata", {}).get("file_path", ""),
                }
            )

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

    return [node_id for node_id, node in hierarchy.items() if node.get("parent") == parent_id]


def _get_latest_journal_excerpt(
    journal_entries: List[Dict[str, Any]],
    task_id: str,
) -> Optional[Dict[str, Any]]:
    """Return the most recent journal entry for the given task."""
    if not journal_entries:
        return None

    filtered = [entry for entry in journal_entries if entry.get("task_id") == task_id]
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
    description = parent_metadata.get("description") or parent_metadata.get("note") or parent.get("description")

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
    summary = phase_metadata.get("description") or phase_metadata.get("note") or phase_node.get("description")
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
    filtered = [entry for entry in journal if entry.get("task_id") == task_id]

    if not filtered:
        return {"entry_count": 0, "entries": []}

    filtered.sort(key=lambda entry: entry.get("timestamp") or "", reverse=True)
    entries = []
    for entry in filtered[:max_entries]:
        summary = (entry.get("content") or "").strip()
        entries.append(
            {
                "timestamp": entry.get("timestamp"),
                "entry_type": entry.get("entry_type"),
                "title": entry.get("title"),
                "summary": summary,
                "author": entry.get("author"),
            }
        )

    return {
        "entry_count": len(filtered),
        "entries": entries,
    }


def _compute_auto_mode_hints(
    spec_data: Dict[str, Any],
    task_id: str,
    task_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute hints for autonomous mode execution.

    These hints help an autonomous agent decide whether to proceed
    without user input or pause for confirmation.

    Args:
        spec_data: Loaded spec data
        task_id: Current task ID
        task_data: Task node data

    Returns:
        Dictionary with autonomous mode hints:
        - estimated_complexity: "low", "medium", or "high"
        - has_sibling_verify: bool (phase has verify tasks)
        - may_require_user_input: bool (task category suggests user input needed)
    """
    hierarchy = spec_data.get("hierarchy", {})
    metadata = task_data.get("metadata", {}) or {}

    # Compute estimated_complexity
    complexity = metadata.get("complexity", "").lower()
    estimated_hours = metadata.get("estimated_hours")

    if complexity in ("complex", "high"):
        estimated_complexity = "high"
    elif complexity in ("medium", "moderate"):
        estimated_complexity = "medium"
    elif complexity in ("simple", "low"):
        estimated_complexity = "low"
    elif estimated_hours is not None:
        # Derive from hours if explicit complexity not set
        if estimated_hours > 2:
            estimated_complexity = "high"
        elif estimated_hours > 0.5:
            estimated_complexity = "medium"
        else:
            estimated_complexity = "low"
    else:
        # Default to medium if no hints
        estimated_complexity = "medium"

    # Check has_sibling_verify - look for verify tasks in same phase
    parent_id = task_data.get("parent")
    has_sibling_verify = False
    if parent_id:
        parent = hierarchy.get(parent_id, {})
        children = parent.get("children", [])
        for sibling_id in children:
            if sibling_id != task_id:
                sibling = hierarchy.get(sibling_id, {})
                if sibling.get("type") == "verify":
                    has_sibling_verify = True
                    break

    # Check may_require_user_input based on task_category
    task_category = metadata.get("task_category", "").lower()
    may_require_user_input = task_category in (
        "decision",
        "investigation",
        "planning",
        "design",
    )

    return {
        "estimated_complexity": estimated_complexity,
        "has_sibling_verify": has_sibling_verify,
        "may_require_user_input": may_require_user_input,
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
            all_tasks = [node for node in hierarchy.values() if node.get("type") in ["task", "subtask", "verify"]]
            completed = sum(1 for t in all_tasks if t.get("status") == "completed")
            pending = sum(1 for t in all_tasks if t.get("status") == "pending")

            if pending == 0 and completed > 0:
                return asdict(success_response(task_id=None, spec_complete=True))
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

    # Compute autonomous mode hints
    auto_mode_hints = _compute_auto_mode_hints(spec_data, task_id, task_data)

    return asdict(
        success_response(
            task_id=task_id,
            task_data=task_data,
            dependencies=deps,
            spec_complete=False,
            context=context,
            auto_mode_hints=auto_mode_hints,
        )
    )
