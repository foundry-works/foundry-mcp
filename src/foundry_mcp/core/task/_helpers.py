"""Shared constants and helpers used by both queries and mutations."""

from typing import Dict, Any, Optional

# Valid task types for add_task
TASK_TYPES = ("task", "subtask", "verify", "research")

# Valid requirement types for update_task_requirements
REQUIREMENT_TYPES = ("acceptance", "technical", "constraint")


def _get_phase_for_node(hierarchy: Dict[str, Any], node_id: str) -> Optional[str]:
    """
    Walk up the hierarchy to find the phase containing a node.

    Args:
        hierarchy: The spec hierarchy dict
        node_id: The node to find the phase for

    Returns:
        Phase ID if found, None otherwise
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

        if node.get("type") == "phase":
            return current_id

        current_id = node.get("parent")

    return None


def check_all_blocked(spec_data: Dict[str, Any]) -> bool:
    """
    Check if all remaining tasks are blocked.

    This is a shared utility used by both batch_operations and the step orchestrator
    to determine if autonomous execution should pause due to all tasks being blocked.

    Args:
        spec_data: Loaded spec data with hierarchy

    Returns:
        True if all pending tasks are blocked, False if any task can proceed
    """
    from foundry_mcp.core.task.queries import is_unblocked

    hierarchy = spec_data.get("hierarchy", {})
    pending_found = False

    for task_id, task_data in hierarchy.items():
        if task_data.get("type") not in ("task", "subtask", "verify"):
            continue
        if task_data.get("status") != "pending":
            continue
        pending_found = True
        # If any task is unblocked, not all are blocked
        if is_unblocked(spec_data, task_id, task_data):
            return False

    # Legacy specs may omit hierarchy and only define phases/tasks.
    # Fall back to phase/task inspection when hierarchy has no pending nodes.
    if not pending_found:
        completed_task_ids = set()
        pending_tasks = []

        for phase in spec_data.get("phases", []):
            for task in phase.get("tasks", []):
                if task.get("type", "task") not in ("task", "subtask", "verify"):
                    continue

                status = task.get("status", "pending")
                task_id = task.get("id", "")
                if status == "completed" and task_id:
                    completed_task_ids.add(task_id)
                if status == "pending":
                    pending_tasks.append(task)

        if not pending_tasks:
            return False

        for task in pending_tasks:
            deps: Any = task.get("depends", [])
            if not deps:
                deps = task.get("dependencies", [])

            if isinstance(deps, dict):
                dep_ids = deps.get("blocked_by") or deps.get("depends") or []
            elif isinstance(deps, list):
                dep_ids = deps
            else:
                dep_ids = []

            if all(dep_id in completed_task_ids for dep_id in dep_ids):
                return False

    return True
