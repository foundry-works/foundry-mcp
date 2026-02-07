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
