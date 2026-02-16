"""
Task mutation operations for SDD workflows.

All functions in this module modify spec data (add, remove, update, move tasks).
Query functions live in ``queries.py``; shared constants in ``_helpers.py``.
"""

import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from foundry_mcp.core.spec import (
    CATEGORIES,
    load_spec,
    save_spec,
    find_spec_file,
    find_specs_directory,
)
from foundry_mcp.core.task._helpers import TASK_TYPES, REQUIREMENT_TYPES, _get_phase_for_node


def _generate_task_id(parent_id: str, existing_children: List[str], task_type: str) -> str:
    """
    Generate a new task ID based on parent and existing siblings.

    For task IDs:
    - If parent is phase-N, generate task-N-M where M is next available
    - If parent is task-N-M, generate task-N-M-P where P is next available

    For verify IDs:
    - Same pattern but with "verify-" prefix

    For research IDs:
    - Same pattern but with "research-" prefix

    Args:
        parent_id: Parent node ID
        existing_children: List of existing child IDs
        task_type: Type of task (task, subtask, verify, research)

    Returns:
        New task ID string
    """
    # Map task_type to ID prefix
    prefix_map = {"verify": "verify", "research": "research"}
    prefix = prefix_map.get(task_type, "task")

    # Extract numeric parts from parent
    if parent_id.startswith("phase-"):
        # Parent is phase-N, new task is task-N-1, task-N-2, etc.
        phase_num = parent_id.replace("phase-", "")
        base = f"{prefix}-{phase_num}"
    elif parent_id.startswith(("task-", "verify-", "research-")):
        # Parent is task-N-M, verify-N-M, or research-N-M; new task appends next number
        # Remove the prefix to get the numeric path
        if parent_id.startswith("task-"):
            base = f"{prefix}-{parent_id[5:]}"  # len("task-") = 5
        elif parent_id.startswith("verify-"):
            base = f"{prefix}-{parent_id[7:]}"  # len("verify-") = 7
        else:  # research-
            base = f"{prefix}-{parent_id[9:]}"  # len("research-") = 9
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
    # Research-specific parameters
    research_type: Optional[str] = None,
    blocking_mode: Optional[str] = None,
    query: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Add a new task to a specification's hierarchy.

    Creates a new task, subtask, verify, or research node under the specified parent.
    Automatically generates the task ID and updates ancestor task counts.

    Args:
        spec_id: Specification ID to add task to.
        parent_id: Parent node ID (phase or task).
        title: Task title.
        description: Optional task description.
        task_type: Type of task (task, subtask, verify, research). Default: task.
        estimated_hours: Optional estimated hours.
        position: Optional position in parent's children list (0-based).
        file_path: Optional file path associated with this task.
        specs_dir: Path to specs directory (auto-detected if not provided).
        research_type: For research nodes - workflow type (chat, consensus, etc).
        blocking_mode: For research nodes - blocking behavior (none, soft, hard).
        query: For research nodes - the research question/topic.

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"task_id": ..., "parent": ..., ...}, None)
        On failure: (None, "error message")
    """
    # Validate task_type
    if task_type not in TASK_TYPES:
        return None, f"Invalid task_type '{task_type}'. Must be one of: {', '.join(TASK_TYPES)}"

    # Validate research-specific parameters
    if task_type == "research":
        from foundry_mcp.core.validation.constants import VALID_RESEARCH_TYPES, RESEARCH_BLOCKING_MODES

        if research_type and research_type not in VALID_RESEARCH_TYPES:
            return None, f"Invalid research_type '{research_type}'. Must be one of: {', '.join(sorted(VALID_RESEARCH_TYPES))}"
        if blocking_mode and blocking_mode not in RESEARCH_BLOCKING_MODES:
            return None, f"Invalid blocking_mode '{blocking_mode}'. Must be one of: {', '.join(sorted(RESEARCH_BLOCKING_MODES))}"

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

    # Add research-specific metadata
    if task_type == "research":
        metadata["research_type"] = research_type or "consensus"  # Default to consensus
        metadata["blocking_mode"] = blocking_mode or "soft"  # Default to soft blocking
        if query:
            metadata["query"] = query.strip()
        metadata["research_history"] = []  # Empty history initially
        metadata["findings"] = {}  # Empty findings initially

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


# Shared hierarchy traversal utilities — canonical implementations live in
# hierarchy_helpers.py.  Private aliases kept for backwards compatibility
# with callers that import from this module.
from foundry_mcp.core.hierarchy_helpers import (  # noqa: F401
    collect_descendants as _collect_descendants,
    count_tasks_in_subtree as _count_tasks_in_subtree,
)


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


from foundry_mcp.core.hierarchy_helpers import (  # noqa: F401
    remove_dependency_references as _remove_dependency_references,
)


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
        hint = " Use `authoring action=\"phase-remove\"` instead." if task_type == "phase" else ""
        return None, f"Cannot remove node type '{task_type}'. Only task, subtask, or verify nodes can be removed.{hint}"

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

# Valid dependency types for manage_task_dependency
DEPENDENCY_TYPES = ("blocks", "blocked_by", "depends")


# Maximum number of requirements per task (to prevent unbounded growth)
MAX_REQUIREMENTS_PER_TASK = 50


def _would_create_circular_dependency(
    hierarchy: Dict[str, Any],
    source_id: str,
    target_id: str,
    dep_type: str,
) -> bool:
    """
    Check if adding a dependency would create a circular reference.

    For blocking dependencies:
    - Adding A blocks B means B is blocked_by A
    - Circular if B already blocks A (directly or transitively)

    Uses breadth-first search to detect cycles in the dependency graph.

    Args:
        hierarchy: The spec hierarchy dict
        source_id: Source task ID
        target_id: Target task ID
        dep_type: Type of dependency being added

    Returns:
        True if adding this dependency would create a cycle
    """
    if source_id == target_id:
        return True

    # For "blocks": source blocks target, so target cannot already block source
    # For "blocked_by": source is blocked_by target, so source cannot already block target
    # For "depends": soft dependency, check for cycles in depends chain

    if dep_type == "blocks":
        # If source blocks target, check if target already blocks source (transitively)
        # i.e., walk from target's "blocks" chain to see if we reach source
        return _can_reach_via_dependency(hierarchy, target_id, source_id, "blocks")
    elif dep_type == "blocked_by":
        # If source is blocked_by target, check if source already blocks target (transitively)
        return _can_reach_via_dependency(hierarchy, source_id, target_id, "blocks")
    elif dep_type == "depends":
        # Check for cycles in depends chain
        return _can_reach_via_dependency(hierarchy, target_id, source_id, "depends")

    return False


def _can_reach_via_dependency(
    hierarchy: Dict[str, Any],
    start_id: str,
    target_id: str,
    dep_key: str,
) -> bool:
    """
    Check if target_id can be reached from start_id via dependency chains.

    Uses BFS to traverse the dependency graph.

    Args:
        hierarchy: The spec hierarchy dict
        start_id: Starting node ID
        target_id: Target node ID to find
        dep_key: Which dependency list to follow ("blocks", "blocked_by", "depends")

    Returns:
        True if target_id is reachable from start_id
    """
    visited = set()
    queue = [start_id]

    while queue:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)

        if current_id == target_id:
            return True

        node = hierarchy.get(current_id)
        if not node:
            continue

        deps = node.get("dependencies", {})
        next_ids = deps.get(dep_key, [])
        if isinstance(next_ids, list):
            for next_id in next_ids:
                if next_id not in visited:
                    queue.append(next_id)

    return False


def manage_task_dependency(
    spec_id: str,
    source_task_id: str,
    target_task_id: str,
    dependency_type: str,
    action: str = "add",
    dry_run: bool = False,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Add or remove a dependency relationship between two tasks.

    Manages blocks, blocked_by, and depends relationships between tasks.
    Updates both source and target tasks atomically.

    Dependency types:
    - blocks: Source task blocks target (target cannot start until source completes)
    - blocked_by: Source task is blocked by target (source cannot start until target completes)
    - depends: Soft dependency (informational, doesn't block)

    When adding:
    - blocks: Adds target to source.blocks AND source to target.blocked_by
    - blocked_by: Adds target to source.blocked_by AND source to target.blocks
    - depends: Only adds target to source.depends (soft, no reciprocal)

    Args:
        spec_id: Specification ID containing the tasks.
        source_task_id: Source task ID.
        target_task_id: Target task ID.
        dependency_type: Type of dependency (blocks, blocked_by, depends).
        action: Action to perform (add or remove). Default: add.
        dry_run: If True, validate and return preview without saving changes.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"source_task": ..., "target_task": ..., "dependency_type": ..., ...}, None)
        On failure: (None, "error message")
    """
    # Validate action
    if action not in ("add", "remove"):
        return None, f"Invalid action '{action}'. Must be 'add' or 'remove'"

    # Validate dependency_type
    if dependency_type not in DEPENDENCY_TYPES:
        return None, f"Invalid dependency_type '{dependency_type}'. Must be one of: {', '.join(DEPENDENCY_TYPES)}"

    # Prevent self-reference
    if source_task_id == target_task_id:
        return None, f"Cannot add dependency: task '{source_task_id}' cannot depend on itself"

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

    # Validate source task exists
    source_task = hierarchy.get(source_task_id)
    if source_task is None:
        return None, f"Source task '{source_task_id}' not found"

    # Validate source task type
    source_type = source_task.get("type")
    if source_type not in ("task", "subtask", "verify", "phase"):
        return None, f"Cannot manage dependencies for node type '{source_type}'"

    # Validate target task exists
    target_task = hierarchy.get(target_task_id)
    if target_task is None:
        return None, f"Target task '{target_task_id}' not found"

    # Validate target task type
    target_type = target_task.get("type")
    if target_type not in ("task", "subtask", "verify", "phase"):
        return None, f"Cannot add dependency to node type '{target_type}'"

    # Get or create dependencies for both tasks
    source_deps = source_task.get("dependencies")
    if source_deps is None:
        source_deps = {"blocks": [], "blocked_by": [], "depends": []}
        source_task["dependencies"] = source_deps

    target_deps = target_task.get("dependencies")
    if target_deps is None:
        target_deps = {"blocks": [], "blocked_by": [], "depends": []}
        target_task["dependencies"] = target_deps

    # Ensure lists exist
    for key in DEPENDENCY_TYPES:
        if not isinstance(source_deps.get(key), list):
            source_deps[key] = []
        if not isinstance(target_deps.get(key), list):
            target_deps[key] = []

    # Determine the reciprocal relationship
    reciprocal_type = None
    if dependency_type == "blocks":
        reciprocal_type = "blocked_by"
    elif dependency_type == "blocked_by":
        reciprocal_type = "blocks"
    # depends has no reciprocal

    if action == "add":
        # Check for circular dependencies
        if _would_create_circular_dependency(hierarchy, source_task_id, target_task_id, dependency_type):
            return None, f"Cannot add dependency: would create circular reference between '{source_task_id}' and '{target_task_id}'"

        # Check if dependency already exists
        if target_task_id in source_deps[dependency_type]:
            return None, f"Dependency already exists: {source_task_id} {dependency_type} {target_task_id}"

        # Add the dependency
        source_deps[dependency_type].append(target_task_id)

        # Add reciprocal if applicable (blocks <-> blocked_by)
        if reciprocal_type:
            if source_task_id not in target_deps[reciprocal_type]:
                target_deps[reciprocal_type].append(source_task_id)

    elif action == "remove":
        # Check if dependency exists
        if target_task_id not in source_deps[dependency_type]:
            return None, f"Dependency does not exist: {source_task_id} {dependency_type} {target_task_id}"

        # Remove the dependency
        source_deps[dependency_type].remove(target_task_id)

        # Remove reciprocal if applicable
        if reciprocal_type and source_task_id in target_deps[reciprocal_type]:
            target_deps[reciprocal_type].remove(source_task_id)

    # Build result
    result = {
        "spec_id": spec_id,
        "source_task": source_task_id,
        "target_task": target_task_id,
        "dependency_type": dependency_type,
        "action": action,
        "dry_run": dry_run,
        "source_dependencies": {
            "blocks": source_deps["blocks"],
            "blocked_by": source_deps["blocked_by"],
            "depends": source_deps["depends"],
        },
        "target_dependencies": {
            "blocks": target_deps["blocks"],
            "blocked_by": target_deps["blocked_by"],
            "depends": target_deps["depends"],
        },
    }

    # Save the spec (unless dry_run)
    if dry_run:
        result["message"] = "Dry run - changes not saved"
    else:
        success = save_spec(spec_id, spec_data, specs_dir)
        if not success:
            return None, "Failed to save specification"

    return result, None


def _is_descendant(hierarchy: Dict[str, Any], ancestor_id: str, potential_descendant_id: str) -> bool:
    """
    Check if a node is a descendant of another node.

    Used to prevent circular references when moving tasks.

    Args:
        hierarchy: The spec hierarchy dict
        ancestor_id: The potential ancestor node ID
        potential_descendant_id: The node to check if it's a descendant

    Returns:
        True if potential_descendant_id is a descendant of ancestor_id
    """
    if ancestor_id == potential_descendant_id:
        return True

    descendants = _collect_descendants(hierarchy, ancestor_id)
    return potential_descendant_id in descendants


def _check_cross_phase_dependencies(
    hierarchy: Dict[str, Any],
    task_id: str,
    old_phase_id: Optional[str],
    new_phase_id: Optional[str],
) -> List[str]:
    """
    Check for potential dependency issues when moving across phases.

    Args:
        hierarchy: The spec hierarchy dict
        task_id: The task being moved
        old_phase_id: The original phase ID
        new_phase_id: The target phase ID

    Returns:
        List of warning messages about potential dependency issues
    """
    warnings = []

    if old_phase_id == new_phase_id:
        return warnings

    task = hierarchy.get(task_id)
    if not task:
        return warnings

    deps = task.get("dependencies", {})

    # Check blocked_by dependencies
    blocked_by = deps.get("blocked_by", [])
    for dep_id in blocked_by:
        dep_phase = _get_phase_for_node(hierarchy, dep_id)
        if dep_phase and dep_phase != new_phase_id:
            dep_node = hierarchy.get(dep_id, {})
            warnings.append(
                f"Task '{task_id}' is blocked by '{dep_id}' ({dep_node.get('title', '')}) "
                f"which is in a different phase ('{dep_phase}')"
            )

    # Check blocks dependencies
    blocks = deps.get("blocks", [])
    for dep_id in blocks:
        dep_phase = _get_phase_for_node(hierarchy, dep_id)
        if dep_phase and dep_phase != new_phase_id:
            dep_node = hierarchy.get(dep_id, {})
            warnings.append(
                f"Task '{task_id}' blocks '{dep_id}' ({dep_node.get('title', '')}) "
                f"which is in a different phase ('{dep_phase}')"
            )

    return warnings


def update_task_metadata(
    spec_id: str,
    task_id: str,
    title: Optional[str] = None,
    file_path: Optional[str] = None,
    description: Optional[str] = None,
    acceptance_criteria: Optional[List[str]] = None,
    task_category: Optional[str] = None,
    actual_hours: Optional[float] = None,
    status_note: Optional[str] = None,
    verification_type: Optional[str] = None,
    command: Optional[str] = None,
    custom_metadata: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Update arbitrary metadata fields on a task.

    Updates various metadata fields on a task including title, file path, description,
    category, hours, notes, verification type, and custom fields.
    At least one field must be provided.

    Args:
        spec_id: Specification ID containing the task.
        task_id: Task ID to update.
        title: Optional new title for the task (cannot be empty/whitespace-only).
        file_path: Optional file path associated with the task.
        description: Optional task description.
        acceptance_criteria: Optional acceptance criteria list.
        task_category: Optional task category (implementation, refactoring, investigation, decision, research).
        actual_hours: Optional actual hours spent on task (must be >= 0).
        status_note: Optional status note or completion note.
        verification_type: Optional verification type (run-tests, fidelity, manual).
        command: Optional command executed for the task.
        custom_metadata: Optional dict of custom metadata fields to merge.
        dry_run: If True, validate and return preview without saving changes.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"task_id": ..., "fields_updated": [...], "previous_values": {...}, ...}, None)
        On failure: (None, "error message")
    """
    # Validate title if provided (cannot be empty/whitespace-only)
    title_update: Optional[str] = None
    if title is not None:
        title_stripped = title.strip()
        if not title_stripped:
            return None, "Title cannot be empty or whitespace-only"
        title_update = title_stripped

    # Collect all provided metadata fields
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

    # Validate at least one field is provided (title or metadata fields)
    if title_update is None and not updates and not custom_metadata:
        return None, "At least one field must be provided (title or metadata fields)"

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

    # Track which fields were updated and their previous values
    fields_updated = []
    previous_values: Dict[str, Any] = {}

    # Apply title update (core field on task, not metadata)
    if title_update is not None:
        previous_values["title"] = task.get("title")
        task["title"] = title_update
        fields_updated.append("title")

    # Apply metadata updates
    for key, value in updates.items():
        if value is not None or key in metadata:
            previous_values[key] = metadata.get(key)
            metadata[key] = value
            fields_updated.append(key)

    # Apply custom metadata
    if custom_metadata and isinstance(custom_metadata, dict):
        for key, value in custom_metadata.items():
            # Don't allow overwriting core fields via custom_metadata
            if key not in ("type", "title", "status", "parent", "children", "dependencies"):
                if key not in previous_values:
                    previous_values[key] = metadata.get(key)
                metadata[key] = value
                if key not in fields_updated:
                    fields_updated.append(key)

    # Build result
    result = {
        "spec_id": spec_id,
        "task_id": task_id,
        "fields_updated": fields_updated,
        "previous_values": previous_values,
        "dry_run": dry_run,
    }

    # Save the spec (unless dry_run)
    if dry_run:
        result["message"] = "Dry run - changes not saved"
    else:
        success = save_spec(spec_id, spec_data, specs_dir)
        if not success:
            return None, "Failed to save specification"

    return result, None


def move_task(
    spec_id: str,
    task_id: str,
    new_parent: Optional[str] = None,
    position: Optional[int] = None,
    dry_run: bool = False,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], List[str]]:
    """
    Move a task to a new position within its parent or to a different parent.

    Supports two modes:
    1. Reorder within parent: only specify position (new_parent=None)
    2. Reparent to different phase/task: specify new_parent, optionally position

    Updates task counts on affected parents. Prevents circular references.
    Emits warnings for cross-phase moves that might affect dependencies.

    Args:
        spec_id: Specification ID containing the task.
        task_id: Task ID to move.
        new_parent: Optional new parent ID (phase or task). If None, reorders
                    within current parent.
        position: Optional position in parent's children list (1-based).
                  If None, appends to end.
        dry_run: If True, validate and return preview without saving changes.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message, warnings_list).
        On success: ({"task_id": ..., "old_parent": ..., "new_parent": ..., ...}, None, [warnings])
        On failure: (None, "error message", [])
    """
    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return None, "No specs directory found. Use specs_dir parameter or set SDD_SPECS_DIR.", []

    # Find and load the spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found", []

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'", []

    hierarchy = spec_data.get("hierarchy", {})

    # Validate task exists
    task = hierarchy.get(task_id)
    if task is None:
        return None, f"Task '{task_id}' not found", []

    # Validate task type (can only move task, subtask, verify)
    task_type = task.get("type")
    if task_type not in ("task", "subtask", "verify"):
        return None, f"Cannot move node type '{task_type}'. Only task, subtask, or verify nodes can be moved.", []

    old_parent_id = task.get("parent")
    if not old_parent_id:
        return None, f"Task '{task_id}' has no parent and cannot be moved", []

    old_parent = hierarchy.get(old_parent_id)
    if not old_parent:
        return None, f"Task's current parent '{old_parent_id}' not found", []

    # Determine effective new parent
    effective_new_parent_id = new_parent if new_parent else old_parent_id
    is_reparenting = effective_new_parent_id != old_parent_id

    # Validate new parent exists
    new_parent_node = hierarchy.get(effective_new_parent_id)
    if new_parent_node is None:
        return None, f"Target parent '{effective_new_parent_id}' not found", []

    # Validate new parent type (can add tasks to phases, groups, or tasks)
    new_parent_type = new_parent_node.get("type")
    if new_parent_type not in ("phase", "group", "task"):
        return None, f"Cannot move to node type '{new_parent_type}'. Target must be a phase, group, or task.", []

    # Prevent self-reference
    if task_id == effective_new_parent_id:
        return None, f"Task '{task_id}' cannot be moved to itself", []

    # Prevent circular reference (can't move a task to one of its descendants)
    if _is_descendant(hierarchy, task_id, effective_new_parent_id):
        return None, f"Cannot move '{task_id}' to '{effective_new_parent_id}': would create circular reference", []

    # Get current children lists
    old_children = old_parent.get("children", [])
    if not isinstance(old_children, list):
        old_children = []

    new_children = new_parent_node.get("children", []) if is_reparenting else old_children.copy()
    if not isinstance(new_children, list):
        new_children = []

    # Validate position
    # Remove task from old position first to calculate valid range
    old_position = None
    if task_id in old_children:
        old_position = old_children.index(task_id)

    # For position validation, consider the list after removal
    max_position = len(new_children) if is_reparenting else len(new_children) - 1
    if position is not None:
        # Convert to 0-based for internal use (user provides 1-based)
        position_0based = position - 1
        if position_0based < 0 or position_0based > max_position:
            return None, f"Invalid position {position}. Must be 1-{max_position + 1}", []
    else:
        # Default: append to end
        position_0based = max_position

    # Check for cross-phase dependency warnings
    warnings: List[str] = []
    if is_reparenting:
        old_phase = _get_phase_for_node(hierarchy, task_id)
        new_phase = _get_phase_for_node(hierarchy, effective_new_parent_id)
        if new_phase != old_phase:
            warnings = _check_cross_phase_dependencies(hierarchy, task_id, old_phase, new_phase)

    # Calculate task counts for the subtree being moved (including the task itself)
    descendants = _collect_descendants(hierarchy, task_id)
    all_moved_nodes = [task_id] + descendants
    total_moved, completed_moved = _count_tasks_in_subtree(hierarchy, all_moved_nodes)

    # Build result for dry run or actual move
    result: Dict[str, Any] = {
        "spec_id": spec_id,
        "task_id": task_id,
        "old_parent": old_parent_id,
        "new_parent": effective_new_parent_id,
        "old_position": old_position + 1 if old_position is not None else None,  # 1-based for output
        "new_position": position_0based + 1,  # 1-based for output
        "is_reparenting": is_reparenting,
        "tasks_in_subtree": total_moved,
        "dry_run": dry_run,
    }

    if dry_run:
        result["message"] = "Dry run - changes not saved"
        if warnings:
            result["dependency_warnings"] = warnings
        return result, None, warnings

    # Perform the move

    # 1. Remove from old parent's children list
    if task_id in old_children:
        old_children.remove(task_id)
        old_parent["children"] = old_children

    # 2. Add to new parent's children list at specified position
    if is_reparenting:
        # Fresh list from new parent
        new_children = new_parent_node.get("children", [])
        if not isinstance(new_children, list):
            new_children = []
    else:
        # Same parent, already removed
        new_children = old_children

    # Insert at position
    if position_0based >= len(new_children):
        new_children.append(task_id)
    else:
        new_children.insert(position_0based, task_id)

    if is_reparenting:
        new_parent_node["children"] = new_children
    else:
        old_parent["children"] = new_children

    # 3. Update task's parent reference
    if is_reparenting:
        task["parent"] = effective_new_parent_id

        # 4. Update ancestor task counts
        # Decrement old parent's ancestors
        _decrement_ancestor_counts(hierarchy, old_parent_id, total_moved, completed_moved)
        # Increment new parent's ancestors
        _update_ancestor_counts(hierarchy, effective_new_parent_id, delta=total_moved)
        # Update completed counts for new ancestors
        if completed_moved > 0:
            current_id = effective_new_parent_id
            visited = set()
            while current_id:
                if current_id in visited:
                    break
                visited.add(current_id)
                node = hierarchy.get(current_id)
                if not node:
                    break
                current_completed = node.get("completed_tasks", 0)
                node["completed_tasks"] = current_completed + completed_moved
                current_id = node.get("parent")

    # Save the spec
    success = save_spec(spec_id, spec_data, specs_dir)
    if not success:
        return None, "Failed to save specification", []

    if warnings:
        result["dependency_warnings"] = warnings

    return result, None, warnings


def _generate_requirement_id(existing_requirements: List[Dict[str, Any]]) -> str:
    """
    Generate a unique requirement ID based on existing requirements.

    Args:
        existing_requirements: List of existing requirement dictionaries

    Returns:
        New requirement ID string (e.g., "req-1", "req-2")
    """
    if not existing_requirements:
        return "req-1"

    max_index = 0
    pattern = re.compile(r"^req-(\d+)$")

    for req in existing_requirements:
        req_id = req.get("id", "")
        match = pattern.match(req_id)
        if match:
            index = int(match.group(1))
            max_index = max(max_index, index)

    return f"req-{max_index + 1}"


def update_task_requirements(
    spec_id: str,
    task_id: str,
    action: str = "add",
    requirement_type: Optional[str] = None,
    text: Optional[str] = None,
    requirement_id: Optional[str] = None,
    dry_run: bool = False,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Add or remove a structured requirement from a task's metadata.

    Requirements are stored in metadata.requirements as a list of objects:
    [{"id": "req-1", "type": "acceptance", "text": "..."}, ...]

    Each requirement has:
    - id: Auto-generated unique ID (e.g., "req-1", "req-2")
    - type: Requirement type (acceptance, technical, constraint)
    - text: Requirement description text

    Args:
        spec_id: Specification ID containing the task.
        task_id: Task ID to update.
        action: Action to perform ("add" or "remove"). Default: "add".
        requirement_type: Requirement type (required for add). One of:
                         acceptance, technical, constraint.
        text: Requirement text (required for add).
        requirement_id: Requirement ID to remove (required for remove action).
        dry_run: If True, validate and return preview without saving changes.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"task_id": ..., "action": ..., "requirement": {...}, ...}, None)
        On failure: (None, "error message")
    """
    # Validate action
    if action not in ("add", "remove"):
        return None, f"Invalid action '{action}'. Must be 'add' or 'remove'"

    # Validate parameters based on action
    if action == "add":
        if requirement_type is None:
            return None, "requirement_type is required for add action"
        if not isinstance(requirement_type, str):
            return None, "requirement_type must be a string"
        requirement_type = requirement_type.lower().strip()
        if requirement_type not in REQUIREMENT_TYPES:
            return None, f"Invalid requirement_type '{requirement_type}'. Must be one of: {', '.join(REQUIREMENT_TYPES)}"

        if text is None:
            return None, "text is required for add action"
        if not isinstance(text, str) or not text.strip():
            return None, "text must be a non-empty string"
        text = text.strip()

    elif action == "remove":
        if requirement_id is None:
            return None, "requirement_id is required for remove action"
        if not isinstance(requirement_id, str) or not requirement_id.strip():
            return None, "requirement_id must be a non-empty string"
        requirement_id = requirement_id.strip()

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
        return None, f"Cannot update requirements for node type '{task_type}'. Only task, subtask, or verify nodes can be updated."

    # Get or create metadata
    metadata = task.get("metadata")
    if metadata is None:
        metadata = {}
        task["metadata"] = metadata

    # Get or create requirements list
    requirements = metadata.get("requirements")
    if requirements is None:
        requirements = []
        metadata["requirements"] = requirements
    elif not isinstance(requirements, list):
        requirements = []
        metadata["requirements"] = requirements

    # Perform the action
    if action == "add":
        # Check limit
        if len(requirements) >= MAX_REQUIREMENTS_PER_TASK:
            return None, f"Cannot add requirement: task already has {MAX_REQUIREMENTS_PER_TASK} requirements (maximum)"

        # Generate new requirement ID
        new_id = _generate_requirement_id(requirements)

        # Create requirement object
        new_requirement = {
            "id": new_id,
            "type": requirement_type,
            "text": text,
        }

        # Add to list
        requirements.append(new_requirement)

        result = {
            "spec_id": spec_id,
            "task_id": task_id,
            "action": "add",
            "requirement": new_requirement,
            "total_requirements": len(requirements),
            "dry_run": dry_run,
        }

    elif action == "remove":
        # Find requirement by ID
        found_index = None
        removed_requirement = None
        for i, req in enumerate(requirements):
            if req.get("id") == requirement_id:
                found_index = i
                removed_requirement = req
                break

        if found_index is None:
            return None, f"Requirement '{requirement_id}' not found in task '{task_id}'"

        # Remove from list
        requirements.pop(found_index)

        result = {
            "spec_id": spec_id,
            "task_id": task_id,
            "action": "remove",
            "requirement": removed_requirement,
            "total_requirements": len(requirements),
            "dry_run": dry_run,
        }

    # Save the spec (unless dry_run)
    if dry_run:
        result["message"] = "Dry run - changes not saved"
    else:
        success = save_spec(spec_id, spec_data, specs_dir)
        if not success:
            return None, "Failed to save specification"

    return result, None
