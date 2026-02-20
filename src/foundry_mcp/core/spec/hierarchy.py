"""
Hierarchy operations for SDD spec nodes.

Functions for traversing and mutating the spec hierarchy: get/update nodes,
add/remove/move phases, recalculate hours, and update phase metadata.

Imports ``io`` (for find/load/save) and ``_constants`` only.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from foundry_mcp.core.spec._constants import (
    CATEGORIES,
    VERIFICATION_TYPES,
)
from foundry_mcp.core.spec.io import (
    find_spec_file,
    find_specs_directory,
    load_spec,
    save_spec,
)




def _normalize_acceptance_criteria(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, list):
        cleaned_items = []
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    cleaned_items.append(cleaned)
        return cleaned_items
    return []


def get_node(spec_data: Dict[str, Any], node_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific node from the hierarchy.

    Args:
        spec_data: JSON spec file data
        node_id: Node identifier

    Returns:
        Node data dictionary or None if not found
    """
    hierarchy = spec_data.get("hierarchy", {})
    return hierarchy.get(node_id)


def update_node(spec_data: Dict[str, Any], node_id: str, updates: Dict[str, Any]) -> bool:
    """
    Update a node in the hierarchy.

    Special handling for metadata: existing metadata fields are preserved
    and merged with new metadata fields.

    Args:
        spec_data: JSON spec file data
        node_id: Node identifier
        updates: Dictionary of fields to update

    Returns:
        True if node exists and was updated, False otherwise
    """
    hierarchy = spec_data.get("hierarchy", {})

    if node_id not in hierarchy:
        return False

    node = hierarchy[node_id]

    if "metadata" in updates:
        existing_metadata = node.get("metadata", {})
        new_metadata = updates["metadata"]
        updates = updates.copy()
        updates["metadata"] = {**existing_metadata, **new_metadata}

    node.update(updates)
    return True


# =============================================================================
# Phase scaffolding helpers
# =============================================================================


def _add_phase_verification(hierarchy: Dict[str, Any], phase_num: int, phase_id: str) -> None:
    """
    Add verify nodes (auto + fidelity) to a phase.

    Args:
        hierarchy: The hierarchy dict to modify.
        phase_num: Phase number (1, 2, 3, etc.).
        phase_id: Phase node ID (e.g., "phase-1").
    """
    verify_auto_id = f"verify-{phase_num}-1"
    verify_fidelity_id = f"verify-{phase_num}-2"

    # Run tests verification
    hierarchy[verify_auto_id] = {
        "type": "verify",
        "title": "Run tests",
        "status": "pending",
        "parent": phase_id,
        "children": [],
        "total_tasks": 1,
        "completed_tasks": 0,
        "metadata": {
            "verification_type": "run-tests",
            "command": "pytest",
            "expected": "All tests pass",
        },
        "dependencies": {
            "blocks": [verify_fidelity_id],
            "blocked_by": [],
            "depends": [],
        },
    }

    # Fidelity verification (spec review)
    hierarchy[verify_fidelity_id] = {
        "type": "verify",
        "title": "Fidelity review",
        "status": "pending",
        "parent": phase_id,
        "children": [],
        "total_tasks": 1,
        "completed_tasks": 0,
        "metadata": {
            "verification_type": "fidelity",
            "mcp_tool": "mcp__foundry-mcp__spec-review-fidelity",
            "scope": "phase",
            "target": phase_id,
            "expected": "Implementation matches specification",
        },
        "dependencies": {
            "blocks": [],
            "blocked_by": [verify_auto_id],
            "depends": [],
        },
    }

    # Update phase children and task count
    hierarchy[phase_id]["children"].extend([verify_auto_id, verify_fidelity_id])
    hierarchy[phase_id]["total_tasks"] += 2


def _generate_phase_id(hierarchy: Dict[str, Any]) -> Tuple[str, int]:
    """Generate the next phase ID and numeric suffix."""
    pattern = re.compile(r"^phase-(\d+)$")
    max_id = 0
    for node_id in hierarchy.keys():
        match = pattern.match(node_id)
        if match:
            max_id = max(max_id, int(match.group(1)))
    next_id = max_id + 1
    return f"phase-{next_id}", next_id


def add_phase(
    spec_id: str,
    title: str,
    description: Optional[str] = None,
    purpose: Optional[str] = None,
    estimated_hours: Optional[float] = None,
    position: Optional[int] = None,
    link_previous: bool = True,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Add a new phase under spec-root and scaffold verification tasks.

    Args:
        spec_id: Specification ID to mutate.
        title: Phase title.
        description: Optional phase description.
        purpose: Optional purpose/goal metadata string.
        estimated_hours: Optional estimated hours for the phase.
        position: Optional zero-based insertion index in spec-root children.
        link_previous: Whether to automatically block on the previous phase when appending.
        specs_dir: Specs directory override.

    Returns:
        Tuple of (result_dict, error_message).
    """
    if not spec_id or not spec_id.strip():
        return None, "Specification ID is required"

    if not title or not title.strip():
        return None, "Phase title is required"

    if estimated_hours is not None and estimated_hours < 0:
        return None, "estimated_hours must be non-negative"

    title = title.strip()

    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set FOUNDRY_SPECS_DIR.",
        )

    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    hierarchy = spec_data.get("hierarchy", {})
    spec_root = hierarchy.get("spec-root")

    if spec_root is None:
        return None, "Specification root node 'spec-root' not found"

    if spec_root.get("type") not in {"spec", "root"}:
        return None, "Specification root node has invalid type"

    children = spec_root.get("children", []) or []
    if not isinstance(children, list):
        children = []

    insert_index = len(children)
    if position is not None and position >= 0:
        insert_index = min(position, len(children))

    phase_id, phase_num = _generate_phase_id(hierarchy)

    metadata: Dict[str, Any] = {
        "purpose": (purpose.strip() if purpose else ""),
    }
    if description:
        metadata["description"] = description.strip()
    if estimated_hours is not None:
        metadata["estimated_hours"] = estimated_hours

    phase_node = {
        "type": "phase",
        "title": title,
        "status": "pending",
        "parent": "spec-root",
        "children": [],
        "total_tasks": 0,
        "completed_tasks": 0,
        "metadata": metadata,
        "dependencies": {
            "blocks": [],
            "blocked_by": [],
            "depends": [],
        },
    }

    hierarchy[phase_id] = phase_node

    if insert_index == len(children):
        children.append(phase_id)
    else:
        children.insert(insert_index, phase_id)
    spec_root["children"] = children

    linked_phase_id: Optional[str] = None
    if link_previous and insert_index > 0 and insert_index == len(children) - 1:
        candidate = children[insert_index - 1]
        previous = hierarchy.get(candidate)
        if previous and previous.get("type") == "phase":
            linked_phase_id = candidate
            prev_deps = previous.setdefault(
                "dependencies",
                {
                    "blocks": [],
                    "blocked_by": [],
                    "depends": [],
                },
            )
            blocks = prev_deps.setdefault("blocks", [])
            if phase_id not in blocks:
                blocks.append(phase_id)
            phase_node["dependencies"]["blocked_by"].append(candidate)

    _add_phase_verification(hierarchy, phase_num, phase_id)

    phase_task_total = phase_node.get("total_tasks", 0)
    total_tasks = spec_root.get("total_tasks", 0)
    spec_root["total_tasks"] = total_tasks + phase_task_total

    # Update spec-level estimated hours if provided
    if estimated_hours is not None:
        spec_metadata = spec_data.setdefault("metadata", {})
        current_hours = spec_metadata.get("estimated_hours")
        if isinstance(current_hours, (int, float)):
            spec_metadata["estimated_hours"] = current_hours + estimated_hours
        else:
            spec_metadata["estimated_hours"] = estimated_hours

    saved = save_spec(spec_id, spec_data, specs_dir)
    if not saved:
        return None, "Failed to save specification"

    verify_ids = [f"verify-{phase_num}-1", f"verify-{phase_num}-2"]

    return {
        "spec_id": spec_id,
        "phase_id": phase_id,
        "title": title,
        "position": insert_index,
        "linked_previous": linked_phase_id,
        "verify_tasks": verify_ids,
    }, None


def add_phase_bulk(
    spec_id: str,
    phase_title: str,
    tasks: List[Dict[str, Any]],
    phase_description: Optional[str] = None,
    phase_purpose: Optional[str] = None,
    phase_estimated_hours: Optional[float] = None,
    metadata_defaults: Optional[Dict[str, Any]] = None,
    position: Optional[int] = None,
    link_previous: bool = True,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Add a new phase with pre-defined tasks in a single atomic operation.

    Creates a phase and all specified tasks/verify nodes without auto-generating
    verification scaffolding. This enables creating complete phase structures
    in one operation.

    Args:
        spec_id: Specification ID to mutate.
        phase_title: Phase title.
        tasks: List of task definitions, each containing:
            - type: "task" or "verify" (required)
            - title: Task title (required)
            - description: Optional description
            - acceptance_criteria: Optional list of acceptance criteria
            - task_category: Optional task category
            - file_path: Optional associated file path
            - estimated_hours: Optional time estimate
            - verification_type: Optional verification type for verify tasks
        phase_description: Optional phase description.
        phase_purpose: Optional purpose/goal metadata string.
        phase_estimated_hours: Optional estimated hours for the phase.
        metadata_defaults: Optional defaults applied to tasks missing explicit values.
            Supported keys: task_category, category, acceptance_criteria, estimated_hours
        position: Optional zero-based insertion index in spec-root children.
        link_previous: Whether to automatically block on the previous phase.
        specs_dir: Specs directory override.

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"phase_id": ..., "tasks_created": [...], ...}, None)
        On failure: (None, "error message")
    """
    # Validate required parameters
    if not spec_id or not spec_id.strip():
        return None, "Specification ID is required"

    if not phase_title or not phase_title.strip():
        return None, "Phase title is required"

    if not tasks or not isinstance(tasks, list) or len(tasks) == 0:
        return None, "At least one task definition is required"

    if phase_estimated_hours is not None and phase_estimated_hours < 0:
        return None, "phase_estimated_hours must be non-negative"

    phase_title = phase_title.strip()
    defaults = metadata_defaults or {}

    # Validate metadata_defaults values
    if defaults:
        default_est_hours = defaults.get("estimated_hours")
        if default_est_hours is not None:
            if not isinstance(default_est_hours, (int, float)) or default_est_hours < 0:
                return None, "metadata_defaults.estimated_hours must be a non-negative number"
        default_category = defaults.get("task_category")
        if default_category is None:
            default_category = defaults.get("category")
        if default_category is not None and not isinstance(default_category, str):
            return None, "metadata_defaults.task_category must be a string"
        default_acceptance = defaults.get("acceptance_criteria")
        if default_acceptance is not None and not isinstance(default_acceptance, (list, str)):
            return None, "metadata_defaults.acceptance_criteria must be a list of strings"
        if isinstance(default_acceptance, list) and any(not isinstance(item, str) for item in default_acceptance):
            return None, "metadata_defaults.acceptance_criteria must be a list of strings"

    # Validate each task definition
    # Valid task types match TASK_TYPES from task.py (avoiding circular import)
    valid_task_types = {"task", "subtask", "verify", "research"}
    for idx, task_def in enumerate(tasks):
        if not isinstance(task_def, dict):
            return None, f"Task at index {idx} must be a dictionary"

        task_type = task_def.get("type")
        if not task_type or task_type not in valid_task_types:
            return None, f"Task at index {idx} must have type: {', '.join(sorted(valid_task_types))}"

        task_title = task_def.get("title")
        if not task_title or not isinstance(task_title, str) or not task_title.strip():
            return None, f"Task at index {idx} must have a non-empty title"

        est_hours = task_def.get("estimated_hours")
        if est_hours is not None:
            if not isinstance(est_hours, (int, float)) or est_hours < 0:
                return None, f"Task at index {idx} has invalid estimated_hours"

        task_category = task_def.get("task_category")
        if task_category is not None and not isinstance(task_category, str):
            return None, f"Task at index {idx} has invalid task_category"

        acceptance_criteria = task_def.get("acceptance_criteria")
        if acceptance_criteria is not None and not isinstance(acceptance_criteria, (list, str)):
            return None, f"Task at index {idx} has invalid acceptance_criteria"
        if isinstance(acceptance_criteria, list) and any(not isinstance(item, str) for item in acceptance_criteria):
            return None, f"Task at index {idx} acceptance_criteria must be a list of strings"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set FOUNDRY_SPECS_DIR.",
        )

    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    hierarchy = spec_data.get("hierarchy", {})
    spec_root = hierarchy.get("spec-root")

    if spec_root is None:
        return None, "Specification root node 'spec-root' not found"

    if spec_root.get("type") not in {"spec", "root"}:
        return None, "Specification root node has invalid type"

    children = spec_root.get("children", []) or []
    if not isinstance(children, list):
        children = []

    insert_index = len(children)
    if position is not None and position >= 0:
        insert_index = min(position, len(children))

    # Generate phase ID
    phase_id, phase_num = _generate_phase_id(hierarchy)

    # Build phase metadata
    phase_metadata: Dict[str, Any] = {
        "purpose": (phase_purpose.strip() if phase_purpose else ""),
    }
    if phase_description:
        phase_metadata["description"] = phase_description.strip()
    if phase_estimated_hours is not None:
        phase_metadata["estimated_hours"] = phase_estimated_hours

    # Create phase node (without children initially)
    phase_node = {
        "type": "phase",
        "title": phase_title,
        "status": "pending",
        "parent": "spec-root",
        "children": [],
        "total_tasks": 0,
        "completed_tasks": 0,
        "metadata": phase_metadata,
        "dependencies": {
            "blocks": [],
            "blocked_by": [],
            "depends": [],
        },
    }

    hierarchy[phase_id] = phase_node

    # Insert phase into spec-root children
    if insert_index == len(children):
        children.append(phase_id)
    else:
        children.insert(insert_index, phase_id)
    spec_root["children"] = children

    # Link to previous phase if requested
    linked_phase_id: Optional[str] = None
    if link_previous and insert_index > 0 and insert_index == len(children) - 1:
        candidate = children[insert_index - 1]
        previous = hierarchy.get(candidate)
        if previous and previous.get("type") == "phase":
            linked_phase_id = candidate
            prev_deps = previous.setdefault(
                "dependencies",
                {"blocks": [], "blocked_by": [], "depends": []},
            )
            blocks = prev_deps.setdefault("blocks", [])
            if phase_id not in blocks:
                blocks.append(phase_id)
            phase_node["dependencies"]["blocked_by"].append(candidate)

    def _nonempty_string(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def _extract_description(task_def: Dict[str, Any]) -> tuple[Optional[str], Any]:
        description = task_def.get("description")
        if _nonempty_string(description) and isinstance(description, str):
            return "description", description.strip()
        details = task_def.get("details")
        if _nonempty_string(details) and isinstance(details, str):
            return "details", details.strip()
        if isinstance(details, list):
            cleaned = [item.strip() for item in details if isinstance(item, str) and item.strip()]
            if cleaned:
                return "details", cleaned
        return None, None

    # Create tasks under the phase
    tasks_created: List[Dict[str, Any]] = []
    task_counter = 0
    verify_counter = 0

    for task_def in tasks:
        task_type = task_def["type"]
        task_title = task_def["title"].strip()

        # Generate task ID based on type
        if task_type == "verify":
            verify_counter += 1
            task_id = f"verify-{phase_num}-{verify_counter}"
        else:
            task_counter += 1
            task_id = f"task-{phase_num}-{task_counter}"

        # Build task metadata with defaults cascade
        task_metadata: Dict[str, Any] = {}

        # Apply description/details
        desc_field, desc_value = _extract_description(task_def)
        if desc_field and desc_value is not None:
            task_metadata[desc_field] = desc_value
        elif task_type == "task":
            return None, f"Task '{task_title}' missing description"

        # Apply file_path
        file_path = task_def.get("file_path")
        if file_path and isinstance(file_path, str):
            task_metadata["file_path"] = file_path.strip()

        # Apply estimated_hours (task-level overrides defaults)
        est_hours = task_def.get("estimated_hours")
        if est_hours is not None:
            task_metadata["estimated_hours"] = float(est_hours)
        elif defaults.get("estimated_hours") is not None:
            task_metadata["estimated_hours"] = float(defaults["estimated_hours"])

        normalized_category = None
        if task_type == "task":
            # Apply acceptance_criteria
            raw_acceptance = task_def.get("acceptance_criteria")
            if raw_acceptance is None:
                raw_acceptance = defaults.get("acceptance_criteria")
            acceptance_criteria = _normalize_acceptance_criteria(raw_acceptance)
            if acceptance_criteria is not None:
                task_metadata["acceptance_criteria"] = acceptance_criteria
            if raw_acceptance is None:
                return None, f"Task '{task_title}' missing acceptance_criteria"
            if not acceptance_criteria:
                return (
                    None,
                    f"Task '{task_title}' acceptance_criteria must include at least one entry",
                )

            # Apply task_category from defaults if not specified
            category = task_def.get("task_category") or task_def.get("category")
            if category is None:
                category = defaults.get("task_category") or defaults.get("category")
            if category and isinstance(category, str):
                normalized_category = category.strip().lower()
                if normalized_category not in CATEGORIES:
                    return (
                        None,
                        f"Task '{task_title}' has invalid task_category '{category}'",
                    )
                task_metadata["task_category"] = normalized_category
            if normalized_category is None:
                return None, f"Task '{task_title}' missing task_category"

            if normalized_category in {"implementation", "refactoring"}:
                if not _nonempty_string(task_metadata.get("file_path")):
                    return (
                        None,
                        f"Task '{task_title}' missing file_path for category '{normalized_category}'",
                    )

        # Apply verification_type for verify tasks
        if task_type == "verify":
            verify_type = task_def.get("verification_type")
            if verify_type and verify_type in VERIFICATION_TYPES:
                task_metadata["verification_type"] = verify_type

        # Create task node
        task_node = {
            "type": task_type,
            "title": task_title,
            "status": "pending",
            "parent": phase_id,
            "children": [],
            "total_tasks": 1,
            "completed_tasks": 0,
            "metadata": task_metadata,
            "dependencies": {
                "blocks": [],
                "blocked_by": [],
                "depends": [],
            },
        }

        hierarchy[task_id] = task_node
        phase_node["children"].append(task_id)
        phase_node["total_tasks"] += 1

        tasks_created.append(
            {
                "task_id": task_id,
                "title": task_title,
                "type": task_type,
            }
        )

    # Update spec-root total_tasks
    total_tasks = spec_root.get("total_tasks", 0)
    spec_root["total_tasks"] = total_tasks + phase_node["total_tasks"]

    # Update spec-level estimated hours if provided
    if phase_estimated_hours is not None:
        spec_metadata = spec_data.setdefault("metadata", {})
        current_hours = spec_metadata.get("estimated_hours")
        if isinstance(current_hours, (int, float)):
            spec_metadata["estimated_hours"] = current_hours + phase_estimated_hours
        else:
            spec_metadata["estimated_hours"] = phase_estimated_hours

    # Save spec atomically
    saved = save_spec(spec_id, spec_data, specs_dir)
    if not saved:
        return None, "Failed to save specification"

    return {
        "spec_id": spec_id,
        "phase_id": phase_id,
        "title": phase_title,
        "position": insert_index,
        "linked_previous": linked_phase_id,
        "tasks_created": tasks_created,
        "total_tasks": len(tasks_created),
    }, None


# Shared hierarchy traversal utilities — canonical implementations live in
# hierarchy_helpers.py.  Private aliases kept for backwards compatibility
# with callers that import from this module.
from foundry_mcp.core.hierarchy_helpers import (  # noqa: F401
    collect_descendants as _collect_descendants,
)
from foundry_mcp.core.hierarchy_helpers import (
    count_tasks_in_subtree as _count_tasks_in_subtree,
)
from foundry_mcp.core.hierarchy_helpers import (
    remove_dependency_references as _remove_dependency_references,
)


def remove_phase(
    spec_id: str,
    phase_id: str,
    force: bool = False,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Remove a phase and all its children from a specification.

    Handles adjacent phase re-linking: if phase B is removed and A blocks B
    which blocks C, then A will be updated to block C directly.

    Args:
        spec_id: Specification ID containing the phase.
        phase_id: Phase ID to remove (e.g., "phase-1").
        force: If True, remove even if phase contains non-completed tasks.
               If False (default), refuse to remove phases with active work.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "phase_id": ..., "children_removed": ..., ...}, None)
        On failure: (None, "error message")
    """
    # Validate inputs
    if not spec_id or not spec_id.strip():
        return None, "Specification ID is required"

    if not phase_id or not phase_id.strip():
        return None, "Phase ID is required"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set FOUNDRY_SPECS_DIR.",
        )

    # Find and load the spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    hierarchy = spec_data.get("hierarchy", {})

    # Validate phase exists
    phase = hierarchy.get(phase_id)
    if phase is None:
        return None, f"Phase '{phase_id}' not found"

    # Validate node type is phase
    node_type = phase.get("type")
    if node_type != "phase":
        return None, f"Node '{phase_id}' is not a phase (type: {node_type})"

    # Collect all descendants
    descendants = _collect_descendants(hierarchy, phase_id)

    # Check for non-completed tasks if force is False
    if not force:
        # Count tasks in phase (excluding verify nodes for the active work check)
        all_nodes = [phase_id] + descendants
        has_active_work = False
        active_task_ids: List[str] = []

        for node_id in all_nodes:
            node = hierarchy.get(node_id)
            if not node:
                continue
            node_status = node.get("status")
            node_node_type = node.get("type")
            # Consider in_progress or pending tasks as active work
            if node_node_type in ("task", "subtask") and node_status in (
                "pending",
                "in_progress",
            ):
                has_active_work = True
                active_task_ids.append(node_id)

        if has_active_work:
            return (
                None,
                f"Phase '{phase_id}' has {len(active_task_ids)} non-completed task(s). "
                f"Use force=True to remove anyway. Active tasks: {', '.join(active_task_ids[:5])}"
                + ("..." if len(active_task_ids) > 5 else ""),
            )

    # Get spec-root and phase position info for re-linking
    spec_root = hierarchy.get("spec-root")
    if spec_root is None:
        return None, "Specification root node 'spec-root' not found"

    children = spec_root.get("children", [])
    if not isinstance(children, list):
        children = []

    # Find phase position
    try:
        phase_index = children.index(phase_id)
    except ValueError:
        return None, f"Phase '{phase_id}' not found in spec-root children"

    # Identify adjacent phases for re-linking
    prev_phase_id: Optional[str] = None
    next_phase_id: Optional[str] = None

    if phase_index > 0:
        candidate = children[phase_index - 1]
        if hierarchy.get(candidate, {}).get("type") == "phase":
            prev_phase_id = candidate

    if phase_index < len(children) - 1:
        candidate = children[phase_index + 1]
        if hierarchy.get(candidate, {}).get("type") == "phase":
            next_phase_id = candidate

    # Re-link adjacent phases: if prev blocks this phase and this phase blocks next,
    # then prev should now block next directly
    relinked_from: Optional[str] = None
    relinked_to: Optional[str] = None

    if prev_phase_id and next_phase_id:
        prev_phase = hierarchy.get(prev_phase_id)
        next_phase = hierarchy.get(next_phase_id)

        if prev_phase and next_phase:
            # Check if prev_phase blocks this phase
            prev_deps = prev_phase.get("dependencies", {})
            prev_blocks = prev_deps.get("blocks", [])

            # Check if this phase blocks next_phase
            phase_deps = phase.get("dependencies", {})
            phase_blocks = phase_deps.get("blocks", [])

            if phase_id in prev_blocks and next_phase_id in phase_blocks:
                # Re-link: prev should now block next
                if next_phase_id not in prev_blocks:
                    prev_blocks.append(next_phase_id)

                # Update next phase's blocked_by
                next_deps = next_phase.setdefault(
                    "dependencies",
                    {
                        "blocks": [],
                        "blocked_by": [],
                        "depends": [],
                    },
                )
                next_blocked_by = next_deps.setdefault("blocked_by", [])
                if prev_phase_id not in next_blocked_by:
                    next_blocked_by.append(prev_phase_id)

                relinked_from = prev_phase_id
                relinked_to = next_phase_id

    # Count tasks being removed
    nodes_to_remove = [phase_id] + descendants
    total_removed, completed_removed = _count_tasks_in_subtree(hierarchy, descendants)

    # Remove all nodes from hierarchy
    for node_id in nodes_to_remove:
        if node_id in hierarchy:
            del hierarchy[node_id]

    # Remove phase from spec-root children
    children.remove(phase_id)
    spec_root["children"] = children

    # Update spec-root task counts
    current_total = spec_root.get("total_tasks", 0)
    current_completed = spec_root.get("completed_tasks", 0)
    spec_root["total_tasks"] = max(0, current_total - total_removed)
    spec_root["completed_tasks"] = max(0, current_completed - completed_removed)

    # Clean up dependency references to removed nodes
    _remove_dependency_references(hierarchy, nodes_to_remove)

    # Save the spec
    saved = save_spec(spec_id, spec_data, specs_dir)
    if not saved:
        return None, "Failed to save specification"

    result: Dict[str, Any] = {
        "spec_id": spec_id,
        "phase_id": phase_id,
        "phase_title": phase.get("title", ""),
        "children_removed": len(descendants),
        "total_tasks_removed": total_removed,
        "completed_tasks_removed": completed_removed,
        "force": force,
    }

    if relinked_from and relinked_to:
        result["relinked"] = {
            "from": relinked_from,
            "to": relinked_to,
        }

    return result, None


def move_phase(
    spec_id: str,
    phase_id: str,
    position: int,
    link_previous: bool = True,
    dry_run: bool = False,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Move a phase to a new position within spec-root's children.

    Supports reordering phases and optionally re-linking phase dependencies
    according to the link_previous pattern (each phase blocked by its predecessor).

    Args:
        spec_id: Specification ID containing the phase.
        phase_id: Phase ID to move (e.g., "phase-2").
        position: Target position (1-based index) in spec-root children.
        link_previous: If True, update dependencies to maintain the sequential
                       blocking pattern. If False, preserve existing dependencies.
        dry_run: If True, validate and return preview without saving changes.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "phase_id": ..., "old_position": ..., "new_position": ..., ...}, None)
        On failure: (None, "error message")
    """
    # Validate inputs
    if not spec_id or not spec_id.strip():
        return None, "Specification ID is required"

    if not phase_id or not phase_id.strip():
        return None, "Phase ID is required"

    if not isinstance(position, int) or position < 1:
        return None, "Position must be a positive integer (1-based)"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set FOUNDRY_SPECS_DIR.",
        )

    # Find and load the spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    hierarchy = spec_data.get("hierarchy", {})

    # Validate phase exists
    phase = hierarchy.get(phase_id)
    if phase is None:
        return None, f"Phase '{phase_id}' not found"

    # Validate node type is phase
    node_type = phase.get("type")
    if node_type != "phase":
        return None, f"Node '{phase_id}' is not a phase (type: {node_type})"

    # Get spec-root
    spec_root = hierarchy.get("spec-root")
    if spec_root is None:
        return None, "Specification root node 'spec-root' not found"

    children = spec_root.get("children", [])
    if not isinstance(children, list):
        children = []

    # Find current position
    try:
        old_index = children.index(phase_id)
    except ValueError:
        return None, f"Phase '{phase_id}' not found in spec-root children"

    # Convert to 0-based index for internal use
    new_index = position - 1

    # Validate position is within bounds
    if new_index < 0 or new_index >= len(children):
        return None, f"Invalid position {position}. Must be 1-{len(children)}"

    # No change needed if same position
    if old_index == new_index:
        return {
            "spec_id": spec_id,
            "phase_id": phase_id,
            "phase_title": phase.get("title", ""),
            "old_position": old_index + 1,
            "new_position": new_index + 1,
            "moved": False,
            "dry_run": dry_run,
            "message": "Phase is already at the specified position",
        }, None

    # Identify old neighbors for dependency cleanup
    old_prev_id: Optional[str] = None
    old_next_id: Optional[str] = None

    if old_index > 0:
        candidate = children[old_index - 1]
        if hierarchy.get(candidate, {}).get("type") == "phase":
            old_prev_id = candidate

    if old_index < len(children) - 1:
        candidate = children[old_index + 1]
        if hierarchy.get(candidate, {}).get("type") == "phase":
            old_next_id = candidate

    # Perform the move in children list
    children.remove(phase_id)
    # After removal, adjust target index if moving forward
    insert_index = new_index if new_index <= old_index else new_index
    if insert_index >= len(children):
        children.append(phase_id)
    else:
        children.insert(insert_index, phase_id)

    # Identify new neighbors
    actual_new_index = children.index(phase_id)
    new_prev_id: Optional[str] = None
    new_next_id: Optional[str] = None

    if actual_new_index > 0:
        candidate = children[actual_new_index - 1]
        if hierarchy.get(candidate, {}).get("type") == "phase":
            new_prev_id = candidate

    if actual_new_index < len(children) - 1:
        candidate = children[actual_new_index + 1]
        if hierarchy.get(candidate, {}).get("type") == "phase":
            new_next_id = candidate

    # Track dependency changes
    dependencies_updated: List[Dict[str, Any]] = []

    if link_previous:
        # Remove old dependency links
        phase_deps = phase.setdefault("dependencies", {"blocks": [], "blocked_by": [], "depends": []})

        # 1. Remove this phase from old_prev's blocks list
        if old_prev_id:
            old_prev = hierarchy.get(old_prev_id)
            if old_prev:
                old_prev_deps = old_prev.get("dependencies", {})
                old_prev_blocks = old_prev_deps.get("blocks", [])
                if phase_id in old_prev_blocks:
                    old_prev_blocks.remove(phase_id)
                    dependencies_updated.append(
                        {
                            "action": "removed",
                            "from": old_prev_id,
                            "relationship": "blocks",
                            "target": phase_id,
                        }
                    )

        # 2. Remove old_prev from this phase's blocked_by
        phase_blocked_by = phase_deps.setdefault("blocked_by", [])
        if old_prev_id and old_prev_id in phase_blocked_by:
            phase_blocked_by.remove(old_prev_id)
            dependencies_updated.append(
                {
                    "action": "removed",
                    "from": phase_id,
                    "relationship": "blocked_by",
                    "target": old_prev_id,
                }
            )

        # 3. Remove this phase from old_next's blocked_by
        if old_next_id:
            old_next = hierarchy.get(old_next_id)
            if old_next:
                old_next_deps = old_next.get("dependencies", {})
                old_next_blocked_by = old_next_deps.get("blocked_by", [])
                if phase_id in old_next_blocked_by:
                    old_next_blocked_by.remove(phase_id)
                    dependencies_updated.append(
                        {
                            "action": "removed",
                            "from": old_next_id,
                            "relationship": "blocked_by",
                            "target": phase_id,
                        }
                    )

        # 4. Remove old_next from this phase's blocks
        phase_blocks = phase_deps.setdefault("blocks", [])
        if old_next_id and old_next_id in phase_blocks:
            phase_blocks.remove(old_next_id)
            dependencies_updated.append(
                {
                    "action": "removed",
                    "from": phase_id,
                    "relationship": "blocks",
                    "target": old_next_id,
                }
            )

        # 5. Link old neighbors to each other (if they were adjacent via this phase)
        if old_prev_id and old_next_id:
            old_prev = hierarchy.get(old_prev_id)
            old_next = hierarchy.get(old_next_id)
            if old_prev and old_next:
                old_prev_deps = old_prev.setdefault("dependencies", {"blocks": [], "blocked_by": [], "depends": []})
                old_prev_blocks = old_prev_deps.setdefault("blocks", [])
                if old_next_id not in old_prev_blocks:
                    old_prev_blocks.append(old_next_id)
                    dependencies_updated.append(
                        {
                            "action": "added",
                            "from": old_prev_id,
                            "relationship": "blocks",
                            "target": old_next_id,
                        }
                    )

                old_next_deps = old_next.setdefault("dependencies", {"blocks": [], "blocked_by": [], "depends": []})
                old_next_blocked_by = old_next_deps.setdefault("blocked_by", [])
                if old_prev_id not in old_next_blocked_by:
                    old_next_blocked_by.append(old_prev_id)
                    dependencies_updated.append(
                        {
                            "action": "added",
                            "from": old_next_id,
                            "relationship": "blocked_by",
                            "target": old_prev_id,
                        }
                    )

        # Add new dependency links
        # 6. New prev blocks this phase
        if new_prev_id:
            new_prev = hierarchy.get(new_prev_id)
            if new_prev:
                new_prev_deps = new_prev.setdefault("dependencies", {"blocks": [], "blocked_by": [], "depends": []})
                new_prev_blocks = new_prev_deps.setdefault("blocks", [])
                if phase_id not in new_prev_blocks:
                    new_prev_blocks.append(phase_id)
                    dependencies_updated.append(
                        {
                            "action": "added",
                            "from": new_prev_id,
                            "relationship": "blocks",
                            "target": phase_id,
                        }
                    )

                # This phase is blocked by new prev
                if new_prev_id not in phase_blocked_by:
                    phase_blocked_by.append(new_prev_id)
                    dependencies_updated.append(
                        {
                            "action": "added",
                            "from": phase_id,
                            "relationship": "blocked_by",
                            "target": new_prev_id,
                        }
                    )

        # 7. This phase blocks new next
        if new_next_id:
            new_next = hierarchy.get(new_next_id)
            if new_next:
                if new_next_id not in phase_blocks:
                    phase_blocks.append(new_next_id)
                    dependencies_updated.append(
                        {
                            "action": "added",
                            "from": phase_id,
                            "relationship": "blocks",
                            "target": new_next_id,
                        }
                    )

                new_next_deps = new_next.setdefault("dependencies", {"blocks": [], "blocked_by": [], "depends": []})
                new_next_blocked_by = new_next_deps.setdefault("blocked_by", [])
                if phase_id not in new_next_blocked_by:
                    new_next_blocked_by.append(phase_id)
                    dependencies_updated.append(
                        {
                            "action": "added",
                            "from": new_next_id,
                            "relationship": "blocked_by",
                            "target": phase_id,
                        }
                    )

                # Remove old link from new prev to new next (now goes through this phase)
                if new_prev_id:
                    new_prev = hierarchy.get(new_prev_id)
                    if new_prev:
                        new_prev_deps = new_prev.get("dependencies", {})
                        new_prev_blocks = new_prev_deps.get("blocks", [])
                        if new_next_id in new_prev_blocks:
                            new_prev_blocks.remove(new_next_id)
                            dependencies_updated.append(
                                {
                                    "action": "removed",
                                    "from": new_prev_id,
                                    "relationship": "blocks",
                                    "target": new_next_id,
                                }
                            )

                    if new_prev_id in new_next_blocked_by:
                        new_next_blocked_by.remove(new_prev_id)
                        dependencies_updated.append(
                            {
                                "action": "removed",
                                "from": new_next_id,
                                "relationship": "blocked_by",
                                "target": new_prev_id,
                            }
                        )

    # Update spec-root children
    spec_root["children"] = children

    # Build result
    result: Dict[str, Any] = {
        "spec_id": spec_id,
        "phase_id": phase_id,
        "phase_title": phase.get("title", ""),
        "old_position": old_index + 1,
        "new_position": actual_new_index + 1,
        "moved": True,
        "link_previous": link_previous,
        "dry_run": dry_run,
    }

    if dependencies_updated:
        result["dependencies_updated"] = dependencies_updated

    if dry_run:
        result["message"] = "Dry run - changes not saved"
        return result, None

    # Save the spec
    saved = save_spec(spec_id, spec_data, specs_dir)
    if not saved:
        return None, "Failed to save specification"

    return result, None


def update_phase_metadata(
    spec_id: str,
    phase_id: str,
    *,
    estimated_hours: Optional[float] = None,
    description: Optional[str] = None,
    purpose: Optional[str] = None,
    dry_run: bool = False,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Update metadata fields of a phase in a specification.

    Allows updating phase-level metadata such as estimated_hours, description,
    and purpose. Tracks previous values for audit purposes.

    Args:
        spec_id: Specification ID containing the phase.
        phase_id: Phase ID to update (e.g., "phase-1").
        estimated_hours: New estimated hours value (must be >= 0 if provided).
        description: New description text for the phase.
        purpose: New purpose text for the phase.
        dry_run: If True, validate and return preview without saving changes.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "phase_id": ..., "updates": [...], ...}, None)
        On failure: (None, "error message")
    """
    # Validate spec_id
    if not spec_id or not spec_id.strip():
        return None, "Specification ID is required"

    # Validate phase_id
    if not phase_id or not phase_id.strip():
        return None, "Phase ID is required"

    # Validate estimated_hours if provided
    if estimated_hours is not None:
        if not isinstance(estimated_hours, (int, float)):
            return None, "estimated_hours must be a number"
        if estimated_hours < 0:
            return None, "estimated_hours must be >= 0"

    # Check that at least one field is being updated
    has_update = any(v is not None for v in [estimated_hours, description, purpose])
    if not has_update:
        return None, "At least one field (estimated_hours, description, purpose) must be provided"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set FOUNDRY_SPECS_DIR.",
        )

    # Find and load the spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    hierarchy = spec_data.get("hierarchy", {})

    # Validate phase exists
    phase = hierarchy.get(phase_id)
    if phase is None:
        return None, f"Phase '{phase_id}' not found"

    # Validate node type is phase
    node_type = phase.get("type")
    if node_type != "phase":
        return None, f"Node '{phase_id}' is not a phase (type: {node_type})"

    # Ensure metadata exists on phase
    if "metadata" not in phase:
        phase["metadata"] = {}

    phase_metadata = phase["metadata"]

    # Track updates with previous values
    updates: List[Dict[str, Any]] = []

    if estimated_hours is not None:
        previous = phase_metadata.get("estimated_hours")
        phase_metadata["estimated_hours"] = estimated_hours
        updates.append(
            {
                "field": "estimated_hours",
                "previous_value": previous,
                "new_value": estimated_hours,
            }
        )

    if description is not None:
        description = description.strip() if description else description
        previous = phase_metadata.get("description")
        phase_metadata["description"] = description
        updates.append(
            {
                "field": "description",
                "previous_value": previous,
                "new_value": description,
            }
        )

    if purpose is not None:
        purpose = purpose.strip() if purpose else purpose
        previous = phase_metadata.get("purpose")
        phase_metadata["purpose"] = purpose
        updates.append(
            {
                "field": "purpose",
                "previous_value": previous,
                "new_value": purpose,
            }
        )

    # Build result
    result: Dict[str, Any] = {
        "spec_id": spec_id,
        "phase_id": phase_id,
        "phase_title": phase.get("title", ""),
        "updates": updates,
        "dry_run": dry_run,
    }

    if dry_run:
        result["message"] = "Dry run - changes not saved"
        return result, None

    # Save the spec
    saved = save_spec(spec_id, spec_data, specs_dir)
    if not saved:
        return None, "Failed to save specification"

    return result, None


def recalculate_estimated_hours(
    spec_id: str,
    dry_run: bool = False,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Recalculate estimated_hours by aggregating from tasks up through the hierarchy.

    Performs full hierarchy rollup:
    1. For each phase: sums estimated_hours from all task/subtask/verify descendants
    2. Updates each phase's metadata.estimated_hours with the calculated sum
    3. Sums all phase estimates to get the spec total
    4. Updates spec metadata.estimated_hours with the calculated sum

    Args:
        spec_id: Specification ID to recalculate.
        dry_run: If True, return report without saving changes.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "phases": [...], "spec_level": {...}, ...}, None)
        On failure: (None, "error message")
    """
    # Validate spec_id
    if not spec_id or not spec_id.strip():
        return None, "Specification ID is required"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return None, "Could not find specs directory"

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Specification '{spec_id}' not found"

    hierarchy = spec_data.get("hierarchy", {})
    spec_root = hierarchy.get("spec-root")
    if not spec_root:
        return None, "Invalid spec: missing spec-root"

    # Get phase children from spec-root
    phase_ids = spec_root.get("children", [])

    # Track results for each phase
    phase_results: List[Dict[str, Any]] = []
    spec_total_calculated = 0.0

    for phase_id in phase_ids:
        phase = hierarchy.get(phase_id)
        if not phase or phase.get("type") != "phase":
            continue

        phase_metadata = phase.get("metadata", {})
        previous_hours = phase_metadata.get("estimated_hours")

        # Collect all descendants of this phase
        descendants = _collect_descendants(hierarchy, phase_id)

        # Sum estimated_hours from task/subtask/verify nodes
        task_count = 0
        calculated_hours = 0.0

        for desc_id in descendants:
            desc_node = hierarchy.get(desc_id)
            if not desc_node:
                continue

            desc_type = desc_node.get("type")
            if desc_type in ("task", "subtask", "verify"):
                task_count += 1
                desc_metadata = desc_node.get("metadata", {})
                est = desc_metadata.get("estimated_hours")
                if isinstance(est, (int, float)) and est >= 0:
                    calculated_hours += float(est)

        # Calculate delta
        prev_value = float(previous_hours) if isinstance(previous_hours, (int, float)) else 0.0
        delta = calculated_hours - prev_value

        phase_results.append(
            {
                "phase_id": phase_id,
                "title": phase.get("title", ""),
                "previous": previous_hours,
                "calculated": calculated_hours,
                "delta": delta,
                "task_count": task_count,
            }
        )

        # Update phase metadata (will be saved if not dry_run)
        if "metadata" not in phase:
            phase["metadata"] = {}
        phase["metadata"]["estimated_hours"] = calculated_hours

        # Add to spec total
        spec_total_calculated += calculated_hours

    # Get spec-level previous value
    spec_metadata = spec_data.get("metadata", {})
    spec_previous = spec_metadata.get("estimated_hours")
    spec_prev_value = float(spec_previous) if isinstance(spec_previous, (int, float)) else 0.0
    spec_delta = spec_total_calculated - spec_prev_value

    # Update spec metadata
    if "metadata" not in spec_data:
        spec_data["metadata"] = {}
    spec_data["metadata"]["estimated_hours"] = spec_total_calculated

    # Build result
    result: Dict[str, Any] = {
        "spec_id": spec_id,
        "dry_run": dry_run,
        "spec_level": {
            "previous": spec_previous,
            "calculated": spec_total_calculated,
            "delta": spec_delta,
        },
        "phases": phase_results,
        "summary": {
            "total_phases": len(phase_results),
            "phases_changed": sum(1 for p in phase_results if p["delta"] != 0),
            "spec_changed": spec_delta != 0,
        },
    }

    if dry_run:
        result["message"] = "Dry run - changes not saved"
        return result, None

    # Save spec
    saved = save_spec(spec_id, spec_data, specs_dir)
    if not saved:
        return None, "Failed to save specification"

    return result, None


def recalculate_actual_hours(
    spec_id: str,
    dry_run: bool = False,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Recalculate actual_hours by aggregating from tasks up through the hierarchy.

    Performs full hierarchy rollup:
    1. For each phase: sums actual_hours from all task/subtask/verify descendants
    2. Updates each phase's metadata.actual_hours with the calculated sum
    3. Sums all phase actuals to get the spec total
    4. Updates spec metadata.actual_hours with the calculated sum

    Args:
        spec_id: Specification ID to recalculate.
        dry_run: If True, return report without saving changes.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "phases": [...], "spec_level": {...}, ...}, None)
        On failure: (None, "error message")
    """
    # Validate spec_id
    if not spec_id or not spec_id.strip():
        return None, "Specification ID is required"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return None, "Could not find specs directory"

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Specification '{spec_id}' not found"

    hierarchy = spec_data.get("hierarchy", {})
    spec_root = hierarchy.get("spec-root")
    if not spec_root:
        return None, "Invalid spec: missing spec-root"

    # Get phase children from spec-root
    phase_ids = spec_root.get("children", [])

    # Track results for each phase
    phase_results: List[Dict[str, Any]] = []
    spec_total_calculated = 0.0

    for phase_id in phase_ids:
        phase = hierarchy.get(phase_id)
        if not phase or phase.get("type") != "phase":
            continue

        phase_metadata = phase.get("metadata", {})
        previous_hours = phase_metadata.get("actual_hours")

        # Collect all descendants of this phase
        descendants = _collect_descendants(hierarchy, phase_id)

        # Sum actual_hours from task/subtask/verify nodes
        task_count = 0
        calculated_hours = 0.0

        for desc_id in descendants:
            desc_node = hierarchy.get(desc_id)
            if not desc_node:
                continue

            desc_type = desc_node.get("type")
            if desc_type in ("task", "subtask", "verify"):
                task_count += 1
                desc_metadata = desc_node.get("metadata", {})
                act = desc_metadata.get("actual_hours")
                if isinstance(act, (int, float)) and act >= 0:
                    calculated_hours += float(act)

        # Calculate delta
        prev_value = float(previous_hours) if isinstance(previous_hours, (int, float)) else 0.0
        delta = calculated_hours - prev_value

        phase_results.append(
            {
                "phase_id": phase_id,
                "title": phase.get("title", ""),
                "previous": previous_hours,
                "calculated": calculated_hours,
                "delta": delta,
                "task_count": task_count,
            }
        )

        # Update phase metadata (will be saved if not dry_run)
        if "metadata" not in phase:
            phase["metadata"] = {}
        phase["metadata"]["actual_hours"] = calculated_hours

        # Add to spec total
        spec_total_calculated += calculated_hours

    # Get spec-level previous value
    spec_metadata = spec_data.get("metadata", {})
    spec_previous = spec_metadata.get("actual_hours")
    spec_prev_value = float(spec_previous) if isinstance(spec_previous, (int, float)) else 0.0
    spec_delta = spec_total_calculated - spec_prev_value

    # Update spec metadata
    if "metadata" not in spec_data:
        spec_data["metadata"] = {}
    spec_data["metadata"]["actual_hours"] = spec_total_calculated

    # Build result
    result: Dict[str, Any] = {
        "spec_id": spec_id,
        "dry_run": dry_run,
        "spec_level": {
            "previous": spec_previous,
            "calculated": spec_total_calculated,
            "delta": spec_delta,
        },
        "phases": phase_results,
        "summary": {
            "total_phases": len(phase_results),
            "phases_changed": sum(1 for p in phase_results if p["delta"] != 0),
            "spec_changed": spec_delta != 0,
        },
    }

    if dry_run:
        result["message"] = "Dry run - changes not saved"
        return result, None

    # Save spec
    saved = save_spec(spec_id, spec_data, specs_dir)
    if not saved:
        return None, "Failed to save specification"

    return result, None
