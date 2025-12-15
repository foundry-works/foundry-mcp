"""
Spec modification operations using direct Python APIs.
Replaces subprocess calls to external CLI tools.
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .spec import load_spec, save_spec, update_node, find_specs_directory


def apply_modifications(
    spec_id: str,
    modifications: List[Dict[str, Any]],
    specs_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Apply a list of modifications to a spec.

    Args:
        spec_id: Specification ID to modify
        modifications: List of modification dictionaries with 'action', 'node_id', etc.
        specs_dir: Path to specs directory (auto-detected if not provided)
        dry_run: If True, don't save changes

    Returns:
        Tuple of (applied_count, skipped_count, changes_list)

    Raises:
        ValueError: If spec not found
        FileNotFoundError: If specs directory not found
    """
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if not specs_dir:
        raise FileNotFoundError("Could not find specs directory")

    spec_data = load_spec(spec_id, specs_dir)
    if not spec_data:
        raise ValueError(f"Specification '{spec_id}' not found")

    applied = 0
    skipped = 0
    changes: List[Dict[str, Any]] = []

    for mod in modifications:
        action = mod.get("action")
        node_id = mod.get("node_id")

        if action == "update_node":
            mod_changes = mod.get("changes", {})
            if update_node(spec_data, node_id, mod_changes):
                applied += 1
                changes.append({
                    "action": action,
                    "node_id": node_id,
                    "status": "applied",
                    "changes": mod_changes,
                })
            else:
                skipped += 1
                changes.append({
                    "action": action,
                    "node_id": node_id,
                    "status": "skipped",
                    "reason": "node not found",
                })

        elif action == "add_node":
            node_data = mod.get("data", {})
            parent_id = mod.get("parent")
            if _add_node(spec_data, node_id, node_data, parent_id):
                applied += 1
                changes.append({
                    "action": action,
                    "node_id": node_id,
                    "status": "applied",
                })
            else:
                skipped += 1
                changes.append({
                    "action": action,
                    "node_id": node_id,
                    "status": "skipped",
                    "reason": "failed to add node",
                })

        elif action == "remove_node":
            cascade = mod.get("cascade", False)
            if _remove_node(spec_data, node_id, cascade=cascade):
                applied += 1
                changes.append({
                    "action": action,
                    "node_id": node_id,
                    "status": "applied",
                })
            else:
                skipped += 1
                changes.append({
                    "action": action,
                    "node_id": node_id,
                    "status": "skipped",
                    "reason": "node not found",
                })

        else:
            skipped += 1
            changes.append({
                "action": action,
                "node_id": node_id,
                "status": "skipped",
                "reason": f"unknown action: {action}",
            })

    if not dry_run and applied > 0:
        save_spec(spec_id, spec_data, specs_dir, backup=True)

    return applied, skipped, changes


def load_modifications_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load modifications from a JSON file.

    Args:
        file_path: Path to the modifications JSON file

    Returns:
        List of modification dictionaries

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Modifications file not found: {file_path}")

    with open(path, 'r') as f:
        data = json.load(f)

    return data.get("modifications", [])


def _add_node(
    spec_data: Dict[str, Any],
    node_id: str,
    node_data: Dict[str, Any],
    parent_id: Optional[str] = None,
) -> bool:
    """
    Add a new node to the spec hierarchy.

    Args:
        spec_data: Spec data dictionary
        node_id: ID for the new node
        node_data: Node data to add
        parent_id: Optional parent node ID

    Returns:
        True if node was added, False otherwise
    """
    hierarchy = spec_data.get("hierarchy", {})

    # Don't overwrite existing nodes
    if node_id in hierarchy:
        return False

    # Add the node
    hierarchy[node_id] = node_data

    # Update parent's children list if parent specified
    if parent_id and parent_id in hierarchy:
        parent = hierarchy[parent_id]
        children = parent.get("children", [])
        if node_id not in children:
            children.append(node_id)
            parent["children"] = children

    return True


def _remove_node(
    spec_data: Dict[str, Any],
    node_id: str,
    cascade: bool = False,
) -> bool:
    """
    Remove a node from the spec hierarchy.

    Args:
        spec_data: Spec data dictionary
        node_id: ID of node to remove
        cascade: If True, also remove children recursively

    Returns:
        True if node was removed, False if not found
    """
    hierarchy = spec_data.get("hierarchy", {})

    if node_id not in hierarchy:
        return False

    node = hierarchy[node_id]

    # Remove children first if cascade
    if cascade:
        children = node.get("children", [])
        for child_id in children:
            _remove_node(spec_data, child_id, cascade=True)

    # Remove this node
    del hierarchy[node_id]

    # Remove from any parent's children list
    for other_id, other_node in hierarchy.items():
        children = other_node.get("children", [])
        if node_id in children:
            children.remove(node_id)
            other_node["children"] = children

    return True
