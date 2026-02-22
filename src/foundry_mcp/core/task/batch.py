"""
Batch operations for SDD task workflows.

Mutation functions live in ``mutations.py``; query functions in ``queries.py``;
shared constants in ``_helpers.py``.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from foundry_mcp.core.spec import (
    find_spec_file,
    find_specs_directory,
    load_spec,
    save_spec,
)

# Valid statuses for batch filtering
BATCH_ALLOWED_STATUSES = {"pending", "in_progress", "completed", "blocked"}

# Safety constraints for batch operations
MAX_PATTERN_LENGTH = 256
DEFAULT_MAX_MATCHES = 100


def _match_tasks_for_batch(
    hierarchy: Dict[str, Any],
    *,
    status_filter: Optional[str] = None,
    parent_filter: Optional[str] = None,
    pattern: Optional[str] = None,
) -> List[str]:
    """Find tasks matching filter criteria (AND logic). Returns sorted task IDs."""
    compiled_pattern = None
    if pattern:
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return []

    matched = []
    target_types = {"task", "subtask", "verify"}

    valid_descendants: Optional[set] = None
    if parent_filter:
        parent_node = hierarchy.get(parent_filter)
        if not parent_node:
            return []
        valid_descendants = set()
        to_visit = list(parent_node.get("children", []))
        while to_visit:
            child_id = to_visit.pop()
            if child_id in valid_descendants:
                continue
            valid_descendants.add(child_id)
            child_node = hierarchy.get(child_id)
            if child_node:
                to_visit.extend(child_node.get("children", []))

    for node_id, node_data in hierarchy.items():
        if node_data.get("type") not in target_types:
            continue
        if status_filter and node_data.get("status") != status_filter:
            continue
        if valid_descendants is not None and node_id not in valid_descendants:
            continue
        if compiled_pattern:
            title = node_data.get("title", "")
            if not (compiled_pattern.search(title) or compiled_pattern.search(node_id)):
                continue
        matched.append(node_id)

    return sorted(matched)


def batch_update_tasks(
    spec_id: str,
    *,
    status_filter: Optional[str] = None,
    parent_filter: Optional[str] = None,
    pattern: Optional[str] = None,
    description: Optional[str] = None,
    file_path: Optional[str] = None,
    estimated_hours: Optional[float] = None,
    category: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
    owners: Optional[List[str]] = None,
    custom_metadata: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    max_matches: int = DEFAULT_MAX_MATCHES,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Batch update metadata across tasks matching filters (AND logic)."""
    # Validate filters
    if not any([status_filter, parent_filter, pattern]):
        return None, "At least one filter must be provided: status_filter, parent_filter, or pattern"
    if status_filter and status_filter not in BATCH_ALLOWED_STATUSES:
        return None, f"Invalid status_filter '{status_filter}'. Must be one of: {sorted(BATCH_ALLOWED_STATUSES)}"
    if pattern:
        if not isinstance(pattern, str) or not pattern.strip():
            return None, "pattern must be a non-empty string"
        pattern = pattern.strip()
        if len(pattern) > MAX_PATTERN_LENGTH:
            return None, f"pattern exceeds maximum length of {MAX_PATTERN_LENGTH} characters"
        try:
            re.compile(pattern)
        except re.error as e:
            return None, f"Invalid regex pattern: {e}"
    if parent_filter:
        if not isinstance(parent_filter, str) or not parent_filter.strip():
            return None, "parent_filter must be a non-empty string"
        parent_filter = parent_filter.strip()

    # Collect metadata updates
    metadata_updates: Dict[str, Any] = {}
    if description is not None:
        metadata_updates["description"] = description.strip() if description else None
    if file_path is not None:
        metadata_updates["file_path"] = file_path.strip() if file_path else None
    if estimated_hours is not None:
        if not isinstance(estimated_hours, (int, float)) or estimated_hours < 0:
            return None, "estimated_hours must be a non-negative number"
        metadata_updates["estimated_hours"] = float(estimated_hours)
    if category is not None:
        metadata_updates["category"] = category.strip() if category else None
    if labels is not None:
        if not isinstance(labels, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in labels.items()
        ):
            return None, "labels must be a dict with string keys and values"
        metadata_updates["labels"] = labels
    if owners is not None:
        if not isinstance(owners, list) or not all(isinstance(o, str) for o in owners):
            return None, "owners must be a list of strings"
        metadata_updates["owners"] = owners
    if custom_metadata:
        if not isinstance(custom_metadata, dict):
            return None, "custom_metadata must be a dict"
        for key, value in custom_metadata.items():
            if key not in metadata_updates:
                metadata_updates[key] = value

    if not metadata_updates:
        return None, "At least one metadata field must be provided"
    if max_matches <= 0:
        return None, "max_matches must be a positive integer"

    # Load spec
    if specs_dir is None:
        specs_dir = find_specs_directory()
    if specs_dir is None:
        return None, "No specs directory found"
    spec_path = find_spec_file(spec_id, specs_dir)
    if not spec_path:
        return None, f"Specification '{spec_id}' not found"
    spec_data = load_spec(spec_id, specs_dir)
    if not spec_data:
        return None, f"Failed to load specification '{spec_id}'"

    hierarchy = spec_data.get("hierarchy", {})
    if parent_filter and parent_filter not in hierarchy:
        return None, f"Parent '{parent_filter}' not found in specification"

    matched_ids = _match_tasks_for_batch(
        hierarchy, status_filter=status_filter, parent_filter=parent_filter, pattern=pattern
    )
    warnings: List[str] = []
    skipped_ids = []
    if len(matched_ids) > max_matches:
        warnings.append(f"Found {len(matched_ids)} matches, limiting to {max_matches}")
        skipped_ids = matched_ids[max_matches:]
        matched_ids = matched_ids[:max_matches]

    if not matched_ids:
        return {
            "spec_id": spec_id,
            "matched_count": 0,
            "updated_count": 0,
            "skipped_count": len(skipped_ids),
            "nodes": [],
            "filters": {"status_filter": status_filter, "parent_filter": parent_filter, "pattern": pattern},
            "metadata_applied": metadata_updates,
            "dry_run": dry_run,
            "message": "No tasks matched",
        }, None

    # Capture originals and build result
    original_metadata: Dict[str, Dict[str, Any]] = {}
    updated_nodes: List[Dict[str, Any]] = []
    for node_id in matched_ids:
        node = hierarchy.get(node_id, {})
        existing_meta = node.get("metadata", {}) or {}
        original_metadata[node_id] = {k: existing_meta.get(k) for k in metadata_updates}
        diff = {
            k: {"old": original_metadata[node_id].get(k), "new": v}
            for k, v in metadata_updates.items()
            if original_metadata[node_id].get(k) != v
        }
        updated_nodes.append(
            {
                "node_id": node_id,
                "title": node.get("title", ""),
                "type": node.get("type", ""),
                "status": node.get("status", ""),
                "fields_updated": list(metadata_updates.keys()),
                "diff": diff,
            }
            if diff
            else {
                "node_id": node_id,
                "title": node.get("title", ""),
                "type": node.get("type", ""),
                "status": node.get("status", ""),
                "fields_updated": list(metadata_updates.keys()),
            }
        )
        if not dry_run:
            if "metadata" not in node:
                node["metadata"] = {}
            node["metadata"].update(metadata_updates)

    if not dry_run:
        if not save_spec(spec_id, spec_data, specs_dir):
            for nid, orig in original_metadata.items():
                n = hierarchy.get(nid, {})
                if "metadata" in n:
                    for k, v in orig.items():
                        if v is None:
                            n["metadata"].pop(k, None)
                        else:
                            n["metadata"][k] = v
            return None, "Failed to save; changes rolled back"

    if len(matched_ids) > 50:
        warnings.append(f"Updated {len(matched_ids)} tasks")

    result = {
        "spec_id": spec_id,
        "matched_count": len(matched_ids),
        "updated_count": len(matched_ids) if not dry_run else 0,
        "skipped_count": len(skipped_ids),
        "nodes": updated_nodes,
        "filters": {"status_filter": status_filter, "parent_filter": parent_filter, "pattern": pattern},
        "metadata_applied": metadata_updates,
        "dry_run": dry_run,
    }
    if warnings:
        result["warnings"] = warnings
    if skipped_ids:
        result["skipped_tasks"] = skipped_ids
    return result, None
