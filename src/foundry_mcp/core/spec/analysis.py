"""Spec analysis functions — completeness checks and duplicate detection.

Read-only operations that inspect spec data without modifying it.
Imports from ``io`` only (load, find); never calls ``save_spec``.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from foundry_mcp.core.spec.io import (
    find_spec_file,
    find_specs_directory,
    load_spec,
)

# Completeness check constants
_CC_WEIGHT_TITLES = 0.20
_CC_WEIGHT_DESCRIPTIONS = 0.30
_CC_WEIGHT_FILE_PATHS = 0.25
_CC_WEIGHT_ESTIMATES = 0.25


def check_spec_completeness(
    spec_id: str,
    *,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Check spec completeness and calculate a score (0-100).

    Evaluates spec quality by checking for:
    - Empty titles
    - Missing task descriptions
    - Missing file_path for implementation/refactoring tasks
    - Missing estimated_hours

    Args:
        spec_id: Specification ID to check.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "completeness_score": ..., ...}, None)
        On failure: (None, "error message")
    """
    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()
    if specs_dir is None:
        return None, "No specs directory found"

    # Load spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if not spec_path:
        return None, f"Specification '{spec_id}' not found"
    spec_data = load_spec(spec_id, specs_dir)
    if not spec_data:
        return None, f"Failed to load specification '{spec_id}'"

    hierarchy = spec_data.get("hierarchy", {})
    if not hierarchy:
        return {
            "spec_id": spec_id,
            "completeness_score": 100,
            "categories": {},
            "issues": [],
            "message": "No hierarchy nodes to check",
        }, None

    # Helper functions
    def _nonempty_string(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def _has_description(metadata: Dict[str, Any]) -> bool:
        if _nonempty_string(metadata.get("description")):
            return True
        details = metadata.get("details")
        if _nonempty_string(details):
            return True
        if isinstance(details, list):
            return any(_nonempty_string(item) for item in details)
        return False

    # Tracking
    issues: List[Dict[str, Any]] = []
    categories: Dict[str, Dict[str, Any]] = {
        "titles": {"complete": 0, "total": 0, "score": 0.0},
        "descriptions": {"complete": 0, "total": 0, "score": 0.0},
        "file_paths": {"complete": 0, "total": 0, "score": 0.0},
        "estimates": {"complete": 0, "total": 0, "score": 0.0},
    }

    # Check each node
    for node_id, node in hierarchy.items():
        if node_id == "spec-root":
            continue
        if not isinstance(node, dict):
            continue

        node_type = node.get("type", "")
        title = node.get("title", "")
        metadata = node.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        # Check title (all nodes)
        categories["titles"]["total"] += 1
        if _nonempty_string(title):
            categories["titles"]["complete"] += 1
        else:
            issues.append(
                {
                    "node_id": node_id,
                    "category": "titles",
                    "message": "Empty or missing title",
                }
            )

        # Check description (tasks and verify nodes only)
        if node_type in ("task", "verify"):
            categories["descriptions"]["total"] += 1
            if _has_description(metadata):
                categories["descriptions"]["complete"] += 1
            else:
                issues.append(
                    {
                        "node_id": node_id,
                        "category": "descriptions",
                        "message": "Missing description",
                    }
                )

            # Check file_path (implementation/refactoring tasks only)
            task_category = metadata.get("task_category", "")
            if task_category in ("implementation", "refactoring"):
                categories["file_paths"]["total"] += 1
                if _nonempty_string(metadata.get("file_path")):
                    categories["file_paths"]["complete"] += 1
                else:
                    issues.append(
                        {
                            "node_id": node_id,
                            "category": "file_paths",
                            "message": "Missing file_path for implementation task",
                        }
                    )

            # Check estimated_hours (tasks only)
            if node_type == "task":
                categories["estimates"]["total"] += 1
                est = metadata.get("estimated_hours")
                if isinstance(est, (int, float)) and est > 0:
                    categories["estimates"]["complete"] += 1
                else:
                    issues.append(
                        {
                            "node_id": node_id,
                            "category": "estimates",
                            "message": "Missing or invalid estimated_hours",
                        }
                    )

    # Calculate category scores
    for cat_data in categories.values():
        if cat_data["total"] > 0:
            cat_data["score"] = round(cat_data["complete"] / cat_data["total"], 2)
        else:
            cat_data["score"] = 1.0  # No items to check = complete

    # Calculate weighted completeness score
    weighted_score = 0.0
    total_weight = 0.0

    if categories["titles"]["total"] > 0:
        weighted_score += categories["titles"]["score"] * _CC_WEIGHT_TITLES
        total_weight += _CC_WEIGHT_TITLES

    if categories["descriptions"]["total"] > 0:
        weighted_score += categories["descriptions"]["score"] * _CC_WEIGHT_DESCRIPTIONS
        total_weight += _CC_WEIGHT_DESCRIPTIONS

    if categories["file_paths"]["total"] > 0:
        weighted_score += categories["file_paths"]["score"] * _CC_WEIGHT_FILE_PATHS
        total_weight += _CC_WEIGHT_FILE_PATHS

    if categories["estimates"]["total"] > 0:
        weighted_score += categories["estimates"]["score"] * _CC_WEIGHT_ESTIMATES
        total_weight += _CC_WEIGHT_ESTIMATES

    # Normalize score
    if total_weight > 0:
        completeness_score = int(round((weighted_score / total_weight) * 100))
    else:
        completeness_score = 100  # Nothing to check

    return {
        "spec_id": spec_id,
        "completeness_score": completeness_score,
        "categories": categories,
        "issues": issues,
        "issue_count": len(issues),
    }, None


# Duplicate detection constants
_DD_DEFAULT_THRESHOLD = 0.8
_DD_MAX_PAIRS = 100
_DD_VALID_SCOPES = {"titles", "descriptions", "both"}


def detect_duplicate_tasks(
    spec_id: str,
    *,
    scope: str = "titles",
    threshold: float = _DD_DEFAULT_THRESHOLD,
    max_pairs: int = _DD_MAX_PAIRS,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Detect duplicate or near-duplicate tasks in a spec.

    Uses text similarity to find tasks with similar titles or descriptions.

    Args:
        spec_id: Specification ID to check.
        scope: What to compare - "titles", "descriptions", or "both".
        threshold: Similarity threshold (0.0-1.0). Default 0.8.
        max_pairs: Maximum duplicate pairs to return. Default 100.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "duplicates": [...], ...}, None)
        On failure: (None, "error message")
    """
    from difflib import SequenceMatcher

    # Validate scope
    if scope not in _DD_VALID_SCOPES:
        return None, f"scope must be one of: {sorted(_DD_VALID_SCOPES)}"

    # Validate threshold
    if not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
        return None, "threshold must be a number between 0.0 and 1.0"

    # Validate max_pairs
    if not isinstance(max_pairs, int) or max_pairs <= 0:
        return None, "max_pairs must be a positive integer"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()
    if specs_dir is None:
        return None, "No specs directory found"

    # Load spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if not spec_path:
        return None, f"Specification '{spec_id}' not found"
    spec_data = load_spec(spec_id, specs_dir)
    if not spec_data:
        return None, f"Failed to load specification '{spec_id}'"

    hierarchy = spec_data.get("hierarchy", {})
    if not hierarchy:
        return {
            "spec_id": spec_id,
            "duplicates": [],
            "duplicate_count": 0,
            "scope": scope,
            "threshold": threshold,
            "message": "No hierarchy nodes to check",
        }, None

    # Collect tasks/verify nodes with their text
    nodes: List[Dict[str, Any]] = []
    for node_id, node in hierarchy.items():
        if node_id == "spec-root":
            continue
        if not isinstance(node, dict):
            continue
        node_type = node.get("type", "")
        if node_type not in ("task", "verify"):
            continue

        title = node.get("title", "") or ""
        metadata = node.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        description = metadata.get("description", "") or ""

        nodes.append(
            {
                "id": node_id,
                "title": title.strip().lower(),
                "description": description.strip().lower(),
            }
        )

    # Compare pairs
    duplicates: List[Dict[str, Any]] = []
    truncated = False
    total_compared = 0

    def similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    for i, node_a in enumerate(nodes):
        if len(duplicates) >= max_pairs:
            truncated = True
            break
        for node_b in nodes[i + 1 :]:
            total_compared += 1
            if len(duplicates) >= max_pairs:
                truncated = True
                break

            # Calculate similarity based on scope
            if scope == "titles":
                sim = similarity(node_a["title"], node_b["title"])
            elif scope == "descriptions":
                sim = similarity(node_a["description"], node_b["description"])
            else:  # both
                title_sim = similarity(node_a["title"], node_b["title"])
                desc_sim = similarity(node_a["description"], node_b["description"])
                sim = max(title_sim, desc_sim)

            if sim >= threshold:
                duplicates.append(
                    {
                        "node_a": node_a["id"],
                        "node_b": node_b["id"],
                        "similarity": round(sim, 2),
                        "scope": scope,
                    }
                )

    result: Dict[str, Any] = {
        "spec_id": spec_id,
        "duplicates": duplicates,
        "duplicate_count": len(duplicates),
        "scope": scope,
        "threshold": threshold,
        "nodes_checked": len(nodes),
        "pairs_compared": total_compared,
    }

    if truncated:
        result["truncated"] = True
        result["warnings"] = [f"Results limited to {max_pairs} pairs"]

    return result, None
