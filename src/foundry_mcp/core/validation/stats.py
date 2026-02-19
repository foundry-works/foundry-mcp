"""Statistics calculation for SDD spec files."""

from pathlib import Path
from typing import Any, Dict, Optional

from foundry_mcp.core.validation.constants import STATUS_FIELDS
from foundry_mcp.core.validation.models import SpecStats


def calculate_stats(spec_data: Dict[str, Any], file_path: Optional[str] = None) -> SpecStats:
    """
    Calculate statistics for a spec file.

    Args:
        spec_data: Parsed JSON spec data
        file_path: Optional path to spec file for size calculation

    Returns:
        SpecStats with calculated metrics
    """
    hierarchy = spec_data.get("hierarchy", {}) or {}

    totals = {
        "nodes": len(hierarchy),
        "tasks": 0,
        "phases": 0,
        "verifications": 0,
    }

    status_counts = {status: 0 for status in STATUS_FIELDS}
    max_depth = 0

    def traverse(node_id: str, depth: int) -> None:
        nonlocal max_depth
        node = hierarchy.get(node_id, {})
        node_type = node.get("type")

        max_depth = max(max_depth, depth)

        if node_type in {"task", "subtask"}:
            totals["tasks"] += 1
            status = node.get("status", "").lower().replace(" ", "_").replace("-", "_")
            if status in status_counts:
                status_counts[status] += 1
        elif node_type == "phase":
            totals["phases"] += 1
        elif node_type == "verify":
            totals["verifications"] += 1

        for child_id in node.get("children", []) or []:
            if child_id in hierarchy:
                traverse(child_id, depth + 1)

    if "spec-root" in hierarchy:
        traverse("spec-root", 0)

    total_tasks = totals["tasks"]
    phase_count = totals["phases"] or 1
    avg_tasks_per_phase = round(total_tasks / phase_count, 2)

    root = hierarchy.get("spec-root", {})
    root_total_tasks = root.get("total_tasks", total_tasks)
    root_completed = root.get("completed_tasks", 0)

    verification_count = totals["verifications"]
    verification_coverage = (verification_count / total_tasks) if total_tasks else 0.0
    progress = (root_completed / root_total_tasks) if root_total_tasks else 0.0

    file_size = 0.0
    if file_path:
        try:
            file_size = Path(file_path).stat().st_size / 1024
        except OSError:
            file_size = 0.0

    return SpecStats(
        spec_id=spec_data.get("spec_id", "unknown"),
        title=spec_data.get("title", ""),
        version=spec_data.get("version", ""),
        status=root.get("status", "unknown"),
        totals=totals,
        status_counts=status_counts,
        max_depth=max_depth,
        avg_tasks_per_phase=avg_tasks_per_phase,
        verification_coverage=verification_coverage,
        progress=progress,
        file_size_kb=file_size,
    )


def _recalculate_counts(spec_data: Dict[str, Any]) -> None:
    """Recalculate task counts for all nodes in hierarchy."""
    hierarchy = spec_data.get("hierarchy", {})
    if not hierarchy:
        return

    # Process bottom-up: leaves first, then parents
    def calculate_node(node_id: str) -> tuple:
        """Return (total_tasks, completed_tasks) for a node."""
        node = hierarchy.get(node_id, {})
        children = node.get("children", [])
        node_type = node.get("type", "")
        status = node.get("status", "")

        if not children:
            # Leaf node
            if node_type in {"task", "subtask", "verify"}:
                total = 1
                completed = 1 if status == "completed" else 0
            else:
                total = 0
                completed = 0
        else:
            # Parent node: sum children
            total = 0
            completed = 0
            for child_id in children:
                if child_id in hierarchy:
                    child_total, child_completed = calculate_node(child_id)
                    total += child_total
                    completed += child_completed

        node["total_tasks"] = total
        node["completed_tasks"] = completed
        return total, completed

    if "spec-root" in hierarchy:
        calculate_node("spec-root")
