"""Adapt hierarchy-based spec data to the phases-array view used by the orchestrator.

Production specs store structure in a flat ``hierarchy`` dict (node-id → node-data),
while the autonomy orchestrator and spec-hash utilities expect a denormalized
``phases`` array.  This module bridges the two representations.

The conversion is **non-destructive** — the original ``hierarchy`` key is preserved
and a ``phases`` key is added alongside it.  If ``phases`` already exists and is
non-empty (e.g. in test fixtures), the spec is returned unchanged.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def load_spec_file(spec_path: Path) -> Dict[str, Any]:
    """Load a spec JSON file and ensure it has the ``phases`` view.

    This is the **single entry point** for loading spec files in the autonomy
    layer.  All code that needs parsed spec data with a ``phases`` array should
    call this instead of inlining ``json.loads`` + ``ensure_phases_view``.

    Raises:
        json.JSONDecodeError: If the file contains invalid JSON.
        OSError: If the file cannot be read.

    Returns:
        Parsed spec dict with ``phases`` guaranteed.
    """
    spec_data = json.loads(spec_path.read_text())
    return ensure_phases_view(spec_data)


def ensure_phases_view(spec_data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure *spec_data* contains a ``phases`` array suitable for the orchestrator.

    If the dict already carries a populated ``phases`` list (test fixtures,
    legacy format) it is returned as-is.  Otherwise the list is built from the
    ``hierarchy`` dict present in production spec files.

    Args:
        spec_data: Parsed spec JSON dictionary (modified **in place**).

    Returns:
        The same *spec_data* dict, now guaranteed to have a ``phases`` key.
    """
    if not spec_data:
        return spec_data

    existing = spec_data.get("phases")
    if isinstance(existing, list) and existing:
        return spec_data

    hierarchy = spec_data.get("hierarchy")
    if not isinstance(hierarchy, dict):
        return spec_data

    spec_root = hierarchy.get("spec-root")
    if not spec_root or not isinstance(spec_root, dict):
        return spec_data

    phases: List[Dict[str, Any]] = []
    for seq_idx, phase_id in enumerate(spec_root.get("children", [])):
        phase_node = hierarchy.get(phase_id)
        if not phase_node or phase_node.get("type") != "phase":
            continue

        tasks: List[Dict[str, Any]] = []
        _collect_leaf_tasks(hierarchy, phase_id, tasks)

        phases.append({
            "id": phase_id,
            "title": phase_node.get("title", ""),
            "sequence_index": seq_idx,
            "metadata": phase_node.get("metadata", {}),
            "tasks": tasks,
        })

    spec_data["phases"] = phases

    task_count = sum(len(p["tasks"]) for p in phases)
    logger.debug(
        "Built phases view from hierarchy: %d phase(s), %d task(s)",
        len(phases),
        task_count,
    )
    return spec_data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_leaf_tasks(
    hierarchy: Dict[str, Any],
    node_id: str,
    out: List[Dict[str, Any]],
) -> None:
    """Recursively collect leaf task/verify/subtask nodes under *node_id*.

    Only **leaf** nodes (those with no children) are emitted.  Parent tasks
    that have subtask children are skipped — their completion is handled by
    the hierarchy rollup logic when the MCP task handler processes completions.

    ``subtask`` nodes are mapped to ``type="task"`` so the orchestrator treats
    them identically to regular implementation tasks.
    """
    node = hierarchy.get(node_id)
    if not node:
        return

    children = node.get("children", [])
    node_type = node.get("type", "")

    if node_type in ("task", "verify", "subtask"):
        if not children:
            # Leaf node — add to output
            deps = node.get("dependencies", {})
            blocked_by = deps.get("blocked_by", []) if isinstance(deps, dict) else []

            out.append({
                "id": node_id,
                "title": node.get("title", ""),
                "type": "task" if node_type in ("task", "subtask") else "verify",
                "status": node.get("status", "pending"),
                "depends": list(blocked_by),
                "dependencies": list(blocked_by),
                "metadata": node.get("metadata", {}),
            })
        else:
            # Parent with children — recurse; don't emit the parent itself
            for child_id in children:
                _collect_leaf_tasks(hierarchy, child_id, out)

    elif node_type in ("phase", "group", "spec"):
        # Container — recurse into children
        for child_id in children:
            _collect_leaf_tasks(hierarchy, child_id, out)
