"""Shared helpers for authoring handler modules."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from foundry_mcp.config.server import ServerConfig
from foundry_mcp.core.observability import get_metrics
from foundry_mcp.core.responses import ErrorCode
from foundry_mcp.core.spec import list_assumptions, load_spec
from foundry_mcp.tools.unified.common import (
    build_request_id,
    make_metric_name,
    make_validation_error_fn,
    resolve_specs_dir,
)

logger = logging.getLogger(__name__)
_metrics = get_metrics()

_ACTION_SUMMARY = {
    "spec-create": "Scaffold a new SDD specification",
    "spec-template": "List/show/apply spec templates",
    "spec-update-frontmatter": "Update a top-level metadata field",
    "spec-find-replace": "Find and replace text across spec titles and descriptions",
    "spec-rollback": "Restore a spec from a backup timestamp",
    "phase-add": "Add a new phase under spec-root with verification scaffolding",
    "phase-add-bulk": "Add a phase with pre-defined tasks in a single atomic operation",
    "phase-template": "List/show/apply phase templates to add pre-configured phases",
    "phase-move": "Reorder a phase within spec-root children",
    "phase-update-metadata": "Update metadata fields of an existing phase",
    "phase-remove": "Remove an existing phase (and optionally dependents)",
    "assumption-add": "Append an assumption entry to spec metadata",
    "assumption-list": "List recorded assumptions for a spec",
    "revision-add": "Record a revision entry in the spec history",
    "intake-add": "Capture a new work idea in the notes intake queue",
    "intake-list": "List new intake items awaiting triage in FIFO order",
    "intake-dismiss": "Dismiss an intake item from the triage queue",
}


def _metric_name(action: str) -> str:
    return make_metric_name("authoring", action)


def _request_id() -> str:
    return build_request_id("authoring")


_validation_error = make_validation_error_fn("authoring")


def _resolve_specs_dir(
    config: ServerConfig, path: Optional[str]
) -> tuple[Optional[Path], Optional[dict]]:
    """Thin wrapper around the shared helper preserving the local call convention."""
    return resolve_specs_dir(config, path)


def _phase_exists(spec_id: str, specs_dir: Path, title: str) -> bool:
    try:
        spec_data = load_spec(spec_id, specs_dir)
    except Exception:  # pragma: no cover - defensive guard
        logger.exception(
            "Failed to inspect spec for duplicate phases", extra={"spec_id": spec_id}
        )
        return False

    if not spec_data:
        return False

    hierarchy = spec_data.get("hierarchy", {})
    if not isinstance(hierarchy, dict):
        return False

    normalized = title.strip().casefold()
    for node in hierarchy.values():
        if isinstance(node, dict) and node.get("type") == "phase":
            node_title = str(node.get("title", "")).strip().casefold()
            if node_title and node_title == normalized:
                return True
    return False


def _assumption_exists(spec_id: str, specs_dir: Path, text: str) -> bool:
    result, error = list_assumptions(spec_id=spec_id, specs_dir=specs_dir)
    if error or not result:
        return False

    normalized = text.strip().casefold()
    for entry in result.get("assumptions", []):
        entry_text = str(entry.get("text", "")).strip().casefold()
        if entry_text and entry_text == normalized:
            return True
    return False
