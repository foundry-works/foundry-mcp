"""Shared helpers for task handler modules."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.autonomy.memory import AutonomyStorage
from foundry_mcp.core.observability import get_metrics
from foundry_mcp.core.pagination import (
    CursorError,
    decode_cursor,
    encode_cursor,
    normalize_page_size,
)
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
)
from foundry_mcp.core.spec import load_spec
from foundry_mcp.tools.unified.common import (
    build_request_id,
    make_metric_name,
    make_validation_error_fn,
    resolve_specs_dir,
)

logger = logging.getLogger(__name__)
_metrics = get_metrics()

_TASK_DEFAULT_PAGE_SIZE = 25
_TASK_MAX_PAGE_SIZE = 100
_TASK_WARNING_THRESHOLD = 75
_ALLOWED_STATUS = {"pending", "in_progress", "completed", "blocked"}


def _request_id() -> str:
    return build_request_id("task")


def _metric(action: str) -> str:
    return make_metric_name("unified_tools.task", action)


_validation_error = make_validation_error_fn("task", default_code=ErrorCode.MISSING_REQUIRED)


def _resolve_specs_dir(
    config: ServerConfig, workspace: Optional[str]
) -> tuple[Optional[Path], Optional[dict]]:
    """Thin wrapper around the shared helper preserving the local call convention."""
    return resolve_specs_dir(config, workspace)


def _load_spec_data(
    spec_id: str, specs_dir: Optional[Path], request_id: str
) -> Tuple[Optional[Dict[str, Any]], Optional[dict]]:
    if specs_dir is None:
        return None, asdict(
            error_response(
                "No specs directory found. Use --specs-dir or set SDD_SPECS_DIR.",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Set SDD_SPECS_DIR or invoke with --specs-dir",
                request_id=request_id,
            )
        )

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, asdict(
            error_response(
                f"Spec not found: {spec_id}",
                error_code=ErrorCode.SPEC_NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation='Verify the spec ID via spec(action="list")',
                request_id=request_id,
            )
        )
    return spec_data, None


def _attach_meta(
    response: dict,
    *,
    request_id: str,
    duration_ms: Optional[float] = None,
    warnings: Optional[List[str]] = None,
) -> dict:
    meta = response.setdefault("meta", {"version": "response-v2"})
    meta["request_id"] = request_id
    if warnings:
        existing = list(meta.get("warnings") or [])
        existing.extend(warnings)
        meta["warnings"] = existing
    if duration_ms is not None:
        telemetry = dict(meta.get("telemetry") or {})
        telemetry["duration_ms"] = round(duration_ms, 2)
        meta["telemetry"] = telemetry
    return response


def _filter_hierarchy(
    hierarchy: Dict[str, Any],
    max_depth: int,
    include_metadata: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    for node_id, node_data in hierarchy.items():
        node_depth = node_id.count("-") if node_id != "spec-root" else 0
        if max_depth > 0 and node_depth > max_depth:
            continue

        filtered_node: Dict[str, Any] = {
            "type": node_data.get("type"),
            "title": node_data.get("title"),
            "status": node_data.get("status"),
        }
        if "children" in node_data:
            filtered_node["children"] = node_data["children"]
        if "parent" in node_data:
            filtered_node["parent"] = node_data["parent"]

        if include_metadata:
            if "metadata" in node_data:
                filtered_node["metadata"] = node_data["metadata"]
            if "dependencies" in node_data:
                filtered_node["dependencies"] = node_data["dependencies"]

        result[node_id] = filtered_node

    return result


def _pagination_warnings(total_count: int, has_more: bool) -> List[str]:
    warnings: List[str] = []
    if total_count > _TASK_WARNING_THRESHOLD:
        warnings.append(
            f"{total_count} results returned; consider using pagination to limit payload size."
        )
    if has_more:
        warnings.append("Additional results available. Follow the cursor to continue.")
    return warnings


def _match_nodes_for_batch(
    hierarchy: Dict[str, Any],
    *,
    phase_id: Optional[str] = None,
    pattern: Optional[str] = None,
    node_type: Optional[str] = None,
) -> List[str]:
    """Filter nodes by phase_id, regex pattern on title/id, and/or node_type.

    All provided filters are combined with AND logic.
    Returns list of matching node IDs.
    """
    matched: List[str] = []
    compiled_pattern = None
    if pattern:
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return []  # Invalid regex returns empty

    for node_id, node_data in hierarchy.items():
        if node_id == "spec-root":
            continue

        # Filter by node_type if specified
        if node_type and node_data.get("type") != node_type:
            continue

        # Filter by phase_id if specified (must be under that phase)
        if phase_id:
            node_parent = node_data.get("parent")
            # Direct children of the phase
            if node_parent != phase_id:
                # Check if it's a nested child (e.g., subtask under task under phase)
                parent_node = hierarchy.get(node_parent, {})
                if parent_node.get("parent") != phase_id:
                    continue

        # Filter by regex pattern on title or node_id
        if compiled_pattern:
            title = node_data.get("title", "")
            if not (compiled_pattern.search(title) or compiled_pattern.search(node_id)):
                continue

        matched.append(node_id)

    return sorted(matched)


_VALID_NODE_TYPES = {"task", "verify", "phase", "subtask"}


# ---------------------------------------------------------------------------
# Autonomy write-lock enforcement helper
# ---------------------------------------------------------------------------

# Import write-lock helpers for autonomy session protection.
try:
    from foundry_mcp.core.autonomy.write_lock import (
        check_autonomy_write_lock as _check_autonomy_write_lock_impl,
        WriteLockStatus as _WriteLockStatus,
    )
    _WRITE_LOCK_AVAILABLE = True
except ImportError:
    _check_autonomy_write_lock_impl = None  # type: ignore[misc,assignment]
    _WriteLockStatus = None  # type: ignore[misc,assignment]
    _WRITE_LOCK_AVAILABLE = False


def _check_autonomy_write_lock(
    spec_id: str,
    workspace: Optional[str],
    bypass_autonomy_lock: bool,
    bypass_reason: Optional[str],
    request_id: str,
) -> Optional[dict]:
    """Check autonomy write-lock and return error response if blocked.

    Args:
        spec_id: The spec ID being modified.
        workspace: Optional workspace path.
        bypass_autonomy_lock: If True, bypass the lock (requires bypass_reason).
        bypass_reason: Reason for bypassing the lock.
        request_id: Request ID for error response.

    Returns:
        None if operation is allowed, error response dict if blocked.
    """
    if not _WRITE_LOCK_AVAILABLE or _check_autonomy_write_lock_impl is None:
        return None

    result = _check_autonomy_write_lock_impl(
        spec_id=spec_id,
        workspace=workspace,
        bypass_flag=bypass_autonomy_lock,
        bypass_reason=bypass_reason,
    )

    if result.status == _WriteLockStatus.LOCKED:
        return asdict(error_response(
            result.message or "Autonomy write lock is active for this spec",
            error_code=ErrorCode.AUTONOMY_WRITE_LOCK_ACTIVE,
            error_type=ErrorType.CONFLICT,
            request_id=request_id,
            details={
                "session_id": result.session_id,
                "session_status": result.session_status,
                "hint": "Use bypass_autonomy_lock=true with bypass_reason to override",
            },
        ))

    # ALLOWED or BYPASSED — operation can proceed
    return None


def _get_storage(config: ServerConfig, workspace: Optional[str] = None) -> AutonomyStorage:
    """Get AutonomyStorage instance for session operations."""
    ws_path = Path(workspace) if workspace else Path.cwd()
    return AutonomyStorage(workspace_path=ws_path)


def _session_not_found_response(
    action: str, request_id: str, spec_id: Optional[str] = None
) -> dict:
    """Return session not found error response."""
    return asdict(error_response(
        "No active session found",
        error_code=ErrorCode.NO_ACTIVE_SESSION,
        error_type=ErrorType.NOT_FOUND,
        request_id=request_id,
        details={
            "action": action,
            "spec_id": spec_id,
            "hint": "Start a session with session-start action",
        },
    ))


def _resolve_session(
    storage: AutonomyStorage,
    action: str,
    request_id: str,
    session_id: Optional[str] = None,
    spec_id: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[dict]]:
    """Resolve a session by session_id, spec_id, or workspace scan.

    Per ADR: when session_id is omitted, find the single non-terminal session.
    If zero → NO_ACTIVE_SESSION. If multiple → AMBIGUOUS_ACTIVE_SESSION.

    Args:
        storage: AutonomyStorage instance
        action: Action name for error messages
        request_id: Request ID for error responses
        session_id: Direct session ID (highest priority)
        spec_id: Spec ID to look up active session pointer

    Returns:
        Tuple of (session, error_response). One will be None.
    """
    from foundry_mcp.core.autonomy.memory import ActiveSessionLookupResult

    # Priority 1: Direct session_id
    if session_id:
        session = storage.load(session_id)
        if not session:
            return None, _session_not_found_response(action, request_id, spec_id)
        return session, None

    # Priority 2: Spec ID pointer lookup
    if spec_id:
        active_session_id = storage.get_active_session(spec_id)
        if not active_session_id:
            return None, _session_not_found_response(action, request_id, spec_id)
        session = storage.load(active_session_id)
        if not session:
            return None, _session_not_found_response(action, request_id, spec_id)
        return session, None

    # Priority 3: Scan all non-terminal sessions
    result, found_id = storage.lookup_active_session()

    if result == ActiveSessionLookupResult.NOT_FOUND:
        return None, _session_not_found_response(action, request_id)

    if result == ActiveSessionLookupResult.AMBIGUOUS:
        return None, asdict(error_response(
            "Multiple active sessions found. Provide spec_id or session_id to disambiguate.",
            error_code=ErrorCode.AMBIGUOUS_ACTIVE_SESSION,
            error_type=ErrorType.VALIDATION,
            request_id=request_id,
            details={
                "action": action,
                "hint": "Provide spec_id or session_id parameter",
            },
        ))

    # FOUND
    if found_id is None:
        return None, _session_not_found_response(action, request_id)
    session = storage.load(found_id)
    if not session:
        return None, _session_not_found_response(action, request_id)
    return session, None
