"""Shared helpers for task handler modules."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from foundry_mcp.config.server import ServerConfig
from foundry_mcp.core.autonomy.memory import AutonomyStorage
from foundry_mcp.core.autonomy.models.signals import (
    derive_loop_signal,
    derive_recommended_actions,
)
from foundry_mcp.core.autonomy.models.state import AutonomousSessionState
from foundry_mcp.core.autonomy.orchestrator import ERROR_SESSION_UNRECOVERABLE
from foundry_mcp.core.observability import get_metrics
from foundry_mcp.core.responses.builders import (
    error_response,
)
from foundry_mcp.core.responses.types import (
    ErrorCode,
    ErrorType,
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


def _validate_context_usage_pct(value: Optional[int], action: str, request_id: str) -> Optional[dict]:
    """Validate context_usage_pct bounds (0-100 integer).

    Returns an error response dict if invalid, None if valid or absent.
    """
    if value is not None:
        if not isinstance(value, int) or not (0 <= value <= 100):
            return _validation_error(
                action=action,
                field="context_usage_pct",
                message="context_usage_pct must be an integer between 0 and 100",
                request_id=request_id,
            )
    return None


# Regex patterns for ID validation
# Session IDs: ULID format (26 chars, Crockford base32) or safe alphanumeric with hyphens
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
# Spec IDs: alphanumeric with hyphens, underscores, dots (for file-safe naming)
_SPEC_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _validate_session_id(value: Optional[str], action: str, request_id: str) -> Optional[dict]:
    """Validate session_id format.

    Returns an error response dict if invalid, None if valid or absent.
    """
    if value is not None:
        if not isinstance(value, str) or not _SESSION_ID_PATTERN.match(value):
            return _validation_error(
                action=action,
                field="session_id",
                message=(
                    "session_id must be 1-128 characters, starting with alphanumeric, "
                    "containing only alphanumeric, hyphens, and underscores"
                ),
                request_id=request_id,
            )
    return None


def _validate_spec_id(value: Optional[str], action: str, request_id: str) -> Optional[dict]:
    """Validate spec_id format.

    Returns an error response dict if invalid, None if valid or absent.
    """
    if value is not None:
        if not isinstance(value, str) or not _SPEC_ID_PATTERN.match(value):
            return _validation_error(
                action=action,
                field="spec_id",
                message=(
                    "spec_id must be 1-128 characters, starting with alphanumeric, "
                    "containing only alphanumeric, hyphens, underscores, and dots"
                ),
                request_id=request_id,
            )
    return None


_SESSION_COMMAND_TO_ACTION: Dict[str, str] = {
    "start": "session-start",
    "status": "session-status",
    "pause": "session-pause",
    "resume": "session-resume",
    "rebase": "session-rebase",
    "end": "session-end",
    "list": "session-list",
    "reset": "session-reset",
}

_SESSION_STEP_COMMAND_TO_ACTION: Dict[str, str] = {
    "next": "session-step-next",
    "report": "session-step-report",
    "replay": "session-step-replay",
    "heartbeat": "session-step-heartbeat",
}

_LEGACY_ACTION_DEPRECATIONS: Dict[str, Dict[str, str]] = {
    "session-start": {
        "action": "session-start",
        "replacement": 'task(action="session", command="start")',
        "removal_target": "2026-05-16_or_2_minor_releases",
    },
    "session-status": {
        "action": "session-status",
        "replacement": 'task(action="session", command="status")',
        "removal_target": "2026-05-16_or_2_minor_releases",
    },
    "session-pause": {
        "action": "session-pause",
        "replacement": 'task(action="session", command="pause")',
        "removal_target": "2026-05-16_or_2_minor_releases",
    },
    "session-resume": {
        "action": "session-resume",
        "replacement": 'task(action="session", command="resume")',
        "removal_target": "2026-05-16_or_2_minor_releases",
    },
    "session-rebase": {
        "action": "session-rebase",
        "replacement": 'task(action="session", command="rebase")',
        "removal_target": "2026-05-16_or_2_minor_releases",
    },
    "session-end": {
        "action": "session-end",
        "replacement": 'task(action="session", command="end")',
        "removal_target": "2026-05-16_or_2_minor_releases",
    },
    "session-list": {
        "action": "session-list",
        "replacement": 'task(action="session", command="list")',
        "removal_target": "2026-05-16_or_2_minor_releases",
    },
    "session-reset": {
        "action": "session-reset",
        "replacement": 'task(action="session", command="reset")',
        "removal_target": "2026-05-16_or_2_minor_releases",
    },
    "session-heartbeat": {
        "action": "session-heartbeat",
        "replacement": 'task(action="session-step", command="heartbeat")',
        "removal_target": "2026-05-16_or_2_minor_releases",
    },
    "session-step-next": {
        "action": "session-step-next",
        "replacement": 'task(action="session-step", command="next")',
        "removal_target": "2026-05-16_or_2_minor_releases",
    },
    "session-step-report": {
        "action": "session-step-report",
        "replacement": 'task(action="session-step", command="report")',
        "removal_target": "2026-05-16_or_2_minor_releases",
    },
    "session-step-replay": {
        "action": "session-step-replay",
        "replacement": 'task(action="session-step", command="replay")',
        "removal_target": "2026-05-16_or_2_minor_releases",
    },
    "session-step-heartbeat": {
        "action": "session-step-heartbeat",
        "replacement": 'task(action="session-step", command="heartbeat")',
        "removal_target": "2026-05-16_or_2_minor_releases",
    },
}


def _check_deprecation_expired(
    deprecation: Dict[str, str],
    request_id: str,
) -> Optional[dict]:
    """Return a hard error if the deprecation removal target date has passed.

    Parses ``removal_target`` by splitting on ``"_or_"`` to extract the ISO
    date prefix.  If ``date.today() >= target_date``, returns an error
    response naming the replacement.

    Escape hatch: ``FOUNDRY_MCP_ALLOW_DEPRECATED_ACTIONS=true`` bypasses
    enforcement.  Unparseable targets fail-open (log warning, skip enforcement).
    """
    if os.environ.get("FOUNDRY_MCP_ALLOW_DEPRECATED_ACTIONS", "").lower() == "true":
        return None

    removal_target = deprecation.get("removal_target", "")
    if not removal_target:
        return None

    date_str = removal_target.split("_or_")[0].strip()
    try:
        target_date = date.fromisoformat(date_str)
    except ValueError:
        logger.warning(
            "Unparseable deprecation removal_target '%s'; skipping enforcement",
            removal_target,
        )
        return None

    if date.today() >= target_date:
        replacement = deprecation.get("replacement", "unknown")
        action = deprecation.get("action", "unknown")
        return _validation_error(
            action=action,
            field="action",
            message=(f"Action '{action}' was removed on {date_str}. Use {replacement} instead."),
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    return None


_REASON_DETAIL_MAX_LENGTH = 2000


def _validate_reason_detail(
    reason_detail: Optional[str],
    action: str,
    request_id: str,
) -> Optional[dict]:
    """Validate reason_detail length. Returns error response if too long, None if valid."""
    if reason_detail is not None and len(reason_detail) > _REASON_DETAIL_MAX_LENGTH:
        return _validation_error(
            action=action,
            field="reason_detail",
            message=f"reason_detail exceeds maximum length of {_REASON_DETAIL_MAX_LENGTH} characters ({len(reason_detail)} provided)",
            request_id=request_id,
        )
    return None


def _normalize_task_action_shape(
    *,
    action: str,
    payload: Dict[str, Any],
    request_id: str,
) -> tuple[str, Dict[str, Any], Optional[Dict[str, str]], Optional[dict]]:
    """Normalize canonical session action shapes to runtime actions.

    Returns:
        Tuple of (normalized_action, normalized_payload, deprecation_metadata, error_response)
    """
    normalized_action = (action or "").strip().lower()
    normalized_payload = dict(payload)

    if normalized_action == "session":
        command = normalized_payload.get("command")
        if not isinstance(command, str) or not command.strip():
            return (
                normalized_action,
                normalized_payload,
                None,
                _validation_error(
                    action="session",
                    field="command",
                    message="command is required (start|status|pause|resume|rebase|end|list|reset)",
                    request_id=request_id,
                    code=ErrorCode.MISSING_REQUIRED,
                ),
            )
        normalized_command = command.strip().lower()
        mapped_action = _SESSION_COMMAND_TO_ACTION.get(normalized_command)
        if mapped_action is None:
            return (
                normalized_action,
                normalized_payload,
                None,
                _validation_error(
                    action="session",
                    field="command",
                    message=(
                        f"unsupported command '{normalized_command}'. "
                        "Expected one of: start, status, pause, resume, rebase, end, list, reset"
                    ),
                    request_id=request_id,
                    code=ErrorCode.INVALID_FORMAT,
                ),
            )
        normalized_payload["command"] = normalized_command
        return mapped_action, normalized_payload, None, None

    if normalized_action == "session-step":
        command = normalized_payload.get("command")
        if not isinstance(command, str) or not command.strip():
            return (
                normalized_action,
                normalized_payload,
                None,
                _validation_error(
                    action="session-step",
                    field="command",
                    message="command is required (next|report|replay|heartbeat)",
                    request_id=request_id,
                    code=ErrorCode.MISSING_REQUIRED,
                ),
            )
        normalized_command = command.strip().lower()
        mapped_action = _SESSION_STEP_COMMAND_TO_ACTION.get(normalized_command)
        if mapped_action is None:
            return (
                normalized_action,
                normalized_payload,
                None,
                _validation_error(
                    action="session-step",
                    field="command",
                    message=(
                        f"unsupported command '{normalized_command}'. Expected one of: next, report, replay, heartbeat"
                    ),
                    request_id=request_id,
                    code=ErrorCode.INVALID_FORMAT,
                ),
            )
        normalized_payload["command"] = normalized_command
        return mapped_action, normalized_payload, None, None

    deprecation = _LEGACY_ACTION_DEPRECATIONS.get(normalized_action)
    if deprecation:
        expired_err = _check_deprecation_expired(deprecation, request_id)
        if expired_err:
            return normalized_action, normalized_payload, deprecation, expired_err
    return normalized_action, normalized_payload, deprecation, None


def _attach_deprecation_metadata(
    response: dict,
    deprecation: Optional[Dict[str, str]],
) -> dict:
    """Attach machine-readable deprecation metadata to response envelope."""
    if not deprecation:
        return response
    meta = response.setdefault("meta", {"version": "response-v2"})
    meta["deprecated"] = dict(deprecation)
    return response


def _emit_legacy_action_warning(
    action: str,
    deprecation: Optional[Dict[str, str]],
) -> None:
    """Emit WARN-level log for deprecated legacy action invocations."""
    if not deprecation:
        return
    logger.warning(
        "Deprecated task action invoked: action=%s replacement=%s removal_target=%s",
        action,
        deprecation.get("replacement"),
        deprecation.get("removal_target"),
    )


def attach_loop_metadata(response: dict, *, overwrite: bool = True) -> dict:
    """Attach loop_signal + recommended_actions to a session-step response.

    This is the single canonical attachment point for loop signal metadata.
    All session-step responses (from handlers, error mappers, and dispatch
    fallbacks) should pass through this function.

    Args:
        response: MCP response dict with ``data`` payload.
        overwrite: When True (default), always set ``loop_signal`` and
            ``recommended_actions``.  When False, only fill in missing
            fields (used by the post-dispatch fallback so handler values
            are preserved).
    """
    data = response.get("data")
    if not isinstance(data, dict):
        return response

    success = bool(response.get("success"))
    details = data.get("details")
    if not isinstance(details, dict):
        details = {}

    error_code = None
    if not success:
        error_code = details.get("error_code") or data.get("error_code")
    repeated_invalid_gate_evidence = bool(details.get("repeated_invalid_gate_evidence"))
    if not repeated_invalid_gate_evidence:
        attempts = details.get("invalid_gate_evidence_attempts")
        if isinstance(attempts, int) and attempts >= 3:
            repeated_invalid_gate_evidence = True

    loop_signal = derive_loop_signal(
        status=data.get("status"),
        pause_reason=data.get("pause_reason"),
        error_code=error_code,
        is_unrecoverable_error=(error_code == ERROR_SESSION_UNRECOVERABLE),
        repeated_invalid_gate_evidence=repeated_invalid_gate_evidence,
    )
    if loop_signal is None:
        return response

    if overwrite:
        data["loop_signal"] = loop_signal.value
    else:
        data.setdefault("loop_signal", loop_signal.value)

    if overwrite or "recommended_actions" not in data:
        recommended_actions = derive_recommended_actions(
            loop_signal=loop_signal,
            pause_reason=data.get("pause_reason"),
            error_code=error_code,
        )
        if recommended_actions:
            data["recommended_actions"] = [ra.model_dump(mode="json", by_alias=True) for ra in recommended_actions]

    return response


def _attach_session_step_loop_metadata(action: str, response: dict) -> dict:
    """Post-dispatch fallback: attach loop signal for session-step responses.

    Ensures authorization/rate-limit/feature-flag errors emitted *before*
    handlers still include machine-readable loop outcomes.  Uses
    ``overwrite=False`` so handler-provided values are preserved.
    """
    normalized_action = (action or "").strip().lower()
    if normalized_action not in {
        "session-step-next",
        "session-step-report",
        "session-step-replay",
    }:
        return response

    return attach_loop_metadata(response, overwrite=False)


def _resolve_specs_dir(config: ServerConfig, workspace: Optional[str]) -> tuple[Optional[Path], Optional[dict]]:
    """Thin wrapper around the shared helper preserving the local call convention."""
    return resolve_specs_dir(config, workspace)


def _load_spec_data(
    spec_id: str, specs_dir: Optional[Path], request_id: str
) -> Tuple[Optional[Dict[str, Any]], Optional[dict]]:
    if specs_dir is None:
        return None, asdict(
            error_response(
                "No specs directory found. Use --specs-dir or set FOUNDRY_SPECS_DIR.",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Set FOUNDRY_SPECS_DIR or invoke with --specs-dir",
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
        warnings.append(f"{total_count} results returned; consider using pagination to limit payload size.")
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
        WriteLockStatus as _WriteLockStatus,
    )
    from foundry_mcp.core.autonomy.write_lock import (
        check_autonomy_write_lock as _check_autonomy_write_lock_impl,
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
    config: Optional[ServerConfig] = None,
) -> Optional[dict]:
    """Check autonomy write-lock and return error response if blocked.

    Args:
        spec_id: The spec ID being modified.
        workspace: Optional workspace path.
        bypass_autonomy_lock: If True, bypass the lock (requires bypass_reason).
        bypass_reason: Reason for bypassing the lock.
        request_id: Request ID for error response.
        config: ServerConfig to check allow_lock_bypass setting.

    Returns:
        None if operation is allowed, error response dict if blocked.
    """
    if not _WRITE_LOCK_AVAILABLE or _check_autonomy_write_lock_impl is None:
        return None

    # Get allow_lock_bypass from config (default False - fail-closed)
    allow_lock_bypass = False
    if config is not None:
        allow_lock_bypass = config.autonomy_security.allow_lock_bypass

    result = _check_autonomy_write_lock_impl(
        spec_id=spec_id,
        workspace=workspace,
        bypass_flag=bypass_autonomy_lock,
        bypass_reason=bypass_reason,
        allow_lock_bypass=allow_lock_bypass,
    )

    assert _WriteLockStatus is not None
    if result.status == _WriteLockStatus.LOCKED:
        return asdict(
            error_response(
                result.message or "Autonomy write lock is active for this spec",
                error_code=ErrorCode.AUTONOMY_WRITE_LOCK_ACTIVE,
                error_type=ErrorType.CONFLICT,
                request_id=request_id,
                details={
                    "session_id": result.session_id,
                    "session_status": result.session_status,
                    "hint": "Use bypass_autonomy_lock=true with bypass_reason to override",
                },
            )
        )

    # ALLOWED or BYPASSED — operation can proceed
    return None


def _workspace_path_error(workspace: str, reason: str, request_id: str) -> dict:
    """Return a validation error for an invalid workspace path."""
    return asdict(
        error_response(
            f"Invalid workspace path: {reason}",
            error_code=ErrorCode.VALIDATION_ERROR,
            error_type=ErrorType.VALIDATION,
            request_id=request_id,
            details={
                "field": "workspace",
                "reason": reason,
                "path": workspace,
                "hint": "Provide an absolute path without '..' components",
            },
        )
    )


def _validate_workspace(
    workspace: str,
    request_id: str,
) -> Tuple[Optional[Path], Optional[dict]]:
    """Validate and canonicalize a workspace path, rejecting traversal attempts.

    Returns:
        (validated_path, None) on success, or (None, error_response) on failure.
    """
    from foundry_mcp.core.authorization import validate_runner_path
    from foundry_mcp.core.errors.authorization import PathValidationError

    try:
        validated = validate_runner_path(workspace, require_within_workspace=False)
        return validated, None
    except PathValidationError as exc:
        logger.warning("Workspace path validation failed: %s (path=%s)", exc.reason, exc.path)
        return None, _workspace_path_error(workspace, exc.reason, request_id)


def _get_storage(
    config: ServerConfig,
    workspace: Optional[str] = None,
    *,
    request_id: Optional[str] = None,
) -> "AutonomyStorage | dict":
    """Get AutonomyStorage instance for session operations.

    When request_id is provided and workspace validation fails, returns an
    error response dict instead of raising.  When request_id is None (legacy
    callers), raises PathValidationError on invalid workspace.
    """
    if workspace:
        if request_id:
            ws_path, err = _validate_workspace(workspace, request_id)
            if err:
                return err
        else:
            from foundry_mcp.core.authorization import validate_runner_path

            ws_path = validate_runner_path(workspace, require_within_workspace=False)
    else:
        ws_path = Path.cwd()
    return AutonomyStorage(workspace_path=ws_path)


def _session_not_found_response(action: str, request_id: str, spec_id: Optional[str] = None) -> dict:
    """Return session not found error response."""
    return asdict(
        error_response(
            "No active session found",
            error_code=ErrorCode.NO_ACTIVE_SESSION,
            error_type=ErrorType.NOT_FOUND,
            request_id=request_id,
            details={
                "action": action,
                "spec_id": spec_id,
                "hint": "Start a session with session-start action",
            },
        )
    )


def _resolve_session(
    storage: AutonomyStorage,
    action: str,
    request_id: str,
    session_id: Optional[str] = None,
    spec_id: Optional[str] = None,
) -> Tuple[Optional[AutonomousSessionState], Optional[dict]]:
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
        return None, asdict(
            error_response(
                "Multiple active sessions found. Provide spec_id or session_id to disambiguate.",
                error_code=ErrorCode.AMBIGUOUS_ACTIVE_SESSION,
                error_type=ErrorType.VALIDATION,
                request_id=request_id,
                details={
                    "action": action,
                    "hint": "Provide spec_id or session_id parameter",
                },
            )
        )

    # FOUND
    if found_id is None:
        return None, _session_not_found_response(action, request_id)
    session = storage.load(found_id)
    if not session:
        return None, _session_not_found_response(action, request_id)
    return session, None
