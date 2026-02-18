"""Session query handlers: status, list, events, heartbeat.

Split from handlers_session.py for maintainability (H3).
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from foundry_mcp.config.server import ServerConfig
from foundry_mcp.core.autonomy.memory import (
    ListSessionsResult,
)
from foundry_mcp.core.responses.types import (
    ErrorCode,
    ErrorType,
)
from foundry_mcp.core.responses.builders import (
    error_response,
    success_response,
)
from foundry_mcp.core.pagination import (
    CursorError,
    decode_cursor,
    encode_cursor,
    normalize_page_size,
)

from foundry_mcp.tools.unified.task_handlers._helpers import (
    _get_storage,
    _request_id,
    _resolve_session,
    _validate_context_usage_pct,
    _is_feature_enabled,
    _feature_disabled_response,
)
from foundry_mcp.tools.unified.task_handlers._session_common import (
    _load_spec_for_session,
    _build_session_response,
    _save_with_version_check,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Events Constants
# =============================================================================

SESSION_EVENTS_DEFAULT_LIMIT = 50
SESSION_EVENTS_MAX_LIMIT = 200
SESSION_EVENTS_CURSOR_KIND = "session-events-v1"


# =============================================================================
# Events Helper Functions
# =============================================================================


def _parse_journal_timestamp(timestamp: Optional[str]) -> datetime:
    """Parse journal timestamp into a timezone-aware datetime for ordering."""
    if not isinstance(timestamp, str) or not timestamp.strip():
        return datetime.min.replace(tzinfo=timezone.utc)

    normalized = timestamp.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _encode_session_events_cursor(
    *,
    session_id: str,
    spec_id: str,
    timestamp: str,
    index: int,
) -> str:
    """Encode opaque cursor for session-events pagination."""
    return encode_cursor(
        {
            "kind": SESSION_EVENTS_CURSOR_KIND,
            "session_id": session_id,
            "spec_id": spec_id,
            "timestamp": timestamp,
            "index": index,
        }
    )


def _decode_session_events_cursor(
    cursor: str,
    *,
    session_id: str,
    spec_id: str,
) -> tuple[datetime, int]:
    """Decode and validate cursor for a specific session/spec pair."""
    try:
        cursor_data = decode_cursor(cursor)
    except CursorError as exc:
        raise ValueError(str(exc)) from exc

    if not isinstance(cursor_data, dict):
        raise ValueError("Cursor payload must be an object")
    if cursor_data.get("kind") != SESSION_EVENTS_CURSOR_KIND:
        raise ValueError("Cursor kind does not match session-events")
    if cursor_data.get("session_id") != session_id:
        raise ValueError("Cursor was issued for a different session_id")
    if cursor_data.get("spec_id") != spec_id:
        raise ValueError("Cursor was issued for a different spec_id")

    timestamp = cursor_data.get("timestamp")
    index = cursor_data.get("index")
    if not isinstance(timestamp, str):
        raise ValueError("Cursor timestamp is missing or invalid")
    if not isinstance(index, int):
        raise ValueError("Cursor index is missing or invalid")

    return _parse_journal_timestamp(timestamp), index


def _collect_session_events(
    *,
    session: Any,
    spec_data: Dict[str, Any],
) -> list[dict]:
    """Return normalized, sorted journal events scoped to a session."""
    journal_entries = spec_data.get("journal", [])
    if not isinstance(journal_entries, list):
        return []

    events: list[dict] = []

    for index, entry in enumerate(journal_entries):
        if not isinstance(entry, dict):
            continue

        metadata = entry.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        if metadata.get("session_id") != session.id:
            continue

        timestamp = entry.get("timestamp")
        if not isinstance(timestamp, str):
            continue

        event: Dict[str, Any] = {
            "event_id": f"{session.id}:{index}",
            "session_id": session.id,
            "spec_id": session.spec_id,
            "timestamp": timestamp,
            "event_type": entry.get("entry_type") if isinstance(entry.get("entry_type"), str) else "note",
            "action": metadata.get("action") if isinstance(metadata.get("action"), str) else None,
            "title": entry.get("title") if isinstance(entry.get("title"), str) else "",
            "summary": entry.get("content") if isinstance(entry.get("content"), str) else "",
            "author": entry.get("author") if isinstance(entry.get("author"), str) else "autonomy",
            "task_id": entry.get("task_id") if isinstance(entry.get("task_id"), str) else None,
            "details": metadata or None,
            "_cursor_timestamp": _parse_journal_timestamp(timestamp),
            "_cursor_index": index,
        }
        events.append(event)

    events.sort(
        key=lambda event: (
            event["_cursor_timestamp"],
            event["_cursor_index"],
        ),
        reverse=True,
    )
    return events


# =============================================================================
# Session Status Handler
# =============================================================================


def _handle_session_status(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    session_id: Optional[str] = None,
    workspace: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Handle session-status action.

    Gets current status of an autonomous session.
    Read-only with effective_status and staleness metadata.

    Args:
        config: Server configuration
        spec_id: Spec ID (optional if session_id provided)
        session_id: Session ID (optional if spec_id provided)
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict with session status or error
    """
    request_id = _request_id()

    # Feature flag check - fail-closed
    if not _is_feature_enabled(config, "autonomy_sessions"):
        return _feature_disabled_response("session-status", request_id)

    storage = _get_storage(config, workspace, request_id=request_id)
    if isinstance(storage, dict):
        return storage

    session, err = _resolve_session(storage, "session-status", request_id, session_id, spec_id)
    if err:
        return err

    return _build_session_response(session, request_id, workspace=workspace)


# =============================================================================
# Session Events Handler
# =============================================================================


def _handle_session_events(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    session_id: Optional[str] = None,
    cursor: Optional[str] = None,
    limit: Optional[int] = None,
    workspace: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Handle session-events action.

    Returns a journal-backed, session-scoped event feed with cursor pagination.
    This is intentionally a filtered view over existing spec journal entries.
    """
    request_id = _request_id()

    # Feature flag check - fail-closed
    if not _is_feature_enabled(config, "autonomy_sessions"):
        return _feature_disabled_response("session-events", request_id)

    page_size = normalize_page_size(
        limit,
        default=SESSION_EVENTS_DEFAULT_LIMIT,
        maximum=SESSION_EVENTS_MAX_LIMIT,
    )

    storage = _get_storage(config, workspace, request_id=request_id)
    if isinstance(storage, dict):
        return storage
    session, err = _resolve_session(
        storage,
        "session-events",
        request_id,
        session_id,
        spec_id,
    )
    if err:
        return err

    spec_data = _load_spec_for_session(session, workspace)
    if spec_data is None:
        return asdict(
            error_response(
                f"Spec not found: {session.spec_id}",
                error_code=ErrorCode.SPEC_NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                request_id=request_id,
                details={
                    "action": "session-events",
                    "session_id": session.id,
                    "spec_id": session.spec_id,
                    "hint": 'Verify spec presence via spec(action="find")',
                },
            )
        )

    started = time.perf_counter()
    events = _collect_session_events(session=session, spec_data=spec_data)

    cursor_position: Optional[tuple[datetime, int]] = None
    if cursor:
        try:
            cursor_position = _decode_session_events_cursor(
                cursor,
                session_id=session.id,
                spec_id=session.spec_id,
            )
        except ValueError as exc:
            return asdict(
                error_response(
                    f"Invalid cursor: {exc}",
                    error_code=ErrorCode.INVALID_CURSOR,
                    error_type=ErrorType.VALIDATION,
                    request_id=request_id,
                    details={
                        "action": "session-events",
                        "session_id": session.id,
                        "spec_id": session.spec_id,
                        "hint": "Reuse cursor from the previous session-events response for this same session",
                    },
                )
            )

    if cursor_position is not None:
        cursor_timestamp, cursor_index = cursor_position
        events = [
            event
            for event in events
            if (
                event["_cursor_timestamp"],
                event["_cursor_index"],
            ) < (
                cursor_timestamp,
                cursor_index,
            )
        ]

    has_more = len(events) > page_size
    page_events = events[:page_size]
    next_cursor = None

    if has_more and page_events:
        last_event = page_events[-1]
        next_cursor = _encode_session_events_cursor(
            session_id=session.id,
            spec_id=session.spec_id,
            timestamp=last_event["timestamp"],
            index=last_event["_cursor_index"],
        )

    serialized_events: List[Dict[str, Any]] = []
    for event in page_events:
        serialized_events.append(
            {
                key: value
                for key, value in event.items()
                if not key.startswith("_") and value is not None
            }
        )

    duration_ms = (time.perf_counter() - started) * 1000

    return asdict(
        success_response(
            data={
                "session_id": session.id,
                "spec_id": session.spec_id,
                "events": serialized_events,
            },
            request_id=request_id,
            pagination={
                "cursor": next_cursor,
                "has_more": has_more,
                "page_size": page_size,
            },
            telemetry={
                "duration_ms": round(duration_ms, 2),
                "journal_entries_scanned": (
                    len(spec_data.get("journal", []))
                    if isinstance(spec_data.get("journal"), list)
                    else 0
                ),
                "session_events_returned": len(serialized_events),
            },
        )
    )


# =============================================================================
# Session List Handler
# =============================================================================


def _handle_session_list(
    *,
    config: ServerConfig,
    status_filter: Optional[str] = None,
    spec_id: Optional[str] = None,
    limit: int = 20,
    cursor: Optional[str] = None,
    workspace: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Handle session-list action.

    Lists autonomous sessions with optional filtering.
    Pagination: cursor-based, limit 20 default/100 max.

    Args:
        config: Server configuration
        status_filter: Filter by session status
        spec_id: Filter by spec ID
        limit: Maximum results to return (1-100)
        cursor: Pagination cursor
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict with session list or error
    """
    request_id = _request_id()

    # Feature flag check - fail-closed
    if not _is_feature_enabled(config, "autonomy_sessions"):
        return _feature_disabled_response("session-list", request_id)

    # Validate limit
    limit = max(1, min(limit, 100))

    storage = _get_storage(config, workspace, request_id=request_id)
    if isinstance(storage, dict):
        return storage

    try:
        result: ListSessionsResult = storage.list_sessions(
            status_filter=status_filter,
            spec_id=spec_id,
            limit=limit,
            cursor=cursor,
            include_total=False,
        )
    except ValueError as e:
        # Invalid cursor
        return asdict(error_response(
            f"Invalid cursor: {e}",
            error_code=ErrorCode.INVALID_CURSOR,
            error_type=ErrorType.VALIDATION,
            request_id=request_id,
        ))

    # Build response
    sessions_data = []
    for summary in result.sessions:
        session_dict = {
            "session_id": summary.session_id,
            "spec_id": summary.spec_id,
            "status": summary.status.value,
            "pause_reason": summary.pause_reason.value if summary.pause_reason else None,
            "created_at": summary.created_at.isoformat() if summary.created_at else None,
            "updated_at": summary.updated_at.isoformat() if summary.updated_at else None,
            "active_phase_id": summary.active_phase_id,
            "tasks_completed": summary.tasks_completed,
        }
        if summary.effective_status:
            session_dict["effective_status"] = summary.effective_status.value
        sessions_data.append(session_dict)

    response_data = {
        "sessions": sessions_data,
        "cursor": result.cursor,
        "has_more": result.has_more,
    }

    return asdict(success_response(
        data=response_data,
        request_id=request_id,
    ))


# =============================================================================
# Session Heartbeat Handler
# =============================================================================


def _handle_session_heartbeat(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    session_id: Optional[str] = None,
    context_usage_pct: Optional[int] = None,
    estimated_tokens_used: Optional[int] = None,
    workspace: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Handle session-heartbeat action.

    Updates session heartbeat and context metrics.

    Args:
        config: Server configuration
        spec_id: Spec ID of the session
        session_id: Session ID (optional, alternative to spec_id)
        context_usage_pct: Current context usage percentage
        estimated_tokens_used: Estimated tokens used
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict confirming heartbeat or error
    """
    request_id = _request_id()

    # Feature flag check - fail-closed
    if not _is_feature_enabled(config, "autonomy_sessions"):
        return _feature_disabled_response("session-heartbeat", request_id)

    # Validate context_usage_pct
    pct_err = _validate_context_usage_pct(context_usage_pct, "session-heartbeat", request_id)
    if pct_err:
        return pct_err

    storage = _get_storage(config, workspace, request_id=request_id)
    if isinstance(storage, dict):
        return storage

    session, err = _resolve_session(storage, "session-heartbeat", request_id, session_id, spec_id)
    if err:
        return err

    # Update heartbeat
    now = datetime.now(timezone.utc)
    session.context.last_heartbeat_at = now
    session.updated_at = now
    pre_mutation_version = session.state_version
    session.state_version += 1

    # Route context_usage_pct through ContextTracker for validation/hardening
    if context_usage_pct is not None:
        from foundry_mcp.core.autonomy.context_tracker import ContextTracker

        ws_path = Path(workspace) if workspace else Path.cwd()
        tracker = ContextTracker(ws_path)
        effective_pct, source = tracker.get_effective_context_pct(
            session, context_usage_pct, now
        )
        session.context.context_usage_pct = effective_pct
        session.context.context_source = source

    if estimated_tokens_used is not None:
        session.context.estimated_tokens_used = estimated_tokens_used

    err = _save_with_version_check(storage, session, pre_mutation_version, "session-heartbeat", request_id)
    if err:
        return err

    # Check if context threshold exceeded
    warnings = []
    if session.context.context_usage_pct >= session.limits.context_threshold_pct:
        warnings.append(f"Context usage at {session.context.context_usage_pct}%")

    response_data = {
        "session_id": session.id,
        "heartbeat_at": now.isoformat(),
        "context_usage_pct": session.context.context_usage_pct,
        "context_source": session.context.context_source,
    }

    return asdict(success_response(
        data=response_data,
        request_id=request_id,
        meta={"warnings": warnings} if warnings else None,
    ))
