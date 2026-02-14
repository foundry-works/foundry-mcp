"""Session lifecycle action handlers: start, pause, resume, end, status, list, rebase, heartbeat, reset.

This module provides handlers for autonomous session management actions.
All actions integrate with:
- AutonomyStorage for persistence
- spec_hash for integrity validation
- write_lock for concurrent access protection
- Journal for lifecycle event logging
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ulid import ULID

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.autonomy.memory import (
    AutonomyStorage,
    ActiveSessionLookupResult,
    ListSessionsResult,
)
from foundry_mcp.core.autonomy.models import (
    AutonomousSessionState,
    SessionStatus,
    PauseReason,
    FailureReason,
    SessionResponseData,
    SessionCounters,
    SessionLimits,
    StopConditions,
    SessionContext,
    ResumeContext,
    CompletedTaskSummary,
    CompletedPhaseSummary,
    PendingTaskSummary,
    RebaseResultDetail,
    GatePolicy,
)
from foundry_mcp.core.autonomy.spec_hash import (
    compute_spec_structure_hash,
    get_spec_file_metadata,
    compute_structural_diff,
    StructuralDiff,
)
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
)
from foundry_mcp.core.spec import load_spec

from foundry_mcp.tools.unified.task_handlers._helpers import (
    _request_id,
    _validation_error,
    _resolve_specs_dir,
)

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Error Code Constants (from ADR)
# =============================================================================

ERROR_SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
ERROR_SESSION_ALREADY_EXISTS = "SESSION_ALREADY_EXISTS"
ERROR_INVALID_STATE_TRANSITION = "INVALID_STATE_TRANSITION"
ERROR_MANUAL_GATE_ACK_REQUIRED = "MANUAL_GATE_ACK_REQUIRED"
ERROR_INVALID_GATE_ACK = "INVALID_GATE_ACK"
ERROR_SPEC_NOT_FOUND = "SPEC_NOT_FOUND"
ERROR_SPEC_STRUCTURE_CHANGED = "SPEC_STRUCTURE_CHANGED"
ERROR_IDEMPOTENCY_MISMATCH = "IDEMPOTENCY_MISMATCH"


# =============================================================================
# Helper Functions
# =============================================================================


def _get_storage(config: ServerConfig, workspace: Optional[str] = None) -> AutonomyStorage:
    """Get AutonomyStorage instance."""
    ws_path = Path(workspace) if workspace else Path.cwd()
    return AutonomyStorage(workspace_path=ws_path)


def _feature_disabled_response(action: str, request_id: str) -> dict:
    """Return feature disabled error response."""
    return asdict(error_response(
        "autonomy_sessions feature is not enabled",
        error_code=ErrorCode.FEATURE_DISABLED,
        error_type=ErrorType.FEATURE_FLAG,
        request_id=request_id,
        details={
            "action": action,
            "feature_flag": "autonomy_sessions",
            "hint": "Enable autonomy_sessions feature flag to use session management",
        },
    ))


def _session_not_found_response(action: str, request_id: str, spec_id: Optional[str] = None) -> dict:
    """Return session not found error response."""
    return asdict(error_response(
        "No active session found",
        error_code=ErrorCode.NOT_FOUND,
        error_type=ErrorType.NOT_FOUND,
        request_id=request_id,
        details={
            "action": action,
            "spec_id": spec_id,
            "hint": "Start a session with session-start action",
        },
    ))


def _invalid_transition_response(
    action: str,
    request_id: str,
    current_status: str,
    target_status: str,
    reason: Optional[str] = None,
) -> dict:
    """Return invalid state transition error response."""
    return asdict(error_response(
        f"Invalid state transition: {current_status} -> {target_status}",
        error_code=ErrorCode.VALIDATION_ERROR,
        error_type=ErrorType.VALIDATION,
        request_id=request_id,
        details={
            "action": action,
            "current_status": current_status,
            "target_status": target_status,
            "reason": reason,
            "hint": "Check session status before attempting transition",
        },
    ))


def _compute_effective_status(session: AutonomousSessionState) -> Optional[SessionStatus]:
    """Compute effective status considering staleness.

    Args:
        session: Session state

    Returns:
        Derived status (paused) if stale, None if actual status applies
    """
    if session.status != SessionStatus.RUNNING:
        return None

    now = datetime.now(timezone.utc)

    # Check step staleness
    if session.last_step_issued:
        from datetime import timedelta
        step_stale_threshold = timedelta(minutes=session.limits.step_stale_minutes)
        if now - session.last_step_issued.issued_at > step_stale_threshold:
            return SessionStatus.PAUSED

    # Check heartbeat staleness (after grace period)
    if session.context.last_heartbeat_at:
        from datetime import timedelta
        heartbeat_stale_threshold = timedelta(minutes=session.limits.heartbeat_stale_minutes)
        if now - session.context.last_heartbeat_at > heartbeat_stale_threshold:
            return SessionStatus.PAUSED

    return None


def _build_session_response(
    session: AutonomousSessionState,
    request_id: str,
    include_resume_context: bool = False,
    rebase_result: Optional[RebaseResultDetail] = None,
) -> dict:
    """Build standard session response data."""
    effective_status = _compute_effective_status(session)

    response_data = SessionResponseData(
        session_id=session.id,
        spec_id=session.spec_id,
        status=session.status,
        pause_reason=session.pause_reason,
        counters=session.counters,
        limits=session.limits,
        stop_conditions=session.stop_conditions,
        write_lock_enforced=session.write_lock_enforced,
        active_phase_id=session.active_phase_id,
        last_heartbeat_at=session.context.last_heartbeat_at,
        next_action_hint=None,  # Filled by step commands
        resume_context=None,
        rebase_result=rebase_result,
    )

    # Include effective status if derived
    data = asdict(response_data)
    if effective_status:
        data["effective_status"] = effective_status.value

    return asdict(success_response(
        data=data,
        request_id=request_id,
    ))


def _write_session_journal(
    spec_id: str,
    action: str,
    summary: str,
    session_id: str,
    workspace: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> bool:
    """Write a journal entry for session lifecycle event.

    Args:
        spec_id: Spec ID
        action: Action being performed
        summary: Summary of the event
        session_id: Session ID
        workspace: Workspace path
        metadata: Optional additional metadata

    Returns:
        True if journal entry was written successfully
    """
    try:
        # Find spec file
        ws_path = Path(workspace) if workspace else Path.cwd()
        specs_dir = ws_path / "specs"
        spec_path = specs_dir / f"{spec_id}.json"

        if not spec_path.exists():
            logger.warning("Spec file not found for journal: %s", spec_path)
            return False

        # Load spec data
        spec_data = json.loads(spec_path.read_text())

        # Add journal entry
        journal_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "entry_type": "status_change",
            "title": f"Session {action}",
            "content": summary,
            "author": "autonomy",
            "task_id": None,
            "metadata": {
                "session_id": session_id,
                "action": action,
                **(metadata or {}),
            },
        }

        # Ensure journal array exists
        if "journal" not in spec_data:
            spec_data["journal"] = []

        spec_data["journal"].append(journal_entry)

        # Write back atomically
        import tempfile
        import os
        fd, temp_path = tempfile.mkstemp(dir=specs_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(spec_data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, spec_path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

        logger.debug("Wrote session journal entry for %s", session_id)
        return True

    except Exception as e:
        logger.warning("Failed to write session journal: %s", e)
        return False


# =============================================================================
# Session Start Handler
# =============================================================================


def _handle_session_start(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    workspace: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Handle session-start action.

    Starts a new autonomous session for a spec.

    Atomic commit sequence:
    1. Acquire per-spec lock (5s timeout)
    2. Check pointer for existing session
    3. Idempotency key check
    4. Read spec + compute hash
    5. Create state (atomic write)
    6. Write journal (mandatory, rollback on failure)
    7. Write pointer
    8. Release lock
    9. GC

    Args:
        config: Server configuration
        spec_id: Spec ID to start session for
        idempotency_key: Optional idempotency key for deduplication
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict with session details or error
    """
    request_id = _request_id()

    if not spec_id:
        return _validation_error(
            action="session-start",
            field="spec_id",
            message="spec_id is required",
            request_id=request_id,
        )

    storage = _get_storage(config, workspace)

    # Step 1: Acquire per-spec lock
    try:
        spec_lock = storage.acquire_spec_lock(spec_id, timeout=5)
    except Exception as e:
        return asdict(error_response(
            f"Failed to acquire spec lock: {e}",
            error_code=ErrorCode.RESOURCE_BUSY,
            error_type=ErrorType.UNAVAILABLE,
            request_id=request_id,
            details={"spec_id": spec_id, "action": "session-start"},
        ))

    with spec_lock:
        # Step 2: Check pointer for existing session
        existing_session_id = storage.get_active_session(spec_id)
        if existing_session_id:
            existing_session = storage.load(existing_session_id)
            if existing_session and existing_session.status in {
                SessionStatus.RUNNING,
                SessionStatus.PAUSED,
            }:
                # Check idempotency key match
                if idempotency_key and existing_session.idempotency_key == idempotency_key:
                    # Idempotent - return existing session
                    return _build_session_response(existing_session, request_id)

                return asdict(error_response(
                    "Active session already exists for this spec",
                    error_code=ErrorCode.CONFLICT,
                    error_type=ErrorType.CONFLICT,
                    request_id=request_id,
                    details={
                        "spec_id": spec_id,
                        "existing_session_id": existing_session_id,
                        "existing_status": existing_session.status.value,
                        "hint": "End existing session before starting a new one",
                    },
                ))

        # Step 3: Load spec and compute hash
        ws_path = Path(workspace) if workspace else Path.cwd()
        specs_dir = ws_path / "specs"
        spec_path = specs_dir / f"{spec_id}.json"

        if not spec_path.exists():
            return asdict(error_response(
                f"Spec not found: {spec_id}",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                request_id=request_id,
                details={"spec_id": spec_id},
            ))

        try:
            spec_data = json.loads(spec_path.read_text())
            spec_structure_hash = compute_spec_structure_hash(spec_data)
            spec_metadata = get_spec_file_metadata(spec_path)
        except Exception as e:
            return asdict(error_response(
                f"Failed to read spec: {e}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                request_id=request_id,
                details={"spec_id": spec_id},
            ))

        # Step 4: Create session state
        now = datetime.now(timezone.utc)
        session = AutonomousSessionState(
            id=str(ULID()),
            spec_id=spec_id,
            idempotency_key=idempotency_key,
            spec_structure_hash=spec_structure_hash,
            spec_file_mtime=spec_metadata.mtime if spec_metadata else None,
            spec_file_size=spec_metadata.file_size if spec_metadata else None,
            status=SessionStatus.RUNNING,
            created_at=now,
            updated_at=now,
            counters=SessionCounters(),
            limits=SessionLimits(),
            stop_conditions=StopConditions(),
            context=SessionContext(),
            write_lock_enforced=True,
            gate_policy=GatePolicy.STRICT,
        )

        # Step 5: Save state (atomic write)
        try:
            storage.save(session)
        except Exception as e:
            logger.error("Failed to save session state: %s", e)
            return asdict(error_response(
                f"Failed to create session: {e}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                request_id=request_id,
            ))

        # Step 6: Write journal (mandatory)
        journal_success = _write_session_journal(
            spec_id=spec_id,
            action="start",
            summary=f"Started autonomous session {session.id}",
            session_id=session.id,
            workspace=workspace,
            metadata={"idempotency_key": idempotency_key},
        )

        if not journal_success:
            # Rollback: delete state before releasing lock
            logger.warning("Journal write failed, rolling back session %s", session.id)
            storage.delete(session.id)
            return asdict(error_response(
                "Failed to write session journal",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                request_id=request_id,
                details={"hint": "Session creation rolled back"},
            ))

        # Step 7: Write pointer
        storage.set_active_session(spec_id, session.id)

        # Lock released by context manager

    # Step 8: GC (best effort)
    try:
        storage.cleanup_expired()
    except Exception as e:
        logger.debug("GC failed (non-critical): %s", e)

    logger.info("Started session %s for spec %s", session.id, spec_id)

    return _build_session_response(session, request_id)


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

    if not spec_id and not session_id:
        return _validation_error(
            action="session-status",
            field="spec_id",
            message="spec_id or session_id is required",
            request_id=request_id,
        )

    storage = _get_storage(config, workspace)

    # Load by session_id or find by spec_id
    session = None
    if session_id:
        session = storage.load(session_id)
    elif spec_id:
        active_session_id = storage.get_active_session(spec_id)
        if active_session_id:
            session = storage.load(active_session_id)

    if not session:
        return _session_not_found_response("session-status", request_id, spec_id)

    return _build_session_response(session, request_id)


# =============================================================================
# Session Pause Handler
# =============================================================================


def _handle_session_pause(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    reason: Optional[str] = None,
    workspace: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Handle session-pause action.

    Pauses an active autonomous session.
    Valid transition: running -> paused

    Args:
        config: Server configuration
        spec_id: Spec ID of the session
        reason: Optional reason for pausing
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict with updated session state or error
    """
    request_id = _request_id()

    if not spec_id:
        return _validation_error(
            action="session-pause",
            field="spec_id",
            message="spec_id is required",
            request_id=request_id,
        )

    storage = _get_storage(config, workspace)

    session_id = storage.get_active_session(spec_id)
    if not session_id:
        return _session_not_found_response("session-pause", request_id, spec_id)

    session = storage.load(session_id)
    if not session:
        return _session_not_found_response("session-pause", request_id, spec_id)

    # Validate state transition
    if session.status != SessionStatus.RUNNING:
        return _invalid_transition_response(
            action="session-pause",
            request_id=request_id,
            current_status=session.status.value,
            target_status="paused",
            reason="Only running sessions can be paused",
        )

    # Update session
    session.status = SessionStatus.PAUSED
    session.pause_reason = PauseReason.USER if not reason else PauseReason(reason)
    session.updated_at = datetime.now(timezone.utc)
    session.paused_at = datetime.now(timezone.utc)

    storage.save(session)

    # Write journal
    _write_session_journal(
        spec_id=spec_id,
        action="pause",
        summary=f"Session paused: {reason or 'user request'}",
        session_id=session.id,
        workspace=workspace,
    )

    logger.info("Paused session %s for spec %s", session.id, spec_id)

    return _build_session_response(session, request_id)


# =============================================================================
# Session Resume Handler
# =============================================================================


def _handle_session_resume(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    gate_ack: Optional[str] = None,
    force: bool = False,
    workspace: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Handle session-resume action.

    Resumes a paused autonomous session.
    Valid transitions: paused -> running, failed -> running (with force)

    Validates manual-gate acknowledgment if pending.

    Args:
        config: Server configuration
        spec_id: Spec ID of the session
        gate_ack: Optional gate acknowledgment ID
        force: Force resume from failed state
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict with session state and next step or error
    """
    request_id = _request_id()

    if not spec_id:
        return _validation_error(
            action="session-resume",
            field="spec_id",
            message="spec_id is required",
            request_id=request_id,
        )

    storage = _get_storage(config, workspace)

    session_id = storage.get_active_session(spec_id)
    if not session_id:
        return _session_not_found_response("session-resume", request_id, spec_id)

    session = storage.load(session_id)
    if not session:
        return _session_not_found_response("session-resume", request_id, spec_id)

    # Validate state transition
    if session.status == SessionStatus.FAILED:
        if not force:
            return _invalid_transition_response(
                action="session-resume",
                request_id=request_id,
                current_status="failed",
                target_status="running",
                reason="Use force=true to resume from failed state",
            )
    elif session.status != SessionStatus.PAUSED:
        return _invalid_transition_response(
            action="session-resume",
            request_id=request_id,
            current_status=session.status.value,
            target_status="running",
            reason="Only paused sessions can be resumed",
        )

    # Check manual gate acknowledgment
    if session.pending_manual_gate_ack:
        if not gate_ack:
            return asdict(error_response(
                "Manual gate acknowledgment required",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                request_id=request_id,
                details={
                    "action": "session-resume",
                    "gate_attempt_id": session.pending_manual_gate_ack.gate_attempt_id,
                    "phase_id": session.pending_manual_gate_ack.phase_id,
                    "hint": "Provide gate_ack with the gate_attempt_id to acknowledge",
                },
            ))

        if gate_ack != session.pending_manual_gate_ack.gate_attempt_id:
            return asdict(error_response(
                "Invalid gate acknowledgment",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                request_id=request_id,
                details={
                    "action": "session-resume",
                    "expected": session.pending_manual_gate_ack.gate_attempt_id,
                    "provided": gate_ack,
                },
            ))

        # Clear pending gate ack
        session.pending_manual_gate_ack = None

    # Update session
    session.status = SessionStatus.RUNNING
    session.pause_reason = None
    session.failure_reason = None
    session.updated_at = datetime.now(timezone.utc)
    session.paused_at = None

    storage.save(session)

    # Write journal
    _write_session_journal(
        spec_id=spec_id,
        action="resume",
        summary=f"Session resumed{', forced from failed' if force else ''}",
        session_id=session.id,
        workspace=workspace,
        metadata={"gate_ack": gate_ack, "force": force},
    )

    logger.info("Resumed session %s for spec %s", session.id, spec_id)

    return _build_session_response(session, request_id, include_resume_context=True)


# =============================================================================
# Session End Handler
# =============================================================================


def _handle_session_end(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    workspace: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Handle session-end action.

    Ends an autonomous session (terminal state).
    Valid transitions: running/paused/failed -> ended

    Args:
        config: Server configuration
        spec_id: Spec ID of the session
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict confirming session ended or error
    """
    request_id = _request_id()

    if not spec_id:
        return _validation_error(
            action="session-end",
            field="spec_id",
            message="spec_id is required",
            request_id=request_id,
        )

    storage = _get_storage(config, workspace)

    session_id = storage.get_active_session(spec_id)
    if not session_id:
        return _session_not_found_response("session-end", request_id, spec_id)

    session = storage.load(session_id)
    if not session:
        return _session_not_found_response("session-end", request_id, spec_id)

    # Validate state transition (any non-terminal -> ended)
    if session.status in {SessionStatus.COMPLETED, SessionStatus.ENDED}:
        return _invalid_transition_response(
            action="session-end",
            request_id=request_id,
            current_status=session.status.value,
            target_status="ended",
            reason="Session is already in terminal state",
        )

    # Update session
    session.status = SessionStatus.ENDED
    session.updated_at = datetime.now(timezone.utc)

    storage.save(session)

    # Remove pointer
    storage.remove_active_session(spec_id)

    # Write journal
    _write_session_journal(
        spec_id=spec_id,
        action="end",
        summary=f"Session ended (was {session.status.value})",
        session_id=session.id,
        workspace=workspace,
    )

    logger.info("Ended session %s for spec %s", session.id, spec_id)

    return _build_session_response(session, request_id)


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

    # Validate limit
    limit = max(1, min(limit, 100))

    storage = _get_storage(config, workspace)

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
            error_code=ErrorCode.VALIDATION_ERROR,
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
# Session Rebase Handler
# =============================================================================

ERROR_REBASE_COMPLETED_TASKS_REMOVED = "REBASE_COMPLETED_TASKS_REMOVED"


def _find_backup_with_hash(
    spec_id: str,
    target_hash: str,
    workspace: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Find a backup spec file that matches the target hash.

    Args:
        spec_id: Spec ID to search for
        target_hash: Target spec_structure_hash to match
        workspace: Workspace path

    Returns:
        Parsed spec data from matching backup, or None if not found
    """
    ws_path = Path(workspace) if workspace else Path.cwd()
    backup_dir = ws_path / "specs" / ".backups" / spec_id

    if not backup_dir.exists():
        return None

    # List all backup files and check their hashes
    backup_files = sorted(backup_dir.glob("*.json"), reverse=True)

    for backup_file in backup_files:
        if backup_file.name == "latest.json":
            continue
        try:
            backup_data = json.loads(backup_file.read_text())
            backup_hash = compute_spec_structure_hash(backup_data)
            if backup_hash == target_hash:
                logger.debug("Found matching backup: %s", backup_file)
                return backup_data
        except Exception as e:
            logger.debug("Failed to read backup %s: %s", backup_file, e)
            continue

    return None


def _handle_session_rebase(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    force: bool = False,
    workspace: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Handle session-rebase action.

    Rebases a paused or failed session to spec changes.
    Computes structural diff, validates completed tasks, and updates session state.

    Valid source states: paused, failed
    INVALID_STATE_TRANSITION for other states.

    Behavior:
    - No-change: returns no_change, transitions to running
    - Hash differs: compute structural diff
    - Completed task removal guarded (REBASE_COMPLETED_TASKS_REMOVED) unless force=true
    - force=true removes missing completed task IDs and adjusts counters

    Args:
        config: Server configuration
        spec_id: Spec ID of the session
        force: Force rebase even if completed tasks were removed
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict with rebase result or error
    """
    request_id = _request_id()

    if not spec_id:
        return _validation_error(
            action="session-rebase",
            field="spec_id",
            message="spec_id is required",
            request_id=request_id,
        )

    storage = _get_storage(config, workspace)

    session_id = storage.get_active_session(spec_id)
    if not session_id:
        return _session_not_found_response("session-rebase", request_id, spec_id)

    session = storage.load(session_id)
    if not session:
        return _session_not_found_response("session-rebase", request_id, spec_id)

    # Only allow rebase for paused or failed sessions
    if session.status not in {SessionStatus.PAUSED, SessionStatus.FAILED}:
        return _invalid_transition_response(
            action="session-rebase",
            request_id=request_id,
            current_status=session.status.value,
            target_status="rebase",
            reason="Only paused or failed sessions can be rebased",
        )

    # Load current spec
    ws_path = Path(workspace) if workspace else Path.cwd()
    specs_dir = ws_path / "specs"
    spec_path = specs_dir / f"{spec_id}.json"

    if not spec_path.exists():
        return asdict(error_response(
            f"Spec not found: {spec_id}",
            error_code=ErrorCode.NOT_FOUND,
            error_type=ErrorType.NOT_FOUND,
            request_id=request_id,
        ))

    try:
        current_spec_data = json.loads(spec_path.read_text())
        current_hash = compute_spec_structure_hash(current_spec_data)
        current_metadata = get_spec_file_metadata(spec_path)
    except Exception as e:
        return asdict(error_response(
            f"Failed to read spec: {e}",
            error_code=ErrorCode.INTERNAL_ERROR,
            error_type=ErrorType.INTERNAL,
            request_id=request_id,
        ))

    # Check if spec changed
    if current_hash == session.spec_structure_hash:
        # No structural changes - transition to running
        session.status = SessionStatus.RUNNING
        session.pause_reason = None
        session.failure_reason = None
        session.paused_at = None
        session.updated_at = datetime.now(timezone.utc)

        storage.save(session)

        rebase_result = RebaseResultDetail(
            result="no_change",
        )

        # Write journal
        _write_session_journal(
            spec_id=spec_id,
            action="rebase",
            summary="Session rebase: no structural changes detected",
            session_id=session.id,
            workspace=workspace,
            metadata={"result": "no_change"},
        )

        return _build_session_response(session, request_id, rebase_result=rebase_result, include_resume_context=True)

    # Compute structural diff - try to find old structure from backups
    old_spec_data = _find_backup_with_hash(spec_id, session.spec_structure_hash, workspace)

    if old_spec_data:
        diff = compute_structural_diff(old_spec_data, current_spec_data)
    else:
        # Fallback: can't compute full diff without old structure
        # Create a minimal diff indicating unknown changes
        diff = StructuralDiff()
        logger.warning(
            "Could not find backup matching hash %s for spec %s, using minimal diff",
            session.spec_structure_hash[:16],
            spec_id,
        )

    # Check for completed tasks in removed tasks
    removed_completed_tasks = [
        task_id for task_id in session.completed_task_ids
        if task_id in diff.removed_tasks
    ]

    if removed_completed_tasks and not force:
        return asdict(error_response(
            "Cannot rebase: completed tasks would be removed",
            error_code=ErrorCode.VALIDATION_ERROR,
            error_type=ErrorType.VALIDATION,
            request_id=request_id,
            details={
                "action": "session-rebase",
                "error_code": ERROR_REBASE_COMPLETED_TASKS_REMOVED,
                "removed_completed_tasks": removed_completed_tasks,
                "hint": "Use force=true to remove these completed tasks and adjust counters",
            },
        ))

    # Apply rebase
    tasks_removed_count = 0
    if removed_completed_tasks and force:
        # Remove missing completed task IDs
        for task_id in removed_completed_tasks:
            if task_id in session.completed_task_ids:
                session.completed_task_ids.remove(task_id)
                session.counters.tasks_completed = max(0, session.counters.tasks_completed - 1)
                tasks_removed_count += 1

    # Update session state
    session.spec_structure_hash = current_hash
    session.spec_file_mtime = current_metadata.mtime if current_metadata else None
    session.spec_file_size = current_metadata.file_size if current_metadata else None
    session.status = SessionStatus.RUNNING
    session.pause_reason = None
    session.failure_reason = None
    session.paused_at = None
    session.updated_at = datetime.now(timezone.utc)

    storage.save(session)

    # Build rebase result
    rebase_result = RebaseResultDetail(
        result="success",
        added_phases=diff.added_phases,
        removed_phases=diff.removed_phases,
        added_tasks=diff.added_tasks,
        removed_tasks=diff.removed_tasks,
        completed_tasks_removed=tasks_removed_count if tasks_removed_count > 0 else None,
    )

    # Write journal
    _write_session_journal(
        spec_id=spec_id,
        action="rebase",
        summary=f"Session rebased: +{len(diff.added_tasks)}/-{len(diff.removed_tasks)} tasks",
        session_id=session.id,
        workspace=workspace,
        metadata={
            "old_hash": session.spec_structure_hash[:16],
            "new_hash": current_hash[:16],
            "added_phases": diff.added_phases,
            "removed_phases": diff.removed_phases,
            "added_tasks": diff.added_tasks,
            "removed_tasks": diff.removed_tasks,
            "force": force,
            "tasks_removed": tasks_removed_count,
        },
    )

    logger.info(
        "Rebased session %s for spec %s: +%d/-%d phases, +%d/-%d tasks",
        session.id,
        spec_id,
        len(diff.added_phases),
        len(diff.removed_phases),
        len(diff.added_tasks),
        len(diff.removed_tasks),
    )

    return _build_session_response(session, request_id, rebase_result=rebase_result, include_resume_context=True)


# =============================================================================
# Session Heartbeat Handler
# =============================================================================


def _handle_session_heartbeat(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
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
        context_usage_pct: Current context usage percentage
        estimated_tokens_used: Estimated tokens used
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict confirming heartbeat or error
    """
    request_id = _request_id()

    if not spec_id:
        return _validation_error(
            action="session-heartbeat",
            field="spec_id",
            message="spec_id is required",
            request_id=request_id,
        )

    # Validate context_usage_pct
    if context_usage_pct is not None:
        if not isinstance(context_usage_pct, int) or not (0 <= context_usage_pct <= 100):
            return _validation_error(
                action="session-heartbeat",
                field="context_usage_pct",
                message="context_usage_pct must be an integer between 0 and 100",
                request_id=request_id,
            )

    storage = _get_storage(config, workspace)

    session_id = storage.get_active_session(spec_id)
    if not session_id:
        return _session_not_found_response("session-heartbeat", request_id, spec_id)

    session = storage.load(session_id)
    if not session:
        return _session_not_found_response("session-heartbeat", request_id, spec_id)

    # Update heartbeat
    now = datetime.now(timezone.utc)
    session.context.last_heartbeat_at = now
    session.updated_at = now

    if context_usage_pct is not None:
        session.context.context_usage_pct = context_usage_pct
    if estimated_tokens_used is not None:
        session.context.estimated_tokens_used = estimated_tokens_used

    storage.save(session)

    # Check if context threshold exceeded
    warnings = []
    if session.context.context_usage_pct >= session.limits.context_threshold_pct:
        warnings.append(f"Context usage at {session.context.context_usage_pct}%")

    response_data = {
        "session_id": session.id,
        "heartbeat_at": now.isoformat(),
        "context_usage_pct": session.context.context_usage_pct,
    }

    return asdict(success_response(
        data=response_data,
        request_id=request_id,
        meta={"warnings": warnings} if warnings else None,
    ))


# =============================================================================
# Session Reset Handler
# =============================================================================


def _handle_session_reset(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    session_id: Optional[str] = None,
    workspace: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Handle session-reset action.

    Resets a failed session to allow retry.
    Only failed sessions can be reset.

    Args:
        config: Server configuration
        spec_id: Spec ID (optional if session_id provided)
        session_id: Session ID to reset
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict confirming reset or error
    """
    request_id = _request_id()

    if not spec_id and not session_id:
        return _validation_error(
            action="session-reset",
            field="session_id",
            message="spec_id or session_id is required",
            request_id=request_id,
        )

    storage = _get_storage(config, workspace)

    # Load session
    session = None
    if session_id:
        session = storage.load(session_id)
    elif spec_id:
        active_id = storage.get_active_session(spec_id)
        if active_id:
            session = storage.load(active_id)

    if not session:
        return _session_not_found_response("session-reset", request_id, spec_id)

    # Only allow reset for failed sessions
    if session.status != SessionStatus.FAILED:
        return asdict(error_response(
            f"Cannot reset session in {session.status.value} state",
            error_code=ErrorCode.VALIDATION_ERROR,
            error_type=ErrorType.VALIDATION,
            request_id=request_id,
            details={
                "action": "session-reset",
                "current_status": session.status.value,
                "hint": "Only failed sessions can be reset",
            },
        ))

    # Reset session to paused state
    session.status = SessionStatus.PAUSED
    session.pause_reason = PauseReason.USER
    session.failure_reason = None
    session.updated_at = datetime.now(timezone.utc)
    session.paused_at = datetime.now(timezone.utc)
    session.counters.consecutive_errors = 0

    storage.save(session)

    # Write journal
    _write_session_journal(
        spec_id=session.spec_id,
        action="reset",
        summary="Session reset from failed state",
        session_id=session.id,
        workspace=workspace,
    )

    logger.info("Reset session %s for spec %s", session.id, session.spec_id)

    return _build_session_response(session, request_id)
