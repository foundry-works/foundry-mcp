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
from typing import Any, Dict, List, Optional

from ulid import ULID

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.autonomy.memory import (
    ActiveSessionLookupResult,
    ListSessionsResult,
)
from foundry_mcp.core.autonomy.models import (
    AutonomousSessionState,
    SessionStatus,
    TERMINAL_STATUSES,
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
    PhaseGateStatus,
    PhaseGateRecord,
    RebaseResultDetail,
    GatePolicy,
    OverrideReasonCode,
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
from foundry_mcp.core.spec import load_spec, resolve_spec_file
from foundry_mcp.core.authorization import (
    get_server_role,
    check_action_allowed,
    Role,
)

from foundry_mcp.tools.unified.task_handlers._helpers import (
    _get_storage,
    _request_id,
    _resolve_session,
    _session_not_found_response,
    _validation_error,
    _resolve_specs_dir,
    _is_feature_enabled,
    _feature_disabled_response,
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
        error_code=ErrorCode.INVALID_STATE_TRANSITION,
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

    Delegates to the shared implementation in models.py to avoid duplication.
    """
    from foundry_mcp.core.autonomy.models import compute_effective_status
    return compute_effective_status(session)


def _build_resume_context(
    session: AutonomousSessionState,
    workspace: Optional[str] = None,
) -> Optional[ResumeContext]:
    """Build resume context from session state and spec data.

    Loads the spec to extract completed/pending tasks, completed phases,
    and journal availability for providing context on resume/rebase.

    Args:
        session: Session state
        workspace: Workspace path

    Returns:
        Populated ResumeContext or None if spec cannot be loaded
    """
    ws_path = Path(workspace) if workspace else Path.cwd()
    specs_dir = ws_path / "specs"

    spec_data = load_spec(session.spec_id, specs_dir)
    if spec_data is None:
        return None

    spec_title = spec_data.get("title") or spec_data.get("name")

    # Extract phase and task information
    phases = spec_data.get("phases", [])
    if not isinstance(phases, list):
        phases = []

    completed_tasks: list[CompletedTaskSummary] = []
    pending_tasks_in_phase: list[PendingTaskSummary] = []
    completed_phases: list[CompletedPhaseSummary] = []

    active_phase_title = None

    for phase in phases:
        if not isinstance(phase, dict):
            continue

        phase_id = phase.get("id", "")
        phase_title = phase.get("title")

        # Track active phase title
        if phase_id == session.active_phase_id:
            active_phase_title = phase_title

        # Check phase completion via gate records
        gate_record = session.phase_gates.get(phase_id)
        if gate_record and gate_record.status in (
            PhaseGateStatus.PASSED,
            PhaseGateStatus.WAIVED,
        ):
            completed_phases.append(CompletedPhaseSummary(
                phase_id=phase_id,
                title=phase_title,
                gate_status=gate_record.status,
            ))

        # Process tasks in this phase
        tasks = phase.get("tasks", [])
        if not isinstance(tasks, list):
            continue

        for task in tasks:
            if not isinstance(task, dict):
                continue
            task_id = task.get("id", "")
            task_title = task.get("title", task_id)

            if task_id in session.completed_task_ids:
                completed_tasks.append(CompletedTaskSummary(
                    task_id=task_id,
                    title=task_title,
                    phase_id=phase_id,
                    files_touched=None,  # Not tracked in spec data
                ))
            elif phase_id == session.active_phase_id:
                task_type = task.get("type", "task")
                if task_type in ("task", "verify"):
                    pending_tasks_in_phase.append(PendingTaskSummary(
                        task_id=task_id,
                        title=task_title,
                    ))

    # Cap recent completed tasks at 10 (most recent = last appended)
    recent_completed = completed_tasks[-10:] if len(completed_tasks) > 10 else completed_tasks

    # Check journal availability
    journal_available = bool(spec_data.get("journal"))
    journal_hint = (
        f"Use journal(action='list', spec_id='{session.spec_id}') to view journal entries"
        if journal_available else None
    )

    return ResumeContext(
        spec_id=session.spec_id,
        spec_title=spec_title,
        active_phase_id=session.active_phase_id,
        active_phase_title=active_phase_title,
        completed_task_count=len(session.completed_task_ids),
        recent_completed_tasks=recent_completed,
        completed_phases=completed_phases,
        pending_tasks_in_phase=pending_tasks_in_phase,
        last_pause_reason=session.pause_reason,
        journal_available=journal_available,
        journal_hint=journal_hint,
    )


def _build_session_response(
    session: AutonomousSessionState,
    request_id: str,
    include_resume_context: bool = False,
    rebase_result: Optional[RebaseResultDetail] = None,
    workspace: Optional[str] = None,
) -> dict:
    """Build standard session response data."""
    effective_status = _compute_effective_status(session)

    # Build resume context if requested
    resume_context = None
    if include_resume_context:
        resume_context = _build_resume_context(session, workspace)

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
        resume_context=resume_context,
        rebase_result=rebase_result,
    )

    # Include effective status if derived
    data = response_data.model_dump(mode="json", by_alias=True)
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

    Uses the existing journal system (add_journal_entry + save_spec)
    to avoid direct spec file mutation and associated race conditions.

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
        from foundry_mcp.core.journal import add_journal_entry
        from foundry_mcp.core.spec import load_spec, save_spec

        ws_path = Path(workspace) if workspace else Path.cwd()
        specs_dir = ws_path / "specs"

        spec_data = load_spec(spec_id, specs_dir)
        if spec_data is None:
            logger.warning("Spec not found for journal: %s", spec_id)
            return False

        add_journal_entry(
            spec_data,
            title=f"Session {action}",
            content=summary,
            entry_type="session",
            author="autonomy",
            metadata={
                "session_id": session_id,
                "action": action,
                **(metadata or {}),
            },
        )

        save_spec(spec_id, spec_data, specs_dir)
        logger.debug("Wrote session journal entry for %s", session_id)
        return True

    except Exception as e:
        logger.warning("Failed to write session journal: %s", e)
        return False


def _compute_required_gates_from_spec(
    spec_data: Dict[str, Any],
    session: AutonomousSessionState,
) -> Dict[str, List[str]]:
    """Compute required gate types for each phase from spec structure.

    Derives the minimum required gate types based on phase composition:
    - Phases with verification tasks require "fidelity" gate
    - Phases without verification tasks require "manual_review" gate
    - Spec authors can expand but minimum gate types cannot be removed

    Args:
        spec_data: Parsed spec data with phases
        session: Current session state (for existing requirements)

    Returns:
        Dict mapping phase_id -> list of required gate types
    """
    required_gates: Dict[str, List[str]] = {}

    phases = spec_data.get("phases", [])
    if not isinstance(phases, list):
        return required_gates

    for phase in phases:
        if not isinstance(phase, dict):
            continue

        phase_id = phase.get("id", "")
        if not phase_id:
            continue

        # Check if phase has verification tasks
        tasks = phase.get("tasks", [])
        has_verify_task = False

        if isinstance(tasks, list):
            for task in tasks:
                if isinstance(task, dict) and task.get("type") == "verify":
                    has_verify_task = True
                    break

        # Minimum gate type based on phase composition
        if has_verify_task:
            minimum_gate = "fidelity"
        else:
            minimum_gate = "manual_review"

        # Preserve existing requirements if they include the minimum
        existing = session.required_phase_gates.get(phase_id, [])
        if minimum_gate in existing:
            # Keep existing (may have additional gates from spec author)
            required_gates[phase_id] = existing
        else:
            # Ensure minimum is present
            required_gates[phase_id] = [minimum_gate]

    return required_gates


def _reconcile_gates_on_rebase(
    session: AutonomousSessionState,
    new_required_gates: Dict[str, List[str]],
    diff: "StructuralDiff",
    *,
    old_spec_data: Optional[Dict[str, Any]] = None,
    new_spec_data: Optional[Dict[str, Any]] = None,
    unknown_structural_changes: bool = False,
) -> None:
    """Reconcile required and satisfied gates after rebase.

    Preserves satisfied gates for unchanged phases, and clears satisfied gates for
    phases impacted by structural edits.

    Args:
        session: Session state to update
        new_required_gates: New required gates computed from updated spec
        diff: Structural diff from rebase
        old_spec_data: Previous spec structure (used to map removed/moved tasks)
        new_spec_data: Current spec structure (used to map added/moved tasks)
        unknown_structural_changes: True when structure hash changed but no old
            structure snapshot is available. In this case, gate satisfaction is
            conservatively reset for all phases.
    """
    def _task_phase_map(spec_data: Optional[Dict[str, Any]]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        if not isinstance(spec_data, dict):
            return mapping

        phases = spec_data.get("phases", [])
        if not isinstance(phases, list):
            return mapping

        for phase in phases:
            if not isinstance(phase, dict):
                continue
            phase_id = phase.get("id")
            if not isinstance(phase_id, str) or not phase_id:
                continue
            tasks = phase.get("tasks", [])
            if not isinstance(tasks, list):
                continue
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                task_id = task.get("id")
                if isinstance(task_id, str) and task_id:
                    mapping[task_id] = phase_id

        return mapping

    old_satisfied = session.satisfied_gates.copy()
    session.required_phase_gates = new_required_gates

    affected_phases = set(diff.added_phases)
    affected_phases.update(diff.removed_phases)

    new_task_phase = _task_phase_map(new_spec_data)
    old_task_phase = _task_phase_map(old_spec_data)

    for task_id in diff.added_tasks:
        phase_id = new_task_phase.get(task_id)
        if phase_id:
            affected_phases.add(phase_id)

    for task_id in diff.removed_tasks:
        phase_id = old_task_phase.get(task_id)
        if phase_id:
            affected_phases.add(phase_id)

    # Detect renamed/moved task placement (same task id, different phase).
    for task_id in set(old_task_phase).intersection(new_task_phase):
        old_phase = old_task_phase[task_id]
        new_phase = new_task_phase[task_id]
        if old_phase != new_phase:
            affected_phases.add(old_phase)
            affected_phases.add(new_phase)

    if unknown_structural_changes:
        affected_phases.update(new_required_gates.keys())

    preserved_satisfied: Dict[str, List[str]] = {}
    for phase_id, required in new_required_gates.items():
        if phase_id in affected_phases:
            continue
        old_satisfied_for_phase = old_satisfied.get(phase_id, [])
        preserved = [gate for gate in old_satisfied_for_phase if gate in required]
        if preserved:
            preserved_satisfied[phase_id] = preserved

    session.satisfied_gates = preserved_satisfied


# =============================================================================
# Session Start Handler
# =============================================================================


def _handle_session_start(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    force: bool = False,
    workspace: Optional[str] = None,
    # Session configuration parameters
    gate_policy: Optional[str] = None,
    max_tasks_per_session: Optional[int] = None,
    max_consecutive_errors: Optional[int] = None,
    context_threshold_pct: Optional[int] = None,
    stop_on_phase_completion: Optional[bool] = None,
    auto_retry_fidelity_gate: Optional[bool] = None,
    heartbeat_stale_minutes: Optional[int] = None,
    heartbeat_grace_minutes: Optional[int] = None,
    step_stale_minutes: Optional[int] = None,
    max_fidelity_review_cycles_per_phase: Optional[int] = None,
    avg_pct_per_step: Optional[int] = None,
    context_staleness_threshold: Optional[int] = None,
    context_staleness_penalty_pct: Optional[int] = None,
    enforce_autonomy_write_lock: Optional[bool] = None,
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

    # Feature flag check - fail-closed
    if not _is_feature_enabled(config, "autonomy_sessions"):
        return _feature_disabled_response("session-start", request_id)

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
            error_code=ErrorCode.LOCK_TIMEOUT,
            error_type=ErrorType.UNAVAILABLE,
            request_id=request_id,
            details={"spec_id": spec_id, "action": "session-start"},
        ))

    with spec_lock:
        # Step 2: Check pointer for existing session
        existing_session_id = storage.get_active_session(spec_id)
        if existing_session_id:
            existing_session = storage.load(existing_session_id)
            if existing_session and existing_session.status not in TERMINAL_STATUSES:
                # Check idempotency key match
                if idempotency_key and existing_session.idempotency_key == idempotency_key:
                    # Idempotent - return existing session
                    return _build_session_response(existing_session, request_id)

                if force:
                    # Force-end existing session before creating new one
                    existing_session.status = SessionStatus.ENDED
                    existing_session.updated_at = datetime.now(timezone.utc)
                    existing_session.state_version += 1
                    storage.save(existing_session)
                    storage.remove_active_session(spec_id)

                    _write_session_journal(
                        spec_id=spec_id,
                        action="end",
                        summary="Session force-ended by new session start",
                        session_id=existing_session.id,
                        workspace=workspace,
                        metadata={"reason": "force_replaced"},
                    )

                    logger.info(
                        "Force-ended existing session %s for spec %s",
                        existing_session.id,
                        spec_id,
                    )
                else:
                    return asdict(error_response(
                        "Active session already exists for this spec",
                        error_code=ErrorCode.SPEC_SESSION_EXISTS,
                        error_type=ErrorType.CONFLICT,
                        request_id=request_id,
                        details={
                            "spec_id": spec_id,
                            "existing_session_id": existing_session_id,
                            "existing_status": existing_session.status.value,
                            "hint": "End existing session or use force=true to replace it",
                        },
                    ))

        # Step 3: Load spec and compute hash
        ws_path = Path(workspace) if workspace else Path.cwd()
        specs_dir = ws_path / "specs"
        spec_path = resolve_spec_file(spec_id, specs_dir)

        if not spec_path:
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

        # Step 4: Create session state with caller-provided configuration
        now = datetime.now(timezone.utc)

        # Build limits from payload, falling back to defaults
        limits_kwargs = {}
        if max_tasks_per_session is not None:
            limits_kwargs["max_tasks_per_session"] = max_tasks_per_session
        if max_consecutive_errors is not None:
            limits_kwargs["max_consecutive_errors"] = max_consecutive_errors
        if context_threshold_pct is not None:
            limits_kwargs["context_threshold_pct"] = context_threshold_pct
        if heartbeat_stale_minutes is not None:
            limits_kwargs["heartbeat_stale_minutes"] = heartbeat_stale_minutes
        if heartbeat_grace_minutes is not None:
            limits_kwargs["heartbeat_grace_minutes"] = heartbeat_grace_minutes
        if step_stale_minutes is not None:
            limits_kwargs["step_stale_minutes"] = step_stale_minutes
        if max_fidelity_review_cycles_per_phase is not None:
            limits_kwargs["max_fidelity_review_cycles_per_phase"] = max_fidelity_review_cycles_per_phase
        if avg_pct_per_step is not None:
            limits_kwargs["avg_pct_per_step"] = avg_pct_per_step
        if context_staleness_threshold is not None:
            limits_kwargs["context_staleness_threshold"] = context_staleness_threshold
        if context_staleness_penalty_pct is not None:
            limits_kwargs["context_staleness_penalty_pct"] = context_staleness_penalty_pct

        # Build stop conditions from payload
        stop_kwargs = {}
        if stop_on_phase_completion is not None:
            stop_kwargs["stop_on_phase_completion"] = stop_on_phase_completion
        if auto_retry_fidelity_gate is not None:
            stop_kwargs["auto_retry_fidelity_gate"] = auto_retry_fidelity_gate

        # Resolve gate policy
        resolved_gate_policy = GatePolicy.STRICT
        if gate_policy is not None:
            try:
                resolved_gate_policy = GatePolicy(gate_policy)
            except ValueError:
                return _validation_error(
                    action="session-start",
                    field="gate_policy",
                    message=f"Invalid gate_policy: {gate_policy}. Must be one of: strict, lenient, manual",
                    request_id=request_id,
                )

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
            limits=SessionLimits(**limits_kwargs),
            stop_conditions=StopConditions(**stop_kwargs),
            context=SessionContext(),
            write_lock_enforced=enforce_autonomy_write_lock if enforce_autonomy_write_lock is not None else True,
            gate_policy=resolved_gate_policy,
        )

        # Compute required gates from spec structure
        session.required_phase_gates = _compute_required_gates_from_spec(spec_data, session)

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
            # Rollback: delete state and index entry before releasing lock
            logger.warning("Journal write failed, rolling back session %s", session.id)
            storage.delete(session.id)
            storage.remove_active_session(spec_id)
            return asdict(error_response(
                "Failed to write session journal",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                request_id=request_id,
                details={"hint": "Session creation rolled back"},
            ))

        # Step 7: Write pointer
        storage.set_active_session(spec_id, session.id)

        # Step 7.5: Auto-verify audit chain for existing trails (P2.1)
        audit_warning = None
        try:
            from foundry_mcp.core.autonomy.audit import get_ledger_path, verify_chain
            from pathlib import Path as _Path

            ws_path = _Path(workspace) if workspace else _Path.cwd()
            ledger_path = get_ledger_path(spec_id=spec_id, workspace_path=ws_path)

            if ledger_path.exists():
                result = verify_chain(spec_id=spec_id, workspace_path=ws_path)
                if not result.valid:
                    audit_warning = {
                        "code": "AUDIT_CHAIN_BROKEN",
                        "divergence_point": result.divergence_point,
                        "divergence_type": result.divergence_type,
                        "detail": result.divergence_detail,
                    }
                    logger.warning(
                        "Audit chain verification failed for spec %s: %s at sequence %d",
                        spec_id,
                        result.divergence_type,
                        result.divergence_point,
                    )
        except Exception as e:
            # Non-blocking - audit verification failure should not prevent session start
            logger.debug("Audit verification skipped (non-critical): %s", e)

        # Lock released by context manager

    # Step 8: GC (best effort)
    try:
        storage.cleanup_expired()
    except Exception as e:
        logger.debug("GC failed (non-critical): %s", e)

    logger.info("Started session %s for spec %s", session.id, spec_id)

    response = _build_session_response(session, request_id)

    # Include audit warning if verification failed (P2.1)
    if audit_warning:
        response["data"]["audit_warning"] = audit_warning

    return response


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

    storage = _get_storage(config, workspace)

    session, err = _resolve_session(storage, "session-status", request_id, session_id, spec_id)
    if err:
        return err

    return _build_session_response(session, request_id)


# =============================================================================
# Session Pause Handler
# =============================================================================


def _handle_session_pause(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    session_id: Optional[str] = None,
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
        session_id: Session ID (optional, alternative to spec_id)
        reason: Optional reason for pausing
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict with updated session state or error
    """
    request_id = _request_id()

    # Feature flag check - fail-closed
    if not _is_feature_enabled(config, "autonomy_sessions"):
        return _feature_disabled_response("session-pause", request_id)

    storage = _get_storage(config, workspace)

    session, err = _resolve_session(storage, "session-pause", request_id, session_id, spec_id)
    if err:
        return err

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
    now = datetime.now(timezone.utc)
    session.status = SessionStatus.PAUSED
    try:
        pause_reason = PauseReason(reason) if reason else PauseReason.USER
    except ValueError:
        pause_reason = PauseReason.USER
    session.pause_reason = pause_reason
    session.updated_at = now
    session.paused_at = now
    session.state_version += 1

    storage.save(session)

    # Write journal
    _write_session_journal(
        spec_id=session.spec_id,
        action="pause",
        summary=f"Session paused: {reason or 'user request'}",
        session_id=session.id,
        workspace=workspace,
    )

    logger.info("Paused session %s for spec %s", session.id, session.spec_id)

    return _build_session_response(session, request_id)


# =============================================================================
# Session Resume Handler
# =============================================================================


def _handle_session_resume(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    session_id: Optional[str] = None,
    acknowledge_gate_review: Optional[bool] = None,
    acknowledged_gate_attempt_id: Optional[str] = None,
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
        session_id: Session ID (optional, alternative to spec_id)
        acknowledge_gate_review: Must be True when resuming from gate_review_required
        acknowledged_gate_attempt_id: Must match pending gate_attempt_id
        force: Force resume from failed state
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict with session state and next step or error
    """
    request_id = _request_id()

    # Feature flag check - fail-closed
    if not _is_feature_enabled(config, "autonomy_sessions"):
        return _feature_disabled_response("session-resume", request_id)

    storage = _get_storage(config, workspace)

    session, err = _resolve_session(storage, "session-resume", request_id, session_id, spec_id)
    if err:
        return err

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

        # Per ADR: if failure_reason == spec_structure_changed, recompute hash
        if session.failure_reason == FailureReason.SPEC_STRUCTURE_CHANGED:
            ws_path = Path(workspace) if workspace else Path.cwd()
            specs_dir = ws_path / "specs"
            spec_path = resolve_spec_file(session.spec_id, specs_dir)

            if spec_path:
                try:
                    current_spec_data = json.loads(spec_path.read_text())
                    current_hash = compute_spec_structure_hash(current_spec_data)
                    if current_hash != session.spec_structure_hash:
                        return asdict(error_response(
                            "Spec structure has changed since session was created. Use session-rebase to reconcile.",
                            error_code=ErrorCode.SPEC_REBASE_REQUIRED,
                            error_type=ErrorType.CONFLICT,
                            request_id=request_id,
                            details={
                                "action": "session-resume",
                                "spec_id": session.spec_id,
                                "old_hash": session.spec_structure_hash[:16],
                                "new_hash": current_hash[:16],
                                "hint": "Use session-rebase to reconcile spec structure changes before resuming",
                            },
                        ))
                except Exception as e:
                    logger.warning("Failed to validate spec structure on resume: %s", e)

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
        if not acknowledge_gate_review:
            return asdict(error_response(
                "Manual gate acknowledgment required",
                error_code=ErrorCode.MANUAL_GATE_ACK_REQUIRED,
                error_type=ErrorType.VALIDATION,
                request_id=request_id,
                details={
                    "action": "session-resume",
                    "gate_attempt_id": session.pending_manual_gate_ack.gate_attempt_id,
                    "phase_id": session.pending_manual_gate_ack.phase_id,
                    "hint": "Provide acknowledge_gate_review=true and acknowledged_gate_attempt_id to acknowledge",
                },
            ))

        if not acknowledged_gate_attempt_id or acknowledged_gate_attempt_id != session.pending_manual_gate_ack.gate_attempt_id:
            return asdict(error_response(
                "Invalid gate acknowledgment",
                error_code=ErrorCode.INVALID_GATE_ACK,
                error_type=ErrorType.VALIDATION,
                request_id=request_id,
                details={
                    "action": "session-resume",
                    "expected": session.pending_manual_gate_ack.gate_attempt_id,
                    "provided": acknowledged_gate_attempt_id,
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
    session.state_version += 1

    storage.save(session)

    # Write journal
    _write_session_journal(
        spec_id=session.spec_id,
        action="resume",
        summary=f"Session resumed{', forced from failed' if force else ''}",
        session_id=session.id,
        workspace=workspace,
        metadata={"acknowledge_gate_review": acknowledge_gate_review, "acknowledged_gate_attempt_id": acknowledged_gate_attempt_id, "force": force},
    )

    logger.info("Resumed session %s for spec %s", session.id, session.spec_id)

    return _build_session_response(session, request_id, include_resume_context=True, workspace=workspace)


# =============================================================================
# Session End Handler
# =============================================================================


def _handle_session_end(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    session_id: Optional[str] = None,
    reason_code: Optional[str] = None,
    reason_detail: Optional[str] = None,
    workspace: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Handle session-end action.

    Ends an autonomous session (terminal state).
    Valid transitions: running/paused/failed -> ended

    Requires reason_code (OverrideReasonCode enum) for audit trail.

    Args:
        config: Server configuration
        spec_id: Spec ID of the session
        session_id: Session ID (optional, alternative to spec_id)
        reason_code: Required structured reason code (OverrideReasonCode enum)
        reason_detail: Optional free-text detail for reason
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict confirming session ended or error
    """
    request_id = _request_id()

    # Feature flag check - fail-closed
    if not _is_feature_enabled(config, "autonomy_sessions"):
        return _feature_disabled_response("session-end", request_id)

    # Validate reason_code is provided
    if not reason_code:
        return _validation_error(
            action="session-end",
            field="reason_code",
            message="reason_code is required for session-end",
            request_id=request_id,
        )

    # Validate reason_code is a valid OverrideReasonCode
    try:
        validated_reason_code = OverrideReasonCode(reason_code)
    except ValueError:
        valid_codes = [e.value for e in OverrideReasonCode]
        return _validation_error(
            action="session-end",
            field="reason_code",
            message=f"Invalid reason_code: {reason_code}. Must be one of: {', '.join(valid_codes)}",
            request_id=request_id,
        )

    storage = _get_storage(config, workspace)

    session, err = _resolve_session(storage, "session-end", request_id, session_id, spec_id)
    if err:
        return err

    # Validate state transition (any non-terminal -> ended)
    if session.status in {SessionStatus.COMPLETED, SessionStatus.ENDED}:
        return _invalid_transition_response(
            action="session-end",
            request_id=request_id,
            current_status=session.status.value,
            target_status="ended",
            reason="Session is already in terminal state",
        )

    # Capture previous status before mutation for journal
    previous_status = session.status
    current_role = get_server_role()

    # Update session
    session.status = SessionStatus.ENDED
    session.updated_at = datetime.now(timezone.utc)
    session.state_version += 1

    storage.save(session)

    # Remove pointer
    storage.remove_active_session(session.spec_id)

    # Write journal with reason code and role
    _write_session_journal(
        spec_id=session.spec_id,
        action="end",
        summary=f"Session ended (was {previous_status.value}): {validated_reason_code.value}",
        session_id=session.id,
        workspace=workspace,
        metadata={
            "reason_code": validated_reason_code.value,
            "reason_detail": reason_detail,
            "server_role": current_role,
            "previous_status": previous_status.value,
        },
    )

    logger.info("Ended session %s for spec %s: %s", session.id, session.spec_id, validated_reason_code.value)

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

    # Feature flag check - fail-closed
    if not _is_feature_enabled(config, "autonomy_sessions"):
        return _feature_disabled_response("session-list", request_id)

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
# Session Rebase Handler
# =============================================================================

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
    session_id: Optional[str] = None,
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
        session_id: Session ID (optional, alternative to spec_id)
        force: Force rebase even if completed tasks were removed
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict with rebase result or error
    """
    request_id = _request_id()

    # Feature flag check - fail-closed
    if not _is_feature_enabled(config, "autonomy_sessions"):
        return _feature_disabled_response("session-rebase", request_id)

    storage = _get_storage(config, workspace)

    session, err = _resolve_session(storage, "session-rebase", request_id, session_id, spec_id)
    if err:
        return err

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
    spec_path = resolve_spec_file(session.spec_id, specs_dir)

    if not spec_path:
        return asdict(error_response(
            f"Spec not found: {session.spec_id}",
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
    now = datetime.now(timezone.utc)
    if current_hash == session.spec_structure_hash:
        # No structural changes - transition to running
        session.status = SessionStatus.RUNNING
        session.pause_reason = None
        session.failure_reason = None
        session.paused_at = None
        session.updated_at = now
        session.state_version += 1

        # Update required gates (spec author may have added gate requirements)
        session.required_phase_gates = _compute_required_gates_from_spec(current_spec_data, session)

        storage.save(session)

        rebase_result = RebaseResultDetail(
            result="no_change",
        )

        # Write journal
        _write_session_journal(
            spec_id=session.spec_id,
            action="rebase",
            summary="Session rebase: no structural changes detected",
            session_id=session.id,
            workspace=workspace,
            metadata={"result": "no_change"},
        )

        return _build_session_response(session, request_id, rebase_result=rebase_result, include_resume_context=True, workspace=workspace)

    # Compute structural diff - try to find old structure from backups
    old_spec_data = _find_backup_with_hash(session.spec_id, session.spec_structure_hash, workspace)

    if old_spec_data:
        diff = compute_structural_diff(old_spec_data, current_spec_data)
    else:
        # Fallback: can't compute full diff without old structure
        # Create a minimal diff indicating unknown changes
        diff = StructuralDiff()
        logger.warning(
            "Could not find backup matching hash %s for spec %s, using minimal diff",
            session.spec_structure_hash[:16],
            session.spec_id,
        )

    # Check for completed tasks in removed tasks
    removed_completed_tasks = [
        task_id for task_id in session.completed_task_ids
        if task_id in diff.removed_tasks
    ]

    if removed_completed_tasks and not force:
        return asdict(error_response(
            "Cannot rebase: completed tasks would be removed",
            error_code=ErrorCode.REBASE_COMPLETED_TASKS_REMOVED,
            error_type=ErrorType.VALIDATION,
            request_id=request_id,
            details={
                "action": "session-rebase",
                "removed_completed_tasks": removed_completed_tasks,
                "hint": "Use force=true to remove these completed tasks and adjust counters",
            },
        ))

    # Capture old hash before mutation for accurate journal metadata
    old_spec_hash = session.spec_structure_hash

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
    session.updated_at = now
    session.state_version += 1

    # Reconcile required and satisfied gates
    new_required_gates = _compute_required_gates_from_spec(current_spec_data, session)
    _reconcile_gates_on_rebase(
        session,
        new_required_gates,
        diff,
        old_spec_data=old_spec_data,
        new_spec_data=current_spec_data,
        unknown_structural_changes=old_spec_data is None,
    )

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
        spec_id=session.spec_id,
        action="rebase",
        summary=f"Session rebased: +{len(diff.added_tasks)}/-{len(diff.removed_tasks)} tasks",
        session_id=session.id,
        workspace=workspace,
        metadata={
            "old_hash": old_spec_hash[:16] if old_spec_hash else None,
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
        session.spec_id,
        len(diff.added_phases),
        len(diff.removed_phases),
        len(diff.added_tasks),
        len(diff.removed_tasks),
    )

    return _build_session_response(session, request_id, rebase_result=rebase_result, include_resume_context=True, workspace=workspace)


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
    if context_usage_pct is not None:
        if not isinstance(context_usage_pct, int) or not (0 <= context_usage_pct <= 100):
            return _validation_error(
                action="session-heartbeat",
                field="context_usage_pct",
                message="context_usage_pct must be an integer between 0 and 100",
                request_id=request_id,
            )

    storage = _get_storage(config, workspace)

    session, err = _resolve_session(storage, "session-heartbeat", request_id, session_id, spec_id)
    if err:
        return err

    # Update heartbeat
    now = datetime.now(timezone.utc)
    session.context.last_heartbeat_at = now
    session.updated_at = now
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

    storage.save(session)

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


# =============================================================================
# Session Reset Handler
# =============================================================================


def _handle_session_reset(
    *,
    config: ServerConfig,
    session_id: Optional[str] = None,
    reason_code: Optional[str] = None,
    reason_detail: Optional[str] = None,
    workspace: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Handle session-reset action.

    Resets a failed session to allow retry.
    Only failed sessions can be reset.

    Per ADR: reset always requires explicit session_id (no active-session lookup).
    Requires reason_code (OverrideReasonCode enum) for audit trail.

    Args:
        config: Server configuration
        session_id: Session ID to reset (required)
        reason_code: Required structured reason code (OverrideReasonCode enum)
        reason_detail: Optional free-text detail for reason
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict confirming reset or error
    """
    request_id = _request_id()

    # Feature flag check - fail-closed
    if not _is_feature_enabled(config, "autonomy_sessions"):
        return _feature_disabled_response("session-reset", request_id)

    # Validate reason_code is provided
    if not reason_code:
        return _validation_error(
            action="session-reset",
            field="reason_code",
            message="reason_code is required for session-reset",
            request_id=request_id,
        )

    # Validate reason_code is a valid OverrideReasonCode
    try:
        validated_reason_code = OverrideReasonCode(reason_code)
    except ValueError:
        valid_codes = [e.value for e in OverrideReasonCode]
        return _validation_error(
            action="session-reset",
            field="reason_code",
            message=f"Invalid reason_code: {reason_code}. Must be one of: {', '.join(valid_codes)}",
            request_id=request_id,
        )

    if not session_id:
        return _validation_error(
            action="session-reset",
            field="session_id",
            message="session_id is required for reset (no active-session lookup allowed)",
            request_id=request_id,
        )

    storage = _get_storage(config, workspace)

    session = storage.load(session_id)
    if not session:
        return _session_not_found_response("session-reset", request_id)

    # Only allow reset for failed sessions
    if session.status != SessionStatus.FAILED:
        return _invalid_transition_response(
            action="session-reset",
            request_id=request_id,
            current_status=session.status.value,
            target_status="deleted",
            reason="Only failed sessions can be reset",
        )

    # Per ADR: reset deletes session state file and removes index entry.
    # This is the escape hatch for corrupt state.
    deleted_session_id = session.id
    deleted_spec_id = session.spec_id
    current_role = get_server_role()

    storage.delete(session.id)
    storage.remove_active_session(session.spec_id)

    # Write journal with reason code and role
    _write_session_journal(
        spec_id=deleted_spec_id,
        action="reset",
        summary=f"Session {deleted_session_id} deleted via reset: {validated_reason_code.value}",
        session_id=deleted_session_id,
        workspace=workspace,
        metadata={
            "reason_code": validated_reason_code.value,
            "reason_detail": reason_detail,
            "server_role": current_role,
        },
    )

    logger.info("Reset (deleted) session %s for spec %s: %s", deleted_session_id, deleted_spec_id, validated_reason_code.value)

    return asdict(success_response(
        data={
            "session_id": deleted_session_id,
            "spec_id": deleted_spec_id,
            "result": "deleted",
            "reason_code": validated_reason_code.value,
        },
        request_id=request_id,
    ))


# =============================================================================
# Gate Waiver Handler
# =============================================================================


ERROR_GATE_WAIVER_DISABLED = "GATE_WAIVER_DISABLED"
ERROR_GATE_WAIVER_UNAUTHORIZED = "GATE_WAIVER_UNAUTHORIZED"
ERROR_GATE_NOT_FOUND = "GATE_NOT_FOUND"
ERROR_GATE_ALREADY_WAIVED = "GATE_ALREADY_WAIVED"
ERROR_GATE_ALREADY_PASSED = "GATE_ALREADY_PASSED"


def _handle_gate_waiver(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    session_id: Optional[str] = None,
    phase_id: Optional[str] = None,
    reason_code: Optional[str] = None,
    reason_detail: Optional[str] = None,
    workspace: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Handle gate-waiver action.

    Privileged break-glass override for required-gate invariant failures.
    Restricted to maintainer role and requires structured reason codes.

    Globally disabled unless allow_gate_waiver=true in config.
    Never available to autonomy_runner or observer roles.

    Args:
        config: Server configuration
        spec_id: Spec ID of the session
        session_id: Session ID (optional, alternative to spec_id)
        phase_id: Phase ID to waive the gate for
        reason_code: Required structured reason code (OverrideReasonCode enum)
        reason_detail: Optional free-text detail for waiver reason
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict with waiver result or error
    """
    request_id = _request_id()

    # Feature flag check - fail-closed
    if not _is_feature_enabled(config, "autonomy_sessions"):
        return _feature_disabled_response("gate-waiver", request_id)

    # Check if gate waiver is globally enabled
    if not config.autonomy_security.allow_gate_waiver:
        return asdict(error_response(
            "Gate waiver is disabled",
            error_code=ErrorCode.FORBIDDEN,
            error_type=ErrorType.AUTHORIZATION,
            request_id=request_id,
            details={
                "action": "gate-waiver",
                "hint": "Enable allow_gate_waiver=true in autonomy_security config",
            },
        ))

    # Role check - only maintainer can waive gates
    current_role = get_server_role()
    authz_result = check_action_allowed(current_role, "session", "gate-waiver")

    # Explicitly deny autonomy_runner and observer roles
    if current_role in (Role.AUTONOMY_RUNNER.value, Role.OBSERVER.value):
        return asdict(error_response(
            f"Gate waiver denied for role: {current_role}",
            error_code=ErrorCode.FORBIDDEN,
            error_type=ErrorType.AUTHORIZATION,
            request_id=request_id,
            details={
                "action": "gate-waiver",
                "configured_role": current_role,
                "required_role": Role.MAINTAINER.value,
                "hint": "Only maintainer role can waive required gates",
            },
        ))

    if not authz_result.allowed:
        return asdict(error_response(
            f"Gate waiver denied: {authz_result.denied_action}",
            error_code=ErrorCode.FORBIDDEN,
            error_type=ErrorType.AUTHORIZATION,
            request_id=request_id,
            details={
                "action": "gate-waiver",
                "configured_role": current_role,
                "required_role": authz_result.required_role,
            },
        ))

    # Validate required parameters
    if not phase_id:
        return _validation_error(
            action="gate-waiver",
            field="phase_id",
            message="phase_id is required",
            request_id=request_id,
        )

    if not reason_code:
        return _validation_error(
            action="gate-waiver",
            field="reason_code",
            message="reason_code is required for gate waiver",
            request_id=request_id,
        )

    # Validate reason_code is a valid OverrideReasonCode
    try:
        validated_reason_code = OverrideReasonCode(reason_code)
    except ValueError:
        valid_codes = [e.value for e in OverrideReasonCode]
        return _validation_error(
            action="gate-waiver",
            field="reason_code",
            message=f"Invalid reason_code: {reason_code}. Must be one of: {', '.join(valid_codes)}",
            request_id=request_id,
        )

    storage = _get_storage(config, workspace)

    session, err = _resolve_session(storage, "gate-waiver", request_id, session_id, spec_id)
    if err:
        return err

    # Check if phase gate exists
    if phase_id not in session.phase_gates:
        return asdict(error_response(
            f"No gate record found for phase: {phase_id}",
            error_code=ErrorCode.NOT_FOUND,
            error_type=ErrorType.NOT_FOUND,
            request_id=request_id,
            details={
                "action": "gate-waiver",
                "phase_id": phase_id,
                "available_phases": list(session.phase_gates.keys()),
            },
        ))

    gate_record = session.phase_gates[phase_id]

    # Check if gate is already passed
    if gate_record.status == PhaseGateStatus.PASSED:
        return asdict(error_response(
            f"Gate already passed for phase: {phase_id}",
            error_code=ErrorCode.VALIDATION_ERROR,
            error_type=ErrorType.VALIDATION,
            request_id=request_id,
            details={
                "action": "gate-waiver",
                "phase_id": phase_id,
                "current_status": gate_record.status.value,
                "hint": "Cannot waive a gate that has already passed",
            },
        ))

    # Check if gate is already waived
    if gate_record.status == PhaseGateStatus.WAIVED:
        return asdict(error_response(
            f"Gate already waived for phase: {phase_id}",
            error_code=ErrorCode.VALIDATION_ERROR,
            error_type=ErrorType.VALIDATION,
            request_id=request_id,
            details={
                "action": "gate-waiver",
                "phase_id": phase_id,
                "current_status": gate_record.status.value,
                "waiver_reason_code": gate_record.waiver_reason_code.value if gate_record.waiver_reason_code else None,
                "hint": "Gate is already waived",
            },
        ))

    # Apply waiver
    now = datetime.now(timezone.utc)
    gate_record.status = PhaseGateStatus.WAIVED
    gate_record.waiver_reason_code = validated_reason_code
    gate_record.waiver_reason_detail = reason_detail
    gate_record.waived_at = now
    gate_record.waived_by_role = current_role

    session.updated_at = now
    session.state_version += 1

    storage.save(session)

    # Write journal entry
    _write_session_journal(
        spec_id=session.spec_id,
        action="gate-waiver",
        summary=f"Gate waived for phase {phase_id}: {validated_reason_code.value}",
        session_id=session.id,
        workspace=workspace,
        metadata={
            "phase_id": phase_id,
            "reason_code": validated_reason_code.value,
            "reason_detail": reason_detail,
            "waived_by_role": current_role,
        },
    )

    logger.info(
        "Gate waived for phase %s in session %s by role %s: %s",
        phase_id,
        session.id,
        current_role,
        validated_reason_code.value,
    )

    response_data = {
        "session_id": session.id,
        "phase_id": phase_id,
        "gate_status": PhaseGateStatus.WAIVED.value,
        "reason_code": validated_reason_code.value,
        "reason_detail": reason_detail,
        "waived_at": now.isoformat(),
        "waived_by_role": current_role,
    }

    return asdict(success_response(
        data=response_data,
        request_id=request_id,
    ))
