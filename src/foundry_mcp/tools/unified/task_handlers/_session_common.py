"""Shared helpers, constants, and utilities for session handler modules.

Extracted from handlers_session.py to reduce file size while keeping
a single canonical location for cross-cutting session concerns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from foundry_mcp.core.autonomy.memory import (
    VersionConflictError,
)
from foundry_mcp.core.autonomy.models import (
    AutonomousSessionState,
    SessionStatus,
    SessionResponseData,
    ResumeContext,
    CompletedTaskSummary,
    CompletedPhaseSummary,
    PendingTaskSummary,
    PhaseGateStatus,
    RebaseResultDetail,
    ActivePhaseProgress,
    RetryCounters,
    derive_loop_signal,
)
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
)
from foundry_mcp.core.spec import load_spec

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


def _save_with_version_check(
    storage: Any,
    session: AutonomousSessionState,
    pre_mutation_version: int,
    action: str,
    request_id: str,
) -> Optional[dict]:
    """Save session with optimistic version check.

    Returns error response dict on conflict, or None on success.
    """
    try:
        storage.save(session, expected_version=pre_mutation_version)
        return None
    except VersionConflictError as exc:
        logger.warning("Version conflict on %s: %s", action, exc)
        return asdict(error_response(
            "Concurrent modification detected: session was modified by another actor",
            error_code=ErrorCode.VERSION_CONFLICT,
            error_type=ErrorType.CONFLICT,
            request_id=request_id,
            details={
                "action": action,
                "session_id": session.id,
                "expected_version": exc.expected_version,
                "actual_version": exc.actual_version,
                "remediation": "Reload session state and retry the operation",
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


def _load_spec_for_session(
    session: AutonomousSessionState,
    workspace: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load spec data for a session, returning None on lookup failure."""
    from foundry_mcp.core.authorization import validate_runner_path
    from foundry_mcp.core.errors.authorization import PathValidationError

    if workspace:
        try:
            ws_path = validate_runner_path(workspace, require_within_workspace=False)
        except PathValidationError:
            logger.warning("Workspace path validation failed in spec loader: %s", workspace)
            return None
    else:
        ws_path = Path.cwd()
    specs_dir = ws_path / "specs"
    return load_spec(session.spec_id, specs_dir)


def _build_active_phase_progress(
    session: AutonomousSessionState,
    spec_data: Optional[Dict[str, Any]],
) -> Optional[ActivePhaseProgress]:
    """Compute active phase progress from spec structure and session state."""
    if not session.active_phase_id or not isinstance(spec_data, dict):
        return None

    phases = spec_data.get("phases", [])
    if not isinstance(phases, list):
        return None

    active_phase: Optional[Dict[str, Any]] = None
    for phase in phases:
        if isinstance(phase, dict) and phase.get("id") == session.active_phase_id:
            active_phase = phase
            break
    if active_phase is None:
        return None

    phase_tasks = active_phase.get("tasks", [])
    if not isinstance(phase_tasks, list):
        phase_tasks = []

    completed_task_ids = set(session.completed_task_ids)
    total_tasks = 0
    completed_tasks = 0
    blocked_tasks = 0

    for task in phase_tasks:
        if not isinstance(task, dict):
            continue
        task_type = task.get("type")
        if task_type not in {"task", "verify", "subtask"}:
            continue

        total_tasks += 1
        task_id = task.get("id")
        status = str(task.get("status") or "").lower()
        is_completed = bool(
            isinstance(task_id, str) and task_id in completed_task_ids
        ) or status == "completed"

        if is_completed:
            completed_tasks += 1
            continue
        if status == "blocked":
            blocked_tasks += 1

    remaining_tasks = max(total_tasks - completed_tasks, 0)
    completion_pct = int(round((completed_tasks / total_tasks) * 100)) if total_tasks else 0

    return ActivePhaseProgress(
        phase_id=session.active_phase_id,
        phase_title=active_phase.get("title") if isinstance(active_phase.get("title"), str) else None,
        total_tasks=total_tasks,
        completed_tasks=completed_tasks,
        blocked_tasks=blocked_tasks,
        remaining_tasks=remaining_tasks,
        completion_pct=completion_pct,
    )


def _build_retry_counters(
    session: AutonomousSessionState,
    spec_data: Optional[Dict[str, Any]],
) -> RetryCounters:
    """Build retry counters including task-level retries when metadata is available."""
    task_retry_counts: Dict[str, int] = {}

    if isinstance(spec_data, dict) and session.active_phase_id:
        phases = spec_data.get("phases", [])
        if isinstance(phases, list):
            for phase in phases:
                if not isinstance(phase, dict) or phase.get("id") != session.active_phase_id:
                    continue
                tasks = phase.get("tasks", [])
                if not isinstance(tasks, list):
                    break
                for task in tasks:
                    if not isinstance(task, dict):
                        continue
                    task_id = task.get("id")
                    if not isinstance(task_id, str) or not task_id:
                        continue
                    metadata = task.get("metadata")
                    if not isinstance(metadata, dict):
                        continue
                    retry_count = metadata.get("retry_count")
                    if isinstance(retry_count, int) and retry_count > 0:
                        task_retry_counts[task_id] = retry_count
                break

    phase_retry_counts: Dict[str, int] = {}
    if session.active_phase_id:
        phase_retry_counts[session.active_phase_id] = (
            session.counters.fidelity_review_cycles_in_active_phase
        )

    return RetryCounters(
        consecutive_errors=session.counters.consecutive_errors,
        fidelity_review_cycles_in_active_phase=session.counters.fidelity_review_cycles_in_active_phase,
        phase_retry_counts=phase_retry_counts,
        task_retry_counts=task_retry_counts,
    )


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
    spec_data = _load_spec_for_session(session, workspace)
    active_phase_progress = _build_active_phase_progress(session, spec_data)
    retry_counters = _build_retry_counters(session, spec_data)
    last_step_issued = session.last_step_issued

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
        last_step_id=last_step_issued.step_id if last_step_issued else None,
        last_step_type=last_step_issued.type if last_step_issued else None,
        current_task_id=(
            last_step_issued.task_id
            if last_step_issued and last_step_issued.task_id
            else session.last_task_id
        ),
        active_phase_progress=active_phase_progress,
        retry_counters=retry_counters,
        session_signal=derive_loop_signal(
            status=session.status,
            pause_reason=session.pause_reason,
        ),
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


def _inject_audit_status(response: dict, *journal_results: bool) -> dict:
    """Add ``meta.audit_status`` to a response based on journal write outcomes.

    Enables operators to determine whether audit writes succeeded for a given
    operation (H2 observability requirement).

    Args:
        response: MCP response dict.
        *journal_results: Boolean results from ``_write_session_journal`` calls.

    Returns:
        The response dict with ``meta.audit_status`` set to one of:
        ``"ok"`` (all writes succeeded), ``"partial"`` (some failed),
        or ``"failed"`` (all failed).
    """
    if not journal_results:
        return response
    if all(journal_results):
        status = "ok"
    elif any(journal_results):
        status = "partial"
    else:
        status = "failed"
    meta = response.get("meta")
    if isinstance(meta, dict):
        meta["audit_status"] = status
    return response


def _write_session_journal(
    spec_id: str,
    action: str,
    summary: str,
    session_id: str,
    workspace: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> bool:
    """Write a journal entry for session lifecycle event (best-effort).

    Policy: Journal writes are best-effort across all handlers. A journal
    write failure is logged as a warning but never blocks the calling
    operation or triggers rollback. This avoids coupling session lifecycle
    correctness to the journal subsystem's availability.

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

    except (OSError, json.JSONDecodeError, ValueError, KeyError) as e:
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
