"""Session rebase and gate waiver handlers.

Split from handlers_session.py for maintainability (H3).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.autonomy.models import (
    AutonomousSessionState,
    SessionStatus,
    PhaseGateStatus,
    RebaseResultDetail,
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
from foundry_mcp.core.spec import resolve_spec_file
from foundry_mcp.core.authorization import (
    check_action_allowed,
    Role,
)

from foundry_mcp.tools.unified.param_schema import Str, validate_payload
from foundry_mcp.tools.unified.task_handlers._helpers import (
    _get_storage,
    _request_id,
    _resolve_session,
    _validate_reason_detail,
    _is_feature_enabled,
    _feature_disabled_response,
)
from foundry_mcp.tools.unified.task_handlers._session_common import (
    _save_with_version_check,
    _invalid_transition_response,
    _build_session_response,
    _inject_audit_status,
    _write_session_journal,
    _compute_required_gates_from_spec,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Declarative validation schemas
# ---------------------------------------------------------------------------

_GATE_WAIVER_PARAMS_SCHEMA = {
    "phase_id": Str(required=True),
    "reason_code": Str(required=True, choices=frozenset(e.value for e in OverrideReasonCode)),
}


# =============================================================================
# Gate Waiver Error Constants
# =============================================================================

ERROR_GATE_WAIVER_DISABLED = "GATE_WAIVER_DISABLED"
ERROR_GATE_WAIVER_UNAUTHORIZED = "GATE_WAIVER_UNAUTHORIZED"
ERROR_GATE_NOT_FOUND = "GATE_NOT_FOUND"
ERROR_GATE_ALREADY_WAIVED = "GATE_ALREADY_WAIVED"
ERROR_GATE_ALREADY_PASSED = "GATE_ALREADY_PASSED"


# =============================================================================
# Rebase Helpers
# =============================================================================


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
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug("Failed to read backup %s: %s", backup_file, e)
            continue

    return None


# =============================================================================
# Session Rebase Handler
# =============================================================================


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

    # Role check - only maintainer can rebase sessions
    # Deferred import: tests mock-patch get_server_role on the handlers_session shim module
    from foundry_mcp.tools.unified.task_handlers.handlers_session import get_server_role as _get_role
    current_role = _get_role()
    if current_role in (Role.AUTONOMY_RUNNER.value, Role.OBSERVER.value):
        return asdict(error_response(
            f"Session rebase denied for role: {current_role}",
            error_code=ErrorCode.FORBIDDEN,
            error_type=ErrorType.AUTHORIZATION,
            request_id=request_id,
            details={
                "action": "session-rebase",
                "configured_role": current_role,
                "required_role": Role.MAINTAINER.value,
                "hint": "Only maintainer role can rebase sessions",
            },
        ))

    storage = _get_storage(config, workspace, request_id=request_id)
    if isinstance(storage, dict):
        return storage

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
    except (OSError, json.JSONDecodeError, ValueError) as e:
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
        pre_mutation_version = session.state_version
        session.state_version += 1

        # Update required gates (spec author may have added gate requirements)
        session.required_phase_gates = _compute_required_gates_from_spec(current_spec_data, session)

        err = _save_with_version_check(storage, session, pre_mutation_version, "session-rebase", request_id)
        if err:
            return err

        rebase_result = RebaseResultDetail(
            result="no_change",
        )

        # Write journal
        journal_ok = _write_session_journal(
            spec_id=session.spec_id,
            action="rebase",
            summary="Session rebase: no structural changes detected",
            session_id=session.id,
            workspace=workspace,
            metadata={"result": "no_change"},
        )

        return _inject_audit_status(
            _build_session_response(session, request_id, rebase_result=rebase_result, include_resume_context=True, workspace=workspace),
            journal_ok,
        )

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

        # Guard: backup missing with completed tasks means we can't verify integrity
        if session.completed_task_ids:
            if not force:
                return asdict(error_response(
                    "Cannot rebase: backup spec not found and session has completed tasks. "
                    "Structural diff cannot verify completed task integrity.",
                    error_code=ErrorCode.REBASE_BACKUP_MISSING,
                    error_type=ErrorType.CONFLICT,
                    request_id=request_id,
                    details={
                        "action": "session-rebase",
                        "completed_task_count": len(session.completed_task_ids),
                        "completed_task_ids": session.completed_task_ids[:10],
                        "missing_backup_hash": session.spec_structure_hash[:16],
                        "hint": "Use force=true to accept potential loss of completion history",
                    },
                ))
            else:
                logger.warning(
                    "Force-rebasing session %s without backup spec; %d completed tasks may lose structural diff coverage",
                    session.id, len(session.completed_task_ids),
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
    pre_mutation_version = session.state_version
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

    err = _save_with_version_check(storage, session, pre_mutation_version, "session-rebase", request_id)
    if err:
        return err

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
    journal_ok = _write_session_journal(
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
            "backup_missing": old_spec_data is None,
            "completed_tasks_at_risk": len(session.completed_task_ids) if old_spec_data is None else 0,
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

    return _inject_audit_status(
        _build_session_response(session, request_id, rebase_result=rebase_result, include_resume_context=True, workspace=workspace),
        journal_ok,
    )


# =============================================================================
# Gate Waiver Handler
# =============================================================================


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
    # Deferred import: tests mock-patch get_server_role on the handlers_session shim module
    from foundry_mcp.tools.unified.task_handlers.handlers_session import get_server_role as _get_role
    current_role = _get_role()
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

    # Validate required parameters via schema
    params = {"phase_id": phase_id, "reason_code": reason_code}
    err = validate_payload(params, _GATE_WAIVER_PARAMS_SCHEMA,
                           tool_name="task", action="gate-waiver",
                           request_id=request_id)
    if err:
        return err
    validated_reason_code = OverrideReasonCode(params["reason_code"])

    # Validate reason_detail length
    reason_detail_err = _validate_reason_detail(reason_detail, "gate-waiver", request_id)
    if reason_detail_err:
        return reason_detail_err

    storage = _get_storage(config, workspace, request_id=request_id)
    if isinstance(storage, dict):
        return storage

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
    pre_mutation_version = session.state_version
    session.state_version += 1

    err = _save_with_version_check(storage, session, pre_mutation_version, "session-gate-waiver", request_id)
    if err:
        return err

    # Write journal entry
    journal_ok = _write_session_journal(
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

    return _inject_audit_status(
        asdict(success_response(
            data=response_data,
            request_id=request_id,
        )),
        journal_ok,
    )
