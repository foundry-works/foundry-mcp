"""Session lifecycle handlers: start, pause, resume, end, reset.

Split from handlers_session.py for maintainability (H3).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ulid import ULID

from foundry_mcp.config.server import ServerConfig
from foundry_mcp.core.authorization import (
    Role,
)
from foundry_mcp.core.autonomy.models.enums import (
    TERMINAL_STATUSES,
    FailureReason,
    GatePolicy,
    OverrideReasonCode,
    PauseReason,
    SessionStatus,
)
from foundry_mcp.core.autonomy.models.session_config import (
    SessionContext,
    SessionCounters,
    SessionLimits,
    StopConditions,
)
from foundry_mcp.core.autonomy.models.state import AutonomousSessionState
from foundry_mcp.core.autonomy.spec_adapter import load_spec_file
from foundry_mcp.core.autonomy.spec_hash import (
    compute_spec_structure_hash,
    get_spec_file_metadata,
)
from foundry_mcp.core.responses.builders import (
    error_response,
    success_response,
)
from foundry_mcp.core.responses.types import (
    ErrorCode,
    ErrorType,
)
from foundry_mcp.core.spec import resolve_spec_file
from foundry_mcp.tools.unified.param_schema import Str, validate_payload
from foundry_mcp.tools.unified.task_handlers._helpers import (
    _get_storage,
    _request_id,
    _resolve_session,
    _session_not_found_response,
    _validate_reason_detail,
    _validation_error,
)
from foundry_mcp.tools.unified.task_handlers._session_common import (
    _build_session_response,
    _compute_required_gates_from_spec,
    _inject_audit_status,
    _invalid_transition_response,
    _save_with_version_check,
    _write_session_journal,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Declarative parameter schemas
# ---------------------------------------------------------------------------

_OVERRIDE_REASON_CHOICES = frozenset(e.value for e in OverrideReasonCode)

_SESSION_START_SCHEMA = {
    "spec_id": Str(required=True),
}

_SESSION_END_SCHEMA = {
    "reason_code": Str(required=True, choices=_OVERRIDE_REASON_CHOICES),
}

_SESSION_RESET_SCHEMA = {
    "reason_code": Str(required=True, choices=_OVERRIDE_REASON_CHOICES),
    "session_id": Str(required=True),
}


# =============================================================================
# Posture Enforcement
# =============================================================================


def _validate_posture_constraints(
    config: ServerConfig,
    *,
    gate_policy: Optional[str],
    enforce_autonomy_write_lock: Optional[bool],
    request_id: str,
) -> Optional[dict]:
    """Validate session-start parameters against the active posture profile.

    Returns an error response dict if posture constraints are violated,
    None if the parameters are acceptable.
    """
    profile = getattr(config.autonomy_posture, "profile", None)
    if not profile:
        return None

    violations: List[str] = []

    if profile == "unattended":
        # Unattended posture enforces strict gate policy
        if gate_policy is not None and gate_policy.lower() != "strict":
            violations.append(f"gate_policy='{gate_policy}' is not allowed under unattended posture (must be 'strict')")
        # Unattended posture requires write lock
        if enforce_autonomy_write_lock is not None and not enforce_autonomy_write_lock:
            violations.append("enforce_autonomy_write_lock=false is not allowed under unattended posture")

    if violations:
        return asdict(
            error_response(
                f"Session configuration violates '{profile}' posture constraints",
                error_code=ErrorCode.AUTHORIZATION,
                error_type=ErrorType.AUTHORIZATION,
                request_id=request_id,
                details={
                    "action": "session-start",
                    "posture_profile": profile,
                    "violations": violations,
                    "hint": "Adjust session parameters to match the active posture profile, or change the posture",
                },
            )
        )

    return None


# =============================================================================
# Session Start Helpers
# =============================================================================


def _handle_existing_session(
    storage: Any,
    spec_id: str,
    idempotency_key: Optional[str],
    force: bool,
    request_id: str,
    workspace: Optional[str],
) -> Optional[dict]:
    """Check for existing active session and handle dedup/force-end.

    Must be called inside the per-spec lock.

    Returns:
        - Response dict to return immediately (idempotent match or conflict error)
        - None if no existing session or it was force-ended
    """
    existing_session_id = storage.get_active_session(spec_id)
    if not existing_session_id:
        return None

    existing_session = storage.load(existing_session_id)
    if not existing_session or existing_session.status in TERMINAL_STATUSES:
        return None

    # Idempotent match — return existing session
    if idempotency_key and existing_session.idempotency_key == idempotency_key:
        return _build_session_response(existing_session, request_id, workspace=workspace)

    if force:
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
        return None

    return asdict(
        error_response(
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
        )
    )


def _load_spec_for_session_start(
    spec_id: str,
    workspace: Optional[str],
    request_id: str,
) -> tuple:
    """Load spec file and compute structure hash.

    Returns:
        (spec_data, spec_structure_hash, spec_metadata, None) on success
        (None, None, None, error_response_dict) on failure
    """
    ws_path = Path(workspace) if workspace else Path.cwd()
    specs_dir = ws_path / "specs"
    spec_path = resolve_spec_file(spec_id, specs_dir)

    if not spec_path:
        return (
            None,
            None,
            None,
            asdict(
                error_response(
                    f"Spec not found: {spec_id}",
                    error_code=ErrorCode.NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    request_id=request_id,
                    details={"spec_id": spec_id},
                )
            ),
        )

    try:
        spec_data = load_spec_file(spec_path)
        spec_structure_hash = compute_spec_structure_hash(spec_data)
        spec_metadata = get_spec_file_metadata(spec_path)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        return (
            None,
            None,
            None,
            asdict(
                error_response(
                    f"Failed to read spec: {e}",
                    error_code=ErrorCode.INTERNAL_ERROR,
                    error_type=ErrorType.INTERNAL,
                    request_id=request_id,
                    details={"spec_id": spec_id},
                )
            ),
        )

    return spec_data, spec_structure_hash, spec_metadata, None


def _resolve_session_config(
    config: ServerConfig,
    overrides: Dict[str, Any],
    request_id: str,
) -> tuple:
    """Resolve session configuration from caller overrides and server defaults.

    Args:
        config: Server configuration with autonomy_session_defaults
        overrides: Caller-provided session config params (gate_policy,
            max_tasks_per_session, enforce_autonomy_write_lock, etc.)
        request_id: Request ID for error responses

    Returns:
        (SessionLimits, StopConditions, GatePolicy, write_lock_enforced, None) on success
        (None, None, None, None, error_response_dict) on validation failure
    """
    session_defaults = getattr(config, "autonomy_session_defaults", None)

    def _resolve(caller_value: Any, attr_name: str, expected_type: type) -> Any:
        if caller_value is not None:
            return caller_value
        if session_defaults is not None and isinstance(getattr(session_defaults, attr_name, None), expected_type):
            return getattr(session_defaults, attr_name)
        return None

    # Build limits — some fields fall back to server defaults, others are direct
    limits_kwargs: Dict[str, Any] = {}
    _LIMITS_WITH_DEFAULTS = [
        "max_tasks_per_session",
        "max_consecutive_errors",
        "max_fidelity_review_cycles_per_phase",
    ]
    _LIMITS_DIRECT = [
        "context_threshold_pct",
        "heartbeat_stale_minutes",
        "heartbeat_grace_minutes",
        "step_stale_minutes",
        "avg_pct_per_step",
        "context_staleness_threshold",
        "context_staleness_penalty_pct",
    ]
    for attr in _LIMITS_WITH_DEFAULTS:
        resolved = _resolve(overrides.get(attr), attr, int)
        if resolved is not None:
            limits_kwargs[attr] = resolved
    for attr in _LIMITS_DIRECT:
        val = overrides.get(attr)
        if val is not None:
            limits_kwargs[attr] = val

    # Build stop conditions
    stop_kwargs: Dict[str, Any] = {}
    for attr in ("stop_on_phase_completion", "auto_retry_fidelity_gate"):
        resolved = _resolve(overrides.get(attr), attr, bool)
        if resolved is not None:
            stop_kwargs[attr] = resolved

    # Resolve gate policy
    gate_policy = overrides.get("gate_policy")
    resolved_gate_policy_raw = "strict"
    if gate_policy is not None:
        resolved_gate_policy_raw = gate_policy
    elif session_defaults is not None and isinstance(getattr(session_defaults, "gate_policy", None), str):
        resolved_gate_policy_raw = session_defaults.gate_policy

    try:
        resolved_gate_policy = GatePolicy(str(resolved_gate_policy_raw).lower())
    except ValueError:
        return (
            None,
            None,
            None,
            None,
            _validation_error(
                action="session-start",
                field="gate_policy",
                message=(f"Invalid gate_policy: {resolved_gate_policy_raw}. Must be one of: strict, lenient, manual"),
                request_id=request_id,
            ),
        )

    enforce_lock = overrides.get("enforce_autonomy_write_lock")
    write_lock = enforce_lock if enforce_lock is not None else True

    return (
        SessionLimits(**limits_kwargs),
        StopConditions(**stop_kwargs),
        resolved_gate_policy,
        write_lock,
        None,
    )


def _verify_audit_chain_for_start(
    spec_id: str,
    workspace: Optional[str],
) -> Optional[dict]:
    """Best-effort audit chain verification for session start.

    Returns audit warning dict if verification failed, None otherwise.
    """
    try:
        from foundry_mcp.core.autonomy.audit import get_ledger_path, verify_chain

        ws_path = Path(workspace) if workspace else Path.cwd()
        ledger_path = get_ledger_path(spec_id=spec_id, workspace_path=ws_path)

        if ledger_path.exists():
            result = verify_chain(spec_id=spec_id, workspace_path=ws_path)
            if not result.valid:
                logger.warning(
                    "Audit chain verification failed for spec %s: %s at sequence %d",
                    spec_id,
                    result.divergence_type,
                    result.divergence_point,
                )
                return {
                    "code": "AUDIT_CHAIN_BROKEN",
                    "divergence_point": result.divergence_point,
                    "divergence_type": result.divergence_type,
                    "detail": result.divergence_detail,
                }
    except (OSError, ValueError, KeyError, ImportError) as e:
        logger.debug("Audit verification skipped (non-critical): %s", e)

    return None


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

    Orchestrates new autonomous session creation for a spec. Delegates to
    focused helpers for each phase: existing session handling, spec loading,
    config resolution, persistence, and post-start cleanup.

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

    params = {"spec_id": spec_id}
    err = validate_payload(
        params, _SESSION_START_SCHEMA, tool_name="task", action="session-start", request_id=request_id
    )
    if err:
        return err
    assert isinstance(spec_id, str)

    posture_err = _validate_posture_constraints(
        config,
        gate_policy=gate_policy,
        enforce_autonomy_write_lock=enforce_autonomy_write_lock,
        request_id=request_id,
    )
    if posture_err:
        return posture_err

    storage = _get_storage(config, workspace, request_id=request_id)
    if isinstance(storage, dict):
        return storage

    # --- Resolve session configuration ---
    config_overrides = {
        k: v
        for k, v in {
            "gate_policy": gate_policy,
            "max_tasks_per_session": max_tasks_per_session,
            "max_consecutive_errors": max_consecutive_errors,
            "context_threshold_pct": context_threshold_pct,
            "stop_on_phase_completion": stop_on_phase_completion,
            "auto_retry_fidelity_gate": auto_retry_fidelity_gate,
            "heartbeat_stale_minutes": heartbeat_stale_minutes,
            "heartbeat_grace_minutes": heartbeat_grace_minutes,
            "step_stale_minutes": step_stale_minutes,
            "max_fidelity_review_cycles_per_phase": max_fidelity_review_cycles_per_phase,
            "avg_pct_per_step": avg_pct_per_step,
            "context_staleness_threshold": context_staleness_threshold,
            "context_staleness_penalty_pct": context_staleness_penalty_pct,
            "enforce_autonomy_write_lock": enforce_autonomy_write_lock,
        }.items()
        if v is not None
    }
    limits, stop_conditions, resolved_gate_policy, write_lock, cfg_err = _resolve_session_config(
        config, config_overrides, request_id
    )
    if cfg_err:
        return cfg_err

    # --- Acquire per-spec lock ---
    try:
        spec_lock = storage.acquire_spec_lock(spec_id, timeout=5)
    except (OSError, TimeoutError) as e:
        return asdict(
            error_response(
                f"Failed to acquire spec lock: {e}",
                error_code=ErrorCode.LOCK_TIMEOUT,
                error_type=ErrorType.UNAVAILABLE,
                request_id=request_id,
                details={"spec_id": spec_id, "action": "session-start"},
            )
        )

    with spec_lock:
        # Check for existing session (dedup / force-end)
        existing_resp = _handle_existing_session(
            storage,
            spec_id,
            idempotency_key,
            force,
            request_id,
            workspace,
        )
        if existing_resp is not None:
            return existing_resp

        # Load spec and compute hash
        spec_data, spec_structure_hash, spec_metadata, spec_err = _load_spec_for_session_start(
            spec_id, workspace, request_id
        )
        if spec_err:
            return spec_err

        # Create session state
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
            paused_at=None,
            pause_reason=None,
            failure_reason=None,
            active_phase_id=None,
            last_task_id=None,
            last_step_issued=None,
            last_issued_response=None,
            pending_gate_evidence=None,
            pending_manual_gate_ack=None,
            pending_verification_receipt=None,
            counters=SessionCounters(),
            limits=limits,
            stop_conditions=stop_conditions,
            context=SessionContext(
                estimated_tokens_used=None,
                last_heartbeat_at=None,
                context_source=None,
                last_context_report_at=None,
                last_context_report_pct=None,
            ),
            write_lock_enforced=write_lock,
            gate_policy=resolved_gate_policy,
        )
        session.required_phase_gates = _compute_required_gates_from_spec(spec_data, session)

        # Persist state (atomic write)
        try:
            storage.save(session)
        except (OSError, ValueError, TimeoutError) as e:
            logger.error("Failed to save session state: %s", e)
            return asdict(
                error_response(
                    f"Failed to create session: {e}",
                    error_code=ErrorCode.INTERNAL_ERROR,
                    error_type=ErrorType.INTERNAL,
                    request_id=request_id,
                )
            )

        # Journal + pointer (best-effort)
        journal_ok = _write_session_journal(
            spec_id=spec_id,
            action="start",
            summary=f"Started autonomous session {session.id}",
            session_id=session.id,
            workspace=workspace,
            metadata={"idempotency_key": idempotency_key},
        )
        storage.set_active_session(spec_id, session.id)

        # Audit chain verification (best-effort)
        audit_warning = _verify_audit_chain_for_start(spec_id, workspace)

    # GC (best-effort, outside lock)
    try:
        storage.cleanup_expired()
    except (OSError, TimeoutError) as e:
        logger.debug("GC failed (non-critical): %s", e)

    logger.info("Started session %s for spec %s", session.id, spec_id)
    response = _build_session_response(session, request_id, workspace=workspace)
    if audit_warning:
        response["data"]["audit_warning"] = audit_warning
    return _inject_audit_status(response, journal_ok)


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

    storage = _get_storage(config, workspace, request_id=request_id)
    if isinstance(storage, dict):
        return storage

    session, err = _resolve_session(storage, "session-pause", request_id, session_id, spec_id)
    if err:
        return err
    assert session is not None

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
    pre_mutation_version = session.state_version
    session.state_version += 1

    err = _save_with_version_check(storage, session, pre_mutation_version, "session-pause", request_id)
    if err:
        return err

    # Write journal
    journal_ok = _write_session_journal(
        spec_id=session.spec_id,
        action="pause",
        summary=f"Session paused: {reason or 'user request'}",
        session_id=session.id,
        workspace=workspace,
    )

    logger.info("Paused session %s for spec %s", session.id, session.spec_id)

    return _inject_audit_status(
        _build_session_response(session, request_id, workspace=workspace),
        journal_ok,
    )


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

    storage = _get_storage(config, workspace, request_id=request_id)
    if isinstance(storage, dict):
        return storage

    session, err = _resolve_session(storage, "session-resume", request_id, session_id, spec_id)
    if err:
        return err
    assert session is not None

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
                    current_spec_data = load_spec_file(spec_path)
                    current_hash = compute_spec_structure_hash(current_spec_data)
                    if current_hash != session.spec_structure_hash:
                        return asdict(
                            error_response(
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
                            )
                        )
                except (OSError, json.JSONDecodeError, ValueError) as e:
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
            return asdict(
                error_response(
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
                )
            )

        if (
            not acknowledged_gate_attempt_id
            or acknowledged_gate_attempt_id != session.pending_manual_gate_ack.gate_attempt_id
        ):
            return asdict(
                error_response(
                    "Invalid gate acknowledgment",
                    error_code=ErrorCode.INVALID_GATE_ACK,
                    error_type=ErrorType.VALIDATION,
                    request_id=request_id,
                    details={
                        "action": "session-resume",
                        "expected": session.pending_manual_gate_ack.gate_attempt_id,
                        "provided": acknowledged_gate_attempt_id,
                    },
                )
            )

        # Clear pending gate ack
        session.pending_manual_gate_ack = None

    # Update session
    session.status = SessionStatus.RUNNING
    session.pause_reason = None
    session.failure_reason = None
    session.updated_at = datetime.now(timezone.utc)
    session.paused_at = None
    pre_mutation_version = session.state_version
    session.state_version += 1

    err = _save_with_version_check(storage, session, pre_mutation_version, "session-resume", request_id)
    if err:
        return err

    # Write journal
    journal_ok = _write_session_journal(
        spec_id=session.spec_id,
        action="resume",
        summary=f"Session resumed{', forced from failed' if force else ''}",
        session_id=session.id,
        workspace=workspace,
        metadata={
            "acknowledge_gate_review": acknowledge_gate_review,
            "acknowledged_gate_attempt_id": acknowledged_gate_attempt_id,
            "force": force,
        },
    )

    logger.info("Resumed session %s for spec %s", session.id, session.spec_id)

    return _inject_audit_status(
        _build_session_response(session, request_id, include_resume_context=True, workspace=workspace),
        journal_ok,
    )


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

    params = {"reason_code": reason_code}
    err = validate_payload(params, _SESSION_END_SCHEMA, tool_name="task", action="session-end", request_id=request_id)
    if err:
        return err
    validated_reason_code = OverrideReasonCode(params["reason_code"])

    # Validate reason_detail length
    reason_detail_err = _validate_reason_detail(reason_detail, "session-end", request_id)
    if reason_detail_err:
        return reason_detail_err

    storage = _get_storage(config, workspace, request_id=request_id)
    if isinstance(storage, dict):
        return storage

    session, err = _resolve_session(storage, "session-end", request_id, session_id, spec_id)
    if err:
        return err
    assert session is not None

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
    # Deferred import: tests mock-patch get_server_role on the handlers_session shim module
    from foundry_mcp.tools.unified.task_handlers.handlers_session import get_server_role as _get_role

    current_role = _get_role()

    # Update session
    session.status = SessionStatus.ENDED
    session.updated_at = datetime.now(timezone.utc)
    pre_mutation_version = session.state_version
    session.state_version += 1

    err = _save_with_version_check(storage, session, pre_mutation_version, "session-end", request_id)
    if err:
        return err

    # Remove pointer
    storage.remove_active_session(session.spec_id)

    # Write journal with reason code and role
    journal_ok = _write_session_journal(
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

    return _inject_audit_status(
        _build_session_response(session, request_id, workspace=workspace),
        journal_ok,
    )


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

    params = {"reason_code": reason_code, "session_id": session_id}
    err = validate_payload(
        params, _SESSION_RESET_SCHEMA, tool_name="task", action="session-reset", request_id=request_id
    )
    if err:
        return err
    assert isinstance(session_id, str)
    validated_reason_code = OverrideReasonCode(params["reason_code"])

    # Validate reason_detail length
    reason_detail_err = _validate_reason_detail(reason_detail, "session-reset", request_id)
    if reason_detail_err:
        return reason_detail_err

    # Role check - only maintainer can reset sessions
    # Deferred import: tests mock-patch get_server_role on the handlers_session shim module
    from foundry_mcp.tools.unified.task_handlers.handlers_session import get_server_role as _get_role

    current_role = _get_role()
    if current_role in (Role.AUTONOMY_RUNNER.value, Role.OBSERVER.value):
        return asdict(
            error_response(
                f"Session reset denied for role: {current_role}",
                error_code=ErrorCode.FORBIDDEN,
                error_type=ErrorType.AUTHORIZATION,
                request_id=request_id,
                details={
                    "action": "session-reset",
                    "configured_role": current_role,
                    "required_role": Role.MAINTAINER.value,
                    "hint": "Only maintainer role can reset failed sessions",
                },
            )
        )

    storage = _get_storage(config, workspace, request_id=request_id)
    if isinstance(storage, dict):
        return storage

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

    storage.delete(session.id)
    storage.remove_active_session(session.spec_id)

    # Write journal with reason code and role
    journal_ok = _write_session_journal(
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

    logger.info(
        "Reset (deleted) session %s for spec %s: %s", deleted_session_id, deleted_spec_id, validated_reason_code.value
    )

    return _inject_audit_status(
        asdict(
            success_response(
                data={
                    "session_id": deleted_session_id,
                    "spec_id": deleted_spec_id,
                    "result": "deleted",
                    "reason_code": validated_reason_code.value,
                },
                request_id=request_id,
            )
        ),
        journal_ok,
    )
