"""Loop signal derivation and recommended action utilities."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from .enums import (
    LoopSignal,
    PauseReason,
    SessionStatus,
)
from .responses import RecommendedAction
from .state import AutonomousSessionState

_PAUSED_NEEDS_ATTENTION_REASONS = frozenset(
    {
        PauseReason.FIDELITY_CYCLE_LIMIT.value,
        PauseReason.GATE_FAILED.value,
        PauseReason.GATE_REVIEW_REQUIRED.value,
        PauseReason.BLOCKED.value,
        PauseReason.ERROR_THRESHOLD.value,
        PauseReason.CONTEXT_LIMIT.value,
        PauseReason.HEARTBEAT_STALE.value,
        PauseReason.STEP_STALE.value,
        PauseReason.TASK_LIMIT.value,
        "spec_rebase_required",
    }
)

_BLOCKED_RUNTIME_ERROR_CODES = frozenset(
    {
        "REQUIRED_GATE_UNSATISFIED",
        "ERROR_REQUIRED_GATE_UNSATISFIED",
        "GATE_AUDIT_FAILURE",
        "ERROR_GATE_AUDIT_FAILURE",
        "GATE_INTEGRITY_CHECKSUM",
        "ERROR_GATE_INTEGRITY_CHECKSUM",
        "AUTHORIZATION",
        "STEP_PROOF_MISSING",
        "STEP_PROOF_MISMATCH",
        "STEP_PROOF_CONFLICT",
        "STEP_PROOF_EXPIRED",
        "VERIFICATION_RECEIPT_MISSING",
        "VERIFICATION_RECEIPT_INVALID",
    }
)


def _normalize_signal_value(value: Any) -> str:
    """Normalize enum/string values for deterministic loop-signal mapping."""
    if value is None:
        return ""
    if isinstance(value, Enum):
        raw = value.value
    else:
        raw = str(value)
    return raw.strip().lower()


def derive_loop_signal(
    *,
    status: Optional[Any] = None,
    pause_reason: Optional[Any] = None,
    error_code: Optional[str] = None,
    is_unrecoverable_error: bool = False,
    repeated_invalid_gate_evidence: bool = False,
) -> Optional[LoopSignal]:
    """Map status/pause/error inputs to the canonical loop signal contract."""
    normalized_status = _normalize_signal_value(status)
    normalized_pause_reason = _normalize_signal_value(pause_reason)
    normalized_error_code = _normalize_signal_value(error_code).upper()

    if normalized_pause_reason == PauseReason.PHASE_COMPLETE.value:
        return LoopSignal.PHASE_COMPLETE

    if normalized_status == SessionStatus.COMPLETED.value or normalized_pause_reason == "spec_complete":
        return LoopSignal.SPEC_COMPLETE

    if normalized_error_code in _BLOCKED_RUNTIME_ERROR_CODES:
        return LoopSignal.BLOCKED_RUNTIME

    if normalized_error_code in {"INVALID_GATE_EVIDENCE", "ERROR_INVALID_GATE_EVIDENCE"}:
        if repeated_invalid_gate_evidence:
            return LoopSignal.BLOCKED_RUNTIME

    if (
        normalized_status == SessionStatus.FAILED.value
        or is_unrecoverable_error
        or normalized_error_code in {"SESSION_UNRECOVERABLE", "ERROR_SESSION_UNRECOVERABLE"}
    ):
        return LoopSignal.FAILED

    if normalized_pause_reason in _PAUSED_NEEDS_ATTENTION_REASONS:
        return LoopSignal.PAUSED_NEEDS_ATTENTION

    return None


def derive_recommended_actions(
    *,
    loop_signal: Optional[LoopSignal],
    pause_reason: Optional[Any] = None,
    error_code: Optional[str] = None,
) -> List[RecommendedAction]:
    """Build recommended_actions payload for escalation outcomes."""
    if loop_signal in (None, LoopSignal.PHASE_COMPLETE, LoopSignal.SPEC_COMPLETE):
        return []

    normalized_pause_reason = _normalize_signal_value(pause_reason)
    normalized_error_code = _normalize_signal_value(error_code).upper()

    if loop_signal == LoopSignal.PAUSED_NEEDS_ATTENTION:
        pause_actions: Dict[str, List[RecommendedAction]] = {
            PauseReason.CONTEXT_LIMIT.value: [
                RecommendedAction(
                    action="resume_in_fresh_context",
                    description="Open a fresh context window, then resume the session.",
                    command='task(action="session", command="resume")',
                )
            ],
            PauseReason.ERROR_THRESHOLD.value: [
                RecommendedAction(
                    action="inspect_recent_failures",
                    description="Inspect recent failed steps before retrying execution.",
                    command='task(action="session-step", command="replay")',
                ),
                RecommendedAction(
                    action="reset_session_if_needed",
                    description="Reset the session only after root-cause triage.",
                    command='task(action="session", command="reset")',
                ),
            ],
            PauseReason.BLOCKED.value: [
                RecommendedAction(
                    action="resolve_task_blockers",
                    description="Unblock dependency chains before resuming automation.",
                    command='task(action="list-blocked")',
                )
            ],
            PauseReason.GATE_FAILED.value: [
                RecommendedAction(
                    action="review_gate_findings",
                    description="Review fidelity findings and apply remediation before retry.",
                    command='review(action="fidelity")',
                )
            ],
            PauseReason.GATE_REVIEW_REQUIRED.value: [
                RecommendedAction(
                    action="acknowledge_manual_gate_review",
                    description="A maintainer must acknowledge manual gate review before resume.",
                    command='task(action="session", command="resume")',
                )
            ],
            PauseReason.FIDELITY_CYCLE_LIMIT.value: [
                RecommendedAction(
                    action="escalate_fidelity_cycle_limit",
                    description="Manual intervention is required after repeated gate retries.",
                    command='task(action="session", command="pause")',
                )
            ],
            PauseReason.HEARTBEAT_STALE.value: [
                RecommendedAction(
                    action="refresh_heartbeat_or_resume",
                    description="Refresh heartbeat or resume the session once the caller is healthy.",
                    command='task(action="session-step", command="heartbeat")',
                )
            ],
            PauseReason.STEP_STALE.value: [
                RecommendedAction(
                    action="replay_or_reset_stale_step",
                    description="Replay the last response first; reset only if replay is invalid.",
                    command='task(action="session-step", command="replay")',
                )
            ],
            PauseReason.TASK_LIMIT.value: [
                RecommendedAction(
                    action="start_followup_session",
                    description="Start a new session to continue once task limit is reached.",
                    command='task(action="session", command="start")',
                )
            ],
            "spec_rebase_required": [
                RecommendedAction(
                    action="rebase_session_to_spec",
                    description="Rebase session state to the latest spec structure before resume.",
                    command='task(action="session", command="rebase")',
                )
            ],
        }
        return pause_actions.get(
            normalized_pause_reason,
            [
                RecommendedAction(
                    action="manual_triage",
                    description="Investigate pause cause and resume only after mitigation.",
                    command='task(action="session", command="status")',
                )
            ],
        )

    if loop_signal == LoopSignal.BLOCKED_RUNTIME:
        if normalized_error_code == "AUTHORIZATION":
            return [
                RecommendedAction(
                    action="use_authorized_role",
                    description="Switch to a role authorized for session-step actions.",
                )
            ]
        if normalized_error_code in {
            "REQUIRED_GATE_UNSATISFIED",
            "ERROR_REQUIRED_GATE_UNSATISFIED",
        }:
            return [
                RecommendedAction(
                    action="satisfy_or_waive_required_gate",
                    description="Satisfy required gates or execute a maintainer gate waiver.",
                    command='task(action="gate-waiver")',
                )
            ]
        if normalized_error_code in {"GATE_AUDIT_FAILURE", "ERROR_GATE_AUDIT_FAILURE"}:
            return [
                RecommendedAction(
                    action="rerun_gate_with_fresh_evidence",
                    description="Rerun gate checks and verify integrity evidence bindings.",
                )
            ]
        if normalized_error_code in {
            "GATE_INTEGRITY_CHECKSUM",
            "ERROR_GATE_INTEGRITY_CHECKSUM",
            "INVALID_GATE_EVIDENCE",
            "ERROR_INVALID_GATE_EVIDENCE",
        }:
            return [
                RecommendedAction(
                    action="request_fresh_gate_evidence",
                    description="Obtain fresh gate evidence and submit it with exact step binding.",
                    command='task(action="session-step", command="next")',
                )
            ]
        if normalized_error_code in {
            "STEP_PROOF_MISSING",
            "STEP_PROOF_MISMATCH",
            "STEP_PROOF_CONFLICT",
            "STEP_PROOF_EXPIRED",
        }:
            return [
                RecommendedAction(
                    action="request_fresh_step_proof",
                    description="Request a fresh step and resubmit with the exact server-issued step_proof token.",
                    command='task(action="session-step", command="next")',
                )
            ]
        if normalized_error_code in {
            "VERIFICATION_RECEIPT_MISSING",
            "VERIFICATION_RECEIPT_INVALID",
        }:
            return [
                RecommendedAction(
                    action="regenerate_verification_receipt",
                    description="Regenerate verification evidence and submit the server-issued verification_receipt fields.",
                    command='task(action="session-step", command="next")',
                )
            ]
        return [
            RecommendedAction(
                action="resolve_runtime_blocker",
                description="Resolve runtime policy or integrity blocker before retrying.",
            )
        ]

    if loop_signal == LoopSignal.FAILED:
        return [
            RecommendedAction(
                action="collect_failure_context",
                description="Inspect failure details and recent session events before retry.",
                command='task(action="session", command="status")',
            ),
            RecommendedAction(
                action="reset_or_restart_session",
                description="Reset or restart the session only after root-cause analysis.",
                command='task(action="session", command="reset")',
            ),
        ]

    return []


def compute_effective_status(session: AutonomousSessionState) -> Optional[SessionStatus]:
    """Compute effective status considering staleness.

    If the session is nominally RUNNING but heartbeat or step timestamps
    indicate staleness, the effective status is PAUSED.

    Args:
        session: Session state

    Returns:
        Derived status (PAUSED) if stale, None if the actual status applies.
    """
    if session.status != SessionStatus.RUNNING:
        return None

    now = datetime.now(timezone.utc)

    # Check step staleness
    if session.last_step_issued:
        step_stale_threshold = timedelta(minutes=session.limits.step_stale_minutes)
        if now - session.last_step_issued.issued_at > step_stale_threshold:
            return SessionStatus.PAUSED

    # Check heartbeat staleness
    if session.context.last_heartbeat_at:
        heartbeat_stale_threshold = timedelta(minutes=session.limits.heartbeat_stale_minutes)
        if now - session.context.last_heartbeat_at > heartbeat_stale_threshold:
            return SessionStatus.PAUSED
    else:
        # Pre-first-heartbeat: use heartbeat_grace_minutes from created_at
        grace_threshold = timedelta(minutes=session.limits.heartbeat_grace_minutes)
        if now - session.created_at > grace_threshold:
            return SessionStatus.PAUSED

    return None
