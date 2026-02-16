"""Step emission and determination logic for the StepOrchestrator.

This module provides the StepEmitterMixin class containing methods for:
- Step determination (steps 11-17 of the orchestration sequence)
- Phase/task helper logic
- Gate policy evaluation and invariant checks
- Step creation methods for all 6 step types

These are separated from the main orchestrator module to reduce file size
while preserving the single public StepOrchestrator API via mixin inheritance.

Usage:
    This module is not intended to be used directly. Import StepOrchestrator
    from the orchestrator module instead.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple

from ulid import ULID

from foundry_mcp.core.autonomy.audit import AuditEventType
from foundry_mcp.core.autonomy.models import (
    AutonomousSessionState,
    GatePolicy,
    GateVerdict,
    LastStepIssued,
    NextStep,
    PauseReason,
    PendingGateEvidence,
    PendingVerificationReceipt,
    PhaseGateRecord,
    PhaseGateStatus,
    SessionStatus,
    StepInstruction,
    StepType,
)

if TYPE_CHECKING:
    from pathlib import Path

    from foundry_mcp.core.autonomy.memory import AutonomyStorage


class _OrchestratorProtocol(Protocol):
    """Protocol describing methods the mixin expects from the host class."""

    storage: AutonomyStorage
    workspace_path: Path

    def _issue_step_proof(
        self, session_id: str, step_id: str, now: datetime
    ) -> str: ...

    def _emit_audit_event(
        self,
        session: AutonomousSessionState,
        event_type: AuditEventType,
        action: str,
        step_id: Optional[str] = None,
        phase_id: Optional[str] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...

logger = logging.getLogger(__name__)


# =============================================================================
# Error Code Constants (from ADR)
# =============================================================================

ERROR_STEP_RESULT_REQUIRED = "STEP_RESULT_REQUIRED"
ERROR_STEP_MISMATCH = "STEP_MISMATCH"
ERROR_STEP_PROOF_MISSING = "STEP_PROOF_MISSING"
ERROR_STEP_PROOF_MISMATCH = "STEP_PROOF_MISMATCH"
ERROR_STEP_PROOF_CONFLICT = "STEP_PROOF_CONFLICT"
ERROR_STEP_PROOF_EXPIRED = "STEP_PROOF_EXPIRED"
ERROR_INVALID_GATE_EVIDENCE = "INVALID_GATE_EVIDENCE"
ERROR_NO_ACTIVE_SESSION = "NO_ACTIVE_SESSION"
ERROR_AMBIGUOUS_ACTIVE_SESSION = "AMBIGUOUS_ACTIVE_SESSION"
ERROR_SESSION_UNRECOVERABLE = "SESSION_UNRECOVERABLE"
ERROR_SPEC_REBASE_REQUIRED = "SPEC_REBASE_REQUIRED"
ERROR_HEARTBEAT_STALE = "HEARTBEAT_STALE"
ERROR_STEP_STALE = "STEP_STALE"
ERROR_ALL_TASKS_BLOCKED = "ALL_TASKS_BLOCKED"
ERROR_GATE_BLOCKED = "GATE_BLOCKED"
ERROR_REQUIRED_GATE_UNSATISFIED = "REQUIRED_GATE_UNSATISFIED"
ERROR_VERIFICATION_RECEIPT_MISSING = "VERIFICATION_RECEIPT_MISSING"
ERROR_VERIFICATION_RECEIPT_INVALID = "VERIFICATION_RECEIPT_INVALID"
ERROR_GATE_INTEGRITY_CHECKSUM = "GATE_INTEGRITY_CHECKSUM"
ERROR_GATE_AUDIT_FAILURE = "GATE_AUDIT_FAILURE"


# =============================================================================
# Orchestration Result Types
# =============================================================================


@dataclass
class OrchestrationResult:
    """Result of computing the next step.

    Attributes:
        success: Whether the orchestration succeeded
        session: Updated session state (may be modified even on error)
        next_step: The next step to execute (None if terminal/paused)
        error_code: Error code if success is False
        error_message: Human-readable error message
        should_persist: Whether session should be persisted before responding
        replay_response: Cached response for replay (if applicable)
    """

    success: bool
    session: AutonomousSessionState
    next_step: Optional[NextStep] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    should_persist: bool = True
    replay_response: Optional[Dict[str, Any]] = None
    warnings: Optional[Dict[str, Any]] = None


# =============================================================================
# Step Emitter Mixin
# =============================================================================


class StepEmitterMixin:
    """Step determination and emission methods for StepOrchestrator.

    This mixin provides all step-emission logic (steps 11-17), phase/task
    helpers, gate policy evaluation, and the 6 step creation methods.

    It relies on the host class (StepOrchestrator) providing:
    - self._issue_step_proof(session_id, step_id, now) -> str
    - self._emit_audit_event(session, event_type, action, **kwargs)
    - self.storage: AutonomyStorage
    - self.workspace_path: Path
    """

    # =========================================================================
    # Gate Invariant Checks
    # =========================================================================

    def _check_required_gates_satisfied(
        self,
        session: AutonomousSessionState,
        phase_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Check if required gates are satisfied for invariant enforcement.

        Args:
            session: Session state
            phase_id: Specific phase to check, or None for all phases

        Returns:
            None if all required gates are satisfied, or a dict with:
            - phase_id: Phase with unsatisfied gate
            - gate_type: Type of the unsatisfied gate
            - blocking_reason: Human-readable reason
            - recovery_action: Dict with action/params to unblock
        """
        if phase_id:
            # Check specific phase
            phases_to_check = [phase_id]
        else:
            # Check all phases with required gates
            phases_to_check = list(session.required_phase_gates.keys())

        for check_phase_id in phases_to_check:
            required_gates = session.required_phase_gates.get(check_phase_id, [])
            satisfied_gates = session.satisfied_gates.get(check_phase_id, [])

            for gate_type in required_gates:
                if gate_type not in satisfied_gates:
                    # Check if gate record shows passed/waived (alternative satisfaction)
                    gate_record = session.phase_gates.get(check_phase_id)
                    if gate_record and gate_record.status in (
                        PhaseGateStatus.PASSED,
                        PhaseGateStatus.WAIVED,
                    ):
                        # Gate is satisfied via record, update satisfied_gates
                        if check_phase_id not in session.satisfied_gates:
                            session.satisfied_gates[check_phase_id] = []
                        if gate_type not in session.satisfied_gates[check_phase_id]:
                            session.satisfied_gates[check_phase_id].append(gate_type)
                        continue

                    # Gate is not satisfied
                    return {
                        "phase_id": check_phase_id,
                        "gate_type": gate_type,
                        "blocking_reason": (
                            f"Required gate '{gate_type}' for phase '{check_phase_id}' "
                            f"is not satisfied. Phase completion blocked."
                        ),
                        "recovery_action": {
                            "action": "gate-waiver",
                            "params": {
                                "phase_id": check_phase_id,
                                "reason_code": "operator_override",
                            },
                            "description": (
                                f"Use gate-waiver action to waive the required '{gate_type}' "
                                f"gate for phase '{check_phase_id}'. Requires maintainer role "
                                f"and allow_gate_waiver=true config."
                            ),
                        },
                    }

        return None

    def _audit_required_gate_integrity(
        self,
        session: AutonomousSessionState,
        spec_data: Optional[Dict[str, Any]],
        phase_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Independently audit gate integrity by rebuilding obligations from spec.

        This method complements _check_required_gates_satisfied by independently
        rebuilding gate obligations directly from spec phases and comparing with
        persisted gate records. This detects tampering where gate records may
        have been modified outside normal orchestrator flow.

        Args:
            session: Session state with persisted gate records
            spec_data: Spec data containing phase definitions
            phase_id: Specific phase to audit, or None for all phases

        Returns:
            None if all gates pass audit, or a dict with:
            - phase_id: Phase with audit failure
            - gate_type: Type of gate with failure
            - audit_failure_type: "missing_record", "invalid_status", "tampered"
            - details: Human-readable description of the failure
        """
        if not spec_data:
            return None

        # Rebuild gate obligations directly from spec phases
        phases_to_audit = []
        if phase_id:
            # Find specific phase in spec
            for phase in spec_data.get("phases", []):
                if phase.get("id") == phase_id:
                    phases_to_audit.append(phase)
                    break
        else:
            # Audit all phases
            phases_to_audit = spec_data.get("phases", [])

        for phase in phases_to_audit:
            check_phase_id = phase.get("id", "")
            if not check_phase_id:
                continue

            # Check if phase requires a fidelity gate (based on spec metadata)
            phase_metadata = phase.get("metadata", {})
            requires_gate = phase_metadata.get("requires_gate", False)

            # Also check session.required_phase_gates for this phase
            session_required = session.required_phase_gates.get(check_phase_id, [])
            gate_types_to_check = session_required if session_required else (["fidelity"] if requires_gate else [])

            for gate_type in gate_types_to_check:
                gate_record = session.phase_gates.get(check_phase_id)

                # Check 1: Missing gate record when required
                if not gate_record:
                    logger.warning(
                        "Gate audit failure: missing gate record for phase %s, gate_type %s",
                        check_phase_id,
                        gate_type,
                    )
                    return {
                        "phase_id": check_phase_id,
                        "gate_type": gate_type,
                        "audit_failure_type": "missing_record",
                        "details": (
                            f"Required gate '{gate_type}' for phase '{check_phase_id}' "
                            f"has no persisted gate record. Gate may have been deleted "
                            f"or never created."
                        ),
                    }

                # Check 2: Gate not in acceptable terminal state (passed or waived)
                if gate_record.status not in (PhaseGateStatus.PASSED, PhaseGateStatus.WAIVED):
                    logger.warning(
                        "Gate audit failure: gate status %s not acceptable for phase %s",
                        gate_record.status.value,
                        check_phase_id,
                    )
                    return {
                        "phase_id": check_phase_id,
                        "gate_type": gate_type,
                        "audit_failure_type": "invalid_status",
                        "details": (
                            f"Required gate '{gate_type}' for phase '{check_phase_id}' "
                            f"has status '{gate_record.status.value}', but only 'passed' "
                            f"or 'waived' are acceptable for terminal transitions."
                        ),
                    }

                # Check 3: Verify waiver has required metadata (if waived)
                if gate_record.status == PhaseGateStatus.WAIVED:
                    if not gate_record.waiver_reason_code:
                        logger.warning(
                            "Gate audit failure: waived gate missing reason_code for phase %s",
                            check_phase_id,
                        )
                        return {
                            "phase_id": check_phase_id,
                            "gate_type": gate_type,
                            "audit_failure_type": "tampered",
                            "details": (
                                f"Gate '{gate_type}' for phase '{check_phase_id}' is waived "
                                f"but missing required waiver_reason_code. This may indicate "
                                f"tampering with the gate record."
                            ),
                        }

        return None

    # =========================================================================
    # Pending Tasks Helpers
    # =========================================================================

    def _get_pending_tasks(
        self,
        spec_data: Dict[str, Any],
        completed_task_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Get all pending (not completed) tasks from spec."""
        pending = []
        for phase in spec_data.get("phases", []):
            for task in phase.get("tasks", []):
                task_id = task.get("id", "")
                if task_id and task_id not in completed_task_ids:
                    pending.append(task)
        return pending

    def _task_can_start(
        self,
        task: Dict[str, Any],
        completed_task_ids: List[str],
        spec_data: Dict[str, Any],
    ) -> bool:
        """Check if a task can start (all dependencies met)."""
        deps = task.get("depends", []) or task.get("dependencies", [])
        for dep_id in deps:
            if dep_id not in completed_task_ids:
                return False
        return True

    # =========================================================================
    # Step Determination (Steps 11-17)
    # =========================================================================

    def _determine_next_step(
        self,
        session: AutonomousSessionState,
        spec_data: Optional[Dict[str, Any]],
        now: datetime,
    ) -> OrchestrationResult:
        """Determine the next step based on current state.

        Implements steps 11-17 of the orchestration sequence.
        """
        if not spec_data:
            return self._create_complete_spec_result(session, now)

        # Get current phase info
        current_phase = self._get_current_phase(session, spec_data)

        # Step 11: Execute verification if phase tasks complete and verifications pending
        verification_task = self._find_pending_verification(session, spec_data)
        if verification_task:
            return self._create_verification_step(session, verification_task, now)

        # Step 12: Run fidelity gate if verifications complete and gate pending
        if current_phase and self._should_run_fidelity_gate(session, current_phase, spec_data):
            return self._create_fidelity_gate_step(session, current_phase, now)

        # Step 13-14: Handle gate failures
        if session.pending_gate_evidence:
            return self._handle_gate_evidence(session, spec_data, now)

        # Step 15: Gate passed + stop_on_phase_completion
        if (
            current_phase
            and self._phase_gate_passed(session, current_phase)
            and session.stop_conditions.stop_on_phase_completion
        ):
            # Enforce gate invariant: required gates must be satisfied
            gate_block = self._check_required_gates_satisfied(session, current_phase.get("id"))
            if gate_block:
                return OrchestrationResult(
                    success=False,
                    session=session,
                    error_code=ERROR_REQUIRED_GATE_UNSATISFIED,
                    error_message=gate_block["blocking_reason"],
                    should_persist=True,
                    warnings={
                        "gate_block": gate_block,
                    },
                )

            # Run independent gate audit before phase-close (P1.4)
            audit_failure = self._audit_required_gate_integrity(
                session, spec_data, current_phase.get("id")
            )
            if audit_failure:
                logger.warning(
                    "Gate audit failure on phase-close: %s",
                    audit_failure["details"],
                )
                return OrchestrationResult(
                    success=False,
                    session=session,
                    error_code=ERROR_GATE_AUDIT_FAILURE,
                    error_message=audit_failure["details"],
                    should_persist=True,
                    warnings={
                        "audit_failure": audit_failure,
                    },
                )

            return self._create_pause_result(
                session,
                PauseReason.PHASE_COMPLETE,
                now,
                f"Phase {current_phase.get('id')} complete. Stopping as configured.",
            )

        # Step 16: Gate passed + next task exists
        next_task = self._find_next_task(session, spec_data)
        if next_task:
            return self._create_implement_task_step(session, next_task, spec_data, now)

        # Step 17: No remaining tasks - complete spec
        # Enforce gate invariant: all phase gates must be satisfied
        gate_block = self._check_required_gates_satisfied(session)
        if gate_block:
            return OrchestrationResult(
                success=False,
                session=session,
                error_code=ERROR_REQUIRED_GATE_UNSATISFIED,
                error_message=gate_block["blocking_reason"],
                should_persist=True,
                warnings={
                    "gate_block": gate_block,
                },
            )

        # Run independent gate audit before spec-complete (P1.4)
        audit_failure = self._audit_required_gate_integrity(session, spec_data)
        if audit_failure:
            logger.warning(
                "Gate audit failure on spec-complete: %s",
                audit_failure["details"],
            )
            return OrchestrationResult(
                success=False,
                session=session,
                error_code=ERROR_GATE_AUDIT_FAILURE,
                error_message=audit_failure["details"],
                should_persist=True,
                warnings={
                    "audit_failure": audit_failure,
                },
            )

        return self._create_complete_spec_result(session, now)

    # =========================================================================
    # Phase / Task Helpers
    # =========================================================================

    def _get_current_phase(
        self,
        session: AutonomousSessionState,
        spec_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Get the current active phase from spec."""
        if not session.active_phase_id:
            # Find first phase with pending tasks
            for phase in spec_data.get("phases", []):
                phase_id = phase.get("id", "")
                if phase_id:
                    return phase
            return None

        for phase in spec_data.get("phases", []):
            if phase.get("id") == session.active_phase_id:
                return phase

        return None

    def _find_pending_verification(
        self,
        session: AutonomousSessionState,
        spec_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Find a pending verification task in the current phase."""
        current_phase = self._get_current_phase(session, spec_data)
        if not current_phase:
            return None

        # Verification steps are only eligible after all implementation tasks in
        # the phase are complete.
        if not self._all_implementation_tasks_complete(
            current_phase,
            session.completed_task_ids,
        ):
            return None

        for task in current_phase.get("tasks", []):
            task_id = task.get("id", "")
            task_type = task.get("type", "task")
            if (
                task_type == "verify"
                and task_id not in session.completed_task_ids
                and task.get("status") != "completed"
            ):
                return task

        return None

    def _all_implementation_tasks_complete(
        self,
        phase: Dict[str, Any],
        completed_task_ids: List[str],
    ) -> bool:
        """Check whether all implementation tasks in a phase are complete."""
        for task in phase.get("tasks", []):
            task_type = task.get("type", "task")
            task_id = task.get("id", "")
            if task_type == "task" and task_id not in completed_task_ids:
                return False
        return True

    def _should_run_fidelity_gate(
        self,
        session: AutonomousSessionState,
        phase: Dict[str, Any],
        spec_data: Dict[str, Any],
    ) -> bool:
        """Check if fidelity gate should run for the phase."""
        phase_id = phase.get("id", "")
        if not phase_id:
            return False

        # Check if gate already passed
        gate_record = session.phase_gates.get(phase_id)
        if gate_record and gate_record.status == PhaseGateStatus.PASSED:
            return False

        # Check if all implementation tasks are complete
        if not self._all_implementation_tasks_complete(
            phase,
            session.completed_task_ids,
        ):
            return False

        # Check if verifications are complete (or no verifications exist)
        for task in phase.get("tasks", []):
            task_type = task.get("type", "task")
            task_id = task.get("id", "")
            if task_type == "verify" and task_id not in session.completed_task_ids:
                return False  # Verifications still pending

        return True

    def _phase_gate_passed(
        self,
        session: AutonomousSessionState,
        phase: Dict[str, Any],
    ) -> bool:
        """Check if the phase gate has passed."""
        phase_id = phase.get("id", "")
        if not phase_id:
            return False

        gate_record = session.phase_gates.get(phase_id)
        return gate_record is not None and gate_record.status == PhaseGateStatus.PASSED

    def _evaluate_gate_policy(
        self,
        session: AutonomousSessionState,
        evidence: PendingGateEvidence,
    ) -> Tuple[bool, Optional[PauseReason]]:
        """Evaluate gate verdict against the configured gate policy.

        Policy behaviors:
        - STRICT: Pass only on verdict=pass
        - LENIENT: Pass on verdict=pass or verdict=warn
        - MANUAL: Always pause for manual review

        Returns:
            Tuple of (should_pass, pause_reason_if_any)
        """
        policy = session.gate_policy

        if policy == GatePolicy.MANUAL:
            # Manual policy always requires human review
            return False, PauseReason.GATE_REVIEW_REQUIRED

        if policy == GatePolicy.STRICT:
            # Strict: only pass on explicit pass
            if evidence.verdict == GateVerdict.PASS:
                return True, None
            return False, PauseReason.GATE_FAILED

        if policy == GatePolicy.LENIENT:
            # Lenient: pass on pass or warn
            if evidence.verdict in (GateVerdict.PASS, GateVerdict.WARN):
                return True, None
            return False, PauseReason.GATE_FAILED

        # Default to strict behavior
        return evidence.verdict == GateVerdict.PASS, None

    def _handle_gate_evidence(
        self,
        session: AutonomousSessionState,
        spec_data: Dict[str, Any],
        now: datetime,
    ) -> OrchestrationResult:
        """Handle pending gate evidence (steps 13-14).

        Applies gate policy evaluation:
        - STRICT: pass only on verdict=pass
        - LENIENT: pass on pass or warn
        - MANUAL: always pause with gate_review_required

        Auto-retry behavior:
        - If auto_retry_fidelity_gate=true and gate fails: address_fidelity_feedback -> retry
        - If auto_retry_fidelity_gate=false and gate fails: pause with gate_failed
        """
        evidence = session.pending_gate_evidence
        if not evidence:
            return self._create_complete_spec_result(session, now)

        # Evaluate gate against policy
        should_pass, pause_reason = self._evaluate_gate_policy(session, evidence)

        if should_pass:
            # Gate passed according to policy - clear evidence and continue
            session.pending_gate_evidence = None
            return self._determine_next_step_after_gate(session, spec_data, now)

        # Gate failed or manual review required
        if pause_reason == PauseReason.GATE_REVIEW_REQUIRED:
            # Manual policy always requires human acknowledgment
            return self._create_pause_result(
                session,
                PauseReason.GATE_REVIEW_REQUIRED,
                now,
                f"Manual gate review required for phase {evidence.phase_id}. "
                f"Verdict: {evidence.verdict.value}. Acknowledge to continue.",
            )

        # Gate failed - check auto-retry setting and cycle cap (ADR line 675)
        if (
            session.stop_conditions.auto_retry_fidelity_gate
            and session.counters.fidelity_review_cycles_in_active_phase
            < session.limits.max_fidelity_review_cycles_per_phase
        ):
            # Create address_fidelity_feedback step for auto-retry cycle
            return self._create_fidelity_feedback_step(session, evidence, now)
        else:
            # No auto-retry - pause for manual intervention
            return self._create_pause_result(
                session,
                PauseReason.GATE_FAILED,
                now,
                f"Fidelity gate failed for phase {evidence.phase_id}. "
                f"Verdict: {evidence.verdict.value}. Manual review required.",
            )

    def _determine_next_step_after_gate(
        self,
        session: AutonomousSessionState,
        spec_data: Dict[str, Any],
        now: datetime,
    ) -> OrchestrationResult:
        """Determine next step after gate passes.

        The fidelity cycle counter is incremented in _handle_gate_evidence
        when the gate is accepted. This method handles phase transition
        and task continuation.
        """
        # Check if we should stop on phase completion
        if session.stop_conditions.stop_on_phase_completion:
            # Enforce gate invariant: required gates must be satisfied
            gate_block = self._check_required_gates_satisfied(session, session.active_phase_id)
            if gate_block:
                return OrchestrationResult(
                    success=False,
                    session=session,
                    error_code=ERROR_REQUIRED_GATE_UNSATISFIED,
                    error_message=gate_block["blocking_reason"],
                    should_persist=True,
                    warnings={
                        "gate_block": gate_block,
                    },
                )

            return self._create_pause_result(
                session,
                PauseReason.PHASE_COMPLETE,
                now,
                f"Phase {session.active_phase_id} complete. Stopping as configured.",
            )

        # Look for more tasks in the current or next phase
        next_task = self._find_next_task(session, spec_data)
        if next_task:
            return self._create_implement_task_step(session, next_task, spec_data, now)

        # No remaining tasks - complete spec
        # Enforce gate invariant: all phase gates must be satisfied
        gate_block = self._check_required_gates_satisfied(session)
        if gate_block:
            return OrchestrationResult(
                success=False,
                session=session,
                error_code=ERROR_REQUIRED_GATE_UNSATISFIED,
                error_message=gate_block["blocking_reason"],
                should_persist=True,
                warnings={
                    "gate_block": gate_block,
                },
            )

        return self._create_complete_spec_result(session, now)

    def _find_next_task(
        self,
        session: AutonomousSessionState,
        spec_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Find the next task to implement."""
        for phase in spec_data.get("phases", []):
            phase_id = phase.get("id", "")

            # Skip phases with failed or pending gates (if gate exists)
            gate_record = session.phase_gates.get(phase_id)
            if gate_record and gate_record.status == PhaseGateStatus.FAILED:
                continue

            for task in phase.get("tasks", []):
                task_id = task.get("id", "")
                task_type = task.get("type", "task")

                # Only consider implementation tasks
                if task_type != "task":
                    continue

                if task_id and task_id not in session.completed_task_ids:
                    # Check if task can start
                    if self._task_can_start(task, session.completed_task_ids, spec_data):
                        # Check for phase advance - reset cycle counter on phase change
                        if session.active_phase_id != phase_id:
                            logger.info(
                                "Phase advance detected: %s -> %s, resetting fidelity cycle counter",
                                session.active_phase_id,
                                phase_id,
                            )
                            session.counters.fidelity_review_cycles_in_active_phase = 0
                        # Update active phase
                        session.active_phase_id = phase_id
                        return task

        return None

    # =========================================================================
    # Step Builders
    # =========================================================================

    def _create_pause_result(
        self,
        session: AutonomousSessionState,
        reason: PauseReason,
        now: datetime,
        message: str,
    ) -> OrchestrationResult:
        """Create a pause step result."""
        session.status = SessionStatus.PAUSED
        session.paused_at = now
        session.pause_reason = reason
        session.updated_at = now
        session.state_version += 1

        step_id = f"step_{ULID()}"
        step_proof = self._issue_step_proof(session.id, step_id, now)
        next_step = NextStep(
            step_id=step_id,
            type=StepType.PAUSE,
            task_id=None,
            phase_id=None,
            task_title=None,
            gate_attempt_id=None,
            instructions=None,
            reason=reason,
            message=message,
            step_proof=step_proof,
        )

        # Update last step issued and cache response
        session.last_step_issued = LastStepIssued(
            step_id=step_id,
            type=StepType.PAUSE,
            task_id=None,
            phase_id=None,
            issued_at=now,
            step_proof=step_proof,
        )

        self._emit_audit_event(
            session,
            AuditEventType.PAUSE,
            action="session_pause",
            step_id=step_id,
            metadata={"reason": reason.value},
        )

        return OrchestrationResult(
            success=True,
            session=session,
            next_step=next_step,
            should_persist=True,
        )

    def _create_implement_task_step(
        self,
        session: AutonomousSessionState,
        task: Dict[str, Any],
        spec_data: Dict[str, Any],
        now: datetime,
    ) -> OrchestrationResult:
        """Create an implement_task step."""
        task_id = task.get("id", "")
        task_title = task.get("title", "")
        phase_id = session.active_phase_id or ""

        step_id = f"step_{ULID()}"
        step_proof = self._issue_step_proof(session.id, step_id, now)

        instructions = [
            StepInstruction(
                tool="task",
                action="prepare",
                description="Load task context and acceptance criteria",
            ),
            StepInstruction(
                tool="task",
                action="update-status",
                description="Mark task as in-progress",
            ),
            StepInstruction(
                tool="task",
                action="complete",
                description="Mark task as complete after implementation",
            ),
        ]

        next_step = NextStep(
            step_id=step_id,
            type=StepType.IMPLEMENT_TASK,
            task_id=task_id,
            phase_id=phase_id,
            task_title=task_title,
            gate_attempt_id=None,
            instructions=instructions,
            reason=None,
            message=None,
            step_proof=step_proof,
        )

        # Update last step issued
        session.last_step_issued = LastStepIssued(
            step_id=step_id,
            type=StepType.IMPLEMENT_TASK,
            task_id=task_id,
            phase_id=phase_id,
            issued_at=now,
            step_proof=step_proof,
        )
        session.last_task_id = task_id
        session.status = SessionStatus.RUNNING
        session.updated_at = now
        session.state_version += 1

        self._emit_audit_event(
            session,
            AuditEventType.STEP_ISSUED,
            action="issue_step",
            step_id=step_id,
            phase_id=phase_id,
            task_id=task_id,
            metadata={"step_type": "implement_task"},
        )

        return OrchestrationResult(
            success=True,
            session=session,
            next_step=next_step,
            should_persist=True,
        )

    def _create_verification_step(
        self,
        session: AutonomousSessionState,
        task: Dict[str, Any],
        now: datetime,
    ) -> OrchestrationResult:
        """Create an execute_verification step with pending receipt."""
        task_id = task.get("id", "")
        task_title = task.get("title", "")
        phase_id = session.active_phase_id or ""

        step_id = f"step_{ULID()}"
        step_proof = self._issue_step_proof(session.id, step_id, now)

        # Extract verification command from task metadata for receipt
        task_metadata = task.get("metadata", {})
        command = task_metadata.get("command", "")
        command_hash = hashlib.sha256(command.encode()).hexdigest() if command else ""

        # Create pending verification receipt for later validation
        if command_hash:
            session.pending_verification_receipt = PendingVerificationReceipt(
                step_id=step_id,
                task_id=task_id,
                expected_command_hash=command_hash,
                issued_at=now,
            )
        else:
            # Clear any stale pending receipt if no command
            session.pending_verification_receipt = None

        instructions = [
            StepInstruction(
                tool="task",
                action="info",
                description="Get verification task details",
            ),
            StepInstruction(
                tool="verification",
                action="execute",
                description="Execute verification commands",
            ),
            StepInstruction(
                tool="task",
                action="complete",
                description="Record verification results",
            ),
        ]

        next_step = NextStep(
            step_id=step_id,
            type=StepType.EXECUTE_VERIFICATION,
            task_id=task_id,
            phase_id=phase_id,
            task_title=task_title,
            gate_attempt_id=None,
            instructions=instructions,
            reason=None,
            message=None,
            step_proof=step_proof,
        )

        session.last_step_issued = LastStepIssued(
            step_id=step_id,
            type=StepType.EXECUTE_VERIFICATION,
            task_id=task_id,
            phase_id=phase_id,
            issued_at=now,
            step_proof=step_proof,
        )
        session.updated_at = now
        session.state_version += 1

        self._emit_audit_event(
            session,
            AuditEventType.STEP_ISSUED,
            action="issue_step",
            step_id=step_id,
            phase_id=phase_id,
            task_id=task_id,
            metadata={"step_type": "execute_verification"},
        )

        return OrchestrationResult(
            success=True,
            session=session,
            next_step=next_step,
            should_persist=True,
        )

    def _create_fidelity_gate_step(
        self,
        session: AutonomousSessionState,
        phase: Dict[str, Any],
        now: datetime,
    ) -> OrchestrationResult:
        """Create a run_fidelity_gate step."""
        phase_id = phase.get("id", "")
        phase_title = phase.get("title", "")
        gate_attempt_id = f"gate_{ULID()}"

        step_id = f"step_{ULID()}"
        step_proof = self._issue_step_proof(session.id, step_id, now)

        # Initialize gate record if not exists
        if phase_id not in session.phase_gates:
            session.phase_gates[phase_id] = PhaseGateRecord(
                required=True,
                status=PhaseGateStatus.PENDING,
                verdict=None,
                gate_attempt_id=None,
                review_path=None,
                evaluated_at=None,
            )

        instructions = [
            StepInstruction(
                tool="review",
                action="fidelity-gate",
                description=f"Run fidelity review for phase {phase_title}",
            ),
            StepInstruction(
                tool="task",
                action="session-step-next",
                description="Report gate outcome and request the next step",
            ),
        ]

        next_step = NextStep(
            step_id=step_id,
            type=StepType.RUN_FIDELITY_GATE,
            task_id=None,
            phase_id=phase_id,
            task_title=None,
            gate_attempt_id=gate_attempt_id,
            instructions=instructions,
            reason=None,
            message=None,
            step_proof=step_proof,
        )

        session.last_step_issued = LastStepIssued(
            step_id=step_id,
            type=StepType.RUN_FIDELITY_GATE,
            task_id=None,
            phase_id=phase_id,
            issued_at=now,
            step_proof=step_proof,
        )
        session.phase_gates[phase_id].gate_attempt_id = gate_attempt_id
        session.updated_at = now
        session.state_version += 1

        self._emit_audit_event(
            session,
            AuditEventType.STEP_ISSUED,
            action="issue_step",
            step_id=step_id,
            phase_id=phase_id,
            metadata={"step_type": "run_fidelity_gate", "gate_attempt_id": gate_attempt_id},
        )

        return OrchestrationResult(
            success=True,
            session=session,
            next_step=next_step,
            should_persist=True,
        )

    def _create_fidelity_feedback_step(
        self,
        session: AutonomousSessionState,
        evidence: Any,  # PendingGateEvidence
        now: datetime,
    ) -> OrchestrationResult:
        """Create an address_fidelity_feedback step."""
        step_id = f"step_{ULID()}"
        step_proof = self._issue_step_proof(session.id, step_id, now)

        instructions = [
            StepInstruction(
                tool="review",
                action="fidelity",
                description="Inspect phase fidelity findings before remediation",
            ),
            StepInstruction(
                tool="task",
                action="prepare",
                description="Load task context to implement remediation",
            ),
            StepInstruction(
                tool="task",
                action="session-step-next",
                description="Report remediation progress and request a gate retry",
            ),
        ]

        next_step = NextStep(
            step_id=step_id,
            type=StepType.ADDRESS_FIDELITY_FEEDBACK,
            task_id=None,
            phase_id=evidence.phase_id,
            task_title=None,
            gate_attempt_id=evidence.gate_attempt_id,
            instructions=instructions,
            reason=None,
            message=None,
            step_proof=step_proof,
        )

        session.last_step_issued = LastStepIssued(
            step_id=step_id,
            type=StepType.ADDRESS_FIDELITY_FEEDBACK,
            task_id=None,
            phase_id=evidence.phase_id,
            issued_at=now,
            step_proof=step_proof,
        )
        session.updated_at = now
        session.state_version += 1

        self._emit_audit_event(
            session,
            AuditEventType.STEP_ISSUED,
            action="issue_step",
            step_id=step_id,
            phase_id=evidence.phase_id,
            metadata={"step_type": "address_fidelity_feedback", "gate_attempt_id": evidence.gate_attempt_id},
        )

        return OrchestrationResult(
            success=True,
            session=session,
            next_step=next_step,
            should_persist=True,
        )

    def _create_complete_spec_result(
        self,
        session: AutonomousSessionState,
        now: datetime,
    ) -> OrchestrationResult:
        """Create a complete_spec result (terminal state)."""
        session.status = SessionStatus.COMPLETED
        session.updated_at = now
        session.state_version += 1

        step_id = f"step_{ULID()}"
        step_proof = self._issue_step_proof(session.id, step_id, now)
        next_step = NextStep(
            step_id=step_id,
            type=StepType.COMPLETE_SPEC,
            task_id=None,
            phase_id=None,
            task_title=None,
            gate_attempt_id=None,
            instructions=None,
            reason=None,
            message=f"Spec {session.spec_id} execution completed. "
            f"Total tasks: {session.counters.tasks_completed}",
            step_proof=step_proof,
        )

        session.last_step_issued = LastStepIssued(
            step_id=step_id,
            type=StepType.COMPLETE_SPEC,
            task_id=None,
            phase_id=None,
            issued_at=now,
            step_proof=step_proof,
        )

        self._emit_audit_event(
            session,
            AuditEventType.SPEC_COMPLETE,
            action="complete_spec",
            step_id=step_id,
            metadata={"tasks_completed": session.counters.tasks_completed},
        )

        return OrchestrationResult(
            success=True,
            session=session,
            next_step=next_step,
            should_persist=True,
        )
