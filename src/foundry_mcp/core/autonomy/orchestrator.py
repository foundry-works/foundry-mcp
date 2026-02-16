"""Step orchestration engine for autonomous spec execution.

This module implements the 18-step orchestration rules defined in ADR-002
for driving autonomous task progression with replay-safe semantics.

The orchestrator handles:
- Replay detection for exactly-once semantics
- Feedback validation and step identity verification
- Spec integrity validation with mtime optimization
- Pause guards (context, error, task limits)
- Staleness detection (step and heartbeat)
- Fidelity gate cycles and phase completion
- Step emission for all 6 step types

Step emission and determination logic (steps 11-17) are provided by
StepEmitterMixin in step_emitters.py.

Usage:
    from foundry_mcp.core.autonomy.orchestrator import StepOrchestrator

    orchestrator = StepOrchestrator(storage, spec_loader)
    result = orchestrator.compute_next_step(session, last_step_result)
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ulid import ULID

from foundry_mcp.core.autonomy.context_tracker import ContextTracker
from foundry_mcp.core.autonomy.memory import AutonomyStorage
from foundry_mcp.core.autonomy.models import (
    AutonomousSessionState,
    FailureReason,
    GateVerdict,
    LastStepIssued,
    LastStepResult,
    PauseReason,
    PhaseGateStatus,
    SessionStatus,
    SessionStepResponseData,
    StepOutcome,
    StepType,
)
from foundry_mcp.core.autonomy.spec_hash import (
    compute_spec_structure_hash,
    get_spec_file_metadata,
)
from foundry_mcp.core.autonomy.server_secret import (
    verify_integrity_checksum,
)
from foundry_mcp.core.autonomy.audit import AuditEventType, AuditLedger
from foundry_mcp.core.spec import resolve_spec_file
from foundry_mcp.core.task._helpers import check_all_blocked

# Import mixin, result type, and error constants from step_emitters
from foundry_mcp.core.autonomy.step_emitters import (
    OrchestrationResult,
    StepEmitterMixin,
    ERROR_STEP_RESULT_REQUIRED,
    ERROR_STEP_MISMATCH,
    ERROR_STEP_PROOF_MISSING,
    ERROR_STEP_PROOF_MISMATCH,
    ERROR_STEP_PROOF_CONFLICT,
    ERROR_STEP_PROOF_EXPIRED,
    ERROR_INVALID_GATE_EVIDENCE,
    ERROR_NO_ACTIVE_SESSION,
    ERROR_AMBIGUOUS_ACTIVE_SESSION,
    ERROR_SESSION_UNRECOVERABLE,
    ERROR_SPEC_REBASE_REQUIRED,
    ERROR_HEARTBEAT_STALE,
    ERROR_STEP_STALE,
    ERROR_ALL_TASKS_BLOCKED,
    ERROR_GATE_BLOCKED,
    ERROR_REQUIRED_GATE_UNSATISFIED,
    ERROR_VERIFICATION_RECEIPT_MISSING,
    ERROR_VERIFICATION_RECEIPT_INVALID,
    ERROR_GATE_INTEGRITY_CHECKSUM,
    ERROR_GATE_AUDIT_FAILURE,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Step Orchestrator
# =============================================================================


class StepOrchestrator(StepEmitterMixin):
    """Orchestrates autonomous session step progression.

    Implements the 18-step priority sequence from ADR-002:
    0. Replay detection - return cached response if step already processed
    1. Validate feedback - require last_step_result on non-initial calls
    2. Validate step identity - check step ID/type/task-or-phase binding
    3. Record step outcome - mark task complete, increment errors, record gate
    4. Validate spec integrity - mtime fast-path, re-hash at phase boundaries
    5. Check terminal states - return terminal status with next_step=null
    6. Enforce step staleness - hard backstop for disappeared callers
    7. Enforce heartbeat staleness - cooperative signal with grace window
    8. Enforce pause guards - context_limit, error_threshold, task_limit
    9. Enforce fidelity-cycle stop heuristic - prevent spinning
    10. Check all-blocked - pause if no unblocked work
    11. Execute verification - if phase tasks complete and verifications pending
    12. Run fidelity gate - if verifications complete and gate pending
    13. Gate fails policy - pause or address_fidelity_feedback (auto-retry)
    14. Fidelity feedback completed - run gate retry
    15. Gate passed + stop_on_phase_completion - pause (phase_complete)
    16. Gate passed + next task exists - implement_task
    17. No remaining tasks - complete_spec, transition to completed

    Steps 11-17 are provided by StepEmitterMixin.
    """

    def __init__(
        self,
        storage: AutonomyStorage,
        spec_loader: Any,
        workspace_path: Optional[Path] = None,
    ) -> None:
        """Initialize the step orchestrator.

        Args:
            storage: AutonomyStorage instance for session persistence
            spec_loader: Function or object to load spec data
            workspace_path: Path to workspace for spec file resolution
        """
        self.storage = storage
        self.spec_loader = spec_loader
        self.workspace_path = workspace_path or Path.cwd()
        # Cache: (spec_id, mtime, file_size) -> spec_data
        self._spec_cache: Optional[Tuple[str, float, int, Dict[str, Any]]] = None
        self._context_tracker = ContextTracker(self.workspace_path)
        # Audit ledger cache keyed by spec_id
        self._audit_ledgers: Dict[str, AuditLedger] = {}

    def invalidate_spec_cache(self, spec_id: Optional[str] = None) -> None:
        """Invalidate the spec data cache.

        Call after rebase or any operation that modifies spec on disk.

        Args:
            spec_id: If provided, only invalidate if cached spec matches.
                     If None, unconditionally clear the cache.
        """
        if spec_id is None:
            self._spec_cache = None
        elif self._spec_cache is not None and self._spec_cache[0] == spec_id:
            self._spec_cache = None

    def _get_ledger(self, spec_id: str) -> AuditLedger:
        """Get or create an audit ledger for the given spec."""
        if spec_id not in self._audit_ledgers:
            self._audit_ledgers[spec_id] = AuditLedger(
                spec_id=spec_id,
                workspace_path=self.workspace_path,
            )
        return self._audit_ledgers[spec_id]

    def _emit_audit_event(
        self,
        session: AutonomousSessionState,
        event_type: AuditEventType,
        action: str,
        step_id: Optional[str] = None,
        phase_id: Optional[str] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit an audit event (best-effort, never fails the calling operation)."""
        try:
            ledger = self._get_ledger(session.spec_id)
            ledger.append(
                event_type=event_type,
                action=action,
                session_id=session.id,
                step_id=step_id,
                phase_id=phase_id,
                task_id=task_id,
                metadata=metadata,
            )
        except (OSError, ValueError, TimeoutError) as e:
            logger.warning("Audit event emission failed (non-critical): %s", e)

    def compute_next_step(
        self,
        session: AutonomousSessionState,
        last_step_result: Optional[LastStepResult] = None,
        context_usage_pct: Optional[int] = None,
        heartbeat_at: Optional[datetime] = None,
    ) -> OrchestrationResult:
        """Compute the next step for an autonomous session.

        This is the main entry point for step orchestration. It implements
        the full 18-step priority sequence.

        Args:
            session: Current session state
            last_step_result: Result of the previous step (required on non-initial calls)
            context_usage_pct: Caller-reported context usage percentage
            heartbeat_at: Timestamp for heartbeat update

        Returns:
            OrchestrationResult with next step or error
        """
        now = datetime.now(timezone.utc)

        # Update context usage via the tracker (Tier 1/2/3 fallthrough)
        effective_pct, source = self._context_tracker.get_effective_context_pct(
            session, context_usage_pct, now
        )
        session.context.context_usage_pct = effective_pct
        session.context.context_source = source
        self._context_tracker.update_step_counter(session)

        if heartbeat_at is not None:
            session.context.last_heartbeat_at = heartbeat_at

        # =================================================================
        # Step 0: Replay Detection
        # =================================================================
        if last_step_result is not None and session.last_step_issued is not None:
            if last_step_result.step_id == session.last_step_issued.step_id:
                proof_enforced = bool(
                    session.last_step_issued.step_proof and last_step_result.step_proof
                )
                # Check if we already processed this step
                if session.last_issued_response is not None and not proof_enforced:
                    logger.info(
                        "Replay detected: returning cached response for step %s",
                        last_step_result.step_id,
                    )
                    return OrchestrationResult(
                        success=True,
                        session=session,
                        next_step=None,  # Response will use cached response
                        replay_response=session.last_issued_response,
                        should_persist=False,
                    )

        # =================================================================
        # Step 1: Validate Feedback
        # =================================================================
        if session.last_step_issued is not None and last_step_result is None:
            logger.warning("STEP_RESULT_REQUIRED: non-initial call without feedback")
            return OrchestrationResult(
                success=False,
                session=session,
                error_code=ERROR_STEP_RESULT_REQUIRED,
                error_message=(
                    f"last_step_result is required on non-initial calls. "
                    f"Previous step: {session.last_step_issued.step_id}"
                ),
                should_persist=False,
            )

        # =================================================================
        # Step 2: Validate Step Identity
        # =================================================================
        if last_step_result is not None and session.last_step_issued is not None:
            proof_error_code, proof_error_message = self._validate_step_proof(
                session.last_step_issued,
                last_step_result,
            )
            if proof_error_message:
                logger.warning("%s: %s", proof_error_code, proof_error_message)
                return OrchestrationResult(
                    success=False,
                    session=session,
                    error_code=proof_error_code,
                    error_message=proof_error_message,
                    should_persist=False,
                )
            mismatch_reason = self._validate_step_identity(
                session.last_step_issued, last_step_result
            )
            if mismatch_reason:
                logger.warning("STEP_MISMATCH: %s", mismatch_reason)
                return OrchestrationResult(
                    success=False,
                    session=session,
                    error_code=ERROR_STEP_MISMATCH,
                    error_message=mismatch_reason,
                    should_persist=False,
                )

        # Validate gate evidence for fidelity gate steps
        if last_step_result is not None and last_step_result.step_type == StepType.RUN_FIDELITY_GATE:
            gate_error = self._validate_gate_evidence(session, last_step_result)
            if gate_error:
                gate_error_code = ERROR_INVALID_GATE_EVIDENCE
                if "integrity checksum" in gate_error.lower():
                    gate_error_code = ERROR_GATE_INTEGRITY_CHECKSUM
                logger.warning("%s: %s", gate_error_code, gate_error)
                return OrchestrationResult(
                    success=False,
                    session=session,
                    error_code=gate_error_code,
                    error_message=gate_error,
                    should_persist=False,
                )

        # Validate verification receipt for EXECUTE_VERIFICATION steps
        if last_step_result is not None and last_step_result.step_type == StepType.EXECUTE_VERIFICATION:
            receipt_error = self._validate_verification_receipt(session, last_step_result)
            if receipt_error:
                error_code = ERROR_VERIFICATION_RECEIPT_MISSING
                if last_step_result.verification_receipt is not None:
                    error_code = ERROR_VERIFICATION_RECEIPT_INVALID
                logger.warning("%s: %s", error_code, receipt_error)
                return OrchestrationResult(
                    success=False,
                    session=session,
                    error_code=error_code,
                    error_message=receipt_error,
                    should_persist=False,
                )

        # =================================================================
        # Step 3: Record Step Outcome
        # =================================================================
        if last_step_result is not None:
            # Clear cached response since we're processing new feedback
            session.last_issued_response = None
            self._record_step_outcome(session, last_step_result, now)

        # =================================================================
        # Step 4: Validate Spec Integrity
        # =================================================================
        spec_data, integrity_error = self._validate_spec_integrity(session, now)
        if integrity_error:
            session.status = SessionStatus.FAILED
            session.failure_reason = FailureReason.SPEC_STRUCTURE_CHANGED
            session.updated_at = now
            session.state_version += 1
            return OrchestrationResult(
                success=False,
                session=session,
                error_code=ERROR_SPEC_REBASE_REQUIRED,
                error_message=integrity_error,
                should_persist=True,
            )

        # =================================================================
        # Step 5: Check Terminal States
        # Per ADR: only COMPLETED and ENDED are truly terminal.
        # FAILED can transition to running via resume(force=true) or rebase.
        # =================================================================
        if session.status in (SessionStatus.COMPLETED, SessionStatus.ENDED):
            logger.info("Session %s is in terminal state: %s", session.id, session.status)
            return OrchestrationResult(
                success=True,
                session=session,
                next_step=None,
                should_persist=False,
            )

        # =================================================================
        # Step 6: Enforce Step Staleness (Hard Backstop)
        # =================================================================
        if session.last_step_issued is not None:
            step_stale_threshold = timedelta(minutes=session.limits.step_stale_minutes)
            if now - session.last_step_issued.issued_at > step_stale_threshold:
                logger.warning(
                    "Step staleness detected: step %s issued at %s",
                    session.last_step_issued.step_id,
                    session.last_step_issued.issued_at.isoformat(),
                )
                return OrchestrationResult(
                    success=False,
                    session=session,
                    error_code=ERROR_STEP_STALE,
                    error_message=(
                        f"Step {session.last_step_issued.step_id} is stale "
                        f"(issued {session.limits.step_stale_minutes}+ minutes ago). "
                        f"Session may need to be reset."
                    ),
                    should_persist=False,
                )

        # =================================================================
        # Step 7: Enforce Heartbeat Staleness
        # =================================================================
        heartbeat_stale_warning = False
        heartbeat_result = self._check_heartbeat_staleness(session, now)
        if isinstance(heartbeat_result, OrchestrationResult):
            return heartbeat_result
        elif heartbeat_result is True:
            # Heartbeat is stale but step is active — ADR: warn, don't pause
            heartbeat_stale_warning = True

        # =================================================================
        # Step 8: Enforce Pause Guards
        # =================================================================
        pause_guard_result = self._check_pause_guards(session)
        if pause_guard_result:
            return pause_guard_result

        # =================================================================
        # Step 9: Enforce Fidelity-Cycle Stop Heuristic
        # =================================================================
        if (
            session.counters.fidelity_review_cycles_in_active_phase
            >= session.limits.max_fidelity_review_cycles_per_phase
        ):
            logger.warning(
                "Fidelity cycle limit reached: %d cycles in phase %s",
                session.counters.fidelity_review_cycles_in_active_phase,
                session.active_phase_id,
            )
            return self._create_pause_result(
                session,
                PauseReason.FIDELITY_CYCLE_LIMIT,
                now,
                (
                    f"Max fidelity review cycles ({session.limits.max_fidelity_review_cycles_per_phase}) "
                    f"reached in phase {session.active_phase_id}. Manual review required."
                ),
            )

        # =================================================================
        # Step 10: Check All-Blocked
        # =================================================================
        all_blocked_result = self._check_all_blocked(session, spec_data)
        if all_blocked_result:
            return all_blocked_result

        # =================================================================
        # Steps 11-17: Determine Next Action Based on Phase State
        # =================================================================
        next_step_result = self._determine_next_step(session, spec_data, now)

        # Propagate heartbeat_stale_warning into result details (ADR line 662)
        if heartbeat_stale_warning:
            if next_step_result.warnings is None:
                next_step_result.warnings = {}
            next_step_result.warnings["heartbeat_stale_warning"] = True

        # Cache the response for replay-safe exactly-once semantics
        if next_step_result.success and next_step_result.next_step is not None:
            # Compute gate invariant fields for observability
            required_phase_gates: list[str] = []
            satisfied_gates: list[str] = []
            missing_required_gates: list[str] = []

            for phase_id, gate_record in session.phase_gates.items():
                if gate_record.required:
                    required_phase_gates.append(phase_id)
                    if gate_record.status in (PhaseGateStatus.PASSED, PhaseGateStatus.WAIVED):
                        satisfied_gates.append(phase_id)
                    elif gate_record.status in (PhaseGateStatus.PENDING, PhaseGateStatus.FAILED):
                        missing_required_gates.append(phase_id)

            response_data = SessionStepResponseData(
                session_id=session.id,
                status=session.status,
                state_version=session.state_version,
                next_step=next_step_result.next_step,
                required_phase_gates=required_phase_gates if required_phase_gates else None,
                satisfied_gates=satisfied_gates if satisfied_gates else None,
                missing_required_gates=missing_required_gates if missing_required_gates else None,
            )
            session.last_issued_response = response_data.model_dump(
                mode="json", by_alias=True
            )

        return next_step_result

    # =========================================================================
    # Validation Methods
    # =========================================================================

    def _validate_step_identity(
        self,
        last_issued: LastStepIssued,
        result: LastStepResult,
    ) -> Optional[str]:
        """Validate that the reported step matches the last issued step.

        Returns:
            None if valid, error message if mismatch
        """
        if result.step_id != last_issued.step_id:
            return (
                f"Step ID mismatch: expected {last_issued.step_id}, "
                f"got {result.step_id}"
            )

        if result.step_type != last_issued.type:
            return (
                f"Step type mismatch: expected {last_issued.type.value}, "
                f"got {result.step_type.value}"
            )

        # Validate task/phase binding
        if last_issued.type in (StepType.IMPLEMENT_TASK, StepType.EXECUTE_VERIFICATION):
            if result.task_id != last_issued.task_id:
                return (
                    f"Task ID mismatch for step type {last_issued.type.value}: "
                    f"expected {last_issued.task_id}, got {result.task_id}"
                )

        if last_issued.type in (
            StepType.RUN_FIDELITY_GATE,
            StepType.ADDRESS_FIDELITY_FEEDBACK,
        ):
            if result.phase_id != last_issued.phase_id:
                return (
                    f"Phase ID mismatch for step type {last_issued.type.value}: "
                    f"expected {last_issued.phase_id}, got {result.phase_id}"
                )

        return None

    def _validate_step_proof(
        self,
        last_issued: LastStepIssued,
        result: LastStepResult,
    ) -> Tuple[str, Optional[str]]:
        """Validate step-proof binding for the current step result."""
        expected = last_issued.step_proof
        provided = result.step_proof

        if expected:
            if not provided:
                return (
                    ERROR_STEP_PROOF_MISSING,
                    (
                        f"step_proof is required for step {last_issued.step_id}. "
                        "Provide the one-time step_proof token from the issued step."
                    ),
                )
            if provided != expected:
                return (
                    ERROR_STEP_PROOF_MISMATCH,
                    (
                        f"step_proof mismatch for step {last_issued.step_id}. "
                        "Use the latest issued step and matching proof token."
                    ),
                )
            return ("", None)

        if provided:
            return (
                ERROR_STEP_PROOF_MISMATCH,
                "Unexpected step_proof provided for a step that does not require proof.",
            )

        return ("", None)

    def _issue_step_proof(
        self,
        session_id: str,
        step_id: str,
        now: datetime,
    ) -> str:
        """Issue a one-time proof token bound to a step ID."""
        seed = f"{session_id}:{step_id}:{now.isoformat()}:{ULID()}"
        return hashlib.sha256(seed.encode()).hexdigest()

    def _validate_gate_evidence(
        self,
        session: AutonomousSessionState,
        result: LastStepResult,
    ) -> Optional[str]:
        """Validate gate attempt ID against pending evidence.

        Checks session/phase/step binding to ensure the gate attempt is valid
        and not stale.

        Returns:
            None if valid, error message if invalid
        """
        if not result.gate_attempt_id:
            return (
                "gate_attempt_id is required for RUN_FIDELITY_GATE step results. "
                "Include the gate_attempt_id from the issued step in last_step_result."
            )

        evidence = session.pending_gate_evidence
        if not evidence:
            return (
                f"No pending gate evidence found for gate_attempt_id {result.gate_attempt_id}. "
                f"The gate evidence may have already been consumed or was never created."
            )

        # Validate gate_attempt_id matches
        if result.gate_attempt_id != evidence.gate_attempt_id:
            return (
                f"Gate attempt ID mismatch: expected {evidence.gate_attempt_id}, "
                f"got {result.gate_attempt_id}"
            )

        # Validate step binding
        if session.last_step_issued and result.step_id != evidence.step_id:
            return (
                f"Gate evidence step binding mismatch: evidence bound to step {evidence.step_id}, "
                f"but result is for step {result.step_id}"
            )

        # Validate phase binding
        if result.phase_id and result.phase_id != evidence.phase_id:
            return (
                f"Gate evidence phase binding mismatch: evidence bound to phase {evidence.phase_id}, "
                f"but result is for phase {result.phase_id}"
            )

        # Validate integrity checksum (P1.3)
        if evidence.integrity_checksum:
            if not verify_integrity_checksum(
                evidence.gate_attempt_id,
                evidence.step_id,
                evidence.phase_id,
                evidence.verdict.value,
                evidence.integrity_checksum,
            ):
                logger.warning(
                    "Gate evidence integrity checksum mismatch for gate_attempt_id %s",
                    evidence.gate_attempt_id,
                )
                return (
                    f"Gate evidence integrity checksum verification failed. "
                    f"The evidence may have been tampered with or the server secret was rotated. "
                    f"Gate attempt ID: {evidence.gate_attempt_id}"
                )

        return None

    def _validate_verification_receipt(
        self,
        session: AutonomousSessionState,
        result: LastStepResult,
    ) -> Optional[str]:
        """Validate verification receipt for EXECUTE_VERIFICATION steps.

        Per P1.2: outcome='success' requires a valid receipt with matching
        command hash, exit code, and output digest. Missing or invalid
        receipts yield deterministic validation errors with recovery guidance.

        Args:
            session: Current session state
            result: Step result being reported

        Returns:
            None if valid, error message if validation fails
        """
        # Only validate for EXECUTE_VERIFICATION steps with outcome='success'
        if result.step_type != StepType.EXECUTE_VERIFICATION:
            return None

        if result.outcome != StepOutcome.SUCCESS:
            return None

        # Check for pending receipt data
        pending = session.pending_verification_receipt
        if not pending:
            logger.warning(
                "VERIFICATION_RECEIPT_MISSING: no pending receipt for step %s",
                result.step_id,
            )
            return (
                f"Verification receipt required for execute_verification step with outcome='success'. "
                f"No pending receipt found for step {result.step_id}. "
                f"This may indicate the step was not properly issued by the orchestrator."
            )

        # Validate step binding
        if result.step_id != pending.step_id:
            return (
                f"Verification receipt step mismatch: expected step {pending.step_id}, "
                f"got step {result.step_id}"
            )
        if result.task_id != pending.task_id:
            return (
                f"Verification receipt task mismatch: expected task {pending.task_id}, "
                f"got task {result.task_id}"
            )

        # Check receipt presence
        receipt = result.verification_receipt
        if not receipt:
            logger.warning(
                "VERIFICATION_RECEIPT_MISSING: step %s reported success without receipt",
                result.step_id,
            )
            return (
                f"Verification receipt is required when outcome='success' for execute_verification steps. "
                f"Include verification_receipt with command_hash, exit_code, and output_digest in last_step_result."
            )

        # Validate step binding in receipt
        if receipt.step_id != result.step_id:
            return (
                f"Verification receipt step_id mismatch: expected {result.step_id}, "
                f"got {receipt.step_id}"
            )

        # Validate command hash matches expected
        if receipt.command_hash != pending.expected_command_hash:
            return (
                f"Verification receipt command_hash mismatch: "
                f"expected {pending.expected_command_hash[:16]}..., "
                f"got {receipt.command_hash[:16]}.... "
                f"The verification command may have been modified or a different verification was run."
            )
        if receipt.issued_at < pending.issued_at:
            return (
                "Verification receipt issued_at is earlier than the server-issued step receipt window. "
                "Use the receipt generated for the currently issued execute_verification step."
            )

        now = datetime.now(timezone.utc)
        future_skew = timedelta(minutes=5)
        if receipt.issued_at > now + future_skew:
            return (
                "Verification receipt issued_at is in the future beyond allowed skew. "
                "Ensure system clocks are synchronized and use server-issued receipt data."
            )

        logger.info(
            "Verification receipt validated for step %s: command_hash=%s, exit_code=%d",
            result.step_id,
            receipt.command_hash[:16],
            receipt.exit_code,
        )

        # Clear pending receipt after successful validation
        session.pending_verification_receipt = None

        return None

    # =========================================================================
    # Step Outcome Recording
    # =========================================================================

    def _record_step_outcome(
        self,
        session: AutonomousSessionState,
        result: LastStepResult,
        now: datetime,
    ) -> None:
        """Record the outcome of a step execution.

        Updates counters, completed tasks, and phase gates based on outcome.
        Gate failures do NOT increment consecutive_errors (per ADR-002).
        """
        # Update counters - gate steps do not affect consecutive_errors at all
        is_gate_step = result.step_type in (
            StepType.RUN_FIDELITY_GATE,
            StepType.ADDRESS_FIDELITY_FEEDBACK,
        )
        if not is_gate_step:
            if result.outcome == StepOutcome.FAILURE:
                session.counters.consecutive_errors += 1
            elif result.outcome == StepOutcome.SUCCESS:
                session.counters.consecutive_errors = 0

        # Record task completion
        if (
            result.step_type in (StepType.IMPLEMENT_TASK, StepType.EXECUTE_VERIFICATION)
            and result.outcome == StepOutcome.SUCCESS
            and result.task_id
        ):
            if result.task_id not in session.completed_task_ids:
                session.completed_task_ids.append(result.task_id)
                session.counters.tasks_completed += 1

        # Record gate result and increment fidelity cycle counter (ADR step 3)
        if result.step_type == StepType.RUN_FIDELITY_GATE and result.gate_attempt_id:
            session.counters.fidelity_review_cycles_in_active_phase += 1
            self._record_gate_outcome(session, result, now)

        # Write journal entry (best-effort)
        self._write_step_journal(session, result)

        # Audit: step consumed
        self._emit_audit_event(
            session,
            AuditEventType.STEP_CONSUMED,
            action="consume_step",
            step_id=result.step_id,
            phase_id=result.phase_id,
            task_id=result.task_id,
            metadata={
                "outcome": result.outcome.value if result.outcome else "unknown",
                "step_type": result.step_type.value if result.step_type else "unknown",
            },
        )

        session.updated_at = now

    def _record_gate_outcome(
        self,
        session: AutonomousSessionState,
        result: LastStepResult,
        now: datetime,
    ) -> None:
        """Record fidelity gate evaluation outcome.

        Per ADR-002 section 4: the caller reports outcome="success" to indicate
        the review ran successfully. The gate verdict and pass/fail determination
        come from pending_gate_evidence.verdict + session gate_policy, NOT from
        the step outcome. A step with outcome="success" and verdict="fail" should
        still record the gate as failed.
        """
        if not result.phase_id:
            return

        gate_record = session.phase_gates.get(result.phase_id)
        if not gate_record:
            return

        # Derive verdict from pending_gate_evidence, not step outcome
        evidence = session.pending_gate_evidence
        if evidence and evidence.phase_id == result.phase_id:
            gate_record.verdict = evidence.verdict
            # Evaluate pass/fail using gate policy
            should_pass, _ = self._evaluate_gate_policy(session, evidence)
            if should_pass:
                gate_record.status = PhaseGateStatus.PASSED
            else:
                gate_record.status = PhaseGateStatus.FAILED
        elif result.outcome == StepOutcome.FAILURE:
            # Fallback: step itself failed (review couldn't run)
            gate_record.status = PhaseGateStatus.FAILED
            gate_record.verdict = GateVerdict.FAIL

        gate_record.evaluated_at = now
        gate_record.gate_attempt_id = result.gate_attempt_id

        # Audit: gate verdict
        audit_type = (
            AuditEventType.GATE_PASSED
            if gate_record.status == PhaseGateStatus.PASSED
            else AuditEventType.GATE_FAILED
        )
        self._emit_audit_event(
            session,
            audit_type,
            action="record_gate_outcome",
            step_id=result.step_id,
            phase_id=result.phase_id,
            metadata={
                "verdict": gate_record.verdict.value if gate_record.verdict else None,
                "gate_attempt_id": result.gate_attempt_id,
                "status": gate_record.status.value,
            },
        )

    def _write_step_journal(
        self,
        session: AutonomousSessionState,
        result: LastStepResult,
    ) -> None:
        """Write journal entry for step outcome (best-effort).

        Uses the existing journal system (add_journal_entry + save_spec)
        to persist step outcomes with files_touched and notes.
        """
        try:
            from foundry_mcp.core.journal import add_journal_entry
            from foundry_mcp.core.spec import load_spec, save_spec

            outcome_str = result.outcome.value if result.outcome else "unknown"
            step_type_str = result.step_type.value if result.step_type else "unknown"

            content = f"Step {result.step_id} ({step_type_str}) completed with outcome: {outcome_str}"
            if result.note:
                content += f"\n\nNote: {result.note}"
            if result.files_touched:
                content += f"\n\nFiles touched: {', '.join(result.files_touched)}"

            logger.info(
                "Step outcome: session=%s step=%s type=%s outcome=%s",
                session.id,
                result.step_id,
                step_type_str,
                outcome_str,
            )

            # Write to spec journal via the standard journal system
            specs_dir = self.workspace_path / "specs"
            spec_data = load_spec(session.spec_id, specs_dir)
            if spec_data is not None:
                add_journal_entry(
                    spec_data,
                    title=f"Step {step_type_str}: {outcome_str}",
                    content=content,
                    entry_type="step",
                    task_id=result.task_id,
                    author="autonomy",
                    metadata={
                        "session_id": session.id,
                        "step_id": result.step_id,
                        "step_type": step_type_str,
                        "outcome": outcome_str,
                        "files_touched": result.files_touched or [],
                    },
                )
                save_spec(session.spec_id, spec_data, specs_dir)
            else:
                logger.debug("Spec not found for step journal: %s", session.spec_id)

        except (OSError, json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug("Failed to write step journal: %s", e)

    # =========================================================================
    # Spec Integrity and File Operations
    # =========================================================================

    def _validate_spec_integrity(
        self,
        session: AutonomousSessionState,
        now: datetime,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Validate spec integrity with mtime optimization.

        Returns:
            Tuple of (spec_data, error_message). spec_data is None on error.
        """
        try:
            # Find spec file
            spec_path = self._find_spec_path(session.spec_id)
            if not spec_path:
                return None, f"Spec not found: {session.spec_id}"

            # Get current file metadata
            current_metadata = get_spec_file_metadata(spec_path)
            if current_metadata is None:
                return None, f"Could not read spec file metadata: {session.spec_id}"

            # Fast path: mtime optimization - skip re-hash if unchanged
            needs_rehash = True
            if (
                session.spec_file_mtime == current_metadata.mtime
                and session.spec_file_size == current_metadata.file_size
            ):
                needs_rehash = False
                logger.debug("Spec mtime unchanged, skipping re-hash")

            # Re-hash at phase boundaries or if mtime changed
            if needs_rehash or session.active_phase_id is None:
                spec_data = self._load_spec_file(spec_path)
                if spec_data is None:
                    return None, f"Could not load spec: {session.spec_id}"

                new_hash = compute_spec_structure_hash(spec_data)
                if new_hash != session.spec_structure_hash:
                    logger.warning(
                        "Spec structure changed: old=%s new=%s",
                        session.spec_structure_hash[:16],
                        new_hash[:16],
                    )
                    return None, (
                        f"Spec structure has changed. Use session-rebase to reconcile. "
                        f"Old hash: {session.spec_structure_hash[:16]}..., "
                        f"New hash: {new_hash[:16]}..."
                    )

                # Update cached metadata
                session.spec_file_mtime = current_metadata.mtime
                session.spec_file_size = current_metadata.file_size

                # Populate spec cache for subsequent calls
                self._spec_cache = (
                    session.spec_id,
                    current_metadata.mtime,
                    current_metadata.file_size,
                    spec_data,
                )

                return spec_data, None

            # Fast path: return cached spec_data if available for this mtime
            if (
                self._spec_cache is not None
                and self._spec_cache[0] == session.spec_id
                and self._spec_cache[1] == current_metadata.mtime
                and self._spec_cache[2] == current_metadata.file_size
            ):
                return self._spec_cache[3], None

            # Load spec without re-hashing
            spec_data = self._load_spec_file(spec_path)
            if spec_data is not None:
                self._spec_cache = (
                    session.spec_id,
                    current_metadata.mtime,
                    current_metadata.file_size,
                    spec_data,
                )
            return spec_data, None

        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.error("Spec integrity validation failed: %s", e)
            self._emit_audit_event(
                session,
                AuditEventType.STEP_CONSUMED,
                action="spec_integrity_failure",
                metadata={"error": str(e)},
            )
            return None, f"Spec integrity validation failed: {e}"

    def _find_spec_path(self, spec_id: str) -> Optional[Path]:
        """Find the path to a spec file using the canonical resolver."""
        specs_dir = self.workspace_path / "specs"
        return resolve_spec_file(spec_id, specs_dir)

    def _load_spec_file(self, spec_path: Path) -> Optional[Dict[str, Any]]:
        """Load and parse a spec file."""
        import json

        try:
            with open(spec_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load spec %s: %s", spec_path, e)
            return None

    # =========================================================================
    # Staleness and Guard Checks
    # =========================================================================

    def _check_heartbeat_staleness(
        self,
        session: AutonomousSessionState,
        now: datetime,
    ):
        """Check for heartbeat staleness with grace window.

        Returns:
            None if not stale, True if stale but step is active (warning only),
            OrchestrationResult if stale and should pause
        """
        if session.context.last_heartbeat_at is None:
            # No heartbeat yet - check if within grace period
            created_delta = now - session.created_at
            grace_delta = timedelta(minutes=session.limits.heartbeat_grace_minutes)
            if created_delta <= grace_delta:
                return None  # Within grace period
        else:
            # Check staleness against last heartbeat
            stale_delta = now - session.context.last_heartbeat_at
            threshold = timedelta(minutes=session.limits.heartbeat_stale_minutes)
            if stale_delta <= threshold:
                return None  # Not stale

        # Heartbeat is stale
        logger.warning(
            "Heartbeat stale for session %s (last: %s)",
            session.id,
            session.context.last_heartbeat_at.isoformat() if session.context.last_heartbeat_at else "never",
        )

        # If step is active, include warning flag but do NOT pause/error (ADR line 662)
        if session.last_step_issued is not None:
            issued_delta = now - session.last_step_issued.issued_at
            step_threshold = timedelta(minutes=session.limits.step_stale_minutes)
            if issued_delta < step_threshold:
                # Step is still active - signal warning to caller, do not pause
                return True

        # Pause due to heartbeat staleness
        return self._create_pause_result(
            session,
            PauseReason.HEARTBEAT_STALE,
            now,
            "Session paused due to heartbeat staleness. Resume to continue.",
        )

    def _check_pause_guards(
        self,
        session: AutonomousSessionState,
    ) -> Optional[OrchestrationResult]:
        """Check pause guards: context, error, task limits.

        Returns:
            OrchestrationResult if guard triggered, None if OK
        """
        now = datetime.now(timezone.utc)

        # Context limit
        if session.context.context_usage_pct >= session.limits.context_threshold_pct:
            logger.info(
                "Context limit reached: %d%% >= %d%% (source: %s)",
                session.context.context_usage_pct,
                session.limits.context_threshold_pct,
                session.context.context_source or "unknown",
            )
            return self._create_pause_result(
                session,
                PauseReason.CONTEXT_LIMIT,
                now,
                (
                    f"Context usage at {session.context.context_usage_pct}% "
                    f"(threshold: {session.limits.context_threshold_pct}%). "
                    f"Resume in a new session."
                ),
            )

        # Error threshold
        if session.counters.consecutive_errors >= session.limits.max_consecutive_errors:
            logger.warning(
                "Error threshold reached: %d consecutive errors",
                session.counters.consecutive_errors,
            )
            return self._create_pause_result(
                session,
                PauseReason.ERROR_THRESHOLD,
                now,
                (
                    f"{session.counters.consecutive_errors} consecutive errors. "
                    f"Manual intervention required."
                ),
            )

        # Task limit
        if session.counters.tasks_completed >= session.limits.max_tasks_per_session:
            logger.info(
                "Task limit reached: %d tasks",
                session.counters.tasks_completed,
            )
            return self._create_pause_result(
                session,
                PauseReason.TASK_LIMIT,
                now,
                (
                    f"Task limit ({session.limits.max_tasks_per_session}) reached. "
                    f"Start a new session to continue."
                ),
            )

        return None

    def _check_all_blocked(
        self,
        session: AutonomousSessionState,
        spec_data: Optional[Dict[str, Any]],
    ) -> Optional[OrchestrationResult]:
        """Check if all remaining tasks are blocked.

        Uses the shared check_all_blocked utility from core/task/_helpers.py.

        Returns:
            OrchestrationResult if all blocked, None if work available
        """
        if not spec_data:
            return None

        # Use shared utility to check if all pending tasks are blocked
        if check_all_blocked(spec_data):
            logger.warning("All remaining tasks are blocked")
            now = datetime.now(timezone.utc)
            return self._create_pause_result(
                session,
                PauseReason.BLOCKED,
                now,
                "All remaining tasks are blocked by dependencies. "
                "Resolve blockers to continue.",
            )

        return None


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "StepOrchestrator",
    "OrchestrationResult",
    "ERROR_STEP_RESULT_REQUIRED",
    "ERROR_STEP_MISMATCH",
    "ERROR_STEP_PROOF_MISSING",
    "ERROR_STEP_PROOF_MISMATCH",
    "ERROR_STEP_PROOF_CONFLICT",
    "ERROR_STEP_PROOF_EXPIRED",
    "ERROR_INVALID_GATE_EVIDENCE",
    "ERROR_NO_ACTIVE_SESSION",
    "ERROR_AMBIGUOUS_ACTIVE_SESSION",
    "ERROR_SESSION_UNRECOVERABLE",
    "ERROR_SPEC_REBASE_REQUIRED",
    "ERROR_HEARTBEAT_STALE",
    "ERROR_STEP_STALE",
    "ERROR_ALL_TASKS_BLOCKED",
    "ERROR_GATE_BLOCKED",
    "ERROR_REQUIRED_GATE_UNSATISFIED",
    "ERROR_VERIFICATION_RECEIPT_MISSING",
    "ERROR_VERIFICATION_RECEIPT_INVALID",
    "ERROR_GATE_INTEGRITY_CHECKSUM",
    "ERROR_GATE_AUDIT_FAILURE",
]
