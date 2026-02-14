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

Usage:
    from foundry_mcp.core.autonomy.orchestrator import StepOrchestrator

    orchestrator = StepOrchestrator(storage, spec_loader)
    result = orchestrator.compute_next_step(session, last_step_result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ulid import ULID

from foundry_mcp.core.autonomy.memory import AutonomyStorage
from foundry_mcp.core.autonomy.models import (
    AutonomousSessionState,
    FailureReason,
    GatePolicy,
    GateVerdict,
    LastStepIssued,
    LastStepResult,
    NextStep,
    PauseReason,
    PendingGateEvidence,
    PhaseGateRecord,
    PhaseGateStatus,
    SessionStatus,
    StepInstruction,
    StepOutcome,
    StepType,
)
from foundry_mcp.core.autonomy.spec_hash import (
    compute_spec_structure_hash,
    get_spec_file_metadata,
)
from foundry_mcp.core.task._helpers import check_all_blocked

logger = logging.getLogger(__name__)


# =============================================================================
# Error Code Constants (from ADR)
# =============================================================================

ERROR_STEP_RESULT_REQUIRED = "STEP_RESULT_REQUIRED"
ERROR_STEP_MISMATCH = "STEP_MISMATCH"
ERROR_INVALID_GATE_EVIDENCE = "INVALID_GATE_EVIDENCE"
ERROR_NO_ACTIVE_SESSION = "NO_ACTIVE_SESSION"
ERROR_AMBIGUOUS_ACTIVE_SESSION = "AMBIGUOUS_ACTIVE_SESSION"
ERROR_SESSION_UNRECOVERABLE = "SESSION_UNRECOVERABLE"
ERROR_SPEC_REBASE_REQUIRED = "SPEC_REBASE_REQUIRED"
ERROR_HEARTBEAT_STALE = "HEARTBEAT_STALE"
ERROR_STEP_STALE = "STEP_STALE"
ERROR_ALL_TASKS_BLOCKED = "ALL_TASKS_BLOCKED"


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


# =============================================================================
# Step Orchestrator
# =============================================================================


class StepOrchestrator:
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
        # Update context if provided
        if context_usage_pct is not None:
            session.context.context_usage_pct = context_usage_pct
        if heartbeat_at is not None:
            session.context.last_heartbeat_at = heartbeat_at

        now = datetime.now(timezone.utc)

        # =================================================================
        # Step 0: Replay Detection
        # =================================================================
        if last_step_result is not None and session.last_step_issued is not None:
            if last_step_result.step_id == session.last_step_issued.step_id:
                # Check if we already processed this step
                if session.last_issued_response is not None:
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
                logger.warning("INVALID_GATE_EVIDENCE: %s", gate_error)
                return OrchestrationResult(
                    success=False,
                    session=session,
                    error_code=ERROR_INVALID_GATE_EVIDENCE,
                    error_message=gate_error,
                    should_persist=False,
                )

        # =================================================================
        # Step 3: Record Step Outcome
        # =================================================================
        if last_step_result is not None:
            self._record_step_outcome(session, last_step_result, now)

        # =================================================================
        # Step 4: Validate Spec Integrity
        # =================================================================
        spec_data, integrity_error = self._validate_spec_integrity(session, now)
        if integrity_error:
            session.status = SessionStatus.FAILED
            session.failure_reason = FailureReason.SPEC_STRUCTURE_CHANGED
            session.updated_at = now
            return OrchestrationResult(
                success=False,
                session=session,
                error_code=ERROR_SPEC_REBASE_REQUIRED,
                error_message=integrity_error,
                should_persist=True,
            )

        # =================================================================
        # Step 5: Check Terminal States
        # =================================================================
        if session.status in (SessionStatus.COMPLETED, SessionStatus.ENDED, SessionStatus.FAILED):
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
        heartbeat_result = self._check_heartbeat_staleness(session, now)
        if heartbeat_result:
            return heartbeat_result

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
        return next_step_result

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
            return None  # No gate attempt to validate

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

        return None

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
        # Update counters - gate failures do not increment consecutive_errors
        is_gate_failure = (
            result.step_type == StepType.RUN_FIDELITY_GATE
            and result.outcome == StepOutcome.FAILURE
        )
        if result.outcome == StepOutcome.FAILURE and not is_gate_failure:
            session.counters.consecutive_errors += 1
        else:
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

        # Record gate result
        if result.step_type == StepType.RUN_FIDELITY_GATE and result.gate_attempt_id:
            self._record_gate_outcome(session, result, now)

        # Write journal entry (best-effort)
        self._write_step_journal(session, result)

        session.updated_at = now

    def _record_gate_outcome(
        self,
        session: AutonomousSessionState,
        result: LastStepResult,
        now: datetime,
    ) -> None:
        """Record fidelity gate evaluation outcome."""
        if not result.phase_id:
            return

        gate_record = session.phase_gates.get(result.phase_id)
        if gate_record:
            if result.outcome == StepOutcome.SUCCESS:
                gate_record.status = PhaseGateStatus.PASSED
                gate_record.verdict = GateVerdict.PASS
            elif result.outcome == StepOutcome.FAILURE:
                gate_record.status = PhaseGateStatus.FAILED
                gate_record.verdict = GateVerdict.FAIL
            gate_record.evaluated_at = now
            gate_record.gate_attempt_id = result.gate_attempt_id

    def _write_step_journal(
        self,
        session: AutonomousSessionState,
        result: LastStepResult,
    ) -> None:
        """Write journal entry for step outcome (best-effort)."""
        try:
            outcome_str = result.outcome.value if result.outcome else "unknown"
            step_type_str = result.step_type.value if result.step_type else "unknown"

            content = f"Step {result.step_id} ({step_type_str}) completed with outcome: {outcome_str}"
            if result.note:
                content += f"\n\nNote: {result.note}"
            if result.files_touched:
                content += f"\n\nFiles touched: {', '.join(result.files_touched)}"

            # Note: Journal writing requires spec_data which we may not have here
            # This is best-effort, so we log instead
            logger.info(
                "Step outcome: session=%s step=%s type=%s outcome=%s",
                session.id,
                result.step_id,
                step_type_str,
                outcome_str,
            )
        except Exception as e:
            logger.debug("Failed to write step journal: %s", e)

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

                return spec_data, None

            # Load spec without re-hashing
            spec_data = self._load_spec_file(spec_path)
            return spec_data, None

        except Exception as e:
            logger.error("Spec integrity validation failed: %s", e)
            return None, f"Spec integrity validation failed: {e}"

    def _find_spec_path(self, spec_id: str) -> Optional[Path]:
        """Find the path to a spec file."""
        search_dirs = [
            self.workspace_path / "specs" / "active",
            self.workspace_path / "specs" / "pending",
            self.workspace_path / "specs" / "completed",
            self.workspace_path / "specs" / "archived",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            # Check for exact match
            spec_path = search_dir / f"{spec_id}.json"
            if spec_path.exists():
                return spec_path

        return None

    def _load_spec_file(self, spec_path: Path) -> Optional[Dict[str, Any]]:
        """Load and parse a spec file."""
        import json

        try:
            with open(spec_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load spec %s: %s", spec_path, e)
            return None

    def _check_heartbeat_staleness(
        self,
        session: AutonomousSessionState,
        now: datetime,
    ) -> Optional[OrchestrationResult]:
        """Check for heartbeat staleness with grace window.

        Returns:
            OrchestrationResult if stale, None if OK
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

        # If step is active, return warning; if idle, pause
        if session.last_step_issued is not None:
            issued_delta = now - session.last_step_issued.issued_at
            step_threshold = timedelta(minutes=session.limits.step_stale_minutes)
            if issued_delta < step_threshold:
                # Step is still active - return error
                return OrchestrationResult(
                    success=False,
                    session=session,
                    error_code=ERROR_HEARTBEAT_STALE,
                    error_message=(
                        "Heartbeat is stale but step is still active. "
                        "Update heartbeat or wait for step to complete."
                    ),
                    should_persist=False,
                )

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
                "Context limit reached: %d%% >= %d%%",
                session.context.context_usage_pct,
                session.limits.context_threshold_pct,
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
            return self._handle_gate_evidence(session, now)

        # Step 15: Gate passed + stop_on_phase_completion
        if (
            current_phase
            and self._phase_gate_passed(session, current_phase)
            and session.stop_conditions.stop_on_phase_completion
        ):
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
        return self._create_complete_spec_result(session, now)

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
        for task in phase.get("tasks", []):
            task_type = task.get("type", "task")
            task_id = task.get("id", "")
            if task_type == "task" and task_id not in session.completed_task_ids:
                return False  # Implementation tasks still pending

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
        now: datetime,
    ) -> OrchestrationResult:
        """Handle pending gate evidence (steps 13-14).

        Applies gate policy evaluation:
        - STRICT: pass only on verdict=pass
        - LENIENT: pass on pass or warn
        - MANUAL: always pause with gate_review_required

        Auto-retry behavior:
        - If auto_retry_fidelity_gate=true and gate fails: address_fidelity_feedback → retry
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
            # Increment fidelity cycle counter on accepted gate
            session.counters.fidelity_review_cycles_in_active_phase += 1
            return self._determine_next_step_after_gate(session, now)

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

        # Gate failed - check auto-retry setting
        if session.stop_conditions.auto_retry_fidelity_gate:
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
        now: datetime,
    ) -> OrchestrationResult:
        """Determine next step after gate passes.

        The fidelity cycle counter is incremented in _handle_gate_evidence
        when the gate is accepted. This method handles phase transition
        and task continuation.
        """
        # Check if we should stop on phase completion
        if session.stop_conditions.stop_on_phase_completion:
            return self._create_pause_result(
                session,
                PauseReason.PHASE_COMPLETE,
                now,
                f"Phase {session.active_phase_id} complete. Stopping as configured.",
            )

        # Look for more tasks in the current or next phase
        # The _find_next_task method will handle phase transition
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

        step_id = f"step_{ULID()}"
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
        )

        # Update last step issued and cache response
        session.last_step_issued = LastStepIssued(
            step_id=step_id,
            type=StepType.PAUSE,
            task_id=None,
            phase_id=None,
            issued_at=now,
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
        )

        # Update last step issued
        session.last_step_issued = LastStepIssued(
            step_id=step_id,
            type=StepType.IMPLEMENT_TASK,
            task_id=task_id,
            phase_id=phase_id,
            issued_at=now,
        )
        session.last_task_id = task_id
        session.status = SessionStatus.RUNNING
        session.updated_at = now

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
        """Create an execute_verification step."""
        task_id = task.get("id", "")
        task_title = task.get("title", "")
        phase_id = session.active_phase_id or ""

        step_id = f"step_{ULID()}"

        instructions = [
            StepInstruction(
                tool="task",
                action="info",
                description="Get verification task details",
            ),
            StepInstruction(
                tool="verification",
                action="run",
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
        )

        session.last_step_issued = LastStepIssued(
            step_id=step_id,
            type=StepType.EXECUTE_VERIFICATION,
            task_id=task_id,
            phase_id=phase_id,
            issued_at=now,
        )
        session.updated_at = now

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
                action="run",
                description=f"Run fidelity review for phase {phase_title}",
            ),
            StepInstruction(
                tool="task",
                action="journal",
                description="Record gate results",
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
        )

        session.last_step_issued = LastStepIssued(
            step_id=step_id,
            type=StepType.RUN_FIDELITY_GATE,
            task_id=None,
            phase_id=phase_id,
            issued_at=now,
        )
        session.phase_gates[phase_id].gate_attempt_id = gate_attempt_id
        session.updated_at = now

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

        instructions = [
            StepInstruction(
                tool="review",
                action="get-findings",
                description="Get fidelity review findings",
            ),
            StepInstruction(
                tool="task",
                action="update",
                description="Address findings in code",
            ),
            StepInstruction(
                tool="task",
                action="complete",
                description="Mark feedback addressed",
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
        )

        session.last_step_issued = LastStepIssued(
            step_id=step_id,
            type=StepType.ADDRESS_FIDELITY_FEEDBACK,
            task_id=None,
            phase_id=evidence.phase_id,
            issued_at=now,
        )
        session.updated_at = now

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

        step_id = f"step_{ULID()}"
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
        )

        session.last_step_issued = LastStepIssued(
            step_id=step_id,
            type=StepType.COMPLETE_SPEC,
            task_id=None,
            phase_id=None,
            issued_at=now,
        )

        return OrchestrationResult(
            success=True,
            session=session,
            next_step=next_step,
            should_persist=True,
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "StepOrchestrator",
    "OrchestrationResult",
    "ERROR_STEP_RESULT_REQUIRED",
    "ERROR_STEP_MISMATCH",
    "ERROR_INVALID_GATE_EVIDENCE",
    "ERROR_NO_ACTIVE_SESSION",
    "ERROR_AMBIGUOUS_ACTIVE_SESSION",
    "ERROR_SESSION_UNRECOVERABLE",
    "ERROR_SPEC_REBASE_REQUIRED",
    "ERROR_HEARTBEAT_STALE",
    "ERROR_STEP_STALE",
    "ERROR_ALL_TASKS_BLOCKED",
]
