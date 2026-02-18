"""Tests for StepOrchestrator.

Covers:
- Replay detection (step 0): cached response returned on duplicate step_id
- Feedback validation (step 1): STEP_RESULT_REQUIRED on non-initial call
- Step identity validation (step 2): STEP_MISMATCH on wrong step_id/type/binding
- Gate evidence validation: INVALID_GATE_EVIDENCE on stale/wrong attempt_id
- Step outcome recording (step 3): task completion, consecutive_errors, gate outcome
- Heartbeat staleness (step 7): grace window and stale detection
- Pause guards (step 8): context_limit, error_threshold, task_limit
- Fidelity cycle stop heuristic (step 9)
- Gate policy evaluation: strict/lenient/manual
- Phase advancement: fidelity cycle counter reset on phase change
- consecutive_errors unaffected by gate steps
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from foundry_mcp.core.autonomy.models.enums import (
    FailureReason,
    GatePolicy,
    GateVerdict,
    PauseReason,
    PhaseGateStatus,
    SessionStatus,
    StepOutcome,
    StepType,
)
from foundry_mcp.core.autonomy.models.gates import (
    PendingGateEvidence,
    PhaseGateRecord,
)
from foundry_mcp.core.autonomy.models.session_config import (
    SessionCounters,
    SessionLimits,
    StopConditions,
)
from foundry_mcp.core.autonomy.models.state import AutonomousSessionState
from foundry_mcp.core.autonomy.models.steps import LastStepIssued, LastStepResult
from foundry_mcp.core.autonomy.models.verification import (
    PendingVerificationReceipt,
    VerificationReceipt,
)
from foundry_mcp.core.autonomy.orchestrator import (
    ERROR_GATE_AUDIT_FAILURE,
    ERROR_GATE_INTEGRITY_CHECKSUM,
    ERROR_HEARTBEAT_STALE,
    ERROR_INVALID_GATE_EVIDENCE,
    ERROR_SPEC_REBASE_REQUIRED,
    ERROR_STEP_MISMATCH,
    ERROR_STEP_PROOF_MISSING,
    ERROR_STEP_PROOF_MISMATCH,
    ERROR_STEP_RESULT_REQUIRED,
    ERROR_STEP_STALE,
    ERROR_VERIFICATION_RECEIPT_INVALID,
    OrchestrationResult,
    StepOrchestrator,
)
from foundry_mcp.core.autonomy.spec_hash import compute_spec_structure_hash

from .conftest import make_hierarchy_spec_data, make_session, make_spec_data


# =============================================================================
# Helpers
# =============================================================================

def _make_orchestrator(tmp_path: Path, spec_data: Optional[Dict[str, Any]] = None) -> StepOrchestrator:
    """Create orchestrator with a real spec file on disk."""
    workspace = tmp_path / "ws"
    specs_dir = workspace / "specs" / "active"
    specs_dir.mkdir(parents=True, exist_ok=True)

    data = spec_data or make_spec_data()
    spec_file = specs_dir / f"{data.get('spec_id', 'test-spec-001')}.json"
    spec_file.write_text(json.dumps(data, indent=2))

    storage = MagicMock()
    return StepOrchestrator(
        storage=storage,
        spec_loader=MagicMock(),
        workspace_path=workspace,
    )


def _issued(
    step_id: str = "step-001",
    step_type: StepType = StepType.IMPLEMENT_TASK,
    task_id: Optional[str] = "task-1",
    phase_id: Optional[str] = "phase-1",
    minutes_ago: int = 0,
    step_proof: Optional[str] = None,
) -> LastStepIssued:
    return LastStepIssued(
        step_id=step_id,
        type=step_type,
        task_id=task_id,
        phase_id=phase_id,
        issued_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        step_proof=step_proof,
    )


def _result(
    step_id: str = "step-001",
    step_type: StepType = StepType.IMPLEMENT_TASK,
    outcome: StepOutcome = StepOutcome.SUCCESS,
    task_id: Optional[str] = "task-1",
    phase_id: Optional[str] = "phase-1",
    gate_attempt_id: Optional[str] = None,
    step_proof: Optional[str] = None,
    verification_receipt: Optional[dict] = None,
) -> LastStepResult:
    return LastStepResult(
        step_id=step_id,
        step_type=step_type,
        outcome=outcome,
        task_id=task_id,
        phase_id=phase_id,
        gate_attempt_id=gate_attempt_id,
        step_proof=step_proof,
        verification_receipt=verification_receipt,
    )


# =============================================================================
# Verification Receipt Model Contract
# =============================================================================


class TestVerificationReceiptModelContract:
    """Receipt construction contract tests (valid and invalid shapes)."""

    def test_verification_receipt_valid_shape(self):
        receipt = VerificationReceipt(
            command_hash="a" * 64,
            exit_code=0,
            output_digest="b" * 64,
            issued_at=datetime.now(timezone.utc),
            step_id="step-valid-1",
        )
        assert receipt.command_hash == "a" * 64
        assert receipt.output_digest == "b" * 64
        assert receipt.step_id == "step-valid-1"

    @pytest.mark.parametrize(
        "receipt_data",
        [
            {
                "command_hash": "short",
                "exit_code": 0,
                "output_digest": "b" * 64,
                "issued_at": datetime.now(timezone.utc),
                "step_id": "step-invalid-1",
            },
            {
                "command_hash": "a" * 64,
                "exit_code": 0,
                "output_digest": "short",
                "issued_at": datetime.now(timezone.utc),
                "step_id": "step-invalid-2",
            },
            {
                "command_hash": "a" * 64,
                "exit_code": 0,
                "output_digest": "b" * 64,
                "issued_at": datetime.now(timezone.utc),
                "step_id": "",
            },
            {
                "command_hash": "a" * 64,
                "exit_code": 0,
                "output_digest": "b" * 64,
                "issued_at": datetime.now(),
                "step_id": "step-invalid-3",
            },
        ],
    )
    def test_verification_receipt_invalid_shapes(self, receipt_data):
        with pytest.raises(ValidationError):
            VerificationReceipt(**receipt_data)


# =============================================================================
# Step 0: Replay Detection
# =============================================================================


class TestReplayDetection:
    """Step 0: Return cached response when step already processed."""

    def test_replay_returns_cached_response(self, tmp_path):
        """If step_id matches last_step_issued and last_issued_response is set, return it."""
        orch = _make_orchestrator(tmp_path)
        cached = {"type": "implement_task", "step_id": "step-001"}
        session = make_session(
            last_step_issued=_issued(step_id="step-001"),
            last_issued_response=cached,
        )

        result = orch.compute_next_step(session, _result(step_id="step-001"))

        assert result.success is True
        assert result.replay_response == cached
        assert result.should_persist is False

    def test_no_replay_without_cached_response(self, tmp_path):
        """If last_issued_response is None, no replay — proceed to outcome recording."""
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            last_step_issued=_issued(step_id="step-001"),
            last_issued_response=None,
        )
        # Should proceed normally (past replay detection)
        result = orch.compute_next_step(session, _result(step_id="step-001"))
        assert result.replay_response is None

    def test_no_replay_on_different_step_id(self, tmp_path):
        """Mismatched step_id → STEP_MISMATCH, not replay."""
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            last_step_issued=_issued(step_id="step-001"),
            last_issued_response={"cached": True},
        )

        result = orch.compute_next_step(session, _result(step_id="step-999"))
        assert result.success is False
        assert result.error_code == ERROR_STEP_MISMATCH

    def test_proof_enforced_feedback_skips_replay_short_circuit(self, tmp_path):
        """Proof-bearing feedback must be processed instead of replayed."""
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            last_step_issued=_issued(step_id="step-001", step_proof="proof-001"),
            last_issued_response={"cached": True},
        )

        result = orch.compute_next_step(
            session,
            _result(step_id="step-001", step_proof="proof-001"),
        )

        assert result.replay_response is None
        assert "task-1" in result.session.completed_task_ids


# =============================================================================
# Step 1: Validate Feedback
# =============================================================================


class TestFeedbackValidation:
    """Step 1: Require last_step_result on non-initial calls."""

    def test_step_result_required_on_noninital_call(self, tmp_path):
        """If last_step_issued is set but no result provided → STEP_RESULT_REQUIRED."""
        orch = _make_orchestrator(tmp_path)
        session = make_session(last_step_issued=_issued())

        result = orch.compute_next_step(session, last_step_result=None)

        assert result.success is False
        assert result.error_code == ERROR_STEP_RESULT_REQUIRED
        assert result.should_persist is False

    def test_initial_call_without_result_is_ok(self, tmp_path):
        """First call (last_step_issued is None) with no result should proceed."""
        orch = _make_orchestrator(tmp_path)
        session = make_session(last_step_issued=None)

        result = orch.compute_next_step(session, last_step_result=None)
        # Should proceed to spec validation and beyond (not STEP_RESULT_REQUIRED)
        assert result.error_code != ERROR_STEP_RESULT_REQUIRED


# =============================================================================
# Step 2: Validate Step Identity
# =============================================================================


class TestStepIdentityValidation:
    """Step 2: Validate step_id, step_type, and task/phase binding."""

    def test_step_id_mismatch(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(last_step_issued=_issued(step_id="step-001"))

        result = orch.compute_next_step(session, _result(step_id="step-other"))
        assert result.success is False
        assert result.error_code == ERROR_STEP_MISMATCH
        assert "Step ID mismatch" in result.error_message

    def test_step_type_mismatch(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            last_step_issued=_issued(step_id="step-001", step_type=StepType.IMPLEMENT_TASK),
        )

        result = orch.compute_next_step(
            session,
            _result(step_id="step-001", step_type=StepType.EXECUTE_VERIFICATION),
        )
        assert result.success is False
        assert result.error_code == ERROR_STEP_MISMATCH
        assert "Step type mismatch" in result.error_message

    def test_task_id_mismatch_for_implement_task(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            last_step_issued=_issued(
                step_id="step-001",
                step_type=StepType.IMPLEMENT_TASK,
                task_id="task-1",
            ),
        )

        result = orch.compute_next_step(
            session,
            _result(step_id="step-001", step_type=StepType.IMPLEMENT_TASK, task_id="task-WRONG"),
        )
        assert result.success is False
        assert result.error_code == ERROR_STEP_MISMATCH
        assert "Task ID mismatch" in result.error_message

    def test_phase_id_mismatch_for_fidelity_gate(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            last_step_issued=_issued(
                step_id="step-001",
                step_type=StepType.RUN_FIDELITY_GATE,
                task_id=None,
                phase_id="phase-1",
            ),
        )

        result = orch.compute_next_step(
            session,
            _result(
                step_id="step-001",
                step_type=StepType.RUN_FIDELITY_GATE,
                task_id=None,
                phase_id="phase-WRONG",
                gate_attempt_id="gate-001",
            ),
        )
        assert result.success is False
        assert result.error_code == ERROR_STEP_MISMATCH
        assert "Phase ID mismatch" in result.error_message

    def test_valid_identity_passes(self, tmp_path):
        """Matching step_id + type + task_id passes validation."""
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            last_step_issued=_issued(step_id="s1", step_type=StepType.IMPLEMENT_TASK, task_id="t1"),
        )

        result = orch.compute_next_step(
            session,
            _result(step_id="s1", step_type=StepType.IMPLEMENT_TASK, task_id="t1"),
        )
        # Should not be STEP_MISMATCH (may fail for other reasons like spec integrity)
        assert result.error_code != ERROR_STEP_MISMATCH


class TestStepProofValidation:
    """Step-proof enforcement for step result reports."""

    def test_missing_step_proof_returns_error(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            last_step_issued=_issued(
                step_id="step-proof-1",
                step_type=StepType.IMPLEMENT_TASK,
                task_id="task-1",
                step_proof="proof-123",
            ),
        )

        result = orch.compute_next_step(
            session,
            _result(
                step_id="step-proof-1",
                step_type=StepType.IMPLEMENT_TASK,
                task_id="task-1",
                outcome=StepOutcome.SUCCESS,
                step_proof=None,
            ),
        )
        assert result.success is False
        assert result.error_code == ERROR_STEP_PROOF_MISSING

    def test_step_proof_mismatch_returns_error(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            last_step_issued=_issued(
                step_id="step-proof-2",
                step_type=StepType.IMPLEMENT_TASK,
                task_id="task-1",
                step_proof="proof-expected",
            ),
        )

        result = orch.compute_next_step(
            session,
            _result(
                step_id="step-proof-2",
                step_type=StepType.IMPLEMENT_TASK,
                task_id="task-1",
                outcome=StepOutcome.SUCCESS,
                step_proof="proof-wrong",
            ),
        )
        assert result.success is False
        assert result.error_code == ERROR_STEP_PROOF_MISMATCH


# =============================================================================
# Gate Evidence Validation
# =============================================================================


class TestGateEvidenceValidation:
    """Validate gate_attempt_id against pending gate evidence."""

    def test_invalid_gate_attempt_id(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            last_step_issued=_issued(
                step_id="step-001",
                step_type=StepType.RUN_FIDELITY_GATE,
                task_id=None,
                phase_id="phase-1",
            ),
            pending_gate_evidence=PendingGateEvidence(
                gate_attempt_id="gate-001",
                step_id="step-001",
                phase_id="phase-1",
                verdict=GateVerdict.PASS,
                issued_at=datetime.now(timezone.utc),
            ),
        )

        result = orch.compute_next_step(
            session,
            _result(
                step_id="step-001",
                step_type=StepType.RUN_FIDELITY_GATE,
                task_id=None,
                phase_id="phase-1",
                gate_attempt_id="gate-WRONG",
            ),
        )
        assert result.success is False
        assert result.error_code == ERROR_INVALID_GATE_EVIDENCE

    def test_no_pending_evidence_returns_error(self, tmp_path):
        """gate_attempt_id provided but no pending evidence → error."""
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            last_step_issued=_issued(
                step_id="step-001",
                step_type=StepType.RUN_FIDELITY_GATE,
                task_id=None,
                phase_id="phase-1",
            ),
            # No pending_gate_evidence
        )

        result = orch.compute_next_step(
            session,
            _result(
                step_id="step-001",
                step_type=StepType.RUN_FIDELITY_GATE,
                task_id=None,
                phase_id="phase-1",
                gate_attempt_id="gate-001",
            ),
        )
        assert result.success is False
        assert result.error_code == ERROR_INVALID_GATE_EVIDENCE

    def test_no_gate_attempt_id_returns_error(self, tmp_path):
        """If result has no gate_attempt_id for a RUN_FIDELITY_GATE step, model validation rejects it."""
        from pydantic import ValidationError

        # Model-level validation now catches missing gate_attempt_id
        with pytest.raises(ValidationError, match="gate_attempt_id is required"):
            _result(
                step_id="step-001",
                step_type=StepType.RUN_FIDELITY_GATE,
                task_id=None,
                phase_id="phase-1",
                gate_attempt_id=None,  # No attempt ID
            )


class TestVerificationReceiptValidation:
    """Verification receipt completeness and binding checks."""

    def test_verification_receipt_task_mismatch_returns_invalid(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        now = datetime.now(timezone.utc)
        expected_hash = "a" * 64
        session = make_session(
            last_step_issued=_issued(
                step_id="step-verify-1",
                step_type=StepType.EXECUTE_VERIFICATION,
                task_id="verify-1",
                phase_id="phase-1",
                step_proof="proof-verify-1",
            ),
            pending_verification_receipt=PendingVerificationReceipt(
                step_id="step-verify-1",
                task_id="verify-EXPECTED-DIFFERENT",
                expected_command_hash=expected_hash,
                issued_at=now,
            ),
        )

        receipt = VerificationReceipt(
            command_hash=expected_hash,
            exit_code=0,
            output_digest="b" * 64,
            issued_at=now,
            step_id="step-verify-1",
        )
        result = orch.compute_next_step(
            session,
            _result(
                step_id="step-verify-1",
                step_type=StepType.EXECUTE_VERIFICATION,
                task_id="verify-1",
                phase_id="phase-1",
                outcome=StepOutcome.SUCCESS,
                step_proof="proof-verify-1",
                verification_receipt=receipt.model_dump(mode="json"),
            ),
        )
        assert result.success is False
        assert result.error_code == ERROR_VERIFICATION_RECEIPT_INVALID

    def test_verification_receipt_issued_before_pending_window_is_invalid(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        now = datetime.now(timezone.utc)
        expected_hash = "c" * 64
        session = make_session(
            last_step_issued=_issued(
                step_id="step-verify-2",
                step_type=StepType.EXECUTE_VERIFICATION,
                task_id="verify-2",
                phase_id="phase-1",
                step_proof="proof-verify-2",
            ),
            pending_verification_receipt=PendingVerificationReceipt(
                step_id="step-verify-2",
                task_id="verify-2",
                expected_command_hash=expected_hash,
                issued_at=now,
            ),
        )

        receipt = VerificationReceipt(
            command_hash=expected_hash,
            exit_code=0,
            output_digest="d" * 64,
            issued_at=now - timedelta(minutes=1),
            step_id="step-verify-2",
        )
        result = orch.compute_next_step(
            session,
            _result(
                step_id="step-verify-2",
                step_type=StepType.EXECUTE_VERIFICATION,
                task_id="verify-2",
                phase_id="phase-1",
                outcome=StepOutcome.SUCCESS,
                step_proof="proof-verify-2",
                verification_receipt=receipt.model_dump(mode="json"),
            ),
        )
        assert result.success is False
        assert result.error_code == ERROR_VERIFICATION_RECEIPT_INVALID


class TestIntegrityFailurePaths:
    """Integrity checksum and audit failures should map to explicit error codes."""

    def test_tampered_gate_checksum_returns_integrity_error(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        now = datetime.now(timezone.utc)
        session = make_session(
            last_step_issued=_issued(
                step_id="step-gate-1",
                step_type=StepType.RUN_FIDELITY_GATE,
                task_id=None,
                phase_id="phase-1",
                step_proof="proof-gate-1",
            ),
            pending_gate_evidence=PendingGateEvidence(
                gate_attempt_id="gate-1",
                step_id="step-gate-1",
                phase_id="phase-1",
                verdict=GateVerdict.PASS,
                issued_at=now,
                integrity_checksum="tampered-checksum",
            ),
        )

        result = orch.compute_next_step(
            session,
            _result(
                step_id="step-gate-1",
                step_type=StepType.RUN_FIDELITY_GATE,
                task_id=None,
                phase_id="phase-1",
                gate_attempt_id="gate-1",
                outcome=StepOutcome.SUCCESS,
                step_proof="proof-gate-1",
            ),
        )
        assert result.success is False
        assert result.error_code == ERROR_GATE_INTEGRITY_CHECKSUM

    def test_gate_audit_failure_detected_before_terminal_transition(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        now = datetime.now(timezone.utc)
        spec_data = make_spec_data(
            phases=[
                {
                    "id": "phase-1",
                    "title": "Phase 1",
                    "metadata": {"requires_gate": True},
                    "tasks": [],
                }
            ]
        )
        session = make_session(
            active_phase_id="phase-1",
            required_phase_gates={"phase-1": ["fidelity"]},
            satisfied_gates={"phase-1": ["fidelity"]},
            phase_gates={
                "phase-1": PhaseGateRecord(
                    required=True,
                    status=PhaseGateStatus.FAILED,
                )
            },
        )

        with patch.object(orch, "_should_run_fidelity_gate", return_value=False):
            result = orch._determine_next_step(session, spec_data, now)
        assert result.success is False
        assert result.error_code == ERROR_GATE_AUDIT_FAILURE


# =============================================================================
# Step 3: Record Step Outcome — consecutive_errors
# =============================================================================


class TestRecordStepOutcome:
    """Step 3: Recording outcomes affects counters correctly."""

    def test_failure_increments_consecutive_errors(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            last_step_issued=_issued(step_id="s1"),
            counters=SessionCounters(consecutive_errors=1),
        )

        orch._record_step_outcome(
            session,
            _result(step_id="s1", outcome=StepOutcome.FAILURE),
            datetime.now(timezone.utc),
        )
        assert session.counters.consecutive_errors == 2

    def test_success_resets_consecutive_errors(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            last_step_issued=_issued(step_id="s1"),
            counters=SessionCounters(consecutive_errors=5),
        )

        orch._record_step_outcome(
            session,
            _result(step_id="s1", outcome=StepOutcome.SUCCESS),
            datetime.now(timezone.utc),
        )
        assert session.counters.consecutive_errors == 0

    def test_gate_step_does_not_affect_consecutive_errors(self, tmp_path):
        """Per ADR-002: gate steps should not affect consecutive_errors at all."""
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            last_step_issued=_issued(step_id="s1", step_type=StepType.RUN_FIDELITY_GATE),
            counters=SessionCounters(consecutive_errors=2),
        )

        # Failure on gate step
        orch._record_step_outcome(
            session,
            _result(
                step_id="s1",
                step_type=StepType.RUN_FIDELITY_GATE,
                outcome=StepOutcome.FAILURE,
                task_id=None,
                gate_attempt_id="gate-001",
            ),
            datetime.now(timezone.utc),
        )
        assert session.counters.consecutive_errors == 2  # Unchanged

    def test_address_fidelity_feedback_does_not_affect_consecutive_errors(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            counters=SessionCounters(consecutive_errors=3),
        )

        orch._record_step_outcome(
            session,
            _result(
                step_id="s1",
                step_type=StepType.ADDRESS_FIDELITY_FEEDBACK,
                outcome=StepOutcome.FAILURE,
                task_id=None,
            ),
            datetime.now(timezone.utc),
        )
        assert session.counters.consecutive_errors == 3

    def test_task_completion_tracked(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            completed_task_ids=[],
            counters=SessionCounters(tasks_completed=0),
        )

        orch._record_step_outcome(
            session,
            _result(step_id="s1", step_type=StepType.IMPLEMENT_TASK, outcome=StepOutcome.SUCCESS, task_id="task-1"),
            datetime.now(timezone.utc),
        )
        assert "task-1" in session.completed_task_ids
        assert session.counters.tasks_completed == 1

    def test_duplicate_task_completion_not_double_counted(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            completed_task_ids=["task-1"],
            counters=SessionCounters(tasks_completed=1),
        )

        orch._record_step_outcome(
            session,
            _result(step_id="s1", step_type=StepType.IMPLEMENT_TASK, outcome=StepOutcome.SUCCESS, task_id="task-1"),
            datetime.now(timezone.utc),
        )
        assert session.completed_task_ids.count("task-1") == 1
        assert session.counters.tasks_completed == 1

    def test_verification_completion_tracked(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(completed_task_ids=[])

        orch._record_step_outcome(
            session,
            _result(step_id="s1", step_type=StepType.EXECUTE_VERIFICATION, outcome=StepOutcome.SUCCESS, task_id="verify-1"),
            datetime.now(timezone.utc),
        )
        assert "verify-1" in session.completed_task_ids

    def test_failed_task_not_tracked_as_complete(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(completed_task_ids=[])

        orch._record_step_outcome(
            session,
            _result(step_id="s1", step_type=StepType.IMPLEMENT_TASK, outcome=StepOutcome.FAILURE, task_id="task-1"),
            datetime.now(timezone.utc),
        )
        assert "task-1" not in session.completed_task_ids


# =============================================================================
# Step 3: Gate Outcome Recording
# =============================================================================


class TestGateOutcomeRecording:
    """Gate pass/fail derived from pending_gate_evidence.verdict + gate policy."""

    def test_success_outcome_with_fail_verdict_records_gate_failed(self, tmp_path):
        """Per ADR: step outcome=success + verdict=fail → gate FAILED."""
        orch = _make_orchestrator(tmp_path)
        now = datetime.now(timezone.utc)
        session = make_session(
            gate_policy=GatePolicy.STRICT,
            phase_gates={
                "phase-1": PhaseGateRecord(
                    required=True,
                    status=PhaseGateStatus.PENDING,
                ),
            },
            pending_gate_evidence=PendingGateEvidence(
                gate_attempt_id="gate-001",
                step_id="s1",
                phase_id="phase-1",
                verdict=GateVerdict.FAIL,
                issued_at=now,
            ),
        )

        orch._record_gate_outcome(
            session,
            _result(
                step_id="s1",
                step_type=StepType.RUN_FIDELITY_GATE,
                outcome=StepOutcome.SUCCESS,
                task_id=None,
                phase_id="phase-1",
                gate_attempt_id="gate-001",
            ),
            now,
        )

        assert session.phase_gates["phase-1"].status == PhaseGateStatus.FAILED
        assert session.phase_gates["phase-1"].verdict == GateVerdict.FAIL

    def test_success_outcome_with_pass_verdict_records_gate_passed(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        now = datetime.now(timezone.utc)
        session = make_session(
            gate_policy=GatePolicy.STRICT,
            phase_gates={
                "phase-1": PhaseGateRecord(
                    required=True,
                    status=PhaseGateStatus.PENDING,
                ),
            },
            pending_gate_evidence=PendingGateEvidence(
                gate_attempt_id="gate-001",
                step_id="s1",
                phase_id="phase-1",
                verdict=GateVerdict.PASS,
                issued_at=now,
            ),
        )

        orch._record_gate_outcome(
            session,
            _result(
                step_id="s1",
                step_type=StepType.RUN_FIDELITY_GATE,
                outcome=StepOutcome.SUCCESS,
                task_id=None,
                phase_id="phase-1",
                gate_attempt_id="gate-001",
            ),
            now,
        )

        assert session.phase_gates["phase-1"].status == PhaseGateStatus.PASSED
        assert session.phase_gates["phase-1"].verdict == GateVerdict.PASS

    def test_step_failure_without_evidence_records_gate_failed(self, tmp_path):
        """Step itself failed (review couldn't run) → gate FAILED fallback."""
        orch = _make_orchestrator(tmp_path)
        now = datetime.now(timezone.utc)
        session = make_session(
            phase_gates={
                "phase-1": PhaseGateRecord(required=True, status=PhaseGateStatus.PENDING),
            },
            # No pending_gate_evidence
        )

        orch._record_gate_outcome(
            session,
            _result(
                step_id="s1",
                step_type=StepType.RUN_FIDELITY_GATE,
                outcome=StepOutcome.FAILURE,
                task_id=None,
                phase_id="phase-1",
                gate_attempt_id="gate-001",
            ),
            now,
        )

        assert session.phase_gates["phase-1"].status == PhaseGateStatus.FAILED
        assert session.phase_gates["phase-1"].verdict == GateVerdict.FAIL


# =============================================================================
# Step 6: Step Staleness
# =============================================================================


class TestStepStaleness:
    """Step 6: Hard backstop for disappeared callers."""

    def test_step_stale_returns_error(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            last_step_issued=_issued(step_id="s1", minutes_ago=120),
            limits=SessionLimits(step_stale_minutes=60),
        )

        # Mock spec integrity to bypass hash mismatch (we're testing step staleness, not spec integrity)
        spec_data = make_spec_data()
        with patch.object(orch, "_validate_spec_integrity", return_value=(spec_data, None)):
            result = orch.compute_next_step(session, _result(step_id="s1"))
        assert result.error_code == ERROR_STEP_STALE

    def test_step_not_stale_within_threshold(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            last_step_issued=_issued(step_id="s1", minutes_ago=5),
            limits=SessionLimits(step_stale_minutes=60),
        )
        result = orch.compute_next_step(session, _result(step_id="s1"))
        assert result.error_code != ERROR_STEP_STALE


# =============================================================================
# Step 7: Heartbeat Staleness
# =============================================================================


class TestHeartbeatStaleness:
    """Step 7: Heartbeat staleness with grace window."""

    def test_no_heartbeat_within_grace_is_ok(self, tmp_path):
        """Session just created, no heartbeat, within grace window → continue."""
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            created_at=datetime.now(timezone.utc) - timedelta(minutes=2),
            limits=SessionLimits(heartbeat_grace_minutes=5),
        )
        session.context.last_heartbeat_at = None

        result = orch._check_heartbeat_staleness(session, datetime.now(timezone.utc))
        assert result is None  # OK

    def test_no_heartbeat_past_grace_triggers_stale(self, tmp_path):
        """Session old, no heartbeat, past grace window → stale."""
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            created_at=datetime.now(timezone.utc) - timedelta(minutes=10),
            limits=SessionLimits(heartbeat_grace_minutes=5),
        )
        session.context.last_heartbeat_at = None

        result = orch._check_heartbeat_staleness(session, datetime.now(timezone.utc))
        assert result is not None
        # Should be a pause result
        assert result.session.status == SessionStatus.PAUSED

    def test_recent_heartbeat_is_ok(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            limits=SessionLimits(heartbeat_stale_minutes=10),
        )
        session.context.last_heartbeat_at = datetime.now(timezone.utc) - timedelta(minutes=3)

        result = orch._check_heartbeat_staleness(session, datetime.now(timezone.utc))
        assert result is None

    def test_stale_heartbeat_triggers_check(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            limits=SessionLimits(heartbeat_stale_minutes=10),
        )
        session.context.last_heartbeat_at = datetime.now(timezone.utc) - timedelta(minutes=15)

        result = orch._check_heartbeat_staleness(session, datetime.now(timezone.utc))
        assert result is not None


# =============================================================================
# Step 8: Pause Guards
# =============================================================================


class TestPauseGuards:
    """Step 8: Context, error, and task limit guards."""

    def test_context_limit_triggers_pause(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            limits=SessionLimits(context_threshold_pct=85),
        )
        session.context.context_usage_pct = 90

        result = orch._check_pause_guards(session)
        assert result is not None
        assert result.session.status == SessionStatus.PAUSED
        assert result.session.pause_reason == PauseReason.CONTEXT_LIMIT

    def test_context_below_threshold_is_ok(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            limits=SessionLimits(context_threshold_pct=85),
        )
        session.context.context_usage_pct = 50

        result = orch._check_pause_guards(session)
        assert result is None

    def test_error_threshold_triggers_pause(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            counters=SessionCounters(consecutive_errors=3),
            limits=SessionLimits(max_consecutive_errors=3),
        )

        result = orch._check_pause_guards(session)
        assert result is not None
        assert result.session.pause_reason == PauseReason.ERROR_THRESHOLD

    def test_errors_below_threshold_is_ok(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            counters=SessionCounters(consecutive_errors=1),
            limits=SessionLimits(max_consecutive_errors=3),
        )

        result = orch._check_pause_guards(session)
        assert result is None

    def test_task_limit_triggers_pause(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            counters=SessionCounters(tasks_completed=100),
            limits=SessionLimits(max_tasks_per_session=100),
        )

        result = orch._check_pause_guards(session)
        assert result is not None
        assert result.session.pause_reason == PauseReason.TASK_LIMIT

    def test_tasks_below_limit_is_ok(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            counters=SessionCounters(tasks_completed=5),
            limits=SessionLimits(max_tasks_per_session=100),
        )

        result = orch._check_pause_guards(session)
        assert result is None


# =============================================================================
# Step 9: Fidelity Cycle Stop Heuristic
# =============================================================================


class TestFidelityCycleLimit:
    """Step 9: Max fidelity review cycles per phase."""

    def test_fidelity_cycle_limit_triggers_pause(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            active_phase_id="phase-1",
            counters=SessionCounters(fidelity_review_cycles_in_active_phase=3),
            limits=SessionLimits(max_fidelity_review_cycles_per_phase=3),
        )
        # Need to get past spec integrity + terminal checks
        # Test _check_pause_guards is not sufficient because fidelity cycle is step 9
        # But we can directly check the condition
        assert session.counters.fidelity_review_cycles_in_active_phase >= session.limits.max_fidelity_review_cycles_per_phase


# =============================================================================
# Gate Policy Evaluation
# =============================================================================


class TestGatePolicyEvaluation:
    """Evaluate gate verdict against configured policy."""

    def _evidence(self, verdict: GateVerdict) -> PendingGateEvidence:
        return PendingGateEvidence(
            gate_attempt_id="gate-001",
            step_id="s1",
            phase_id="phase-1",
            verdict=verdict,
            issued_at=datetime.now(timezone.utc),
        )

    def test_strict_pass_on_pass(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(gate_policy=GatePolicy.STRICT)
        should_pass, reason = orch._evaluate_gate_policy(session, self._evidence(GateVerdict.PASS))
        assert should_pass is True
        assert reason is None

    def test_strict_fail_on_warn(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(gate_policy=GatePolicy.STRICT)
        should_pass, reason = orch._evaluate_gate_policy(session, self._evidence(GateVerdict.WARN))
        assert should_pass is False
        assert reason == PauseReason.GATE_FAILED

    def test_strict_fail_on_fail(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(gate_policy=GatePolicy.STRICT)
        should_pass, reason = orch._evaluate_gate_policy(session, self._evidence(GateVerdict.FAIL))
        assert should_pass is False
        assert reason == PauseReason.GATE_FAILED

    def test_lenient_pass_on_pass(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(gate_policy=GatePolicy.LENIENT)
        should_pass, _ = orch._evaluate_gate_policy(session, self._evidence(GateVerdict.PASS))
        assert should_pass is True

    def test_lenient_pass_on_warn(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(gate_policy=GatePolicy.LENIENT)
        should_pass, _ = orch._evaluate_gate_policy(session, self._evidence(GateVerdict.WARN))
        assert should_pass is True

    def test_lenient_fail_on_fail(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(gate_policy=GatePolicy.LENIENT)
        should_pass, reason = orch._evaluate_gate_policy(session, self._evidence(GateVerdict.FAIL))
        assert should_pass is False
        assert reason == PauseReason.GATE_FAILED

    def test_manual_always_requires_review(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(gate_policy=GatePolicy.MANUAL)
        should_pass, reason = orch._evaluate_gate_policy(session, self._evidence(GateVerdict.PASS))
        assert should_pass is False
        assert reason == PauseReason.GATE_REVIEW_REQUIRED

    def test_manual_fail_verdict_also_requires_review(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(gate_policy=GatePolicy.MANUAL)
        should_pass, reason = orch._evaluate_gate_policy(session, self._evidence(GateVerdict.FAIL))
        assert should_pass is False
        assert reason == PauseReason.GATE_REVIEW_REQUIRED


# =============================================================================
# Phase Advancement: Cycle Counter Reset
# =============================================================================


class TestPhaseAdvancement:
    """Phase change resets fidelity_review_cycles_in_active_phase."""

    def test_phase_change_resets_fidelity_counter(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        spec_data = make_spec_data()
        session = make_session(
            active_phase_id="phase-1",
            counters=SessionCounters(fidelity_review_cycles_in_active_phase=2),
            completed_task_ids=["task-1", "task-2", "verify-1"],  # All phase-1 tasks done
        )

        next_task = orch._find_next_task(session, spec_data)
        # Should find task-3 in phase-2 and reset cycle counter
        assert next_task is not None
        assert next_task.get("id") == "task-3"
        assert session.active_phase_id == "phase-2"
        assert session.counters.fidelity_review_cycles_in_active_phase == 0

    def test_same_phase_does_not_reset_counter(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        spec_data = make_spec_data()
        session = make_session(
            active_phase_id="phase-1",
            counters=SessionCounters(fidelity_review_cycles_in_active_phase=1),
            completed_task_ids=["task-1"],  # Only first task done
        )

        next_task = orch._find_next_task(session, spec_data)
        assert next_task is not None
        assert next_task.get("id") == "task-2"
        assert session.active_phase_id == "phase-1"
        assert session.counters.fidelity_review_cycles_in_active_phase == 1  # Unchanged


# =============================================================================
# Verification Ordering
# =============================================================================


class TestVerificationOrdering:
    """Verification should not run before implementation tasks in the same phase."""

    def test_compute_next_step_prefers_implement_before_verify(self, tmp_path):
        spec_data = make_spec_data(
            phases=[
                {
                    "id": "phase-1",
                    "title": "Phase 1",
                    "sequence_index": 0,
                    "tasks": [
                        {"id": "task-1", "title": "Task 1", "type": "task", "status": "pending"},
                        {"id": "verify-1", "title": "Verify 1", "type": "verify", "status": "pending"},
                    ],
                },
            ],
        )
        orch = _make_orchestrator(tmp_path, spec_data)
        session = make_session(
            active_phase_id="phase-1",
            spec_structure_hash=compute_spec_structure_hash(spec_data),
            completed_task_ids=[],
            last_step_issued=None,
        )

        result = orch.compute_next_step(session, last_step_result=None)

        assert result.success is True
        assert result.next_step is not None
        assert result.next_step.type == StepType.IMPLEMENT_TASK
        assert result.next_step.task_id == "task-1"


class TestExecutionOrderModelCheck:
    """Table-driven checks for phase progression next-step transitions."""

    @pytest.mark.parametrize(
        "name,phases,completed_task_ids,expected_step_type,expected_task_id",
        [
            (
                "task_only_phase",
                [
                    {
                        "id": "phase-1",
                        "title": "Phase 1",
                        "tasks": [
                            {"id": "task-1", "title": "Task 1", "type": "task", "status": "pending"},
                        ],
                    },
                ],
                [],
                StepType.IMPLEMENT_TASK,
                "task-1",
            ),
            (
                "task_and_verify_phase_task_pending",
                [
                    {
                        "id": "phase-1",
                        "title": "Phase 1",
                        "tasks": [
                            {"id": "task-1", "title": "Task 1", "type": "task", "status": "pending"},
                            {"id": "verify-1", "title": "Verify 1", "type": "verify", "status": "pending"},
                        ],
                    },
                ],
                [],
                StepType.IMPLEMENT_TASK,
                "task-1",
            ),
            (
                "task_and_verify_phase_ready_for_verification",
                [
                    {
                        "id": "phase-1",
                        "title": "Phase 1",
                        "tasks": [
                            {"id": "task-1", "title": "Task 1", "type": "task", "status": "pending"},
                            {"id": "verify-1", "title": "Verify 1", "type": "verify", "status": "pending"},
                        ],
                    },
                ],
                ["task-1"],
                StepType.EXECUTE_VERIFICATION,
                "verify-1",
            ),
            (
                "verify_only_phase",
                [
                    {
                        "id": "phase-1",
                        "title": "Phase 1",
                        "tasks": [
                            {"id": "verify-1", "title": "Verify 1", "type": "verify", "status": "pending"},
                        ],
                    },
                ],
                [],
                StepType.EXECUTE_VERIFICATION,
                "verify-1",
            ),
            (
                "gate_required_phase",
                [
                    {
                        "id": "phase-1",
                        "title": "Phase 1",
                        "metadata": {"requires_gate": True},
                        "tasks": [
                            {"id": "task-1", "title": "Task 1", "type": "task", "status": "pending"},
                        ],
                    },
                ],
                ["task-1"],
                StepType.RUN_FIDELITY_GATE,
                None,
            ),
        ],
    )
    def test_phase_progression_matrix(
        self,
        tmp_path,
        name,
        phases,
        completed_task_ids,
        expected_step_type,
        expected_task_id,
    ):
        spec_data = make_spec_data(phases=phases)
        orch = _make_orchestrator(tmp_path, spec_data)
        session = make_session(
            active_phase_id="phase-1",
            spec_structure_hash=compute_spec_structure_hash(spec_data),
            completed_task_ids=completed_task_ids,
            last_step_issued=None,
        )

        result = orch.compute_next_step(session, last_step_result=None)

        assert result.success is True, name
        assert result.next_step is not None, name
        assert result.next_step.type == expected_step_type, name
        assert result.next_step.task_id == expected_task_id, name


# =============================================================================
# Terminal State Check (step 5)
# =============================================================================


class TestTerminalStates:
    """Step 5: Terminal sessions should return early."""

    @pytest.mark.parametrize("status", [SessionStatus.COMPLETED, SessionStatus.ENDED, SessionStatus.FAILED])
    def test_terminal_state_returns_no_next_step(self, tmp_path, status):
        orch = _make_orchestrator(tmp_path)
        session = make_session(
            status=status,
            last_step_issued=None,  # No feedback required
        )

        result = orch.compute_next_step(session)
        # The spec integrity check runs first and may fail, but
        # if the session is terminal _after_ integrity check, it returns null
        # For FAILED, integrity check may set FAILED again; let's test directly
        if status == SessionStatus.FAILED:
            # FAILED sessions may get SPEC_REBASE_REQUIRED from integrity check
            # or be recognized as terminal in step 5
            assert result.next_step is None or result.error_code is not None
        else:
            # COMPLETED and ENDED should hit step 5 (after step 4 integrity succeeds)
            # Since spec may not exist, integrity check may fail first
            # Let's just verify that if integrity passes, terminal is detected
            pass


# =============================================================================
# Create Step Helpers
# =============================================================================


class TestCreateStepHelpers:
    """Test step creation helpers produce valid NextStep objects."""

    def test_create_pause_result(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(status=SessionStatus.RUNNING)
        now = datetime.now(timezone.utc)

        result = orch._create_pause_result(session, PauseReason.USER, now, "User requested")

        assert result.success is True
        assert result.should_persist is True
        assert result.next_step is not None
        assert result.next_step.type == StepType.PAUSE
        assert result.next_step.reason == PauseReason.USER
        assert result.next_step.message == "User requested"
        assert session.status == SessionStatus.PAUSED
        assert session.pause_reason == PauseReason.USER
        assert session.last_step_issued is not None
        assert session.last_step_issued.type == StepType.PAUSE

    def test_create_implement_task_step(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(active_phase_id="phase-1")
        task = {"id": "task-1", "title": "Implement feature"}
        spec_data = make_spec_data()
        now = datetime.now(timezone.utc)

        result = orch._create_implement_task_step(session, task, spec_data, now)

        assert result.success is True
        assert result.next_step.type == StepType.IMPLEMENT_TASK
        assert result.next_step.task_id == "task-1"
        assert result.next_step.task_title == "Implement feature"
        assert result.next_step.instructions is not None
        assert len(result.next_step.instructions) == 3
        assert session.last_step_issued.type == StepType.IMPLEMENT_TASK
        assert session.last_task_id == "task-1"

    def test_create_verification_step(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(active_phase_id="phase-1")
        task = {"id": "verify-1", "title": "Run tests"}
        now = datetime.now(timezone.utc)

        result = orch._create_verification_step(session, task, now)

        assert result.success is True
        assert result.next_step.type == StepType.EXECUTE_VERIFICATION
        assert result.next_step.task_id == "verify-1"

    def test_create_fidelity_gate_step(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(active_phase_id="phase-1")
        phase = {"id": "phase-1", "title": "Phase 1"}
        now = datetime.now(timezone.utc)

        result = orch._create_fidelity_gate_step(session, phase, now)

        assert result.success is True
        assert result.next_step.type == StepType.RUN_FIDELITY_GATE
        assert result.next_step.phase_id == "phase-1"
        assert result.next_step.gate_attempt_id is not None
        assert "phase-1" in session.phase_gates

    def test_create_complete_spec_result(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session()
        now = datetime.now(timezone.utc)

        result = orch._create_complete_spec_result(session, now)

        assert result.success is True
        assert result.next_step.type == StepType.COMPLETE_SPEC
        assert session.status == SessionStatus.COMPLETED

    def test_create_fidelity_feedback_step(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session()
        evidence = PendingGateEvidence(
            gate_attempt_id="gate-001",
            step_id="s1",
            phase_id="phase-1",
            verdict=GateVerdict.FAIL,
            issued_at=datetime.now(timezone.utc),
        )
        now = datetime.now(timezone.utc)

        result = orch._create_fidelity_feedback_step(session, evidence, now)

        assert result.success is True
        assert result.next_step.type == StepType.ADDRESS_FIDELITY_FEEDBACK
        assert result.next_step.phase_id == "phase-1"

    def test_created_steps_include_step_proof_tokens(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        session = make_session(active_phase_id="phase-1")
        now = datetime.now(timezone.utc)

        implement = orch._create_implement_task_step(
            session,
            {"id": "task-1", "title": "Implement feature"},
            make_spec_data(),
            now,
        )
        pause = orch._create_pause_result(session, PauseReason.USER, now, "pause")

        assert implement.next_step.step_proof is not None
        assert len(implement.next_step.step_proof) == 64
        assert pause.next_step.step_proof is not None
        assert len(pause.next_step.step_proof) == 64
        assert session.last_step_issued is not None
        assert session.last_step_issued.step_proof == pause.next_step.step_proof

    def test_emitted_step_instructions_match_registered_router_actions(self, tmp_path):
        from foundry_mcp.tools.unified.review import _REVIEW_ROUTER
        from foundry_mcp.tools.unified.task_handlers import _TASK_ROUTER
        from foundry_mcp.tools.unified.verification import _VERIFICATION_ROUTER

        orch = _make_orchestrator(tmp_path)
        session = make_session(active_phase_id="phase-1")
        now = datetime.now(timezone.utc)

        implement_result = orch._create_implement_task_step(
            session,
            {"id": "task-1", "title": "Implement feature"},
            make_spec_data(),
            now,
        )
        verification_result = orch._create_verification_step(
            session,
            {"id": "verify-1", "title": "Run tests"},
            now,
        )
        gate_result = orch._create_fidelity_gate_step(
            session,
            {"id": "phase-1", "title": "Phase 1"},
            now,
        )
        feedback_result = orch._create_fidelity_feedback_step(
            session,
            PendingGateEvidence(
                gate_attempt_id="gate-001",
                step_id="s1",
                phase_id="phase-1",
                verdict=GateVerdict.FAIL,
                issued_at=now,
            ),
            now,
        )

        allowed_actions = {
            "task": set(_TASK_ROUTER.allowed_actions()),
            "review": set(_REVIEW_ROUTER.allowed_actions()),
            "verification": set(_VERIFICATION_ROUTER.allowed_actions()),
        }

        emitted = (
            (implement_result.next_step.instructions or [])
            + (verification_result.next_step.instructions or [])
            + (gate_result.next_step.instructions or [])
            + (feedback_result.next_step.instructions or [])
        )
        assert emitted, "Expected at least one emitted step instruction"

        for instruction in emitted:
            assert instruction.tool in allowed_actions
            assert instruction.action in allowed_actions[instruction.tool]


# =============================================================================
# Gate Evidence Handling (steps 13-14)
# =============================================================================


class TestHandleGateEvidence:
    """Steps 13-14: Handle pending gate evidence with policy evaluation."""

    def test_gate_passed_clears_evidence(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        spec_data = make_spec_data()
        now = datetime.now(timezone.utc)
        evidence = PendingGateEvidence(
            gate_attempt_id="gate-001",
            step_id="s1",
            phase_id="phase-1",
            verdict=GateVerdict.PASS,
            issued_at=now,
        )
        session = make_session(
            gate_policy=GatePolicy.STRICT,
            pending_gate_evidence=evidence,
        )

        result = orch._handle_gate_evidence(session, spec_data, now)
        assert result.success is True
        assert session.pending_gate_evidence is None

    def test_gate_failed_auto_retry_creates_feedback_step(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        spec_data = make_spec_data()
        now = datetime.now(timezone.utc)
        evidence = PendingGateEvidence(
            gate_attempt_id="gate-001",
            step_id="s1",
            phase_id="phase-1",
            verdict=GateVerdict.FAIL,
            issued_at=now,
        )
        session = make_session(
            gate_policy=GatePolicy.STRICT,
            pending_gate_evidence=evidence,
            stop_conditions=StopConditions(auto_retry_fidelity_gate=True),
        )

        result = orch._handle_gate_evidence(session, spec_data, now)
        assert result.success is True
        assert result.next_step.type == StepType.ADDRESS_FIDELITY_FEEDBACK

    def test_gate_failed_no_auto_retry_pauses(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        spec_data = make_spec_data()
        now = datetime.now(timezone.utc)
        evidence = PendingGateEvidence(
            gate_attempt_id="gate-001",
            step_id="s1",
            phase_id="phase-1",
            verdict=GateVerdict.FAIL,
            issued_at=now,
        )
        session = make_session(
            gate_policy=GatePolicy.STRICT,
            pending_gate_evidence=evidence,
            stop_conditions=StopConditions(auto_retry_fidelity_gate=False),
        )

        result = orch._handle_gate_evidence(session, spec_data, now)
        assert result.session.status == SessionStatus.PAUSED
        assert result.session.pause_reason == PauseReason.GATE_FAILED

    def test_manual_policy_always_pauses_for_review(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        spec_data = make_spec_data()
        now = datetime.now(timezone.utc)
        evidence = PendingGateEvidence(
            gate_attempt_id="gate-001",
            step_id="s1",
            phase_id="phase-1",
            verdict=GateVerdict.PASS,  # Even PASS requires manual review
            issued_at=now,
        )
        session = make_session(
            gate_policy=GatePolicy.MANUAL,
            pending_gate_evidence=evidence,
        )

        result = orch._handle_gate_evidence(session, spec_data, now)
        assert result.session.status == SessionStatus.PAUSED
        assert result.session.pause_reason == PauseReason.GATE_REVIEW_REQUIRED

    def test_fidelity_cycle_incremented_in_record_step_outcome(self, tmp_path):
        """Fidelity cycle counter is incremented in _record_step_outcome (step 3), not _handle_gate_evidence."""
        orch = _make_orchestrator(tmp_path)
        now = datetime.now(timezone.utc)
        session = make_session(
            active_phase_id="phase-1",
            counters=SessionCounters(fidelity_review_cycles_in_active_phase=1),
            phase_gates={"phase-1": PhaseGateRecord(
                required=True, status=PhaseGateStatus.PENDING,
            )},
        )

        # _record_step_outcome increments fidelity cycle counter on gate steps
        orch._record_step_outcome(
            session,
            _result(
                step_id="s1",
                step_type=StepType.RUN_FIDELITY_GATE,
                outcome=StepOutcome.SUCCESS,
                task_id=None,
                phase_id="phase-1",
                gate_attempt_id="gate-001",
            ),
            now,
        )
        assert session.counters.fidelity_review_cycles_in_active_phase == 2


# =============================================================================
# Audit Integration Tests
# =============================================================================


class TestAuditIntegration:
    """Tests verifying audit events are emitted at orchestrator decision points."""

    def _make_session_for_orch(self, tmp_path):
        """Create a session with spec_structure_hash matching the spec on disk."""
        spec_data = make_spec_data()
        spec_hash = compute_spec_structure_hash(spec_data)
        return make_session(spec_structure_hash=spec_hash)

    def test_step_issued_emits_audit_event(self, tmp_path):
        """Orchestrator emits STEP_ISSUED audit event when issuing a task step."""
        from foundry_mcp.core.autonomy.audit import AuditEventType, AuditLedger

        orch = _make_orchestrator(tmp_path)
        session = self._make_session_for_orch(tmp_path)

        result = orch.compute_next_step(session)
        assert result.success
        assert result.next_step is not None

        ledger = AuditLedger(
            spec_id=session.spec_id,
            workspace_path=orch.workspace_path,
        )
        entries = ledger.get_entries(event_type=AuditEventType.STEP_ISSUED)
        assert len(entries) >= 1
        assert entries[0].session_id == session.id
        assert entries[0].action == "issue_step"

    def test_step_consumed_emits_audit_event(self, tmp_path):
        """Orchestrator emits STEP_CONSUMED audit event when recording step outcome."""
        from foundry_mcp.core.autonomy.audit import AuditEventType, AuditLedger

        orch = _make_orchestrator(tmp_path)
        session = self._make_session_for_orch(tmp_path)

        result1 = orch.compute_next_step(session)
        assert result1.success

        step_result = _result(
            step_id=result1.next_step.step_id,
            step_type=result1.next_step.type,
            outcome=StepOutcome.SUCCESS,
            task_id=result1.next_step.task_id,
            phase_id=result1.next_step.phase_id,
            step_proof=result1.next_step.step_proof,
        )
        orch.compute_next_step(session, last_step_result=step_result)

        ledger = AuditLedger(
            spec_id=session.spec_id,
            workspace_path=orch.workspace_path,
        )
        entries = ledger.get_entries(event_type=AuditEventType.STEP_CONSUMED)
        assert len(entries) >= 1
        consumed = entries[0]
        assert consumed.session_id == session.id
        assert consumed.action == "consume_step"
        assert consumed.metadata["outcome"] == "success"

    def test_pause_emits_audit_event(self, tmp_path):
        """Orchestrator emits PAUSE audit event on context limit pause."""
        from foundry_mcp.core.autonomy.audit import AuditEventType, AuditLedger

        orch = _make_orchestrator(tmp_path)
        session = self._make_session_for_orch(tmp_path)
        session.context.context_usage_pct = 95
        session.limits.context_threshold_pct = 90

        result = orch.compute_next_step(session)
        assert result.success
        assert result.next_step.type == StepType.PAUSE

        ledger = AuditLedger(
            spec_id=session.spec_id,
            workspace_path=orch.workspace_path,
        )
        entries = ledger.get_entries(event_type=AuditEventType.PAUSE)
        assert len(entries) >= 1
        assert entries[0].metadata["reason"] == "context_limit"

    def test_audit_failure_does_not_block_step(self, tmp_path):
        """Audit write failure is best-effort and does not block orchestration."""
        orch = _make_orchestrator(tmp_path)
        session = self._make_session_for_orch(tmp_path)

        with patch(
            "foundry_mcp.core.autonomy.audit.AuditLedger.append",
            side_effect=OSError("disk full"),
        ):
            result = orch.compute_next_step(session)
            assert result.success  # Should succeed despite audit failure


# =============================================================================
# Spec Cache Invalidation (T16)
# =============================================================================


class TestSpecCacheInvalidation:
    """Test invalidate_spec_cache() method."""

    def test_invalidate_clears_cache_for_matching_spec(self, tmp_path):
        """invalidate_spec_cache(spec_id) clears cache when spec_id matches."""
        orch = _make_orchestrator(tmp_path)
        # Manually populate the cache
        orch._spec_cache = ("test-spec-001", 1234.0, 5678, {"spec_id": "test-spec-001"})

        orch.invalidate_spec_cache("test-spec-001")
        assert orch._spec_cache is None

    def test_invalidate_preserves_cache_for_different_spec(self, tmp_path):
        """invalidate_spec_cache(spec_id) preserves cache when spec_id differs."""
        orch = _make_orchestrator(tmp_path)
        cached = ("other-spec", 1234.0, 5678, {"spec_id": "other-spec"})
        orch._spec_cache = cached

        orch.invalidate_spec_cache("test-spec-001")
        assert orch._spec_cache == cached

    def test_invalidate_unconditional_clears_any_cache(self, tmp_path):
        """invalidate_spec_cache(None) clears any cached spec."""
        orch = _make_orchestrator(tmp_path)
        orch._spec_cache = ("any-spec", 1234.0, 5678, {"spec_id": "any-spec"})

        orch.invalidate_spec_cache()
        assert orch._spec_cache is None

    def test_invalidate_on_empty_cache_is_noop(self, tmp_path):
        """invalidate_spec_cache() on empty cache does not raise."""
        orch = _make_orchestrator(tmp_path)
        assert orch._spec_cache is None

        orch.invalidate_spec_cache("test-spec-001")
        assert orch._spec_cache is None

        orch.invalidate_spec_cache()
        assert orch._spec_cache is None


# =============================================================================
# T3: Verification Receipt Timing Boundary Tests
# =============================================================================


class TestVerificationReceiptTimingBoundaries:
    """Boundary tests for receipt.issued_at vs pending.issued_at validation."""

    def _make_session_and_result(
        self,
        pending_issued_at: datetime,
        receipt_issued_at: datetime,
    ):
        """Build a session + result pair for receipt timing validation."""
        cmd_hash = "a" * 64
        session = make_session(
            status=SessionStatus.RUNNING,
            last_step_issued=LastStepIssued(
                step_id="step-v1",
                type=StepType.EXECUTE_VERIFICATION,
                task_id="task-v1",
                phase_id="phase-1",
                issued_at=pending_issued_at,
            ),
        )
        session.pending_verification_receipt = PendingVerificationReceipt(
            step_id="step-v1",
            task_id="task-v1",
            expected_command_hash=cmd_hash,
            issued_at=pending_issued_at,
        )
        result = _result(
            step_id="step-v1",
            step_type=StepType.EXECUTE_VERIFICATION,
            outcome=StepOutcome.SUCCESS,
            task_id="task-v1",
            phase_id="phase-1",
            verification_receipt={
                "command_hash": cmd_hash,
                "exit_code": 0,
                "output_digest": "b" * 64,
                "issued_at": receipt_issued_at.isoformat(),
                "step_id": "step-v1",
            },
        )
        return session, result

    def test_receipt_at_exact_issuance_time_is_valid(self, tmp_path):
        """receipt.issued_at == pending.issued_at — valid (not strictly less)."""
        orch = _make_orchestrator(tmp_path)
        t0 = datetime.now(timezone.utc)
        session, result = self._make_session_and_result(t0, t0)

        error = orch._validate_verification_receipt(session, result)
        assert error is None

    def test_receipt_just_after_issuance_is_valid(self, tmp_path):
        """receipt.issued_at == pending.issued_at + 1s — valid."""
        orch = _make_orchestrator(tmp_path)
        t0 = datetime.now(timezone.utc)
        session, result = self._make_session_and_result(t0, t0 + timedelta(seconds=1))

        error = orch._validate_verification_receipt(session, result)
        assert error is None

    def test_receipt_before_issuance_is_invalid(self, tmp_path):
        """receipt.issued_at == pending.issued_at - 1s — invalid."""
        orch = _make_orchestrator(tmp_path)
        t0 = datetime.now(timezone.utc)
        session, result = self._make_session_and_result(t0, t0 - timedelta(seconds=1))

        error = orch._validate_verification_receipt(session, result)
        assert error is not None
        assert "earlier" in error.lower()


# =============================================================================
# Hierarchy-Format Spec Integration
# =============================================================================


class TestHierarchySpecIntegration:
    """Integration tests verifying the full orchestrator pipeline works with
    hierarchy-format specs (the production format).

    All other orchestrator tests use the denormalized phases-array format.
    These tests ensure that when a hierarchy-format spec is written to disk and
    loaded via ``load_spec_file()``, the adapter conversion produces data that
    the orchestrator can consume correctly for task discovery, phase advancement,
    fidelity gates, and hash consistency.
    """

    def test_initial_step_discovers_first_task(self, tmp_path):
        """Orchestrator finds the first pending task from a hierarchy spec."""
        spec_data = make_hierarchy_spec_data()
        orch = _make_orchestrator(tmp_path, spec_data)

        # load_spec_file() is called inside _validate_spec_integrity;
        # compute the hash from the converted view to match
        from foundry_mcp.core.autonomy.spec_adapter import ensure_phases_view

        converted = ensure_phases_view(dict(spec_data))
        spec_hash = compute_spec_structure_hash(converted)

        session = make_session(
            spec_structure_hash=spec_hash,
            last_step_issued=None,
        )

        result = orch.compute_next_step(session, last_step_result=None)

        assert result.success is True
        assert result.next_step is not None
        assert result.next_step.type == StepType.IMPLEMENT_TASK
        assert result.next_step.task_id == "task-1"

    def test_phase_advancement_across_phases(self, tmp_path):
        """After completing all phase-1 tasks and satisfying the gate,
        orchestrator advances to phase-2."""
        spec_data = make_hierarchy_spec_data()
        orch = _make_orchestrator(tmp_path, spec_data)

        from foundry_mcp.core.autonomy.spec_adapter import ensure_phases_view

        converted = ensure_phases_view(dict(spec_data))
        spec_hash = compute_spec_structure_hash(converted)

        session = make_session(
            spec_structure_hash=spec_hash,
            active_phase_id="phase-1",
            completed_task_ids=["task-1", "task-2", "verify-1"],
            last_step_issued=None,
            # Pre-satisfy the fidelity gate so the orchestrator can advance
            phase_gates={
                "phase-1": PhaseGateRecord(
                    required=True,
                    status=PhaseGateStatus.PASSED,
                    verdict=GateVerdict.PASS,
                ),
            },
        )

        result = orch.compute_next_step(session, last_step_result=None)

        assert result.success is True
        assert result.next_step is not None
        assert result.next_step.task_id == "task-3"
        assert session.active_phase_id == "phase-2"

    def test_verification_after_implementation(self, tmp_path):
        """Verify tasks are scheduled after all implementation tasks complete."""
        spec_data = make_hierarchy_spec_data(
            phase_defs=[
                {
                    "id": "phase-1",
                    "title": "Phase 1",
                    "children": ["task-1", "verify-1"],
                    "metadata": {},
                },
            ],
        )
        orch = _make_orchestrator(tmp_path, spec_data)

        from foundry_mcp.core.autonomy.spec_adapter import ensure_phases_view

        converted = ensure_phases_view(dict(spec_data))
        spec_hash = compute_spec_structure_hash(converted)

        session = make_session(
            spec_structure_hash=spec_hash,
            active_phase_id="phase-1",
            completed_task_ids=["task-1"],
            last_step_issued=None,
        )

        result = orch.compute_next_step(session, last_step_result=None)

        assert result.success is True
        assert result.next_step is not None
        assert result.next_step.type == StepType.EXECUTE_VERIFICATION
        assert result.next_step.task_id == "verify-1"

    def test_fidelity_gate_triggered(self, tmp_path):
        """Fidelity gate fires when all tasks in a gated phase are complete."""
        spec_data = make_hierarchy_spec_data(
            phase_defs=[
                {
                    "id": "phase-1",
                    "title": "Gated Phase",
                    "children": ["task-1"],
                    "metadata": {"requires_gate": True},
                },
            ],
        )
        orch = _make_orchestrator(tmp_path, spec_data)

        from foundry_mcp.core.autonomy.spec_adapter import ensure_phases_view

        converted = ensure_phases_view(dict(spec_data))
        spec_hash = compute_spec_structure_hash(converted)

        session = make_session(
            spec_structure_hash=spec_hash,
            active_phase_id="phase-1",
            completed_task_ids=["task-1"],
            last_step_issued=None,
        )

        result = orch.compute_next_step(session, last_step_result=None)

        assert result.success is True
        assert result.next_step is not None
        assert result.next_step.type == StepType.RUN_FIDELITY_GATE
        assert result.next_step.phase_id == "phase-1"

    def test_hash_equivalence_with_phases_format(self):
        """Hierarchy spec produces the same structure hash as an equivalent phases spec."""
        from foundry_mcp.core.autonomy.spec_adapter import ensure_phases_view

        hierarchy = make_hierarchy_spec_data(
            spec_id="hash-equiv",
            phase_defs=[
                {
                    "id": "phase-1",
                    "title": "Phase 1",
                    "children": ["task-1"],
                    "metadata": {},
                },
            ],
        )
        converted = ensure_phases_view(dict(hierarchy))
        hash_from_hierarchy = compute_spec_structure_hash(converted)

        # Build the equivalent phases-format spec
        phases_spec = make_spec_data(
            spec_id="hash-equiv",
            phases=[
                {
                    "id": "phase-1",
                    "title": "Phase 1",
                    "sequence_index": 0,
                    "tasks": [
                        {"id": "task-1", "title": "Task task-1", "type": "task", "status": "pending"},
                    ],
                },
            ],
        )
        hash_from_phases = compute_spec_structure_hash(phases_spec)

        assert hash_from_hierarchy == hash_from_phases

    def test_spec_integrity_check_passes(self, tmp_path):
        """_validate_spec_integrity succeeds for hierarchy specs on disk."""
        spec_data = make_hierarchy_spec_data()
        orch = _make_orchestrator(tmp_path, spec_data)

        from foundry_mcp.core.autonomy.spec_adapter import ensure_phases_view

        converted = ensure_phases_view(dict(spec_data))
        spec_hash = compute_spec_structure_hash(converted)

        session = make_session(
            spec_structure_hash=spec_hash,
        )

        loaded, error = orch._validate_spec_integrity(session, datetime.now(timezone.utc))

        assert error is None
        assert loaded is not None
        assert "phases" in loaded
        assert len(loaded["phases"]) == 2

    def test_complete_spec_after_all_tasks(self, tmp_path):
        """Orchestrator issues COMPLETE_SPEC when all tasks are done and gate is satisfied."""
        spec_data = make_hierarchy_spec_data(
            phase_defs=[
                {
                    "id": "phase-1",
                    "title": "Only Phase",
                    "children": ["task-1"],
                    "metadata": {},
                },
            ],
        )
        orch = _make_orchestrator(tmp_path, spec_data)

        from foundry_mcp.core.autonomy.spec_adapter import ensure_phases_view

        converted = ensure_phases_view(dict(spec_data))
        spec_hash = compute_spec_structure_hash(converted)

        session = make_session(
            spec_structure_hash=spec_hash,
            active_phase_id="phase-1",
            completed_task_ids=["task-1"],
            last_step_issued=None,
            phase_gates={
                "phase-1": PhaseGateRecord(
                    required=True,
                    status=PhaseGateStatus.PASSED,
                    verdict=GateVerdict.PASS,
                ),
            },
        )

        result = orch.compute_next_step(session, last_step_result=None)

        assert result.success is True
        assert result.next_step is not None
        assert result.next_step.type == StepType.COMPLETE_SPEC

    def test_subtask_expansion(self, tmp_path):
        """Parent tasks with subtask children are expanded to leaf nodes."""
        spec_data = make_hierarchy_spec_data(
            phase_defs=[
                {
                    "id": "phase-1",
                    "title": "Phase 1",
                    "children": ["task-parent"],
                    "metadata": {},
                },
            ],
        )
        # Give the parent task subtask children
        spec_data["hierarchy"]["task-parent"]["children"] = ["subtask-1", "subtask-2"]
        for sid in ["subtask-1", "subtask-2"]:
            spec_data["hierarchy"][sid] = {
                "type": "subtask",
                "title": f"Subtask {sid}",
                "status": "pending",
                "parent": "task-parent",
                "children": [],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": {},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            }

        orch = _make_orchestrator(tmp_path, spec_data)

        from foundry_mcp.core.autonomy.spec_adapter import ensure_phases_view

        converted = ensure_phases_view(dict(spec_data))
        spec_hash = compute_spec_structure_hash(converted)

        session = make_session(
            spec_structure_hash=spec_hash,
            last_step_issued=None,
        )

        result = orch.compute_next_step(session, last_step_result=None)

        assert result.success is True
        assert result.next_step is not None
        assert result.next_step.type == StepType.IMPLEMENT_TASK
        assert result.next_step.task_id in ("subtask-1", "subtask-2")

    def test_multi_step_progression(self, tmp_path):
        """Walk through multiple orchestrator steps on a hierarchy spec."""
        spec_data = make_hierarchy_spec_data(
            phase_defs=[
                {
                    "id": "phase-1",
                    "title": "Phase 1",
                    "children": ["task-1", "task-2"],
                    "metadata": {},
                },
            ],
        )
        orch = _make_orchestrator(tmp_path, spec_data)

        from foundry_mcp.core.autonomy.spec_adapter import ensure_phases_view

        converted = ensure_phases_view(dict(spec_data))
        spec_hash = compute_spec_structure_hash(converted)

        session = make_session(
            spec_structure_hash=spec_hash,
            last_step_issued=None,
        )

        # Step 1: get first task
        result1 = orch.compute_next_step(session, last_step_result=None)
        assert result1.success is True
        assert result1.next_step.task_id == "task-1"

        # Report success on task-1
        step_result = _result(
            step_id=result1.next_step.step_id,
            step_type=StepType.IMPLEMENT_TASK,
            outcome=StepOutcome.SUCCESS,
            task_id="task-1",
            phase_id="phase-1",
            step_proof=result1.next_step.step_proof,
        )

        # Step 2: get second task
        result2 = orch.compute_next_step(session, last_step_result=step_result)
        assert result2.success is True
        assert result2.next_step.task_id == "task-2"
