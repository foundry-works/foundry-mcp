"""T13: Exercise untested error paths through the handler layer.

Tests:
- ERROR_ALL_TASKS_BLOCKED via orchestrator (all pending tasks have unmet deps)
- ERROR_STEP_PROOF_EXPIRED via time manipulation
- Step proof conflict (same proof, different payload)
- Repeated invalid gate evidence accumulation (3+ attempts triggers blocked_runtime)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from foundry_mcp.core.autonomy.models.enums import (
    PauseReason,
    SessionStatus,
    StepOutcome,
    StepType,
)
from foundry_mcp.core.autonomy.models.responses import NextStep
from foundry_mcp.core.autonomy.models.steps import (
    LastStepIssued,
    LastStepResult,
    StepInstruction,
)
from foundry_mcp.core.autonomy.orchestrator import (
    ERROR_ALL_TASKS_BLOCKED,
    ERROR_INVALID_GATE_EVIDENCE,
    ERROR_STEP_PROOF_CONFLICT,
    ERROR_STEP_PROOF_EXPIRED,
    OrchestrationResult,
    StepOrchestrator,
)
from .conftest import make_session, make_spec_data


# =============================================================================
# Helpers
# =============================================================================


def _make_config(workspace: Path) -> MagicMock:
    config = MagicMock()
    config.workspace_path = str(workspace)
    config.specs_dir = str(workspace / "specs")
    config.feature_flags = {"autonomy_sessions": True}
    return config


def _setup_workspace(tmp_path: Path, spec_id: str = "test-spec-001") -> Path:
    workspace = tmp_path / "ws"
    specs_dir = workspace / "specs" / "active"
    specs_dir.mkdir(parents=True)

    spec_data = make_spec_data(spec_id=spec_id)
    spec_data["title"] = "Test Spec"
    spec_data["journal"] = []
    spec_path = specs_dir / f"{spec_id}.json"
    spec_path.write_text(json.dumps(spec_data, indent=2))

    return workspace


def _assert_success(resp: dict) -> dict:
    assert resp["success"] is True, f"Expected success, got error: {resp.get('error')}"
    assert resp["error"] is None
    return resp["data"]


def _assert_error(resp: dict) -> dict:
    assert resp["success"] is False
    assert resp["error"] is not None
    return resp


# =============================================================================
# T13.1: ERROR_ALL_TASKS_BLOCKED through orchestrator
# =============================================================================


class TestAllTasksBlocked:
    """Test the all-tasks-blocked error path through the orchestrator."""

    def test_orchestrator_returns_all_blocked_when_all_pending_tasks_have_deps(self, tmp_path):
        """When all pending tasks have unmet dependencies, orchestrator pauses with BLOCKED."""
        from foundry_mcp.core.autonomy.memory import AutonomyStorage

        workspace = tmp_path / "ws"
        specs_dir = workspace / "specs" / "active"
        specs_dir.mkdir(parents=True)

        # Create spec where ALL pending tasks have unresolvable dependencies
        blocked_spec = {
            "spec_id": "blocked-spec",
            "phases": [
                {
                    "id": "phase-1",
                    "title": "Phase 1",
                    "sequence_index": 0,
                    "tasks": [
                        {
                            "id": "task-a",
                            "title": "Task A",
                            "type": "task",
                            "status": "pending",
                            "depends": ["task-b"],
                        },
                        {
                            "id": "task-b",
                            "title": "Task B",
                            "type": "task",
                            "status": "pending",
                            "depends": ["task-a"],
                        },
                    ],
                },
            ],
            "hierarchy": {
                "spec-root": {
                    "type": "spec",
                    "children": ["phase-1"],
                },
                "phase-1": {
                    "type": "phase",
                    "parent": "spec-root",
                    "children": ["task-a", "task-b"],
                },
                "task-a": {
                    "type": "task",
                    "status": "pending",
                    "parent": "phase-1",
                    "dependencies": {
                        "blocked_by": ["task-b"],
                    },
                },
                "task-b": {
                    "type": "task",
                    "status": "pending",
                    "parent": "phase-1",
                    "dependencies": {
                        "blocked_by": ["task-a"],
                    },
                },
            },
        }

        spec_path = specs_dir / "blocked-spec.json"
        spec_path.write_text(json.dumps(blocked_spec, indent=2))

        storage = AutonomyStorage(
            storage_path=tmp_path / "sessions",
            workspace_path=workspace,
        )

        session = make_session(
            session_id="blocked-session",
            spec_id="blocked-spec",
            active_phase_id="phase-1",
            spec_structure_hash="b" * 64,
        )
        storage.save(session)

        orchestrator = StepOrchestrator(
            storage=storage,
            spec_loader=None,
            workspace_path=workspace,
        )

        # Mock _validate_spec_integrity to return the blocked spec data (no error)
        with patch.object(
            orchestrator, "_validate_spec_integrity", return_value=(blocked_spec, None)
        ):
            result = orchestrator.compute_next_step(session)

        # The result should be a pause with BLOCKED reason
        assert result.session.status == SessionStatus.PAUSED
        assert result.session.pause_reason == PauseReason.BLOCKED

    def test_all_blocked_mapped_to_resource_busy_at_handler(self):
        """ERROR_ALL_TASKS_BLOCKED maps to RESOURCE_BUSY at handler layer."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _map_orchestrator_error_to_response,
        )

        resp = _map_orchestrator_error_to_response(
            error_code=ERROR_ALL_TASKS_BLOCKED,
            error_message="All tasks blocked",
            request_id="test-req",
            session_id="sess-1",
            state_version=5,
        )
        assert resp["success"] is False
        details = resp["data"].get("details", resp["data"])
        assert details["error_code"] == ERROR_ALL_TASKS_BLOCKED
        assert "remediation" in details

    def test_all_blocked_through_handler_step_next(self, tmp_path):
        """End-to-end: session-step-next returns paused status when all tasks blocked."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        # Start a session
        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        session_id = data["session_id"]

        # Mock orchestrator to return all-blocked pause
        mock_result = OrchestrationResult(
            success=True,
            session=make_session(
                session_id=session_id,
                spec_id="test-spec-001",
                status=SessionStatus.PAUSED,
                pause_reason=PauseReason.BLOCKED,
            ),
            next_step=None,
            should_persist=True,
        )

        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session_step.StepOrchestrator"
        ) as MockOrch:
            MockOrch.return_value.compute_next_step.return_value = mock_result
            resp = _handle_session_step_next(
                config=config,
                spec_id="test-spec-001",
                workspace=str(workspace),
            )

        data = _assert_success(resp)
        assert data["status"] == "paused"
        # SessionStepResponseData derives loop_signal from status
        assert data.get("loop_signal") == "paused_needs_attention"


# =============================================================================
# T13.2: ERROR_STEP_PROOF_EXPIRED via time manipulation
# =============================================================================


class TestStepProofExpired:
    """Test proof expiration through the handler layer with time manipulation."""

    def test_expired_proof_returns_error_through_handler(self, tmp_path):
        """A proof consumed beyond grace window returns STEP_PROOF_EXPIRED."""
        from foundry_mcp.core.autonomy.memory import AutonomyStorage
        from foundry_mcp.core.autonomy.models.steps import StepProofRecord

        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        # Start a session
        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        session_id = data["session_id"]

        # Issue a first step with a step_proof so the session has last_step_issued
        now = datetime.now(timezone.utc)
        mock_next = NextStep(
            step_id="step-001",
            type=StepType.IMPLEMENT_TASK,
            task_id="task-1",
            phase_id="phase-1",
            task_title="Task 1",
            instructions=[StepInstruction(
                tool="foundry",
                action="implement",
                description="Implement task 1",
            )],
            message="Implement task 1",
            step_proof="current-proof",
        )
        mock_result = OrchestrationResult(
            success=True,
            session=make_session(
                session_id=session_id,
                spec_id="test-spec-001",
                last_step_issued=LastStepIssued(
                    step_id="step-001",
                    type=StepType.IMPLEMENT_TASK,
                    task_id="task-1",
                    phase_id="phase-1",
                    issued_at=now,
                    step_proof="current-proof",
                ),
            ),
            next_step=mock_next,
            should_persist=True,
        )

        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session_step.StepOrchestrator"
        ) as MockOrch:
            MockOrch.return_value.compute_next_step.return_value = mock_result
            _handle_session_step_next(
                config=config,
                spec_id="test-spec-001",
                workspace=str(workspace),
            )

        # Now create an expired proof record directly in storage for a DIFFERENT proof.
        # The payload_hash must match what the handler computes from the last_step_result.
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _hash_last_step_result_payload,
        )

        storage_path = Path(config.workspace_path) / "specs" / ".autonomy" / "sessions"
        storage = AutonomyStorage(
            storage_path=storage_path,
            workspace_path=Path(config.workspace_path),
        )

        # Build the same LastStepResult the handler will parse from our dict
        parsed_result = LastStepResult(
            step_id="step-000",
            step_type=StepType.IMPLEMENT_TASK,
            task_id="task-1",
            outcome=StepOutcome.SUCCESS,
            step_proof="old-proof-token",
        )
        expected_hash = _hash_last_step_result_payload(parsed_result)

        expired_record = StepProofRecord(
            step_proof="old-proof-token",
            step_id="step-000",
            payload_hash=expected_hash,
            consumed_at=datetime.now(timezone.utc) - timedelta(minutes=10),
            grace_expires_at=datetime.now(timezone.utc) - timedelta(minutes=5),
            response_hash=None,
            cached_response=None,
        )
        storage.save_proof_record(session_id, expired_record)

        # Now try to use the expired proof — include task_id since implement_task requires it
        resp = _handle_session_step_next(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
            last_step_result={
                "step_id": "step-000",
                "step_type": "implement_task",
                "task_id": "task-1",
                "outcome": "success",
                "step_proof": "old-proof-token",
            },
        )

        err = _assert_error(resp)
        details = err["data"].get("details", err["data"])
        assert details["error_code"] == ERROR_STEP_PROOF_EXPIRED


# =============================================================================
# T13.3: Step Proof Conflict Detection
# =============================================================================


class TestStepProofConflict:
    """Test proof conflict detection (same proof, different payload)."""

    def test_conflict_detected_when_payload_differs(self, tmp_path):
        """Same proof token with different payload hash returns PROOF_CONFLICT."""
        from foundry_mcp.core.autonomy.memory import AutonomyStorage
        from foundry_mcp.core.autonomy.models.steps import StepProofRecord

        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        # Start session and issue a step with a step_proof
        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        session_id = data["session_id"]

        now = datetime.now(timezone.utc)
        mock_next = NextStep(
            step_id="step-001",
            type=StepType.IMPLEMENT_TASK,
            task_id="task-1",
            phase_id="phase-1",
            task_title="Task 1",
            instructions=[StepInstruction(
                tool="foundry",
                action="implement",
                description="Implement task 1",
            )],
            message="Implement task 1",
            step_proof="current-proof",
        )
        mock_result = OrchestrationResult(
            success=True,
            session=make_session(
                session_id=session_id,
                spec_id="test-spec-001",
                last_step_issued=LastStepIssued(
                    step_id="step-001",
                    type=StepType.IMPLEMENT_TASK,
                    task_id="task-1",
                    phase_id="phase-1",
                    issued_at=now,
                    step_proof="current-proof",
                ),
            ),
            next_step=mock_next,
            should_persist=True,
        )

        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session_step.StepOrchestrator"
        ) as MockOrch:
            MockOrch.return_value.compute_next_step.return_value = mock_result
            _handle_session_step_next(
                config=config,
                spec_id="test-spec-001",
                workspace=str(workspace),
            )

        # Create a consumed proof record for a different proof token with known hash
        storage_path = Path(config.workspace_path) / "specs" / ".autonomy" / "sessions"
        storage = AutonomyStorage(
            storage_path=storage_path,
            workspace_path=Path(config.workspace_path),
        )

        consumed_record = StepProofRecord(
            step_proof="conflict-proof-token",
            step_id="step-000",
            payload_hash="original-hash-abc123",
            consumed_at=datetime.now(timezone.utc),
            grace_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            response_hash=None,
            cached_response=None,
        )
        storage.save_proof_record(session_id, consumed_record)

        # Now try same proof with different payload (different outcome) — include task_id
        resp = _handle_session_step_next(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
            last_step_result={
                "step_id": "step-000",
                "step_type": "implement_task",
                "task_id": "task-1",
                "outcome": "failure",  # Different from original
                "step_proof": "conflict-proof-token",
            },
        )

        err = _assert_error(resp)
        details = err["data"].get("details", err["data"])
        assert details["error_code"] == ERROR_STEP_PROOF_CONFLICT

    def test_proof_conflict_through_storage_layer(self, tmp_path):
        """Direct storage-level proof conflict detection."""
        from foundry_mcp.core.autonomy.memory import AutonomyStorage

        storage = AutonomyStorage(
            storage_path=tmp_path / "sessions",
            workspace_path=tmp_path,
        )

        session_id = "conflict-test-session"
        proof = "conflict-proof"
        original_hash = hashlib.sha256(b"original-payload").hexdigest()
        different_hash = hashlib.sha256(b"different-payload").hexdigest()

        # First consumption succeeds
        success, record, err = storage.consume_proof_with_lock(
            session_id=session_id,
            step_proof=proof,
            payload_hash=original_hash,
            grace_window_seconds=60,
        )
        assert success is True
        assert record is None
        assert err == ""

        # Second consumption with different payload -> PROOF_CONFLICT
        success, record, err = storage.consume_proof_with_lock(
            session_id=session_id,
            step_proof=proof,
            payload_hash=different_hash,
            grace_window_seconds=60,
        )
        assert success is False
        assert err == "PROOF_CONFLICT"


# =============================================================================
# T13.4: Repeated Invalid Gate Evidence Accumulation
# =============================================================================


class TestRepeatedInvalidGateEvidence:
    """Test that 3+ invalid gate evidence attempts triggers blocked_runtime signal."""

    def test_invalid_gate_evidence_error_mapped_correctly(self):
        """INVALID_GATE_EVIDENCE maps to VALIDATION_ERROR at handler layer."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _map_orchestrator_error_to_response,
        )

        resp = _map_orchestrator_error_to_response(
            error_code=ERROR_INVALID_GATE_EVIDENCE,
            error_message="Gate evidence invalid",
            request_id="test-req",
        )
        assert resp["success"] is False
        details = resp["data"].get("details", resp["data"])
        assert details["error_code"] == ERROR_INVALID_GATE_EVIDENCE

    def test_repeated_invalid_gate_evidence_triggers_blocked_runtime(self):
        """When invalid_gate_evidence_attempts >= 3, loop_signal becomes blocked_runtime."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _attach_loop_fields,
        )

        # Build a response that mimics 3+ invalid gate evidence attempts
        response = {
            "success": False,
            "error": "Gate evidence invalid",
            "data": {
                "status": "running",
                "details": {
                    "error_code": "INVALID_GATE_EVIDENCE",
                    "invalid_gate_evidence_attempts": 3,
                },
            },
            "meta": {"version": "response-v2"},
        }

        result = _attach_loop_fields(response)
        assert result["data"]["loop_signal"] == "blocked_runtime"

    def test_fewer_than_three_attempts_does_not_trigger_blocked(self):
        """With < 3 invalid gate evidence attempts, loop_signal is NOT blocked_runtime."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _attach_loop_fields,
        )

        response = {
            "success": False,
            "error": "Gate evidence invalid",
            "data": {
                "status": "running",
                "details": {
                    "error_code": "INVALID_GATE_EVIDENCE",
                    "invalid_gate_evidence_attempts": 2,
                },
            },
            "meta": {"version": "response-v2"},
        }

        result = _attach_loop_fields(response)
        # With < 3 attempts, it should NOT be blocked_runtime
        loop_signal = result["data"].get("loop_signal")
        assert loop_signal != "blocked_runtime"

    def test_explicit_repeated_flag_triggers_blocked_runtime(self):
        """When repeated_invalid_gate_evidence flag is explicitly True, blocked_runtime."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _attach_loop_fields,
        )

        response = {
            "success": False,
            "error": "Gate evidence invalid",
            "data": {
                "status": "running",
                "details": {
                    "error_code": "INVALID_GATE_EVIDENCE",
                    "repeated_invalid_gate_evidence": True,
                },
            },
            "meta": {"version": "response-v2"},
        }

        result = _attach_loop_fields(response)
        assert result["data"]["loop_signal"] == "blocked_runtime"

    def test_gate_evidence_validation_through_orchestrator(self, tmp_path):
        """Test gate evidence validation error path through orchestrator directly."""
        from foundry_mcp.core.autonomy.memory import AutonomyStorage

        storage = AutonomyStorage(
            storage_path=tmp_path / "sessions",
            workspace_path=tmp_path,
        )

        workspace = tmp_path / "ws"
        specs_dir = workspace / "specs" / "active"
        specs_dir.mkdir(parents=True)

        spec_data = make_spec_data()
        spec_path = specs_dir / "test-spec-001.json"
        spec_path.write_text(json.dumps(spec_data, indent=2))

        orchestrator = StepOrchestrator(
            storage=storage,
            spec_loader=None,
            workspace_path=workspace,
        )

        # Create a session with last_step_issued as RUN_FIDELITY_GATE
        # but NO pending_gate_evidence → validation will fail
        now = datetime.now(timezone.utc)
        session = make_session(
            session_id="gate-test-session",
            spec_id="test-spec-001",
            active_phase_id="phase-1",
            last_step_issued=LastStepIssued(
                step_id="gate-step-001",
                type=StepType.RUN_FIDELITY_GATE,
                task_id=None,
                phase_id="phase-1",
                issued_at=now,
                step_proof="gate-proof",
            ),
        )

        result_feedback = LastStepResult(
            step_id="gate-step-001",
            step_type=StepType.RUN_FIDELITY_GATE,
            task_id=None,
            phase_id="phase-1",
            outcome=StepOutcome.SUCCESS,
            note=None,
            files_touched=None,
            gate_attempt_id="nonexistent-gate-attempt",
            step_proof="gate-proof",  # Must match last_step_issued.step_proof
            verification_receipt=None,
        )

        # Mock _validate_spec_integrity to skip spec loading
        with patch.object(
            orchestrator, "_validate_spec_integrity", return_value=(spec_data, None)
        ):
            result = orchestrator.compute_next_step(session, result_feedback)

        # Should be an error about invalid gate evidence
        assert result.success is False
        assert result.error_code in (
            ERROR_INVALID_GATE_EVIDENCE,
            "INVALID_GATE_EVIDENCE",
        )
        assert "gate" in (result.error_message or "").lower()


# =============================================================================
# T13.5: Step proof expiration at storage layer
# =============================================================================


class TestStepProofExpirationStorage:
    """Test proof expiration directly at the storage layer."""

    def test_consume_expired_proof_returns_proof_expired(self, tmp_path):
        """Consuming an already-consumed proof after grace window returns PROOF_EXPIRED."""
        import time

        from foundry_mcp.core.autonomy.memory import AutonomyStorage

        storage = AutonomyStorage(
            storage_path=tmp_path / "sessions",
            workspace_path=tmp_path,
        )

        session_id = "expiry-test"
        proof = "test-proof-token"
        payload_hash = hashlib.sha256(b"test-payload").hexdigest()

        # First: consume the proof with a very short grace window
        success, record, err = storage.consume_proof_with_lock(
            session_id=session_id,
            step_proof=proof,
            payload_hash=payload_hash,
            grace_window_seconds=1,  # 1 second grace
        )
        assert success is True
        assert err == ""

        # Wait for grace to expire
        time.sleep(1.5)

        # Now try same proof + same payload → should get PROOF_EXPIRED
        success, record, err = storage.consume_proof_with_lock(
            session_id=session_id,
            step_proof=proof,
            payload_hash=payload_hash,
            grace_window_seconds=1,
        )
        assert success is False
        assert err == "PROOF_EXPIRED"
        assert record is not None  # Returns the expired record
