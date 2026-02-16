"""Tests for session-step handlers: next, report, replay.

Covers:
- session-step-next: initial call (no feedback), with feedback, missing step_type,
  invalid outcome, invalid step_type, orchestrator error mapping, replay response,
  persist failure handling, heartbeat flag
- session-step-report: required field validation (spec_id, step_id, outcome),
  delegation to step-next
- session-step-replay: returns cached response, no cache returns error,
  session not found
- _map_orchestrator_error_to_response: all error code mappings, remediation hints
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.autonomy.models import (
    LastStepIssued,
    NextStep,
    PauseReason,
    SessionStatus,
    StepOutcome,
    StepType,
)
from foundry_mcp.core.autonomy.orchestrator import (
    ERROR_ALL_TASKS_BLOCKED,
    ERROR_GATE_AUDIT_FAILURE,
    ERROR_INVALID_GATE_EVIDENCE,
    ERROR_GATE_INTEGRITY_CHECKSUM,
    ERROR_HEARTBEAT_STALE,
    ERROR_REQUIRED_GATE_UNSATISFIED,
    ERROR_SESSION_UNRECOVERABLE,
    ERROR_SPEC_REBASE_REQUIRED,
    ERROR_STEP_MISMATCH,
    ERROR_STEP_PROOF_MISSING,
    ERROR_STEP_PROOF_MISMATCH,
    ERROR_STEP_PROOF_CONFLICT,
    ERROR_STEP_PROOF_EXPIRED,
    ERROR_STEP_RESULT_REQUIRED,
    ERROR_STEP_STALE,
    ERROR_VERIFICATION_RECEIPT_INVALID,
    ERROR_VERIFICATION_RECEIPT_MISSING,
    OrchestrationResult,
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


def _create_session_for_step_tests(tmp_path: Path) -> tuple:
    """Create a workspace + running session, return (workspace, config, session_id)."""
    from foundry_mcp.tools.unified.task_handlers.handlers_session import (
        _handle_session_start,
    )

    workspace = _setup_workspace(tmp_path)
    config = _make_config(workspace)

    resp = _handle_session_start(
        config=config, spec_id="test-spec-001", workspace=str(workspace),
    )
    data = _assert_success(resp)
    return workspace, config, data["session_id"]


# =============================================================================
# _map_orchestrator_error_to_response Tests
# =============================================================================


class TestMapOrchestratorError:
    """Tests for _map_orchestrator_error_to_response."""

    def test_step_result_required_mapping(self):
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _map_orchestrator_error_to_response,
        )

        resp = _map_orchestrator_error_to_response(
            error_code=ERROR_STEP_RESULT_REQUIRED,
            error_message="Step result required",
            request_id="test-req",
            session_id="sess-1",
            state_version=5,
        )
        assert resp["success"] is False
        details = resp["data"].get("details", resp["data"])
        assert details["error_code"] == ERROR_STEP_RESULT_REQUIRED
        assert "remediation" in details
        assert details["state_version"] == 5

    def test_step_mismatch_mapping(self):
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _map_orchestrator_error_to_response,
        )

        resp = _map_orchestrator_error_to_response(
            error_code=ERROR_STEP_MISMATCH,
            error_message="Step mismatch",
            request_id="test-req",
        )
        assert resp["success"] is False

    @pytest.mark.parametrize(
        "error_code",
        [
            ERROR_STEP_PROOF_MISSING,
            ERROR_STEP_PROOF_MISMATCH,
            ERROR_STEP_PROOF_CONFLICT,
            ERROR_STEP_PROOF_EXPIRED,
        ],
    )
    def test_step_proof_error_mappings(self, error_code: str):
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _map_orchestrator_error_to_response,
        )

        resp = _map_orchestrator_error_to_response(
            error_code=error_code,
            error_message="step proof issue",
            request_id="test-req",
        )
        assert resp["success"] is False
        assert resp["data"]["loop_signal"] == "blocked_runtime"
        details = resp["data"].get("details", {})
        assert "remediation" in details

    def test_spec_rebase_required_mapping(self):
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _map_orchestrator_error_to_response,
        )

        resp = _map_orchestrator_error_to_response(
            error_code=ERROR_SPEC_REBASE_REQUIRED,
            error_message="Spec rebase required",
            request_id="test-req",
        )
        assert resp["success"] is False
        details = resp["data"].get("details", resp["data"])
        assert "rebase" in details["remediation"].lower()

    def test_heartbeat_stale_mapping(self):
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _map_orchestrator_error_to_response,
        )

        resp = _map_orchestrator_error_to_response(
            error_code=ERROR_HEARTBEAT_STALE,
            error_message="Heartbeat stale",
            request_id="test-req",
        )
        assert resp["success"] is False

    def test_step_stale_mapping(self):
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _map_orchestrator_error_to_response,
        )

        resp = _map_orchestrator_error_to_response(
            error_code=ERROR_STEP_STALE,
            error_message="Step stale",
            request_id="test-req",
        )
        assert resp["success"] is False

    def test_all_tasks_blocked_mapping(self):
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _map_orchestrator_error_to_response,
        )

        resp = _map_orchestrator_error_to_response(
            error_code=ERROR_ALL_TASKS_BLOCKED,
            error_message="All tasks blocked",
            request_id="test-req",
        )
        assert resp["success"] is False

    def test_session_unrecoverable_mapping(self):
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _map_orchestrator_error_to_response,
        )

        resp = _map_orchestrator_error_to_response(
            error_code=ERROR_SESSION_UNRECOVERABLE,
            error_message="Unrecoverable",
            request_id="test-req",
        )
        assert resp["success"] is False

    def test_verification_receipt_missing_mapping(self):
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _map_orchestrator_error_to_response,
        )

        resp = _map_orchestrator_error_to_response(
            error_code=ERROR_VERIFICATION_RECEIPT_MISSING,
            error_message="Receipt missing",
            request_id="test-req",
        )
        assert resp["success"] is False
        assert resp["data"]["error_type"] == "validation"
        details = resp["data"].get("details", {})
        assert "verification_receipt" in details.get("remediation", "")

    def test_verification_receipt_invalid_mapping(self):
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _map_orchestrator_error_to_response,
        )

        resp = _map_orchestrator_error_to_response(
            error_code=ERROR_VERIFICATION_RECEIPT_INVALID,
            error_message="Receipt invalid",
            request_id="test-req",
        )
        assert resp["success"] is False
        assert resp["data"]["error_type"] == "validation"

    def test_gate_integrity_checksum_mapping(self):
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _map_orchestrator_error_to_response,
        )

        resp = _map_orchestrator_error_to_response(
            error_code=ERROR_GATE_INTEGRITY_CHECKSUM,
            error_message="Gate integrity checksum mismatch",
            request_id="test-req",
        )
        assert resp["success"] is False
        assert resp["data"]["error_type"] == "validation"

    def test_gate_audit_failure_mapping(self):
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _map_orchestrator_error_to_response,
        )

        resp = _map_orchestrator_error_to_response(
            error_code=ERROR_GATE_AUDIT_FAILURE,
            error_message="Gate audit failed",
            request_id="test-req",
        )
        assert resp["success"] is False
        assert resp["data"]["error_type"] == "conflict"
        details = resp["data"].get("details", {})
        assert "audit" in details.get("remediation", "").lower()

    def test_unknown_error_code_maps_to_internal(self):
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _map_orchestrator_error_to_response,
        )

        resp = _map_orchestrator_error_to_response(
            error_code="UNKNOWN_ERROR",
            error_message="Something unexpected",
            request_id="test-req",
        )
        assert resp["success"] is False

    def test_none_error_code_maps_to_internal(self):
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _map_orchestrator_error_to_response,
        )

        resp = _map_orchestrator_error_to_response(
            error_code=None,
            error_message="No error code",
            request_id="test-req",
        )
        assert resp["success"] is False

    def test_required_gate_unsatisfied_maps_to_blocked_runtime_loop_signal(self):
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _map_orchestrator_error_to_response,
        )

        resp = _map_orchestrator_error_to_response(
            error_code=ERROR_REQUIRED_GATE_UNSATISFIED,
            error_message="Required gate unsatisfied",
            request_id="test-req",
        )
        assert resp["success"] is False
        assert resp["data"]["loop_signal"] == "blocked_runtime"
        assert resp["data"]["recommended_actions"]


class TestLoopSignalMapping:
    """Contract tests for loop_signal mapping rows in WS3."""

    @staticmethod
    def _attach(response: dict) -> dict:
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _attach_loop_fields,
        )

        return _attach_loop_fields(response)

    def test_phase_complete_pause_maps_to_phase_complete(self):
        resp = self._attach(
            {
                "success": True,
                "data": {
                    "status": SessionStatus.PAUSED.value,
                    "pause_reason": PauseReason.PHASE_COMPLETE.value,
                },
                "error": None,
                "meta": {"version": "response-v2"},
            }
        )
        assert resp["data"]["loop_signal"] == "phase_complete"
        assert "recommended_actions" not in resp["data"]

    def test_completed_status_maps_to_spec_complete(self):
        resp = self._attach(
            {
                "success": True,
                "data": {
                    "status": SessionStatus.COMPLETED.value,
                    "pause_reason": None,
                },
                "error": None,
                "meta": {"version": "response-v2"},
            }
        )
        assert resp["data"]["loop_signal"] == "spec_complete"

    def test_spec_complete_pause_reason_maps_to_spec_complete(self):
        resp = self._attach(
            {
                "success": True,
                "data": {
                    "status": SessionStatus.PAUSED.value,
                    "pause_reason": "spec_complete",
                },
                "error": None,
                "meta": {"version": "response-v2"},
            }
        )
        assert resp["data"]["loop_signal"] == "spec_complete"

    @pytest.mark.parametrize(
        "pause_reason",
        [
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
        ],
    )
    def test_attention_pause_reasons_map_to_paused_needs_attention(self, pause_reason: str):
        resp = self._attach(
            {
                "success": True,
                "data": {
                    "status": SessionStatus.PAUSED.value,
                    "pause_reason": pause_reason,
                },
                "error": None,
                "meta": {"version": "response-v2"},
            }
        )
        assert resp["data"]["loop_signal"] == "paused_needs_attention"
        assert resp["data"]["recommended_actions"]

    def test_failed_status_maps_to_failed_signal(self):
        resp = self._attach(
            {
                "success": True,
                "data": {
                    "status": SessionStatus.FAILED.value,
                    "pause_reason": None,
                },
                "error": None,
                "meta": {"version": "response-v2"},
            }
        )
        assert resp["data"]["loop_signal"] == "failed"
        assert resp["data"]["recommended_actions"]

    @pytest.mark.parametrize(
        "error_code",
        [
            ERROR_REQUIRED_GATE_UNSATISFIED,
            ERROR_GATE_AUDIT_FAILURE,
            ERROR_GATE_INTEGRITY_CHECKSUM,
            ERROR_STEP_PROOF_MISSING,
            ERROR_STEP_PROOF_CONFLICT,
            "FEATURE_DISABLED",
            "AUTHORIZATION",
        ],
    )
    def test_blocked_runtime_error_codes_map_to_blocked_runtime(self, error_code: str):
        resp = self._attach(
            {
                "success": False,
                "data": {
                    "error_code": error_code,
                    "error_type": "authorization",
                },
                "error": "blocked",
                "meta": {"version": "response-v2"},
            }
        )
        assert resp["data"]["loop_signal"] == "blocked_runtime"
        assert resp["data"]["recommended_actions"]

    def test_repeated_invalid_gate_evidence_maps_to_blocked_runtime(self):
        resp = self._attach(
            {
                "success": False,
                "data": {
                    "error_code": ERROR_INVALID_GATE_EVIDENCE,
                    "details": {"invalid_gate_evidence_attempts": 3},
                },
                "error": "invalid gate evidence",
                "meta": {"version": "response-v2"},
            }
        )
        assert resp["data"]["loop_signal"] == "blocked_runtime"


# =============================================================================
# Session Step Next Tests
# =============================================================================


class TestSessionStepNext:
    """Tests for _handle_session_step_next."""

    def test_next_session_not_found(self, tmp_path):
        """Step-next with no session returns error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_step_next(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        assert resp["success"] is False

    def test_next_initial_call_delegates_to_orchestrator(self, tmp_path):
        """Initial step-next (no feedback) calls orchestrator.compute_next_step."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
        )

        workspace, config, session_id = _create_session_for_step_tests(tmp_path)

        # Mock the orchestrator to return a success result
        mock_next_step = NextStep(
            step_id="step-001",
            type=StepType.IMPLEMENT_TASK,
            task_id="task-1",
            phase_id="phase-1",
            task_title="Implement task 1",
        )
        mock_result = OrchestrationResult(
            success=True,
            session=make_session(session_id=session_id),
            next_step=mock_next_step,
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
        assert data["session_id"] == session_id
        assert data["next_step"] is not None
        assert data["next_step"]["step_id"] == "step-001"

    def test_next_with_feedback(self, tmp_path):
        """Step-next with last_step_result passes parsed feedback to orchestrator."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
        )

        workspace, config, session_id = _create_session_for_step_tests(tmp_path)

        mock_result = OrchestrationResult(
            success=True,
            session=make_session(session_id=session_id),
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
                last_step_result={
                    "step_id": "step-001",
                    "step_type": "implement_task",
                    "outcome": "success",
                    "task_id": "task-1",
                    "phase_id": "phase-1",
                },
            )

        data = _assert_success(resp)
        # Verify orchestrator was called with parsed LastStepResult
        call_args = MockOrch.return_value.compute_next_step.call_args
        assert call_args.kwargs["last_step_result"] is not None
        assert call_args.kwargs["last_step_result"].step_id == "step-001"
        assert call_args.kwargs["last_step_result"].outcome == StepOutcome.SUCCESS

    def test_next_passes_step_proof_and_verification_receipt(self, tmp_path):
        """step_proof and verification_receipt are parsed into LastStepResult."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
        )

        workspace, config, session_id = _create_session_for_step_tests(tmp_path)

        mock_result = OrchestrationResult(
            success=True,
            session=make_session(session_id=session_id),
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
                last_step_result={
                    "step_id": "step-verify-001",
                    "step_type": "execute_verification",
                    "outcome": "success",
                    "task_id": "verify-1",
                    "phase_id": "phase-1",
                    "step_proof": "proof-abc-123",
                    "verification_receipt": {
                        "command_hash": "a" * 64,
                        "exit_code": 0,
                        "output_digest": "b" * 64,
                        "issued_at": datetime.now(timezone.utc).isoformat(),
                        "step_id": "step-verify-001",
                    },
                },
            )

        _assert_success(resp)
        call_args = MockOrch.return_value.compute_next_step.call_args
        parsed = call_args.kwargs["last_step_result"]
        assert parsed.step_proof == "proof-abc-123"
        assert parsed.verification_receipt is not None
        assert parsed.verification_receipt.command_hash == "a" * 64
        assert parsed.verification_receipt.output_digest == "b" * 64

    def test_next_step_proof_idempotent_replay_returns_cached_response(self, tmp_path):
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
        )

        workspace, config, session_id = _create_session_for_step_tests(tmp_path)
        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        assert session is not None
        session.last_step_issued = LastStepIssued(
            step_id="step-proof-replay-1",
            type=StepType.IMPLEMENT_TASK,
            task_id="task-1",
            phase_id="phase-1",
            issued_at=datetime.now(timezone.utc),
            step_proof="proof-replay-1",
        )
        storage.save(session)

        mock_result = OrchestrationResult(
            success=True,
            session=make_session(session_id=session_id),
            next_step=None,
            should_persist=False,
        )
        payload = {
            "step_id": "step-proof-replay-1",
            "step_type": "implement_task",
            "outcome": "success",
            "task_id": "task-1",
            "phase_id": "phase-1",
            "step_proof": "proof-replay-1",
        }

        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session_step.StepOrchestrator"
        ) as MockOrch:
            MockOrch.return_value.compute_next_step.return_value = mock_result
            first = _handle_session_step_next(
                config=config,
                session_id=session_id,
                workspace=str(workspace),
                last_step_result=payload,
            )
        _assert_success(first)

        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session_step.StepOrchestrator"
        ) as MockOrch:
            replay = _handle_session_step_next(
                config=config,
                session_id=session_id,
                workspace=str(workspace),
                last_step_result=payload,
            )
            MockOrch.return_value.compute_next_step.assert_not_called()

        _assert_success(replay)
        assert replay["data"] == first["data"]

    def test_next_step_proof_conflict_returns_error(self, tmp_path):
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
        )

        workspace, config, session_id = _create_session_for_step_tests(tmp_path)
        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        assert session is not None
        session.last_step_issued = LastStepIssued(
            step_id="step-proof-conflict-1",
            type=StepType.IMPLEMENT_TASK,
            task_id="task-1",
            phase_id="phase-1",
            issued_at=datetime.now(timezone.utc),
            step_proof="proof-conflict-1",
        )
        storage.save(session)

        mock_result = OrchestrationResult(
            success=True,
            session=make_session(session_id=session_id),
            next_step=None,
            should_persist=False,
        )
        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session_step.StepOrchestrator"
        ) as MockOrch:
            MockOrch.return_value.compute_next_step.return_value = mock_result
            _handle_session_step_next(
                config=config,
                session_id=session_id,
                workspace=str(workspace),
                last_step_result={
                    "step_id": "step-proof-conflict-1",
                    "step_type": "implement_task",
                    "outcome": "success",
                    "task_id": "task-1",
                    "phase_id": "phase-1",
                    "step_proof": "proof-conflict-1",
                },
            )

        conflict = _handle_session_step_next(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
            last_step_result={
                "step_id": "step-proof-conflict-1",
                "step_type": "implement_task",
                "outcome": "success",
                "task_id": "task-1",
                "phase_id": "phase-1",
                "note": "different payload",
                "step_proof": "proof-conflict-1",
            },
        )
        assert conflict["success"] is False
        assert conflict["data"]["details"]["error_code"] == ERROR_STEP_PROOF_CONFLICT

    def test_next_step_proof_expired_returns_error(self, tmp_path):
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
        )

        workspace, config, session_id = _create_session_for_step_tests(tmp_path)
        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        assert session is not None
        session.last_step_issued = LastStepIssued(
            step_id="step-proof-expired-1",
            type=StepType.IMPLEMENT_TASK,
            task_id="task-1",
            phase_id="phase-1",
            issued_at=datetime.now(timezone.utc),
            step_proof="proof-expired-1",
        )
        storage.save(session)

        mock_result = OrchestrationResult(
            success=True,
            session=make_session(session_id=session_id),
            next_step=None,
            should_persist=False,
        )
        payload = {
            "step_id": "step-proof-expired-1",
            "step_type": "implement_task",
            "outcome": "success",
            "task_id": "task-1",
            "phase_id": "phase-1",
            "step_proof": "proof-expired-1",
        }
        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session_step.StepOrchestrator"
        ) as MockOrch:
            MockOrch.return_value.compute_next_step.return_value = mock_result
            _handle_session_step_next(
                config=config,
                session_id=session_id,
                workspace=str(workspace),
                last_step_result=payload,
            )

        record = storage.get_proof_record(
            session_id,
            "proof-expired-1",
            include_expired=True,
        )
        assert record is not None
        record.grace_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        storage.save_proof_record(session_id, record)

        expired = _handle_session_step_next(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
            last_step_result=payload,
        )
        assert expired["success"] is False
        assert expired["data"]["details"]["error_code"] == ERROR_STEP_PROOF_EXPIRED

    def test_next_missing_step_type_in_feedback(self, tmp_path):
        """last_step_result without step_type returns validation error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
        )

        workspace, config, session_id = _create_session_for_step_tests(tmp_path)

        resp = _handle_session_step_next(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
            last_step_result={
                "step_id": "step-001",
                "outcome": "success",
                # step_type missing
            },
        )
        assert resp["success"] is False

    def test_next_invalid_verification_receipt_shape_returns_validation_error(self, tmp_path):
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
        )

        workspace, config, _ = _create_session_for_step_tests(tmp_path)
        resp = _handle_session_step_next(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
            last_step_result={
                "step_id": "step-verify-shape-1",
                "step_type": "execute_verification",
                "outcome": "success",
                "task_id": "verify-1",
                "phase_id": "phase-1",
                "verification_receipt": {
                    "command_hash": "not-a-sha256",
                    "exit_code": 0,
                    "output_digest": "b" * 64,
                    "issued_at": datetime.now(timezone.utc).isoformat(),
                    "step_id": "step-verify-shape-1",
                },
            },
        )
        assert resp["success"] is False
        assert resp["data"]["error_code"] == "VALIDATION_ERROR"

    def test_next_invalid_outcome_in_feedback(self, tmp_path):
        """Invalid outcome value returns validation error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
        )

        workspace, config, session_id = _create_session_for_step_tests(tmp_path)

        resp = _handle_session_step_next(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
            last_step_result={
                "step_id": "step-001",
                "step_type": "implement_task",
                "outcome": "invalid_outcome",
            },
        )
        assert resp["success"] is False

    def test_next_invalid_step_type_in_feedback(self, tmp_path):
        """Invalid step_type value returns validation error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
        )

        workspace, config, session_id = _create_session_for_step_tests(tmp_path)

        resp = _handle_session_step_next(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
            last_step_result={
                "step_id": "step-001",
                "step_type": "nonexistent_type",
                "outcome": "success",
            },
        )
        assert resp["success"] is False

    def test_next_replay_returns_cached(self, tmp_path):
        """When orchestrator returns replay_response, it is wrapped in success_response envelope."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
        )

        workspace, config, session_id = _create_session_for_step_tests(tmp_path)

        cached_response = {
            "session_id": session_id,
            "status": "running",
            "state_version": 2,
            "next_step": None,
        }
        mock_result = OrchestrationResult(
            success=True,
            session=make_session(session_id=session_id),
            replay_response=cached_response,
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

        # Replay response is wrapped in a success_response envelope (fix #1: full SessionStepResponseData caching)
        assert resp["success"] is True
        assert resp["data"] == cached_response
        assert resp["error"] is None
        assert resp["meta"]["version"] == "response-v2"

    def test_next_orchestrator_error_mapped(self, tmp_path):
        """Orchestrator error is mapped to appropriate response."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
        )

        workspace, config, session_id = _create_session_for_step_tests(tmp_path)

        mock_result = OrchestrationResult(
            success=False,
            session=make_session(session_id=session_id),
            error_code=ERROR_STEP_RESULT_REQUIRED,
            error_message="Must provide step result",
            should_persist=False,
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

        assert resp["success"] is False

    def test_next_persist_failure(self, tmp_path):
        """Persistence failure returns INTERNAL_ERROR."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
        )

        workspace, config, session_id = _create_session_for_step_tests(tmp_path)

        mock_result = OrchestrationResult(
            success=True,
            session=make_session(session_id=session_id),
            next_step=None,
            should_persist=True,
        )

        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session_step.StepOrchestrator"
        ) as MockOrch, patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session_step._get_storage"
        ) as MockGetStorage:
            mock_storage = MagicMock()
            mock_storage.load.return_value = make_session(session_id=session_id)
            # lookup_active_session not needed since we provide spec_id
            mock_storage.get_active_session.return_value = session_id
            mock_storage.save.side_effect = IOError("Disk full")
            MockGetStorage.return_value = mock_storage

            MockOrch.return_value.compute_next_step.return_value = mock_result

            resp = _handle_session_step_next(
                config=config,
                spec_id="test-spec-001",
                workspace=str(workspace),
            )

        assert resp["success"] is False

    def test_next_heartbeat_flag(self, tmp_path):
        """heartbeat=True passes heartbeat_at to orchestrator."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
        )

        workspace, config, session_id = _create_session_for_step_tests(tmp_path)

        mock_result = OrchestrationResult(
            success=True,
            session=make_session(session_id=session_id),
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
                heartbeat=True,
            )

        call_args = MockOrch.return_value.compute_next_step.call_args
        assert call_args.kwargs["heartbeat_at"] is not None


# =============================================================================
# Session Step Report Tests
# =============================================================================


class TestSessionStepReport:
    """Tests for _handle_session_step_report."""

    def test_report_missing_spec_id(self, tmp_path):
        """Report without spec_id returns validation error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_report,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_step_report(
            config=config,
            step_id="step-001",
            outcome="success",
            workspace=str(workspace),
        )
        assert resp["success"] is False

    def test_report_missing_step_id(self, tmp_path):
        """Report without step_id returns validation error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_report,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_step_report(
            config=config,
            spec_id="test-spec-001",
            outcome="success",
            workspace=str(workspace),
        )
        assert resp["success"] is False

    def test_report_missing_outcome(self, tmp_path):
        """Report without outcome returns validation error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_report,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_step_report(
            config=config,
            spec_id="test-spec-001",
            step_id="step-001",
            workspace=str(workspace),
        )
        assert resp["success"] is False

    def test_report_delegates_to_step_next(self, tmp_path):
        """Report delegates to _handle_session_step_next with constructed last_step_result."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_report,
        )

        workspace, config, session_id = _create_session_for_step_tests(tmp_path)

        # Patch _handle_session_step_next directly to verify delegation
        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session_step._handle_session_step_next"
        ) as mock_next:
            mock_next.return_value = {"success": True, "data": {}, "error": None, "meta": {"version": "response-v2"}}
            _handle_session_step_report(
                config=config,
                spec_id="test-spec-001",
                step_id="step-001",
                step_type="implement_task",
                outcome="success",
                note="all good",
                files_touched=["src/main.py"],
                workspace=str(workspace),
            )

        # Verify delegation happened with correct last_step_result
        mock_next.assert_called_once()
        call_kwargs = mock_next.call_args.kwargs
        assert call_kwargs["spec_id"] == "test-spec-001"
        assert call_kwargs["last_step_result"]["step_id"] == "step-001"
        assert call_kwargs["last_step_result"]["step_type"] == "implement_task"
        assert call_kwargs["last_step_result"]["outcome"] == "success"
        assert call_kwargs["last_step_result"]["note"] == "all good"
        assert call_kwargs["last_step_result"]["files_touched"] == ["src/main.py"]

    def test_report_accepts_session_id_without_spec_id(self, tmp_path):
        """session-step-report should resolve by session_id when spec_id is omitted."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_report,
        )

        workspace, config, session_id = _create_session_for_step_tests(tmp_path)

        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session_step._handle_session_step_next"
        ) as mock_next:
            mock_next.return_value = {"success": True, "data": {}, "error": None, "meta": {"version": "response-v2"}}
            resp = _handle_session_step_report(
                config=config,
                session_id=session_id,
                step_id="step-001",
                step_type="implement_task",
                outcome="success",
                workspace=str(workspace),
            )

        assert resp["success"] is True
        call_kwargs = mock_next.call_args.kwargs
        assert call_kwargs["session_id"] == session_id
        assert call_kwargs["spec_id"] is None


# =============================================================================
# Session Step Replay Tests
# =============================================================================


class TestSessionStepReplay:
    """Tests for _handle_session_step_replay."""

    def test_replay_no_session(self, tmp_path):
        """Replay with no session returns error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_replay,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_step_replay(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        assert resp["success"] is False

    def test_replay_no_cached_response(self, tmp_path):
        """Replay with no cached response returns NOT_FOUND."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_replay,
        )

        workspace, config, session_id = _create_session_for_step_tests(tmp_path)

        resp = _handle_session_step_replay(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        assert resp["success"] is False

    def test_replay_returns_cached_response(self, tmp_path):
        """Replay returns the cached response data inside a standard envelope."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_replay,
        )
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage

        workspace, config, session_id = _create_session_for_step_tests(tmp_path)

        # Set up a cached response
        cached = {
            "session_id": session_id,
            "status": "running",
            "state_version": 2,
            "next_step": None,
        }
        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        session.last_issued_response = cached
        storage.save(session)

        resp = _handle_session_step_replay(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        assert resp["success"] is True
        assert resp["error"] is None
        assert resp["meta"]["version"] == "response-v2"
        assert resp["data"] == cached

    def test_replay_by_session_id(self, tmp_path):
        """Replay works when resolved by session_id."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_replay,
        )
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage

        workspace, config, session_id = _create_session_for_step_tests(tmp_path)

        cached = {
            "session_id": session_id,
            "status": "running",
            "state_version": 3,
            "next_step": None,
        }
        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        session.last_issued_response = cached
        storage.save(session)

        resp = _handle_session_step_replay(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
        )
        assert resp["success"] is True
        assert resp["error"] is None
        assert resp["meta"]["version"] == "response-v2"
        assert resp["data"] == cached

    def test_replay_matches_step_next_envelope_shape(self, tmp_path):
        """Replay response uses the same envelope shape as session-step-next."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
            _handle_session_step_replay,
        )

        workspace, config, _session_id = _create_session_for_step_tests(tmp_path)

        next_resp = _handle_session_step_next(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        assert next_resp["success"] is True
        assert next_resp["error"] is None
        assert next_resp["meta"]["version"] == "response-v2"

        replay_resp = _handle_session_step_replay(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        assert replay_resp["success"] is True
        assert replay_resp["error"] is None
        assert replay_resp["meta"]["version"] == "response-v2"
        assert set(replay_resp.keys()) == set(next_resp.keys())
        assert replay_resp["data"] == next_resp["data"]


class TestSessionStepContractConformance:
    """Contract checks for session-step-next/session-step-replay envelopes."""

    def test_next_and_replay_success_paths_follow_response_v2(
        self,
        tmp_path,
        assert_response_contract,
    ):
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
            _handle_session_step_replay,
        )

        workspace, config, _session_id = _create_session_for_step_tests(tmp_path)

        next_resp = _handle_session_step_next(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        assert_response_contract(next_resp)

        replay_resp = _handle_session_step_replay(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        assert_response_contract(replay_resp)

    def test_next_error_path_follows_response_v2(self, tmp_path, assert_response_contract):
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_next,
        )

        workspace, config, _session_id = _create_session_for_step_tests(tmp_path)

        resp = _handle_session_step_next(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
            last_step_result={
                "step_id": "step-001",
                "outcome": "success",
                # step_type missing -> validation error path
            },
        )
        assert_response_contract(resp)
        assert resp["success"] is False

    def test_replay_error_path_follows_response_v2(self, tmp_path, assert_response_contract):
        from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
            _handle_session_step_replay,
        )

        workspace, config, _session_id = _create_session_for_step_tests(tmp_path)
        resp = _handle_session_step_replay(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        assert_response_contract(resp)
        assert resp["success"] is False
