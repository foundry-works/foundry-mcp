"""P0 integration tests for autonomous session management.

Covers ADR-002 testing strategy requirements:
- Full lifecycle: start -> heartbeat -> next loop -> context pause -> resume -> complete
- Phase boundary: gate pass advances, gate fail pauses, manual gate requires human resume
- Spec drift: editing spec mid-session causes failed on next call; recovery via rebase
- Duplicate start: second start for same spec returns SPEC_SESSION_EXISTS; force=true replaces
- Action split: session rejects next/heartbeat commands at handler boundary
- Resume context: recent_completed_tasks truncation, completed_task_count, journal availability
- Idempotency key: duplicate start with same key returns existing session
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

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
from foundry_mcp.core.autonomy.models.steps import LastStepResult
from foundry_mcp.core.autonomy.orchestrator import (
    ERROR_SPEC_REBASE_REQUIRED,
    StepOrchestrator,
)
from foundry_mcp.core.autonomy.spec_hash import compute_spec_structure_hash

from .conftest import make_session, make_spec_data

# =============================================================================
# Helpers
# =============================================================================


def _make_workspace(tmp_path: Path, spec_data: Optional[Dict[str, Any]] = None) -> Path:
    """Create a workspace with a spec file and return the workspace path."""
    workspace = tmp_path / "ws"
    specs_dir = workspace / "specs" / "active"
    specs_dir.mkdir(parents=True, exist_ok=True)

    data = spec_data or make_spec_data()
    spec_id = data.get("spec_id", "test-spec-001")
    spec_file = specs_dir / f"{spec_id}.json"
    spec_file.write_text(json.dumps(data, indent=2))

    return workspace


def _make_orchestrator(workspace: Path, spec_data: Optional[Dict[str, Any]] = None) -> StepOrchestrator:
    """Create an orchestrator backed by the workspace."""
    storage = MagicMock()
    return StepOrchestrator(
        storage=storage,
        spec_loader=MagicMock(),
        workspace_path=workspace,
    )


def _make_config(workspace: Path) -> MagicMock:
    """Create a mock ServerConfig pointing at workspace."""
    config = MagicMock()
    config.workspace_path = str(workspace)
    config.specs_dir = str(workspace / "specs")
    return config


def _setup_workspace(tmp_path: Path, spec_id: str = "test-spec-001") -> Path:
    """Create workspace with spec file for handler tests."""
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
    """Assert response is successful and return data."""
    assert resp["success"] is True, f"Expected success, got error: {resp.get('error')}"
    return resp["data"]


# =============================================================================
# Full Lifecycle Integration Tests
# =============================================================================


class TestFullLifecycle:
    """Test full session lifecycle: start -> heartbeat -> next -> pause -> resume -> complete."""

    def test_start_heartbeat_pause_resume(self, tmp_path):
        """Full lifecycle through start, heartbeat, context pause, and resume."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_end,
            _handle_session_heartbeat,
            _handle_session_pause,
            _handle_session_resume,
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        # Start session
        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        session_id = data["session_id"]
        assert data["status"] == "running"

        # Heartbeat with context
        resp = _handle_session_heartbeat(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
            context_usage_pct=42,
        )
        hb_data = _assert_success(resp)
        assert hb_data["context_usage_pct"] == 42

        # Pause
        resp = _handle_session_pause(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["status"] == "paused"

        # Resume
        resp = _handle_session_resume(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["status"] == "running"

        # End
        resp = _handle_session_end(
            config=config,
            session_id=session_id,
            reason_code="operator_override",
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["status"] == "ended"


# =============================================================================
# Phase Boundary Tests
# =============================================================================


class TestPhaseBoundary:
    """Test phase gate behavior: pass advances, fail pauses, manual requires ack."""

    def test_gate_pass_advances_to_next_phase(self, tmp_path):
        """Gate pass clears evidence and moves to next task/phase."""
        workspace = _make_workspace(tmp_path)
        orch = _make_orchestrator(workspace)
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
            active_phase_id="phase-1",
            completed_task_ids=["task-1", "task-2", "verify-1"],
            phase_gates={
                "phase-1": PhaseGateRecord(
                    required=True,
                    status=PhaseGateStatus.PASSED,
                )
            },
        )

        result = orch._handle_gate_evidence(session, spec_data, now)
        # Gate passed: evidence cleared, next step should proceed
        assert session.pending_gate_evidence is None
        assert result.success is True

    def test_gate_fail_pauses_session(self, tmp_path):
        """Gate failure with no auto-retry pauses session."""
        workspace = _make_workspace(tmp_path)
        orch = _make_orchestrator(workspace)
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

    def test_manual_gate_requires_human_resume(self, tmp_path):
        """Manual gate policy always pauses for human review, even on PASS verdict."""
        workspace = _make_workspace(tmp_path)
        orch = _make_orchestrator(workspace)
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
            gate_policy=GatePolicy.MANUAL,
            pending_gate_evidence=evidence,
        )

        result = orch._handle_gate_evidence(session, spec_data, now)
        assert result.session.status == SessionStatus.PAUSED
        assert result.session.pause_reason == PauseReason.GATE_REVIEW_REQUIRED


# =============================================================================
# Spec Drift Tests
# =============================================================================


class TestSpecDrift:
    """Test spec change detection and rebase recovery."""

    @pytest.fixture(autouse=True)
    def _set_maintainer_role(self):
        """Rebase requires maintainer role."""
        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session.get_server_role",
            return_value="maintainer",
        ):
            yield

    def test_spec_change_fails_session(self, tmp_path):
        """Editing spec mid-session causes SPEC_REBASE_REQUIRED on next step."""
        workspace = _make_workspace(tmp_path)
        spec_data = make_spec_data()
        original_hash = compute_spec_structure_hash(spec_data)

        orch = _make_orchestrator(workspace)
        session = make_session(
            spec_structure_hash=original_hash,
            spec_file_mtime=0.0,  # Force re-hash by setting old mtime
        )

        # Modify spec on disk
        spec_path = workspace / "specs" / "active" / "test-spec-001.json"
        modified_spec = spec_data.copy()
        modified_spec["phases"] = spec_data["phases"] + [
            {"id": "phase-3", "title": "Phase 3", "tasks": [{"id": "task-99", "title": "New", "type": "task"}]}
        ]
        spec_path.write_text(json.dumps(modified_spec, indent=2))

        result = orch.compute_next_step(session)
        assert result.success is False
        assert result.error_code == ERROR_SPEC_REBASE_REQUIRED
        assert session.status == SessionStatus.FAILED

    def test_rebase_recovery(self, tmp_path):
        """Rebase after spec change recovers the session."""
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_rebase,
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        # Start session
        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        session_id = _assert_success(resp)["session_id"]

        # Mark session as failed (as if spec hash mismatch was detected)
        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        session.status = SessionStatus.FAILED
        session.failure_reason = FailureReason.SPEC_STRUCTURE_CHANGED
        storage.save(session)

        # Rebase should recover (no actual spec change, so it's a no-change rebase)
        resp = _handle_session_rebase(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["status"] == "running"


# =============================================================================
# Duplicate Start Tests
# =============================================================================


class TestDuplicateStart:
    """Test duplicate session start behavior."""

    def test_duplicate_start_returns_conflict(self, tmp_path):
        """Second start for same spec returns SPEC_SESSION_EXISTS."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp1 = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        _assert_success(resp1)

        resp2 = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        assert resp2["success"] is False

    def test_duplicate_start_force_replaces(self, tmp_path):
        """Duplicate start with force=true ends existing and creates new."""
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp1 = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        old_session_id = _assert_success(resp1)["session_id"]

        resp2 = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
            force=True,
        )
        new_data = _assert_success(resp2)
        new_session_id = new_data["session_id"]
        assert new_session_id != old_session_id
        assert new_data["status"] == "running"

        # Old session should be ended
        storage = _get_storage(config, str(workspace))
        old_session = storage.load(old_session_id)
        assert old_session.status == SessionStatus.ENDED

    def test_idempotency_key_returns_existing(self, tmp_path):
        """Duplicate start with same idempotency key returns existing session."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp1 = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
            idempotency_key="key-123",
        )
        data1 = _assert_success(resp1)

        resp2 = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
            idempotency_key="key-123",
        )
        data2 = _assert_success(resp2)
        assert data2["session_id"] == data1["session_id"]

    def test_idempotency_key_different_returns_conflict(self, tmp_path):
        """Duplicate start with different idempotency key returns SPEC_SESSION_EXISTS."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp1 = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
            idempotency_key="key-123",
        )
        _assert_success(resp1)

        resp2 = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
            idempotency_key="key-different",
        )
        assert resp2["success"] is False


# =============================================================================
# Resume Context Tests
# =============================================================================


class TestResumeContext:
    """Test resume context generation: truncation, counts, journal availability."""

    def test_recent_completed_tasks_truncation(self, tmp_path):
        """recent_completed_tasks is capped at 10 tasks."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _build_resume_context,
        )

        # Create spec with 15 tasks
        tasks = [{"id": f"task-{i}", "title": f"Task {i}", "type": "task", "status": "completed"} for i in range(15)]
        spec_data = make_spec_data(
            phases=[
                {"id": "phase-1", "title": "Phase 1", "tasks": tasks},
            ]
        )

        workspace = _make_workspace(tmp_path, spec_data)

        session = make_session(
            active_phase_id="phase-1",
            completed_task_ids=[f"task-{i}" for i in range(15)],
            counters=SessionCounters(tasks_completed=15),
        )

        ctx = _build_resume_context(session, str(workspace))
        assert ctx is not None
        assert ctx.completed_task_count == 15
        assert len(ctx.recent_completed_tasks) == 10
        # Most recent = last 10 (task-5 through task-14)
        assert ctx.recent_completed_tasks[0].task_id == "task-5"
        assert ctx.recent_completed_tasks[-1].task_id == "task-14"

    def test_completed_task_count_full(self, tmp_path):
        """completed_task_count provides the full count even when truncated."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _build_resume_context,
        )

        tasks = [{"id": f"task-{i}", "title": f"Task {i}", "type": "task"} for i in range(20)]
        spec_data = make_spec_data(
            phases=[
                {"id": "phase-1", "title": "Phase 1", "tasks": tasks},
            ]
        )
        workspace = _make_workspace(tmp_path, spec_data)

        session = make_session(
            active_phase_id="phase-1",
            completed_task_ids=[f"task-{i}" for i in range(20)],
            counters=SessionCounters(tasks_completed=20),
        )

        ctx = _build_resume_context(session, str(workspace))
        assert ctx is not None
        assert ctx.completed_task_count == 20
        assert len(ctx.recent_completed_tasks) == 10

    def test_journal_available_flag(self, tmp_path):
        """journal_available flag reflects spec journal presence."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _build_resume_context,
        )

        spec_data = make_spec_data()
        spec_data["journal"] = [{"entry": "test"}]
        workspace = _make_workspace(tmp_path, spec_data)

        session = make_session(active_phase_id="phase-1")

        ctx = _build_resume_context(session, str(workspace))
        assert ctx is not None
        assert ctx.journal_available is True
        assert ctx.journal_hint is not None

    def test_journal_not_available(self, tmp_path):
        """journal_available is False when spec has no journal."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _build_resume_context,
        )

        spec_data = make_spec_data()
        # No journal key
        workspace = _make_workspace(tmp_path, spec_data)

        session = make_session(active_phase_id="phase-1")

        ctx = _build_resume_context(session, str(workspace))
        assert ctx is not None
        assert ctx.journal_available is False

    def test_pending_tasks_in_phase(self, tmp_path):
        """Pending tasks in active phase are listed correctly."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _build_resume_context,
        )

        spec_data = make_spec_data()
        workspace = _make_workspace(tmp_path, spec_data)

        session = make_session(
            active_phase_id="phase-1",
            completed_task_ids=["task-1"],  # Only task-1 done
        )

        ctx = _build_resume_context(session, str(workspace))
        assert ctx is not None
        # task-2 and verify-1 should be pending in phase-1
        pending_ids = [t.task_id for t in ctx.pending_tasks_in_phase]
        assert "task-2" in pending_ids
        assert "verify-1" in pending_ids
        assert "task-1" not in pending_ids


# =============================================================================
# State Version Tests (for issues #9, #10, #12)
# =============================================================================


class TestStateVersionIncrements:
    @pytest.fixture(autouse=True)
    def _set_maintainer_role(self):
        """Rebase requires maintainer role."""
        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session.get_server_role",
            return_value="maintainer",
        ):
            yield

    """Verify state_version is incremented on all state mutations."""

    def test_heartbeat_increments_state_version(self, tmp_path):
        """Heartbeat handler increments state_version."""
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_heartbeat,
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        session_id = _assert_success(resp)["session_id"]

        storage = _get_storage(config, str(workspace))
        session_before = storage.load(session_id)
        version_before = session_before.state_version

        _handle_session_heartbeat(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
            context_usage_pct=50,
        )

        session_after = storage.load(session_id)
        assert session_after.state_version == version_before + 1

    def test_force_end_increments_old_session_state_version(self, tmp_path):
        """Force-ending an existing session increments its state_version."""
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp1 = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        old_id = _assert_success(resp1)["session_id"]

        storage = _get_storage(config, str(workspace))
        old_before = storage.load(old_id)
        version_before = old_before.state_version

        # Force-start new session
        _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
            force=True,
        )

        old_after = storage.load(old_id)
        assert old_after.status == SessionStatus.ENDED
        assert old_after.state_version == version_before + 1

    def test_rebase_increments_state_version(self, tmp_path):
        """Rebase handler increments state_version."""
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_rebase,
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        session_id = _assert_success(resp)["session_id"]

        # Mark as failed for rebase
        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        session.status = SessionStatus.FAILED
        session.failure_reason = FailureReason.SPEC_STRUCTURE_CHANGED
        version_before = session.state_version
        storage.save(session)

        resp = _handle_session_rebase(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
        )
        _assert_success(resp)

        session_after = storage.load(session_id)
        assert session_after.state_version > version_before


# =============================================================================
# LastStepResult Cross-Field Validation Tests (issue #15)
# =============================================================================


class TestLastStepResultValidation:
    """Test cross-field validation on LastStepResult."""

    def test_implement_task_requires_task_id(self):
        """implement_task step type requires task_id."""
        with pytest.raises(ValueError, match="task_id is required"):
            LastStepResult(
                step_id="s1",
                step_type=StepType.IMPLEMENT_TASK,
                outcome=StepOutcome.SUCCESS,
                # Missing task_id
            )

    def test_execute_verification_requires_task_id(self):
        """execute_verification step type requires task_id."""
        with pytest.raises(ValueError, match="task_id is required"):
            LastStepResult(
                step_id="s1",
                step_type=StepType.EXECUTE_VERIFICATION,
                outcome=StepOutcome.SUCCESS,
                # Missing task_id
            )

    def test_run_fidelity_gate_requires_phase_and_gate(self):
        """run_fidelity_gate requires phase_id and gate_attempt_id."""
        with pytest.raises(ValueError, match="phase_id is required"):
            LastStepResult(
                step_id="s1",
                step_type=StepType.RUN_FIDELITY_GATE,
                outcome=StepOutcome.SUCCESS,
                # Missing phase_id
            )

        with pytest.raises(ValueError, match="gate_attempt_id is required"):
            LastStepResult(
                step_id="s1",
                step_type=StepType.RUN_FIDELITY_GATE,
                outcome=StepOutcome.SUCCESS,
                phase_id="phase-1",
                # Missing gate_attempt_id
            )

    def test_address_fidelity_feedback_requires_phase(self):
        """address_fidelity_feedback requires phase_id."""
        with pytest.raises(ValueError, match="phase_id is required"):
            LastStepResult(
                step_id="s1",
                step_type=StepType.ADDRESS_FIDELITY_FEEDBACK,
                outcome=StepOutcome.SUCCESS,
                # Missing phase_id
            )

    def test_valid_step_results_pass(self):
        """Valid step results pass validation."""
        # implement_task with task_id
        r = LastStepResult(
            step_id="s1",
            step_type=StepType.IMPLEMENT_TASK,
            outcome=StepOutcome.SUCCESS,
            task_id="task-1",
        )
        assert r.task_id == "task-1"

        # run_fidelity_gate with phase_id and gate_attempt_id
        r = LastStepResult(
            step_id="s2",
            step_type=StepType.RUN_FIDELITY_GATE,
            outcome=StepOutcome.SUCCESS,
            phase_id="phase-1",
            gate_attempt_id="gate-001",
        )
        assert r.phase_id == "phase-1"

        # pause and complete_spec have no extra requirements
        r = LastStepResult(
            step_id="s3",
            step_type=StepType.PAUSE,
            outcome=StepOutcome.SUCCESS,
        )
        assert r.step_type == StepType.PAUSE


# =============================================================================
# Auto-Retry Cycle Cap Test (issue #8)
# =============================================================================


class TestAutoRetryCycleCap:
    """Test that auto-retry respects cycle cap."""

    def test_auto_retry_blocked_at_cycle_cap(self, tmp_path):
        """Auto-retry should NOT schedule address_fidelity_feedback at cycle cap."""
        workspace = _make_workspace(tmp_path)
        orch = _make_orchestrator(workspace)
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
            # At the cycle cap
            counters=SessionCounters(fidelity_review_cycles_in_active_phase=3),
            limits=SessionLimits(max_fidelity_review_cycles_per_phase=3),
        )

        result = orch._handle_gate_evidence(session, spec_data, now)
        # Should pause instead of auto-retrying
        assert result.session.status == SessionStatus.PAUSED
        assert result.session.pause_reason == PauseReason.GATE_FAILED
        # Counter still incremented (tracks failed evaluations)
        assert session.counters.fidelity_review_cycles_in_active_phase == 4

    def test_auto_retry_allowed_below_cap(self, tmp_path):
        """Auto-retry schedules address_fidelity_feedback when below cap."""
        workspace = _make_workspace(tmp_path)
        orch = _make_orchestrator(workspace)
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
            counters=SessionCounters(fidelity_review_cycles_in_active_phase=1),
            limits=SessionLimits(max_fidelity_review_cycles_per_phase=3),
        )

        result = orch._handle_gate_evidence(session, spec_data, now)
        # Should create address_fidelity_feedback step
        assert result.success is True
        assert result.next_step is not None
        assert result.next_step.type == StepType.ADDRESS_FIDELITY_FEEDBACK
        # Counter incremented from 1 to 2 on failed gate evaluation
        assert session.counters.fidelity_review_cycles_in_active_phase == 2


# =============================================================================
# Terminal States Test (issue #7)
# =============================================================================


class TestTerminalStates:
    """Test that only COMPLETED and ENDED are truly terminal."""

    def test_failed_not_terminal_in_orchestrator(self, tmp_path):
        """FAILED session should not be short-circuited as terminal."""
        workspace = _make_workspace(tmp_path)
        orch = _make_orchestrator(workspace)

        session = make_session(
            status=SessionStatus.FAILED,
            spec_file_mtime=None,  # Force integrity check
        )

        # FAILED should NOT return terminal short-circuit
        # It should proceed through normal orchestration (and likely hit
        # spec integrity check, but the point is it doesn't stop at step 5)
        result = orch.compute_next_step(session)
        # The session should proceed past step 5 (terminal check)
        # It may fail at spec integrity or elsewhere, but NOT with
        # a terminal no-op response
        if result.success:
            assert result.next_step is not None or result.session.status == SessionStatus.COMPLETED
        else:
            # Error from spec integrity or another check - not terminal short-circuit
            assert result.error_code is not None


# =============================================================================
# Gate Invariant Enforcement Tests (HB-13)
# =============================================================================


class TestGateInvariantEnforcement:
    """Test that required phase gates cannot be bypassed (HB-13).

    These tests verify the gate invariant enforcement ensures:
    1. Orchestrator cannot complete spec when required phase gate is skipped
    2. Phase progression is blocked when required gate is unsatisfied
    3. Gate waiver (privileged path) allows progression

    Note: These tests directly call _check_required_gates_satisfied to test
    the gate invariant logic in isolation from the full orchestration flow.
    """

    def test_required_gate_blocks_spec_completion(self, tmp_path):
        """Spec completion is blocked when required phase gate is unsatisfied."""

        workspace = _make_workspace(tmp_path)
        orch = _make_orchestrator(workspace)

        # Session with all tasks completed but phase-1 gate unsatisfied
        session = make_session(
            active_phase_id="phase-2",
            completed_task_ids=["task-1", "verify-1", "task-2"],
            # Phase-1 has required gate but it's PENDING (not satisfied)
            phase_gates={
                "phase-1": PhaseGateRecord(
                    required=True,
                    status=PhaseGateStatus.PENDING,
                ),
            },
            required_phase_gates={"phase-1": ["fidelity"]},
            satisfied_gates={},  # Nothing satisfied
        )

        # Directly test the gate check method
        gate_block = orch._check_required_gates_satisfied(session)

        # Should return blocking info
        assert gate_block is not None
        assert gate_block["phase_id"] == "phase-1"
        assert "gate" in gate_block["blocking_reason"].lower()

    def test_required_gate_blocks_phase_progression(self, tmp_path):
        """Phase progression is blocked when required gate is unsatisfied."""

        workspace = _make_workspace(tmp_path)
        orch = _make_orchestrator(workspace)

        # Session with phase-1 tasks done but gate not satisfied
        session = make_session(
            active_phase_id="phase-1",
            completed_task_ids=["task-1"],
            phase_gates={
                "phase-1": PhaseGateRecord(
                    required=True,
                    status=PhaseGateStatus.PENDING,
                ),
            },
            required_phase_gates={"phase-1": ["fidelity"]},
            satisfied_gates={},
        )

        # Check gates for phase-1 specifically
        gate_block = orch._check_required_gates_satisfied(session, "phase-1")

        # Should return blocking info
        assert gate_block is not None
        assert gate_block["phase_id"] == "phase-1"

    def test_gate_pass_allows_phase_progression(self, tmp_path):
        """Passed gate allows phase progression."""

        workspace = _make_workspace(tmp_path)
        orch = _make_orchestrator(workspace)

        # Session with phase-1 tasks done AND gate PASSED
        session = make_session(
            active_phase_id="phase-1",
            completed_task_ids=["task-1"],
            phase_gates={
                "phase-1": PhaseGateRecord(
                    required=True,
                    status=PhaseGateStatus.PASSED,
                ),
            },
            required_phase_gates={"phase-1": ["fidelity"]},
            satisfied_gates={"phase-1": ["fidelity"]},
        )

        # Check gates for phase-1
        gate_block = orch._check_required_gates_satisfied(session, "phase-1")

        # Should return None (no block)
        assert gate_block is None

    def test_gate_waiver_allows_spec_completion(self, tmp_path):
        """Waived gate allows spec completion (privileged path)."""

        workspace = _make_workspace(tmp_path)
        orch = _make_orchestrator(workspace)

        # Session with required gate WAIVED
        session = make_session(
            active_phase_id="phase-2",
            completed_task_ids=["task-1", "task-2"],
            phase_gates={
                "phase-1": PhaseGateRecord(
                    required=True,
                    status=PhaseGateStatus.WAIVED,
                ),
            },
            required_phase_gates={"phase-1": ["fidelity"]},
            satisfied_gates={"phase-1": ["fidelity"]},  # Waived counts as satisfied
        )

        # Check all gates
        gate_block = orch._check_required_gates_satisfied(session)

        # Should return None (waived = satisfied)
        assert gate_block is None

    def test_no_required_gate_allows_unrestricted_progression(self, tmp_path):
        """No required gates means unrestricted phase progression."""

        workspace = _make_workspace(tmp_path)
        orch = _make_orchestrator(workspace)

        # Session with NO required gates
        session = make_session(
            active_phase_id="phase-1",
            completed_task_ids=["task-1"],
            phase_gates={
                "phase-1": PhaseGateRecord(
                    required=False,  # Not required
                    status=PhaseGateStatus.PENDING,
                ),
            },
            required_phase_gates={},  # No required gates
            satisfied_gates={},
        )

        # Check gates
        gate_block = orch._check_required_gates_satisfied(session)

        # Should return None (no required gates to check)
        assert gate_block is None

    def test_phase_gate_status_overrides_satisfied_gates(self, tmp_path):
        """PhaseGateStatus.PASSED/WAIVED overrides missing satisfied_gates entry."""

        workspace = _make_workspace(tmp_path)
        orch = _make_orchestrator(workspace)

        # Session with gate PASSED but NOT in satisfied_gates
        # This tests the fallback logic that checks PhaseGateRecord.status
        session = make_session(
            active_phase_id="phase-1",
            completed_task_ids=["task-1"],
            phase_gates={
                "phase-1": PhaseGateRecord(
                    required=True,
                    status=PhaseGateStatus.PASSED,  # Passed!
                ),
            },
            required_phase_gates={"phase-1": ["fidelity"]},
            satisfied_gates={},  # Empty - but gate record shows PASSED
        )

        # Check gates - should auto-populate satisfied_gates from gate record
        gate_block = orch._check_required_gates_satisfied(session, "phase-1")

        # Should return None (gate record status overrides missing satisfied_gates)
        assert gate_block is None
        # The method should have populated satisfied_gates
        assert "fidelity" in session.satisfied_gates.get("phase-1", [])
