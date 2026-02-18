"""Tests for session lifecycle handlers.

Covers:
- session-start: basic creation, force-end existing, idempotency key, spec not found,
  missing spec_id, gate_policy validation, best-effort journal writes
- session-pause: running -> paused, invalid state transitions, invalid reason fallback
- session-resume: paused -> running, failed + force -> running,
  spec_structure_changed guard, manual gate ack, gate_ack mismatch
- session-end: running/paused/failed -> ended, terminal state rejection
- session-status: read-only, delegates to _resolve_session
- session-list: empty, pagination, cursor, status/spec filter, invalid cursor
- session-rebase: no-change, structural change, completed task removal guard,
  force rebase with task removal, invalid state
- session-heartbeat: updates heartbeat, context_usage_pct validation
- session-reset: failed -> deleted, non-failed rejection, reason_code validation
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.autonomy.models.enums import (
    FailureReason,
    GatePolicy,
    PauseReason,
    SessionStatus,
    StepType,
)
from foundry_mcp.core.autonomy.models.gates import PendingManualGateAck
from foundry_mcp.core.autonomy.models.session_config import (
    SessionCounters,
    SessionLimits,
    StopConditions,
)
from foundry_mcp.core.autonomy.models.state import AutonomousSessionState
from foundry_mcp.core.autonomy.models.steps import LastStepIssued
from .conftest import make_session, make_spec_data


# =============================================================================
# Helpers
# =============================================================================

def _make_config(workspace: Path) -> MagicMock:
    """Create a mock ServerConfig that points at a workspace."""
    config = MagicMock()
    config.workspace_path = str(workspace)
    config.specs_dir = str(workspace / "specs")
    return config


def _setup_workspace(tmp_path: Path, spec_id: str = "test-spec-001") -> Path:
    """Create workspace with spec file, return workspace path."""
    workspace = tmp_path / "ws"
    specs_dir = workspace / "specs" / "active"
    specs_dir.mkdir(parents=True)

    spec_data = make_spec_data(spec_id=spec_id)
    spec_data["title"] = "Test Spec"
    spec_data["journal"] = []
    spec_path = specs_dir / f"{spec_id}.json"
    spec_path.write_text(json.dumps(spec_data, indent=2))

    return workspace


def _append_session_journal_entry(
    *,
    workspace: Path,
    spec_id: str,
    session_id: str,
    action: str,
    title: str,
    content: str,
    entry_type: str = "session",
) -> None:
    """Append a session-scoped journal entry directly to a test spec file."""
    spec_path = workspace / "specs" / "active" / f"{spec_id}.json"
    spec_data = json.loads(spec_path.read_text())
    journal = spec_data.setdefault("journal", [])
    journal.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "entry_type": entry_type,
            "title": title,
            "content": content,
            "author": "autonomy",
            "metadata": {
                "session_id": session_id,
                "action": action,
            },
        }
    )
    spec_path.write_text(json.dumps(spec_data, indent=2))


def _call_handler(handler_func, **kwargs) -> dict:
    """Call a handler and return the response dict."""
    return handler_func(**kwargs)


def _assert_success(resp: dict) -> dict:
    """Assert response is successful and return data."""
    assert resp["success"] is True, f"Expected success, got error: {resp.get('error')}"
    assert resp["error"] is None
    assert resp["meta"]["version"] == "response-v2"
    return resp["data"]


def _assert_error(resp: dict, error_code: Optional[str] = None) -> dict:
    """Assert response is an error and optionally check the error code."""
    assert resp["success"] is False
    assert resp["error"] is not None
    if error_code:
        # Error code appears in data.error_code or data.code depending on format
        details = resp.get("data", {})
        code = details.get("error_code") or details.get("code")
        assert code == error_code, f"Expected error_code={error_code}, got {code}"
    return resp


# =============================================================================
# Session Start Tests
# =============================================================================


class TestSessionStart:
    """Tests for _handle_session_start."""

    def test_start_creates_session(self, tmp_path):
        """Basic session creation with a valid spec."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )

        data = _assert_success(resp)
        assert data["spec_id"] == "test-spec-001"
        assert data["status"] == "running"
        assert data["session_id"]  # non-empty ULID

    def test_start_missing_spec_id(self, tmp_path):
        """Missing spec_id returns validation error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_start(
            config=config,
            workspace=str(workspace),
        )

        assert resp["success"] is False

    def test_start_spec_not_found(self, tmp_path):
        """Non-existent spec returns NOT_FOUND."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_start(
            config=config,
            spec_id="nonexistent-spec",
            workspace=str(workspace),
        )

        assert resp["success"] is False

    def test_start_conflict_without_force(self, tmp_path):
        """Starting a second session without force returns SPEC_SESSION_EXISTS."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        # Start first session
        resp1 = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        _assert_success(resp1)

        # Attempt second session without force
        resp2 = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        assert resp2["success"] is False

    def test_start_force_ends_existing(self, tmp_path):
        """force=True ends existing session and creates a new one."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        # Start first session
        resp1 = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        data1 = _assert_success(resp1)
        first_session_id = data1["session_id"]

        # Force start a second session
        resp2 = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
            force=True,
        )
        data2 = _assert_success(resp2)
        assert data2["session_id"] != first_session_id
        assert data2["status"] == "running"

    def test_start_idempotency_key_returns_existing(self, tmp_path):
        """Same idempotency key returns existing session (idempotent)."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp1 = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            idempotency_key="my-key-123",
            workspace=str(workspace),
        )
        data1 = _assert_success(resp1)

        resp2 = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            idempotency_key="my-key-123",
            workspace=str(workspace),
        )
        data2 = _assert_success(resp2)
        assert data2["session_id"] == data1["session_id"]

    def test_start_invalid_gate_policy(self, tmp_path):
        """Invalid gate_policy returns validation error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            gate_policy="invalid_policy",
            workspace=str(workspace),
        )

        assert resp["success"] is False

    def test_start_custom_limits(self, tmp_path):
        """Custom session limits are applied."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
            max_tasks_per_session=50,
            max_consecutive_errors=10,
            heartbeat_stale_minutes=30,
        )

        data = _assert_success(resp)
        assert data["limits"]["max_tasks_per_session"] == 50
        assert data["limits"]["max_consecutive_errors"] == 10
        assert data["limits"]["heartbeat_stale_minutes"] == 30

    def test_start_uses_configured_session_defaults(self, tmp_path):
        """Session-start uses config.autonomy_session_defaults when args omitted."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)
        config.autonomy_session_defaults = type(
            "SessionDefaults",
            (),
            {
                "gate_policy": "manual",
                "stop_on_phase_completion": True,
                "auto_retry_fidelity_gate": False,
                "max_tasks_per_session": 42,
                "max_consecutive_errors": 7,
                "max_fidelity_review_cycles_per_phase": 5,
            },
        )()

        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )

        data = _assert_success(resp)
        assert data["limits"]["max_tasks_per_session"] == 42
        assert data["limits"]["max_consecutive_errors"] == 7
        assert data["limits"]["max_fidelity_review_cycles_per_phase"] == 5
        assert data["stop_conditions"]["stop_on_phase_completion"] is True
        assert data["stop_conditions"]["auto_retry_fidelity_gate"] is False

        from foundry_mcp.core.autonomy.memory import AutonomyStorage

        storage = AutonomyStorage(workspace_path=workspace)
        session_id = storage.get_active_session("test-spec-001")
        assert session_id is not None
        session = storage.load(session_id)
        assert session is not None
        assert session.gate_policy.value == "manual"

    def test_start_succeeds_when_journal_write_fails(self, tmp_path):
        """Session start succeeds even if journal write fails (best-effort policy)."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session._write_session_journal",
            return_value=False,
        ):
            resp = _handle_session_start(
                config=config,
                spec_id="test-spec-001",
                workspace=str(workspace),
            )
        _assert_success(resp)

    def test_start_rejects_path_traversal(self, tmp_path):
        """Workspace with path traversal is rejected."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace) + "/../../../etc",
        )
        assert resp["success"] is False

    def test_start_rejects_dotdot_workspace(self, tmp_path):
        """Workspace containing .. anywhere is rejected."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        config = _make_config(tmp_path)

        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace="../../../etc",
        )
        assert resp["success"] is False

    def test_start_rejects_lenient_gate_policy_under_unattended_posture(self, tmp_path):
        """Unattended posture rejects non-strict gate_policy at runtime."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)
        config.autonomy_posture.profile = "unattended"

        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
            gate_policy="lenient",
        )
        assert resp["success"] is False
        details = resp.get("data", {}).get("details", {})
        assert details.get("posture_profile") == "unattended"
        assert any("gate_policy" in v for v in details.get("violations", []))

    def test_start_rejects_write_lock_bypass_under_unattended_posture(self, tmp_path):
        """Unattended posture rejects enforce_autonomy_write_lock=false."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)
        config.autonomy_posture.profile = "unattended"

        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
            enforce_autonomy_write_lock=False,
        )
        assert resp["success"] is False
        details = resp.get("data", {}).get("details", {})
        assert any("write_lock" in v for v in details.get("violations", []))

    def test_start_allows_strict_policy_under_unattended_posture(self, tmp_path):
        """Unattended posture allows session-start with strict gate_policy."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)
        config.autonomy_posture.profile = "unattended"

        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
            gate_policy="strict",
        )
        assert resp["success"] is True

    def test_start_allows_lenient_policy_without_posture(self, tmp_path):
        """Without a posture profile, lenient gate_policy is allowed."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)
        config.autonomy_posture.profile = None

        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
            gate_policy="lenient",
        )
        assert resp["success"] is True


# =============================================================================
# Session Pause Tests
# =============================================================================


class TestSessionPause:
    """Tests for _handle_session_pause."""

    def test_pause_running_session(self, tmp_path):
        """Pausing a running session transitions to paused."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_pause,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        data_start = _assert_success(resp_start)

        resp_pause = _handle_session_pause(
            config=config,
            spec_id="test-spec-001",
            reason="user",
            workspace=str(workspace),
        )
        data = _assert_success(resp_pause)
        assert data["status"] == "paused"
        assert data["pause_reason"] == "user"

    def test_pause_invalid_reason_fallback(self, tmp_path):
        """Invalid PauseReason string falls back to USER."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_pause,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        resp = _handle_session_pause(
            config=config,
            spec_id="test-spec-001",
            reason="totally_invalid_reason",
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["status"] == "paused"
        assert data["pause_reason"] == "user"

    def test_pause_non_running_session(self, tmp_path):
        """Pausing a non-running session returns INVALID_STATE_TRANSITION."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_pause,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        # Pause it first
        _handle_session_pause(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        # Try to pause again
        resp = _handle_session_pause(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        assert resp["success"] is False


# =============================================================================
# Session Resume Tests
# =============================================================================


class TestSessionResume:
    """Tests for _handle_session_resume."""

    def test_resume_paused_session(self, tmp_path):
        """Resuming a paused session transitions to running."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_pause,
            _handle_session_resume,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        _handle_session_pause(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        resp = _handle_session_resume(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["status"] == "running"
        assert data["pause_reason"] is None

    def test_resume_running_session_rejected(self, tmp_path):
        """Resuming a running session returns INVALID_STATE_TRANSITION."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_resume,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        resp = _handle_session_resume(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        assert resp["success"] is False

    def test_resume_failed_without_force_rejected(self, tmp_path):
        """Resuming a failed session without force returns INVALID_STATE_TRANSITION."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_resume,
        )
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        data_start = _assert_success(resp_start)
        session_id = data_start["session_id"]

        # Manually mark session as failed
        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        session.status = SessionStatus.FAILED
        session.failure_reason = FailureReason.STATE_CORRUPT
        storage.save(session)

        resp = _handle_session_resume(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        assert resp["success"] is False

    def test_resume_failed_with_force(self, tmp_path):
        """force=True resumes a failed session."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_resume,
        )
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        # Manually mark as failed
        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        session.status = SessionStatus.FAILED
        session.failure_reason = FailureReason.STATE_CORRUPT
        storage.save(session)

        resp = _handle_session_resume(
            config=config,
            spec_id="test-spec-001",
            force=True,
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["status"] == "running"

    def test_resume_failed_spec_structure_changed_requires_rebase(self, tmp_path):
        """Resume from SPEC_STRUCTURE_CHANGED failure checks hash; if changed, returns SPEC_REBASE_REQUIRED."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_resume,
        )
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        # Mark as failed due to spec structure change
        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        session.status = SessionStatus.FAILED
        session.failure_reason = FailureReason.SPEC_STRUCTURE_CHANGED
        # Set a different hash to simulate spec having changed
        session.spec_structure_hash = "b" * 64
        storage.save(session)

        resp = _handle_session_resume(
            config=config,
            spec_id="test-spec-001",
            force=True,
            workspace=str(workspace),
        )
        # Spec hash differs from stored, so should require rebase
        assert resp["success"] is False

    def test_resume_manual_gate_ack_required(self, tmp_path):
        """Resuming with pending manual gate ack but no gate_ack returns error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_pause,
            _handle_session_resume,
        )
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        # Pause, then set pending gate ack
        _handle_session_pause(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        session.pending_manual_gate_ack = PendingManualGateAck(
            gate_attempt_id="gate-attempt-123",
            phase_id="phase-1",
            issued_at=datetime.now(timezone.utc),
        )
        storage.save(session)

        resp = _handle_session_resume(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        assert resp["success"] is False

    def test_resume_manual_gate_ack_wrong_id(self, tmp_path):
        """Providing wrong gate_ack returns INVALID_GATE_ACK."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_pause,
            _handle_session_resume,
        )
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        _handle_session_pause(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        session.pending_manual_gate_ack = PendingManualGateAck(
            gate_attempt_id="gate-attempt-123",
            phase_id="phase-1",
            issued_at=datetime.now(timezone.utc),
        )
        storage.save(session)

        resp = _handle_session_resume(
            config=config,
            spec_id="test-spec-001",
            acknowledge_gate_review=True,
            acknowledged_gate_attempt_id="wrong-gate-id",
            workspace=str(workspace),
        )
        assert resp["success"] is False

    def test_resume_manual_gate_ack_correct(self, tmp_path):
        """Correct gate acknowledgment clears pending ack and resumes."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_pause,
            _handle_session_resume,
        )
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        _handle_session_pause(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        session.pending_manual_gate_ack = PendingManualGateAck(
            gate_attempt_id="gate-attempt-123",
            phase_id="phase-1",
            issued_at=datetime.now(timezone.utc),
        )
        storage.save(session)

        resp = _handle_session_resume(
            config=config,
            spec_id="test-spec-001",
            acknowledge_gate_review=True,
            acknowledged_gate_attempt_id="gate-attempt-123",
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["status"] == "running"


# =============================================================================
# Session End Tests
# =============================================================================


class TestSessionEnd:
    """Tests for _handle_session_end."""

    def test_end_running_session(self, tmp_path):
        """Ending a running session transitions to ended."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_end,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        resp = _handle_session_end(
            config=config,
            spec_id="test-spec-001",
            reason_code="operator_override",
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["status"] == "ended"

    def test_end_paused_session(self, tmp_path):
        """Ending a paused session transitions to ended."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_pause,
            _handle_session_end,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        _handle_session_pause(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        resp = _handle_session_end(
            config=config,
            spec_id="test-spec-001",
            reason_code="operator_override",
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["status"] == "ended"

    def test_end_already_ended_rejected(self, tmp_path):
        """Ending an already ended session returns INVALID_STATE_TRANSITION."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_end,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        _handle_session_end(
            config=config,
            spec_id="test-spec-001",
            reason_code="operator_override",
            workspace=str(workspace),
        )

        # Session is now ended, pointer removed. Look up by session_id directly.
        resp = _handle_session_end(
            config=config,
            session_id=session_id,
            reason_code="operator_override",
            workspace=str(workspace),
        )
        assert resp["success"] is False

    def test_end_requires_reason_code(self, tmp_path):
        """Ending a session without reason_code returns validation error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_end,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)
        _handle_session_start(config=config, spec_id="test-spec-001", workspace=str(workspace))

        resp = _handle_session_end(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        assert resp["success"] is False
        assert resp["data"]["details"]["field"] == "reason_code"

    def test_end_rejects_invalid_reason_code(self, tmp_path):
        """Ending a session with invalid reason_code returns validation error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_end,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)
        _handle_session_start(config=config, spec_id="test-spec-001", workspace=str(workspace))

        resp = _handle_session_end(
            config=config,
            spec_id="test-spec-001",
            reason_code="not_a_valid_code",
            workspace=str(workspace),
        )
        assert resp["success"] is False
        assert resp["data"]["details"]["field"] == "reason_code"


# =============================================================================
# Session Status Tests
# =============================================================================


class TestSessionStatus:
    """Tests for _handle_session_status."""

    def test_status_returns_session_data(self, tmp_path):
        """Status returns session details."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_status,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        data_start = _assert_success(resp_start)

        resp = _handle_session_status(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["session_id"] == data_start["session_id"]
        assert data["status"] == "running"
        assert data["session_signal"] is None

    def test_status_includes_phase_complete_session_signal(self, tmp_path):
        """session-status includes derived session_signal for phase completion pauses."""
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_status,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        assert session is not None
        session.status = SessionStatus.PAUSED
        session.pause_reason = PauseReason.PHASE_COMPLETE
        storage.save(session)

        resp = _handle_session_status(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["session_signal"] == "phase_complete"

    def test_status_includes_operator_progress_fields(self, tmp_path):
        """session-status includes P2 operator fields for progress and retries."""
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_status,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        assert session is not None
        session.active_phase_id = "phase-1"
        session.last_task_id = "task-1"
        session.completed_task_ids = ["task-1"]
        session.counters.consecutive_errors = 2
        session.counters.fidelity_review_cycles_in_active_phase = 1
        session.last_step_issued = LastStepIssued(
            step_id="01TESTSTEP000000000000000000",
            type=StepType.IMPLEMENT_TASK,
            task_id="task-1",
            phase_id="phase-1",
            issued_at=datetime.now(timezone.utc),
        )
        storage.save(session)

        spec_path = workspace / "specs" / "active" / "test-spec-001.json"
        spec_data = json.loads(spec_path.read_text())
        spec_data["phases"][0]["tasks"][1]["metadata"] = {"retry_count": 3}
        spec_path.write_text(json.dumps(spec_data, indent=2))

        resp = _handle_session_status(
            config=config, session_id=session_id, workspace=str(workspace),
        )
        data = _assert_success(resp)

        assert data["last_step_id"] == "01TESTSTEP000000000000000000"
        assert data["last_step_type"] == "implement_task"
        assert data["current_task_id"] == "task-1"

        active_phase_progress = data["active_phase_progress"]
        assert active_phase_progress["phase_id"] == "phase-1"
        assert active_phase_progress["total_tasks"] == 3
        assert active_phase_progress["completed_tasks"] == 1
        assert active_phase_progress["remaining_tasks"] == 2

        retry_counters = data["retry_counters"]
        assert retry_counters["consecutive_errors"] == 2
        assert retry_counters["fidelity_review_cycles_in_active_phase"] == 1
        assert retry_counters["phase_retry_counts"]["phase-1"] == 1
        assert retry_counters["task_retry_counts"]["task-2"] == 3

    def test_status_no_session_returns_error(self, tmp_path):
        """Status with no active session returns NO_ACTIVE_SESSION."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_status,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_status(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        assert resp["success"] is False


# =============================================================================
# Session List Tests
# =============================================================================


class TestSessionList:
    """Tests for _handle_session_list."""

    def test_list_empty(self, tmp_path):
        """Listing with no sessions returns empty list."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_list,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_list(
            config=config, workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["sessions"] == []
        assert data["has_more"] is False

    def test_list_returns_sessions(self, tmp_path):
        """Listing returns created sessions."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_list,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        resp = _handle_session_list(
            config=config, workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["spec_id"] == "test-spec-001"

    def test_list_pagination_limit(self, tmp_path):
        """Pagination limit is respected."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_list,
            _handle_session_end,
        )

        workspace = _setup_workspace(tmp_path)
        # Create additional spec files for multiple sessions
        for i in range(3):
            spec_id = f"spec-{i:03d}"
            spec_dir = workspace / "specs" / "active"
            spec_data = make_spec_data(spec_id=spec_id)
            spec_data["title"] = f"Spec {i}"
            spec_data["journal"] = []
            (spec_dir / f"{spec_id}.json").write_text(json.dumps(spec_data))

        config = _make_config(workspace)

        # Create 3 sessions (force end each to allow creating next)
        for i in range(3):
            _handle_session_start(
                config=config,
                spec_id=f"spec-{i:03d}",
                workspace=str(workspace),
            )

        resp = _handle_session_list(
            config=config, workspace=str(workspace), limit=2,
        )
        data = _assert_success(resp)
        assert len(data["sessions"]) == 2
        assert data["has_more"] is True

    def test_list_filter_by_status(self, tmp_path):
        """Status filter returns only matching sessions."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_pause,
            _handle_session_list,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        # Pause the session
        _handle_session_pause(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        # List running sessions (should be empty)
        resp = _handle_session_list(
            config=config,
            workspace=str(workspace),
            status_filter="running",
        )
        data = _assert_success(resp)
        assert len(data["sessions"]) == 0

        # List paused sessions (should have 1)
        resp = _handle_session_list(
            config=config,
            workspace=str(workspace),
            status_filter="paused",
        )
        data = _assert_success(resp)
        assert len(data["sessions"]) == 1


# =============================================================================
# Session Events Tests
# =============================================================================


class TestSessionEvents:
    """Tests for _handle_session_events."""

    def test_events_returns_journal_backed_session_view(self, tmp_path):
        """session-events returns journal entries filtered to the target session."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_events,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        _append_session_journal_entry(
            workspace=workspace,
            spec_id="test-spec-001",
            session_id=session_id,
            action="start",
            title="Session start",
            content="Started autonomous session",
        )
        _append_session_journal_entry(
            workspace=workspace,
            spec_id="test-spec-001",
            session_id=session_id,
            action="pause",
            title="Session pause",
            content="Paused autonomous session",
        )
        _append_session_journal_entry(
            workspace=workspace,
            spec_id="test-spec-001",
            session_id=session_id,
            action="resume",
            title="Session resume",
            content="Resumed autonomous session",
        )

        resp = _handle_session_events(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
        )
        data = _assert_success(resp)

        assert data["session_id"] == session_id
        assert data["spec_id"] == "test-spec-001"
        assert len(data["events"]) >= 3
        assert all(event["session_id"] == session_id for event in data["events"])
        assert all("event_id" in event for event in data["events"])
        assert all("timestamp" in event for event in data["events"])

        actions = {event.get("action") for event in data["events"]}
        assert "start" in actions
        assert "pause" in actions
        assert "resume" in actions

    def test_events_pagination_cursor(self, tmp_path):
        """session-events paginates with stable cursor semantics."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_events,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        for i in range(5):
            _append_session_journal_entry(
                workspace=workspace,
                spec_id="test-spec-001",
                session_id=session_id,
                action="status",
                title=f"Status {i}",
                content=f"Synthetic event {i}",
            )

        first_page = _handle_session_events(
            config=config,
            session_id=session_id,
            limit=2,
            workspace=str(workspace),
        )
        first_data = _assert_success(first_page)
        assert len(first_data["events"]) == 2
        assert first_page["meta"]["pagination"]["has_more"] is True

        cursor = first_page["meta"]["pagination"]["cursor"]
        assert isinstance(cursor, str)

        second_page = _handle_session_events(
            config=config,
            session_id=session_id,
            limit=2,
            cursor=cursor,
            workspace=str(workspace),
        )
        second_data = _assert_success(second_page)
        assert len(second_data["events"]) >= 1

        first_ids = {event["event_id"] for event in first_data["events"]}
        second_ids = {event["event_id"] for event in second_data["events"]}
        assert first_ids.isdisjoint(second_ids)

    def test_events_invalid_cursor_returns_validation_error(self, tmp_path):
        """session-events rejects malformed cursors."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_events,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        resp = _handle_session_events(
            config=config,
            session_id=session_id,
            cursor="not-a-valid-cursor",
            workspace=str(workspace),
        )

        assert resp["success"] is False
        assert resp["data"]["error_code"] == "INVALID_CURSOR"

    def test_events_filter_to_requested_session(self, tmp_path):
        """session-events does not leak entries across sessions."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_end,
            _handle_session_events,
        )

        workspace = _setup_workspace(tmp_path)

        # Add a second spec so we can create an additional session.
        spec_two = make_spec_data(spec_id="test-spec-002")
        spec_two["title"] = "Spec 2"
        spec_two["journal"] = []
        (workspace / "specs" / "active" / "test-spec-002.json").write_text(
            json.dumps(spec_two, indent=2)
        )

        config = _make_config(workspace)

        first_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        first_session_id = _assert_success(first_start)["session_id"]
        _handle_session_end(
            config=config,
            session_id=first_session_id,
            reason_code="operator_override",
            workspace=str(workspace),
        )

        second_start = _handle_session_start(
            config=config, spec_id="test-spec-002", workspace=str(workspace),
        )
        second_session_id = _assert_success(second_start)["session_id"]

        _append_session_journal_entry(
            workspace=workspace,
            spec_id="test-spec-001",
            session_id=first_session_id,
            action="start",
            title="First session",
            content="First session event",
        )
        _append_session_journal_entry(
            workspace=workspace,
            spec_id="test-spec-002",
            session_id=second_session_id,
            action="start",
            title="Second session",
            content="Second session event",
        )

        resp = _handle_session_events(
            config=config,
            session_id=second_session_id,
            workspace=str(workspace),
        )
        data = _assert_success(resp)

        assert data["session_id"] == second_session_id
        assert len(data["events"]) >= 1
        assert all(event["session_id"] == second_session_id for event in data["events"])


# =============================================================================
# Session Rebase Tests
# =============================================================================


class TestSessionRebase:
    """Tests for _handle_session_rebase."""

    @pytest.fixture(autouse=True)
    def _set_maintainer_role(self):
        """Rebase requires maintainer role; set it by default for existing tests."""
        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session.get_server_role",
            return_value="maintainer",
        ):
            yield

    def test_rebase_no_change(self, tmp_path):
        """Rebase with no structural change transitions to running."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_pause,
            _handle_session_rebase,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        _handle_session_pause(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        resp = _handle_session_rebase(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["status"] == "running"
        assert data.get("rebase_result", {}).get("result") == "no_change"

    def test_rebase_running_session_rejected(self, tmp_path):
        """Rebase on running session returns INVALID_STATE_TRANSITION."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_rebase,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        resp = _handle_session_rebase(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        assert resp["success"] is False

    def test_rebase_detects_structural_change(self, tmp_path):
        """Rebase detects when spec has structurally changed."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_pause,
            _handle_session_rebase,
        )
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        _handle_session_pause(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        # Modify spec on disk (add a new task, changing structure)
        spec_path = workspace / "specs" / "active" / "test-spec-001.json"
        spec_data = json.loads(spec_path.read_text())
        spec_data["phases"][0]["tasks"].append(
            {"id": "task-new", "title": "New Task", "type": "task", "status": "pending"}
        )
        spec_path.write_text(json.dumps(spec_data, indent=2))

        resp = _handle_session_rebase(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["status"] == "running"
        assert data.get("rebase_result", {}).get("result") == "success"

    def test_rebase_preserves_unaffected_phase_gates(self, tmp_path):
        """Added tasks clear gate satisfaction only for the impacted phase."""
        from foundry_mcp.core.autonomy.spec_hash import compute_spec_structure_hash
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_rebase,
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        assert session is not None
        session.status = SessionStatus.PAUSED
        session.pause_reason = PauseReason.USER
        session.satisfied_gates = {
            "phase-1": ["fidelity"],
            "phase-2": ["manual_review"],
        }

        spec_path = workspace / "specs" / "active" / "test-spec-001.json"
        old_spec_data = json.loads(spec_path.read_text())
        session.spec_structure_hash = compute_spec_structure_hash(old_spec_data)

        backup_dir = workspace / "specs" / ".backups" / "test-spec-001"
        backup_dir.mkdir(parents=True)
        (backup_dir / "backup-001.json").write_text(json.dumps(old_spec_data))
        storage.save(session)

        spec_data = json.loads(spec_path.read_text())
        spec_data["phases"][1]["tasks"].append(
            {"id": "task-new-phase-2", "title": "New phase 2 task", "type": "task", "status": "pending"}
        )
        spec_path.write_text(json.dumps(spec_data, indent=2))

        resp = _handle_session_rebase(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        _assert_success(resp)

        updated = storage.load(session_id)
        assert updated is not None
        assert updated.satisfied_gates == {"phase-1": ["fidelity"]}

    def test_rebase_without_old_snapshot_clears_gate_satisfaction(self, tmp_path):
        """When old structure is unavailable, rebase resets satisfied gates conservatively."""
        from foundry_mcp.core.autonomy.spec_hash import compute_spec_structure_hash
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_rebase,
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        assert session is not None
        session.status = SessionStatus.PAUSED
        session.pause_reason = PauseReason.USER
        session.satisfied_gates = {
            "phase-1": ["fidelity"],
            "phase-2": ["manual_review"],
        }

        # Keep a stale hash and avoid writing a matching backup snapshot.
        spec_path = workspace / "specs" / "active" / "test-spec-001.json"
        old_spec_data = json.loads(spec_path.read_text())
        old_hash = compute_spec_structure_hash(old_spec_data)
        session.spec_structure_hash = "0" * len(old_hash)
        storage.save(session)

        spec_data = json.loads(spec_path.read_text())
        spec_data["phases"][0]["tasks"].append(
            {"id": "task-new-phase-1", "title": "New phase 1 task", "type": "task", "status": "pending"}
        )
        spec_path.write_text(json.dumps(spec_data, indent=2))

        resp = _handle_session_rebase(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        _assert_success(resp)

        updated = storage.load(session_id)
        assert updated is not None
        assert updated.satisfied_gates == {}

    def test_rebase_completed_task_removal_guarded(self, tmp_path):
        """Rebase with removed completed tasks returns error without force."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_pause,
            _handle_session_rebase,
        )
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage
        from foundry_mcp.core.autonomy.spec_hash import compute_spec_structure_hash

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        # Mark task-1 as completed in session
        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        session.completed_task_ids.append("task-1")
        session.counters.tasks_completed = 1
        session.status = SessionStatus.PAUSED
        session.pause_reason = PauseReason.USER

        # Save the old spec hash as a backup so diff computation works
        spec_path = workspace / "specs" / "active" / "test-spec-001.json"
        old_spec_data = json.loads(spec_path.read_text())
        old_hash = compute_spec_structure_hash(old_spec_data)
        session.spec_structure_hash = old_hash

        # Create backup directory with old spec
        backup_dir = workspace / "specs" / ".backups" / "test-spec-001"
        backup_dir.mkdir(parents=True)
        (backup_dir / "backup-001.json").write_text(json.dumps(old_spec_data))

        storage.save(session)

        # Now remove task-1 from spec on disk
        spec_data = json.loads(spec_path.read_text())
        spec_data["phases"][0]["tasks"] = [
            t for t in spec_data["phases"][0]["tasks"] if t["id"] != "task-1"
        ]
        spec_path.write_text(json.dumps(spec_data, indent=2))

        resp = _handle_session_rebase(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        assert resp["success"] is False

    def test_rebase_force_removes_completed_tasks(self, tmp_path):
        """force=True allows rebase even when completed tasks are removed."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_rebase,
        )
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage
        from foundry_mcp.core.autonomy.spec_hash import compute_spec_structure_hash

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        # Set up session with completed task
        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        session.completed_task_ids.append("task-1")
        session.counters.tasks_completed = 1
        session.status = SessionStatus.PAUSED
        session.pause_reason = PauseReason.USER

        spec_path = workspace / "specs" / "active" / "test-spec-001.json"
        old_spec_data = json.loads(spec_path.read_text())
        old_hash = compute_spec_structure_hash(old_spec_data)
        session.spec_structure_hash = old_hash

        # Create backup
        backup_dir = workspace / "specs" / ".backups" / "test-spec-001"
        backup_dir.mkdir(parents=True)
        (backup_dir / "backup-001.json").write_text(json.dumps(old_spec_data))

        storage.save(session)

        # Remove task-1 from spec
        spec_data = json.loads(spec_path.read_text())
        spec_data["phases"][0]["tasks"] = [
            t for t in spec_data["phases"][0]["tasks"] if t["id"] != "task-1"
        ]
        spec_path.write_text(json.dumps(spec_data, indent=2))

        resp = _handle_session_rebase(
            config=config,
            spec_id="test-spec-001",
            force=True,
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["status"] == "running"

    def test_rebase_denied_for_autonomy_runner(self, tmp_path):
        """Non-maintainer roles are denied session-rebase."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_rebase,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session.get_server_role",
            return_value="autonomy_runner",
        ):
            resp = _handle_session_rebase(
                config=config,
                spec_id="test-spec-001",
                workspace=str(workspace),
            )
        assert resp["success"] is False
        assert resp["data"]["error_code"] == "FORBIDDEN"
        assert resp["data"]["details"]["action"] == "session-rebase"
        assert resp["data"]["details"]["required_role"] == "maintainer"

    def test_rebase_denied_for_observer(self, tmp_path):
        """Observer role is denied session-rebase."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_rebase,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session.get_server_role",
            return_value="observer",
        ):
            resp = _handle_session_rebase(
                config=config,
                spec_id="test-spec-001",
                workspace=str(workspace),
            )
        assert resp["success"] is False
        assert resp["data"]["error_code"] == "FORBIDDEN"


# =============================================================================
# Rebase Backup Guard Tests (C2)
# =============================================================================


class TestRebaseBackupGuard:
    """Tests for the backup-missing guard in rebase when completed tasks exist."""

    @pytest.fixture(autouse=True)
    def _set_maintainer_role(self):
        """Rebase requires maintainer role."""
        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session.get_server_role",
            return_value="maintainer",
        ):
            yield

    def test_rebase_missing_backup_with_completed_tasks_fails(self, tmp_path):
        """Rebase fails when backup spec is missing and session has completed tasks."""
        from foundry_mcp.core.autonomy.memory import AutonomyStorage
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_rebase,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        # Create a session with completed tasks and a hash that won't match any backup
        storage = AutonomyStorage(
            storage_path=workspace / "specs" / ".autonomy" / "sessions",
            workspace_path=workspace,
        )
        session = make_session(
            session_id="rebase-guard-1",
            spec_id="test-spec-001",
            status=SessionStatus.PAUSED,
            spec_structure_hash="b" * 64,  # Different from actual spec hash
            completed_task_ids=["task-1", "task-2"],
        )
        storage.save(session)
        storage.set_active_session("test-spec-001", "rebase-guard-1")

        resp = _handle_session_rebase(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )

        assert resp["success"] is False
        assert resp["data"]["error_code"] == "REBASE_BACKUP_MISSING"
        assert resp["data"]["details"]["completed_task_count"] == 2

    def test_rebase_missing_backup_with_completed_tasks_force_succeeds(self, tmp_path):
        """Force rebase proceeds even when backup is missing and completed tasks exist."""
        from foundry_mcp.core.autonomy.memory import AutonomyStorage
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_rebase,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        storage = AutonomyStorage(
            storage_path=workspace / "specs" / ".autonomy" / "sessions",
            workspace_path=workspace,
        )
        session = make_session(
            session_id="rebase-guard-2",
            spec_id="test-spec-001",
            status=SessionStatus.PAUSED,
            spec_structure_hash="b" * 64,
            completed_task_ids=["task-1"],
        )
        storage.save(session)
        storage.set_active_session("test-spec-001", "rebase-guard-2")

        resp = _handle_session_rebase(
            config=config,
            spec_id="test-spec-001",
            force=True,
            workspace=str(workspace),
        )

        _assert_success(resp)

    def test_rebase_missing_backup_no_completed_tasks_succeeds(self, tmp_path):
        """Rebase proceeds when backup is missing but no completed tasks exist."""
        from foundry_mcp.core.autonomy.memory import AutonomyStorage
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_rebase,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        storage = AutonomyStorage(
            storage_path=workspace / "specs" / ".autonomy" / "sessions",
            workspace_path=workspace,
        )
        session = make_session(
            session_id="rebase-guard-3",
            spec_id="test-spec-001",
            status=SessionStatus.PAUSED,
            spec_structure_hash="b" * 64,
            completed_task_ids=[],
        )
        storage.save(session)
        storage.set_active_session("test-spec-001", "rebase-guard-3")

        resp = _handle_session_rebase(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )

        _assert_success(resp)


# =============================================================================
# Session Heartbeat Tests
# =============================================================================


class TestSessionHeartbeat:
    """Tests for _handle_session_heartbeat."""

    def test_heartbeat_updates_timestamp(self, tmp_path):
        """Heartbeat updates last_heartbeat_at."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_heartbeat,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        resp = _handle_session_heartbeat(
            config=config,
            spec_id="test-spec-001",
            context_usage_pct=50,
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["heartbeat_at"] is not None
        assert data["context_usage_pct"] == 50

    def test_heartbeat_invalid_context_pct(self, tmp_path):
        """context_usage_pct out of range returns validation error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_heartbeat,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        resp = _handle_session_heartbeat(
            config=config,
            spec_id="test-spec-001",
            context_usage_pct=150,
            workspace=str(workspace),
        )
        assert resp["success"] is False

    def test_heartbeat_no_session(self, tmp_path):
        """Heartbeat with no active session returns error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_heartbeat,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_heartbeat(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        assert resp["success"] is False

    def test_heartbeat_with_estimated_tokens(self, tmp_path):
        """Heartbeat with estimated_tokens_used updates context."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_heartbeat,
        )
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        _handle_session_heartbeat(
            config=config,
            spec_id="test-spec-001",
            estimated_tokens_used=50000,
            workspace=str(workspace),
        )

        # Verify tokens were stored
        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        assert session.context.estimated_tokens_used == 50000


# =============================================================================
# Session Reset Tests
# =============================================================================


class TestSessionReset:
    """Tests for _handle_session_reset."""

    @pytest.fixture(autouse=True)
    def _set_maintainer_role(self):
        """Reset requires maintainer role; set it by default for existing tests."""
        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session.get_server_role",
            return_value="maintainer",
        ):
            yield

    def test_reset_failed_session(self, tmp_path):
        """Resetting a failed session deletes it (ADR escape hatch)."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_reset,
        )
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        # Manually mark as failed
        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        session.status = SessionStatus.FAILED
        session.failure_reason = FailureReason.STATE_CORRUPT
        session.counters.consecutive_errors = 5
        storage.save(session)

        # Reset requires explicit session_id per ADR
        resp = _handle_session_reset(
            config=config,
            session_id=session_id,
            reason_code="corrupt_state",
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["result"] == "deleted"
        assert data["session_id"] == session_id
        assert data["spec_id"] == "test-spec-001"
        # Verify session is actually deleted from storage
        assert storage.load(session_id) is None

    def test_reset_requires_session_id(self, tmp_path):
        """Reset without session_id returns validation error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_reset,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_reset(
            config=config,
            reason_code="testing",
            workspace=str(workspace),
        )
        assert resp["success"] is False

    def test_reset_running_rejected(self, tmp_path):
        """Resetting a running session returns INVALID_STATE_TRANSITION."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_reset,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        resp = _handle_session_reset(
            config=config,
            session_id=session_id,
            reason_code="operator_override",
            workspace=str(workspace),
        )
        assert resp["success"] is False

    def test_reset_paused_rejected(self, tmp_path):
        """Resetting a paused (not failed) session returns error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_pause,
            _handle_session_reset,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        _handle_session_pause(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        resp = _handle_session_reset(
            config=config,
            session_id=session_id,
            reason_code="operator_override",
            workspace=str(workspace),
        )
        assert resp["success"] is False

    def test_reset_requires_reason_code(self, tmp_path):
        """Reset without reason_code returns validation error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_reset,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)
        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        resp = _handle_session_reset(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
        )
        assert resp["success"] is False
        assert resp["data"]["details"]["field"] == "reason_code"

    def test_reset_rejects_invalid_reason_code(self, tmp_path):
        """Reset with invalid reason_code returns validation error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_reset,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)
        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        resp = _handle_session_reset(
            config=config,
            session_id=session_id,
            reason_code="bad_reason",
            workspace=str(workspace),
        )
        assert resp["success"] is False
        assert resp["data"]["details"]["field"] == "reason_code"

    def test_reset_denied_for_autonomy_runner(self, tmp_path):
        """Non-maintainer roles are denied session-reset."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_reset,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session.get_server_role",
            return_value="autonomy_runner",
        ):
            resp = _handle_session_reset(
                config=config,
                session_id="any-session-id",
                reason_code="operator_override",
                workspace=str(workspace),
            )
        assert resp["success"] is False
        assert resp["data"]["error_code"] == "FORBIDDEN"
        assert resp["data"]["details"]["action"] == "session-reset"
        assert resp["data"]["details"]["required_role"] == "maintainer"

    def test_reset_denied_for_observer(self, tmp_path):
        """Observer role is denied session-reset."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_reset,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session.get_server_role",
            return_value="observer",
        ):
            resp = _handle_session_reset(
                config=config,
                session_id="any-session-id",
                reason_code="operator_override",
                workspace=str(workspace),
            )
        assert resp["success"] is False
        assert resp["data"]["error_code"] == "FORBIDDEN"

    def test_reset_allowed_for_maintainer(self, tmp_path):
        """Maintainer role is allowed to reset failed sessions."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_reset,
        )
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        session.status = SessionStatus.FAILED
        session.failure_reason = FailureReason.STATE_CORRUPT
        storage.save(session)

        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_session.get_server_role",
            return_value="maintainer",
        ):
            resp = _handle_session_reset(
                config=config,
                session_id=session_id,
                reason_code="corrupt_state",
                workspace=str(workspace),
            )
        data = _assert_success(resp)
        assert data["result"] == "deleted"


# =============================================================================
# Session Resolution Tests (via _resolve_session)
# =============================================================================


class TestSessionResolution:
    """Tests for _resolve_session integration through handlers."""

    def test_resolve_by_session_id(self, tmp_path):
        """Direct session_id lookup works."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_status,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        resp = _handle_session_status(
            config=config, session_id=session_id, workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["session_id"] == session_id

    def test_resolve_by_spec_id(self, tmp_path):
        """Spec ID pointer lookup works."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_status,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )

        resp = _handle_session_status(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        data = _assert_success(resp)
        assert data["spec_id"] == "test-spec-001"


class TestSessionEventsBenchmark:
    """Performance benchmark for session-events at design-scale (WS4 200ms target)."""

    @pytest.mark.benchmark
    def test_events_query_under_200ms_at_10k_entries(self, tmp_path):
        """session-events query must stay under 200ms with 10k journal entries.

        Design target from PLAN.md WS4: 10 concurrent sessions, each polled
        at 10-30s intervals, with journal volumes up to 10,000 entries per session.
        """
        import time as time_mod

        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_events,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp_start = _handle_session_start(
            config=config, spec_id="test-spec-001", workspace=str(workspace),
        )
        session_id = _assert_success(resp_start)["session_id"]

        # Write 10,000 journal entries directly for speed.
        spec_path = workspace / "specs" / "active" / "test-spec-001.json"
        spec_data = json.loads(spec_path.read_text())
        journal = spec_data.setdefault("journal", [])
        base_time = datetime.now(timezone.utc)
        for i in range(10_000):
            ts = (base_time + timedelta(seconds=i)).isoformat().replace("+00:00", "Z")
            journal.append({
                "timestamp": ts,
                "entry_type": "session",
                "title": f"Step {i}",
                "content": f"Executed step {i}",
                "author": "autonomy",
                "metadata": {
                    "session_id": session_id,
                    "action": "step",
                },
            })
        spec_path.write_text(json.dumps(spec_data))

        # Benchmark: first page query should be under 200ms.
        start = time_mod.perf_counter()
        resp = _handle_session_events(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
            limit=50,
        )
        elapsed_ms = (time_mod.perf_counter() - start) * 1000

        data = _assert_success(resp)
        assert len(data["events"]) == 50
        assert resp["meta"]["telemetry"]["journal_entries_scanned"] == 10_000
        assert elapsed_ms < 200, (
            f"session-events query took {elapsed_ms:.1f}ms, exceeds 200ms design target"
        )


# =============================================================================
# Workspace Path Traversal Tests (T2)
# =============================================================================


class TestWorkspacePathTraversal:
    """Tests for workspace path traversal rejection in _load_spec_for_session."""

    def test_load_spec_rejects_path_traversal(self):
        """_load_spec_for_session returns None for workspace with path traversal."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _load_spec_for_session,
        )

        session = make_session(spec_id="test-spec-001")
        result = _load_spec_for_session(session, workspace="../../../etc")
        assert result is None

    def test_load_spec_rejects_dotdot_in_path(self):
        """_load_spec_for_session returns None for embedded .. in workspace."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _load_spec_for_session,
        )

        session = make_session(spec_id="test-spec-001")
        result = _load_spec_for_session(session, workspace="/tmp/safe/../../../etc")
        assert result is None


# =============================================================================
# WS4 Operator Observability Field Tests (T18)
# =============================================================================


class TestOperatorObservabilityFields:
    """Verify that session-status responses include WS4 operator observability fields.

    Tests that last_step_id, current_task_id, active_phase_progress,
    and retry_counters are populated correctly.
    """

    def test_status_includes_last_step_id_and_type(self, tmp_path):
        """last_step_id and last_step_type reflect the most recent step issued."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_status,
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

        # Manually set last_step_issued on the session to simulate step issuance
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage
        from foundry_mcp.core.autonomy.models.enums import StepType
        from foundry_mcp.core.autonomy.models.steps import LastStepIssued

        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        session.last_step_issued = LastStepIssued(
            step_id="step-abc-123",
            type=StepType.IMPLEMENT_TASK,
            task_id="task-1",
            phase_id="phase-1",
            issued_at=datetime.now(timezone.utc),
        )
        storage.save(session)

        # Query status
        status_resp = _handle_session_status(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
        )
        status_data = _assert_success(status_resp)

        assert status_data["last_step_id"] == "step-abc-123"
        assert status_data["last_step_type"] == "implement_task"

    def test_status_includes_current_task_id_from_last_step(self, tmp_path):
        """current_task_id is derived from the last issued step's task_id."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_status,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        session_id = data["session_id"]

        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage
        from foundry_mcp.core.autonomy.models.enums import StepType
        from foundry_mcp.core.autonomy.models.steps import LastStepIssued

        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        session.last_step_issued = LastStepIssued(
            step_id="step-xyz-456",
            type=StepType.IMPLEMENT_TASK,
            task_id="task-2",
            phase_id="phase-1",
            issued_at=datetime.now(timezone.utc),
        )
        storage.save(session)

        status_resp = _handle_session_status(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
        )
        status_data = _assert_success(status_resp)

        assert status_data["current_task_id"] == "task-2"

    def test_status_includes_active_phase_progress(self, tmp_path):
        """active_phase_progress reports correct completed/total/remaining counts."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_status,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        session_id = data["session_id"]

        # Set active_phase_id and mark some tasks completed
        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage

        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        session.active_phase_id = "phase-1"
        session.completed_task_ids = ["task-1"]  # 1 of 3 tasks in phase-1 completed
        storage.save(session)

        status_resp = _handle_session_status(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
        )
        status_data = _assert_success(status_resp)

        progress = status_data.get("active_phase_progress")
        assert progress is not None
        assert progress["phase_id"] == "phase-1"
        assert progress["total_tasks"] == 3  # task-1, task-2, verify-1 from make_spec_data
        assert progress["completed_tasks"] == 1
        assert progress["remaining_tasks"] == 2
        assert 0 <= progress["completion_pct"] <= 100

    def test_status_includes_retry_counters(self, tmp_path):
        """retry_counters reports fidelity review cycles and consecutive errors."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_status,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        session_id = data["session_id"]

        from foundry_mcp.tools.unified.task_handlers._helpers import _get_storage

        storage = _get_storage(config, str(workspace))
        session = storage.load(session_id)
        session.active_phase_id = "phase-1"
        session.counters.consecutive_errors = 2
        session.counters.fidelity_review_cycles_in_active_phase = 1
        storage.save(session)

        status_resp = _handle_session_status(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
        )
        status_data = _assert_success(status_resp)

        counters = status_data.get("retry_counters")
        assert counters is not None
        assert counters["consecutive_errors"] == 2
        assert counters["fidelity_review_cycles_in_active_phase"] == 1
        assert "phase-1" in counters.get("phase_retry_counts", {})

    def test_status_no_step_fields_are_null_initially(self, tmp_path):
        """Before any step is issued, last_step_id and current_task_id are null."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_status,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        data = _assert_success(resp)
        session_id = data["session_id"]

        status_resp = _handle_session_status(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
        )
        status_data = _assert_success(status_resp)

        assert status_data.get("last_step_id") is None
        assert status_data.get("last_step_type") is None
        assert status_data.get("current_task_id") is None


class TestSessionEventsCursorPagination:
    """Test cursor-based pagination for session-events."""

    def test_events_cursor_pagination(self, tmp_path):
        """Cursor pagination returns non-overlapping pages of events."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_events,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        # Start session to create events
        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        start_data = _assert_success(resp)
        session_id = start_data["session_id"]

        # Add more journal entries to create pagination
        spec_path = workspace / "specs" / "active" / "test-spec-001.json"
        spec_data = json.loads(spec_path.read_text())
        journal = spec_data.setdefault("journal", [])
        for i in range(10):
            journal.append({
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "entry_type": "session",
                "title": f"Event {i}",
                "content": f"Content {i}",
                "author": "autonomy",
                "metadata": {
                    "session_id": session_id,
                    "action": "test",
                },
            })
        spec_path.write_text(json.dumps(spec_data, indent=2))

        # First page with small limit
        page1 = _handle_session_events(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
            limit=3,
        )
        page1_data = _assert_success(page1)
        page1_events = page1_data["events"]

        # If there are enough events for pagination
        if page1["meta"].get("pagination", {}).get("has_more"):
            cursor = page1["meta"]["pagination"]["cursor"]
            assert cursor is not None

            # Second page
            page2 = _handle_session_events(
                config=config,
                session_id=session_id,
                workspace=str(workspace),
                cursor=cursor,
                limit=3,
            )
            page2_data = _assert_success(page2)
            page2_events = page2_data["events"]

            # Pages should not overlap (check by title uniqueness)
            page1_titles = {e.get("title") for e in page1_events}
            page2_titles = {e.get("title") for e in page2_events}
            assert page1_titles & page2_titles == set(), (
                f"Overlapping events between pages: {page1_titles & page2_titles}"
            )

    def test_events_invalid_cursor_returns_error(self, tmp_path):
        """Invalid cursor returns a validation error."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_events,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        start_data = _assert_success(resp)
        session_id = start_data["session_id"]

        error_resp = _handle_session_events(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
            cursor="bad-cursor-value",
        )
        assert error_resp["success"] is False

    def test_events_response_includes_telemetry(self, tmp_path):
        """session-events includes telemetry metadata."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_events,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        start_data = _assert_success(resp)
        session_id = start_data["session_id"]

        events_resp = _handle_session_events(
            config=config,
            session_id=session_id,
            workspace=str(workspace),
        )
        events_data = _assert_success(events_resp)
        telemetry = events_resp["meta"].get("telemetry", {})

        assert "duration_ms" in telemetry
        assert "journal_entries_scanned" in telemetry
        assert "session_events_returned" in telemetry


# =============================================================================
# H2: Audit Status Observability Tests
# =============================================================================


class TestAuditStatusObservability:
    """Verify meta.audit_status is attached to session lifecycle responses."""

    def test_start_includes_audit_status_ok(self, tmp_path):
        """Session-start response includes audit_status=ok on successful journal write."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )

        _assert_success(resp)
        assert resp["meta"]["audit_status"] == "ok"

    def test_pause_includes_audit_status_ok(self, tmp_path):
        """Session-pause response includes audit_status=ok on successful journal write."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_pause,
            _handle_session_start,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        start_resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        _assert_success(start_resp)

        resp = _handle_session_pause(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )

        _assert_success(resp)
        assert resp["meta"]["audit_status"] == "ok"

    def test_audit_status_failed_when_journal_write_fails(self, tmp_path):
        """audit_status=failed when journal write returns False."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _handle_session_start,
            _handle_session_pause,
        )

        workspace = _setup_workspace(tmp_path)
        config = _make_config(workspace)

        start_resp = _handle_session_start(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )
        _assert_success(start_resp)

        # Make journal write fail by removing the spec file
        spec_path = workspace / "specs" / "active" / "test-spec-001.json"
        spec_path.unlink()

        resp = _handle_session_pause(
            config=config,
            spec_id="test-spec-001",
            workspace=str(workspace),
        )

        _assert_success(resp)
        assert resp["meta"]["audit_status"] == "failed"

    def test_inject_audit_status_helper_partial(self):
        """_inject_audit_status correctly reports partial when mixed results."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _inject_audit_status,
        )

        response = {"success": True, "data": {}, "error": None, "meta": {"version": "response-v2"}}
        result = _inject_audit_status(response, True, False)
        assert result["meta"]["audit_status"] == "partial"

    def test_inject_audit_status_helper_all_ok(self):
        """_inject_audit_status correctly reports ok when all succeed."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _inject_audit_status,
        )

        response = {"success": True, "data": {}, "error": None, "meta": {"version": "response-v2"}}
        result = _inject_audit_status(response, True, True)
        assert result["meta"]["audit_status"] == "ok"

    def test_inject_audit_status_helper_all_failed(self):
        """_inject_audit_status correctly reports failed when all fail."""
        from foundry_mcp.tools.unified.task_handlers.handlers_session import (
            _inject_audit_status,
        )

        response = {"success": True, "data": {}, "error": None, "meta": {"version": "response-v2"}}
        result = _inject_audit_status(response, False, False)
        assert result["meta"]["audit_status"] == "failed"
