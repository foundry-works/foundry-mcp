"""Tests for session lifecycle handlers.

Covers:
- session-start: basic creation, force-end existing, idempotency key, spec not found,
  missing spec_id, gate_policy validation, journal rollback
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

from foundry_mcp.core.autonomy.models import (
    AutonomousSessionState,
    FailureReason,
    GatePolicy,
    PauseReason,
    PendingManualGateAck,
    SessionCounters,
    SessionLimits,
    SessionStatus,
    StopConditions,
)
from .conftest import make_session, make_spec_data


# =============================================================================
# Helpers
# =============================================================================

def _make_config(workspace: Path) -> MagicMock:
    """Create a mock ServerConfig that points at a workspace."""
    config = MagicMock()
    config.workspace_path = str(workspace)
    config.specs_dir = str(workspace / "specs")
    config.feature_flags = {"autonomy_sessions": True}
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
# Session Rebase Tests
# =============================================================================


class TestSessionRebase:
    """Tests for _handle_session_rebase."""

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
