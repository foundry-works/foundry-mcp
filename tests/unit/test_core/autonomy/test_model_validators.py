"""Tests for cross-field model validators (M1).

Covers:
- SessionLimits.validate_heartbeat_ordering
- StepProofRecord.validate_grace_after_consumed
- AutonomousSessionState.validate_satisfied_gates_subset
"""

from datetime import datetime, timedelta, timezone

import pytest

from foundry_mcp.core.autonomy.models import (
    AutonomousSessionState,
    SessionLimits,
    SessionStatus,
    StepProofRecord,
)


class TestSessionLimitsHeartbeatOrdering:
    """SessionLimits: heartbeat_grace_minutes < heartbeat_stale_minutes."""

    def test_valid_defaults(self):
        limits = SessionLimits()
        assert limits.heartbeat_grace_minutes == 5
        assert limits.heartbeat_stale_minutes == 10

    def test_valid_custom(self):
        limits = SessionLimits(heartbeat_grace_minutes=2, heartbeat_stale_minutes=8)
        assert limits.heartbeat_grace_minutes == 2

    def test_equal_raises(self):
        with pytest.raises(ValueError, match="heartbeat_grace_minutes.*must be.*less than"):
            SessionLimits(heartbeat_grace_minutes=10, heartbeat_stale_minutes=10)

    def test_grace_greater_raises(self):
        with pytest.raises(ValueError, match="heartbeat_grace_minutes.*must be.*less than"):
            SessionLimits(heartbeat_grace_minutes=15, heartbeat_stale_minutes=10)


class TestStepProofRecordGraceAfterConsumed:
    """StepProofRecord: grace_expires_at > consumed_at."""

    def test_valid_proof(self):
        now = datetime.now(timezone.utc)
        record = StepProofRecord(
            step_proof="proof-1",
            step_id="step-1",
            payload_hash="a" * 64,
            consumed_at=now,
            grace_expires_at=now + timedelta(seconds=30),
        )
        assert record.grace_expires_at > record.consumed_at

    def test_grace_before_consumed_raises(self):
        now = datetime.now(timezone.utc)
        with pytest.raises(ValueError, match="grace_expires_at.*must be.*after.*consumed_at"):
            StepProofRecord(
                step_proof="proof-1",
                step_id="step-1",
                payload_hash="a" * 64,
                consumed_at=now,
                grace_expires_at=now - timedelta(seconds=1),
            )

    def test_grace_equal_consumed_raises(self):
        now = datetime.now(timezone.utc)
        with pytest.raises(ValueError, match="grace_expires_at.*must be.*after.*consumed_at"):
            StepProofRecord(
                step_proof="proof-1",
                step_id="step-1",
                payload_hash="a" * 64,
                consumed_at=now,
                grace_expires_at=now,
            )


class TestAutonomousSessionStateSatisfiedGatesSubset:
    """AutonomousSessionState: satisfied_gates subset of required_phase_gates."""

    def _make_session(self, required_phase_gates=None, satisfied_gates=None):
        now = datetime.now(timezone.utc)
        return AutonomousSessionState(
            **{"_schema_version": 3},
            id="sess-001",
            spec_id="spec-001",
            spec_structure_hash="a" * 64,
            status=SessionStatus.RUNNING,
            created_at=now,
            updated_at=now,
            required_phase_gates=required_phase_gates or {},
            satisfied_gates=satisfied_gates or {},
        )

    def test_valid_empty(self):
        session = self._make_session()
        assert session.satisfied_gates == {}

    def test_valid_subset(self):
        session = self._make_session(
            required_phase_gates={"phase-1": ["fidelity", "review"]},
            satisfied_gates={"phase-1": ["fidelity"]},
        )
        assert session.satisfied_gates == {"phase-1": ["fidelity"]}

    def test_unknown_phase_raises(self):
        with pytest.raises(ValueError, match="unknown phase.*phase-2"):
            self._make_session(
                required_phase_gates={"phase-1": ["fidelity"]},
                satisfied_gates={"phase-2": ["fidelity"]},
            )

    def test_extra_gate_raises(self):
        with pytest.raises(ValueError, match="not in required_phase_gates.*extra_gate"):
            self._make_session(
                required_phase_gates={"phase-1": ["fidelity"]},
                satisfied_gates={"phase-1": ["fidelity", "extra_gate"]},
            )
