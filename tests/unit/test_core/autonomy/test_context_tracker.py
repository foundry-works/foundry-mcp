"""Tests for ContextTracker.

Covers:
- is_sandbox_mode() with various env var values
- Sidecar reading: fresh, stale, missing, malformed
- Caller-reported validation: monotonicity, staleness penalty
- Pessimistic estimation: growth from last known, clamped at 100
- Tier fallthrough: sidecar > caller > estimated
- Integration: tracker wired through orchestrator triggers pause at threshold
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.autonomy.context_tracker import (
    SIDECAR_REL_PATH,
    ContextTracker,
    is_sandbox_mode,
)
from foundry_mcp.core.autonomy.models import (
    PauseReason,
    SessionContext,
    SessionLimits,
    SessionStatus,
)

from .conftest import make_session, make_spec_data


# =============================================================================
# Helpers
# =============================================================================


def _write_sidecar(workspace: Path, pct: int, ts: Optional[datetime] = None) -> Path:
    """Write a sidecar context.json file."""
    sidecar_dir = workspace / SIDECAR_REL_PATH.parent
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = workspace / SIDECAR_REL_PATH

    if ts is None:
        ts = datetime.now(timezone.utc)

    data = {
        "context_usage_pct": pct,
        "estimated_tokens_used": pct * 1000,
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    sidecar_path.write_text(json.dumps(data), encoding="utf-8")
    return sidecar_path


# =============================================================================
# is_sandbox_mode() Tests
# =============================================================================


class TestIsSandboxMode:
    """Test sandbox mode detection from env var."""

    def test_sandbox_enabled_1(self, monkeypatch):
        monkeypatch.setenv("FOUNDRY_SANDBOX", "1")
        assert is_sandbox_mode() is True

    def test_sandbox_enabled_true(self, monkeypatch):
        monkeypatch.setenv("FOUNDRY_SANDBOX", "true")
        assert is_sandbox_mode() is True

    def test_sandbox_enabled_yes(self, monkeypatch):
        monkeypatch.setenv("FOUNDRY_SANDBOX", "yes")
        assert is_sandbox_mode() is True

    def test_sandbox_enabled_TRUE_uppercase(self, monkeypatch):
        monkeypatch.setenv("FOUNDRY_SANDBOX", "TRUE")
        assert is_sandbox_mode() is True

    def test_sandbox_disabled_0(self, monkeypatch):
        monkeypatch.setenv("FOUNDRY_SANDBOX", "0")
        assert is_sandbox_mode() is False

    def test_sandbox_disabled_empty(self, monkeypatch):
        monkeypatch.setenv("FOUNDRY_SANDBOX", "")
        assert is_sandbox_mode() is False

    def test_sandbox_disabled_missing(self, monkeypatch):
        monkeypatch.delenv("FOUNDRY_SANDBOX", raising=False)
        assert is_sandbox_mode() is False

    def test_sandbox_disabled_no(self, monkeypatch):
        monkeypatch.setenv("FOUNDRY_SANDBOX", "no")
        assert is_sandbox_mode() is False


# =============================================================================
# Sidecar Reading Tests
# =============================================================================


class TestReadSidecar:
    """Tier 1: sidecar file reading and validation."""

    def test_fresh_sidecar_returns_pct(self, tmp_path):
        now = datetime.now(timezone.utc)
        _write_sidecar(tmp_path, 42, ts=now)

        tracker = ContextTracker(tmp_path)
        result = tracker._read_sidecar(now)
        assert result == 42

    def test_stale_sidecar_returns_none(self, tmp_path):
        now = datetime.now(timezone.utc)
        stale_ts = now - timedelta(minutes=3)
        _write_sidecar(tmp_path, 42, ts=stale_ts)

        tracker = ContextTracker(tmp_path)
        result = tracker._read_sidecar(now)
        assert result is None

    def test_missing_sidecar_returns_none(self, tmp_path):
        tracker = ContextTracker(tmp_path)
        now = datetime.now(timezone.utc)
        result = tracker._read_sidecar(now)
        assert result is None

    def test_malformed_json_returns_none(self, tmp_path):
        sidecar_dir = tmp_path / SIDECAR_REL_PATH.parent
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = tmp_path / SIDECAR_REL_PATH
        sidecar_path.write_text("not json {{{", encoding="utf-8")

        tracker = ContextTracker(tmp_path)
        now = datetime.now(timezone.utc)
        result = tracker._read_sidecar(now)
        assert result is None

    def test_missing_pct_field_returns_none(self, tmp_path):
        sidecar_dir = tmp_path / SIDECAR_REL_PATH.parent
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = tmp_path / SIDECAR_REL_PATH
        now = datetime.now(timezone.utc)
        data = {"timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
        sidecar_path.write_text(json.dumps(data), encoding="utf-8")

        tracker = ContextTracker(tmp_path)
        result = tracker._read_sidecar(now)
        assert result is None

    def test_pct_clamped_to_100(self, tmp_path):
        now = datetime.now(timezone.utc)
        _write_sidecar(tmp_path, 150, ts=now)

        tracker = ContextTracker(tmp_path)
        result = tracker._read_sidecar(now)
        assert result == 100

    def test_pct_clamped_to_0(self, tmp_path):
        now = datetime.now(timezone.utc)
        _write_sidecar(tmp_path, -5, ts=now)

        tracker = ContextTracker(tmp_path)
        result = tracker._read_sidecar(now)
        assert result == 0

    def test_unparseable_timestamp_returns_none(self, tmp_path):
        sidecar_dir = tmp_path / SIDECAR_REL_PATH.parent
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = tmp_path / SIDECAR_REL_PATH
        data = {
            "context_usage_pct": 50,
            "timestamp": "not-a-timestamp",
        }
        sidecar_path.write_text(json.dumps(data), encoding="utf-8")

        tracker = ContextTracker(tmp_path)
        now = datetime.now(timezone.utc)
        result = tracker._read_sidecar(now)
        assert result is None


# =============================================================================
# Validate and Harden Tests (Tier 2)
# =============================================================================


class TestValidateAndHarden:
    """Tier 2: caller-reported value validation and hardening."""

    def test_accepts_normal_increase(self, tmp_path):
        tracker = ContextTracker(tmp_path)
        session = make_session()
        session.context.last_context_report_pct = 30
        now = datetime.now(timezone.utc)

        result = tracker._validate_and_harden(session, 50, now)
        assert result == 50

    def test_rejects_decrease_keeps_higher(self, tmp_path):
        tracker = ContextTracker(tmp_path)
        session = make_session()
        session.context.last_context_report_pct = 50
        now = datetime.now(timezone.utc)

        result = tracker._validate_and_harden(session, 40, now)
        assert result == 50  # Keeps the higher value

    def test_accepts_reset_below_threshold(self, tmp_path):
        """Drops to <10% are accepted as /clear resets."""
        tracker = ContextTracker(tmp_path)
        session = make_session()
        session.context.last_context_report_pct = 70
        now = datetime.now(timezone.utc)

        result = tracker._validate_and_harden(session, 5, now)
        assert result == 5

    def test_reset_clears_consecutive_count(self, tmp_path):
        tracker = ContextTracker(tmp_path)
        session = make_session()
        session.context.last_context_report_pct = 70
        session.context.consecutive_same_reports = 3
        now = datetime.now(timezone.utc)

        tracker._validate_and_harden(session, 5, now)
        assert session.context.consecutive_same_reports == 0

    def test_staleness_penalty_applied(self, tmp_path):
        """After N identical reports, penalty is added."""
        tracker = ContextTracker(tmp_path)
        session = make_session(
            limits=SessionLimits(
                context_staleness_threshold=3,
                context_staleness_penalty_pct=5,
            ),
        )
        session.context.last_context_report_pct = 60
        session.context.consecutive_same_reports = 2  # Will become 3 (>= threshold)
        now = datetime.now(timezone.utc)

        result = tracker._validate_and_harden(session, 60, now)
        assert session.context.consecutive_same_reports == 3
        assert result == 65  # 60 + 5 penalty

    def test_staleness_penalty_capped_at_100(self, tmp_path):
        tracker = ContextTracker(tmp_path)
        session = make_session(
            limits=SessionLimits(
                context_staleness_threshold=2,
                context_staleness_penalty_pct=10,
            ),
        )
        session.context.last_context_report_pct = 95
        session.context.consecutive_same_reports = 1  # Will become 2
        now = datetime.now(timezone.utc)

        result = tracker._validate_and_harden(session, 95, now)
        assert result == 100  # Capped at 100

    def test_no_penalty_below_threshold(self, tmp_path):
        tracker = ContextTracker(tmp_path)
        session = make_session(
            limits=SessionLimits(context_staleness_threshold=5),
        )
        session.context.last_context_report_pct = 60
        session.context.consecutive_same_reports = 1  # Will become 2, below 5
        now = datetime.now(timezone.utc)

        result = tracker._validate_and_harden(session, 60, now)
        assert result == 60  # No penalty

    def test_consecutive_count_resets_on_change(self, tmp_path):
        tracker = ContextTracker(tmp_path)
        session = make_session()
        session.context.last_context_report_pct = 50
        session.context.consecutive_same_reports = 4
        now = datetime.now(timezone.utc)

        tracker._validate_and_harden(session, 55, now)
        assert session.context.consecutive_same_reports == 0

    def test_first_report_no_previous(self, tmp_path):
        tracker = ContextTracker(tmp_path)
        session = make_session()
        # No previous report
        assert session.context.last_context_report_pct is None
        now = datetime.now(timezone.utc)

        result = tracker._validate_and_harden(session, 25, now)
        assert result == 25

    def test_clamps_to_0_100(self, tmp_path):
        tracker = ContextTracker(tmp_path)
        session = make_session()
        now = datetime.now(timezone.utc)

        assert tracker._validate_and_harden(session, -10, now) == 0
        session.context.last_context_report_pct = None
        assert tracker._validate_and_harden(session, 200, now) == 100


# =============================================================================
# Estimation Tests (Tier 3)
# =============================================================================


class TestEstimateGrowth:
    """Tier 3: pessimistic estimation from last known value."""

    def test_estimates_from_base_plus_steps(self, tmp_path):
        tracker = ContextTracker(tmp_path)
        session = make_session(limits=SessionLimits(avg_pct_per_step=3))
        session.context.context_usage_pct = 30
        session.context.steps_since_last_report = 5

        result = tracker._estimate_growth(session)
        assert result == 45  # 30 + (5 * 3)

    def test_estimation_clamped_at_100(self, tmp_path):
        tracker = ContextTracker(tmp_path)
        session = make_session(limits=SessionLimits(avg_pct_per_step=10))
        session.context.context_usage_pct = 80
        session.context.steps_since_last_report = 5

        result = tracker._estimate_growth(session)
        assert result == 100  # Capped

    def test_estimation_starts_from_0_when_no_history(self, tmp_path):
        tracker = ContextTracker(tmp_path)
        session = make_session()
        session.context.context_usage_pct = 0
        session.context.steps_since_last_report = 0

        result = tracker._estimate_growth(session)
        assert result == 0

    def test_estimation_with_zero_steps(self, tmp_path):
        tracker = ContextTracker(tmp_path)
        session = make_session()
        session.context.context_usage_pct = 40
        session.context.steps_since_last_report = 0

        result = tracker._estimate_growth(session)
        assert result == 40  # No growth


# =============================================================================
# get_effective_context_pct() Tier Fallthrough Tests
# =============================================================================


class TestGetEffectiveContextPct:
    """Full tier fallthrough: sidecar > caller > estimated."""

    def test_sidecar_wins_in_sandbox_mode(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FOUNDRY_SANDBOX", "1")
        now = datetime.now(timezone.utc)
        _write_sidecar(tmp_path, 42, ts=now)

        tracker = ContextTracker(tmp_path)
        session = make_session()

        pct, source = tracker.get_effective_context_pct(session, 30, now)
        assert pct == 42
        assert source == "sidecar"

    def test_caller_used_when_no_sidecar(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FOUNDRY_SANDBOX", "1")
        now = datetime.now(timezone.utc)
        # No sidecar file

        tracker = ContextTracker(tmp_path)
        session = make_session()

        pct, source = tracker.get_effective_context_pct(session, 30, now)
        assert pct == 30
        assert source == "caller"

    def test_caller_used_when_not_sandbox(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FOUNDRY_SANDBOX", raising=False)
        now = datetime.now(timezone.utc)
        _write_sidecar(tmp_path, 42, ts=now)  # Sidecar exists but not sandbox mode

        tracker = ContextTracker(tmp_path)
        session = make_session()

        pct, source = tracker.get_effective_context_pct(session, 30, now)
        assert pct == 30
        assert source == "caller"

    def test_estimation_used_when_no_caller_report(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FOUNDRY_SANDBOX", raising=False)
        now = datetime.now(timezone.utc)

        tracker = ContextTracker(tmp_path)
        session = make_session(limits=SessionLimits(avg_pct_per_step=5))
        session.context.context_usage_pct = 20
        session.context.steps_since_last_report = 3

        pct, source = tracker.get_effective_context_pct(session, None, now)
        assert pct == 35  # 20 + (3 * 5)
        assert source == "estimated"

    def test_step_counter_reset_on_sidecar_report(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FOUNDRY_SANDBOX", "1")
        now = datetime.now(timezone.utc)
        _write_sidecar(tmp_path, 50, ts=now)

        tracker = ContextTracker(tmp_path)
        session = make_session()
        session.context.steps_since_last_report = 10

        tracker.get_effective_context_pct(session, None, now)
        assert session.context.steps_since_last_report == 0

    def test_step_counter_reset_on_caller_report(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FOUNDRY_SANDBOX", raising=False)
        now = datetime.now(timezone.utc)

        tracker = ContextTracker(tmp_path)
        session = make_session()
        session.context.steps_since_last_report = 7

        tracker.get_effective_context_pct(session, 45, now)
        assert session.context.steps_since_last_report == 0

    def test_step_counter_not_reset_on_estimation(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FOUNDRY_SANDBOX", raising=False)
        now = datetime.now(timezone.utc)

        tracker = ContextTracker(tmp_path)
        session = make_session()
        session.context.steps_since_last_report = 7

        tracker.get_effective_context_pct(session, None, now)
        assert session.context.steps_since_last_report == 7  # Unchanged

    def test_update_step_counter_increments(self, tmp_path):
        tracker = ContextTracker(tmp_path)
        session = make_session()
        assert session.context.steps_since_last_report == 0

        tracker.update_step_counter(session)
        assert session.context.steps_since_last_report == 1

        tracker.update_step_counter(session)
        assert session.context.steps_since_last_report == 2

    def test_report_tracking_updated(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FOUNDRY_SANDBOX", raising=False)
        now = datetime.now(timezone.utc)

        tracker = ContextTracker(tmp_path)
        session = make_session()

        tracker.get_effective_context_pct(session, 55, now)
        assert session.context.last_context_report_at == now
        assert session.context.last_context_report_pct == 55


# =============================================================================
# Integration: Orchestrator with Context Tracker
# =============================================================================


class TestOrchestratorContextIntegration:
    """Verify tracker is wired through orchestrator and triggers pause."""

    def _make_orchestrator_and_session(self, tmp_path, spec_data=None, **session_kwargs):
        """Create orchestrator and session with matching spec hash."""
        from foundry_mcp.core.autonomy.orchestrator import StepOrchestrator
        from foundry_mcp.core.autonomy.spec_hash import compute_spec_structure_hash

        workspace = tmp_path / "ws"
        specs_dir = workspace / "specs" / "active"
        specs_dir.mkdir(parents=True, exist_ok=True)

        data = spec_data or make_spec_data()
        spec_file = specs_dir / f"{data.get('spec_id', 'test-spec-001')}.json"
        spec_file.write_text(json.dumps(data, indent=2))

        # Compute the real hash so spec integrity check passes
        real_hash = compute_spec_structure_hash(data)

        storage = MagicMock()
        orch = StepOrchestrator(
            storage=storage,
            spec_loader=MagicMock(),
            workspace_path=workspace,
        )
        session = make_session(spec_structure_hash=real_hash, **session_kwargs)
        return orch, session

    def test_caller_reported_pct_triggers_context_pause(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FOUNDRY_SANDBOX", raising=False)
        orch, session = self._make_orchestrator_and_session(
            tmp_path,
            limits=SessionLimits(context_threshold_pct=80),
        )

        result = orch.compute_next_step(session, context_usage_pct=90)
        assert result.session.status == SessionStatus.PAUSED
        assert result.session.pause_reason == PauseReason.CONTEXT_LIMIT
        assert result.session.context.context_source == "caller"

    def test_sidecar_triggers_context_pause(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FOUNDRY_SANDBOX", "1")
        orch, session = self._make_orchestrator_and_session(
            tmp_path,
            limits=SessionLimits(context_threshold_pct=80),
        )
        now = datetime.now(timezone.utc)
        _write_sidecar(tmp_path / "ws", 90, ts=now)

        result = orch.compute_next_step(session)
        assert result.session.status == SessionStatus.PAUSED
        assert result.session.pause_reason == PauseReason.CONTEXT_LIMIT
        assert result.session.context.context_source == "sidecar"

    def test_estimated_pct_triggers_pause(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FOUNDRY_SANDBOX", raising=False)
        orch, session = self._make_orchestrator_and_session(
            tmp_path,
            limits=SessionLimits(
                context_threshold_pct=80,
                avg_pct_per_step=10,
            ),
        )
        session.context.context_usage_pct = 70
        session.context.steps_since_last_report = 2  # +1 from update_step_counter = 3 steps

        result = orch.compute_next_step(session)
        # 70 + (3 * 10) = 100, well above threshold
        assert result.session.status == SessionStatus.PAUSED
        assert result.session.pause_reason == PauseReason.CONTEXT_LIMIT
        assert result.session.context.context_source == "estimated"

    def test_low_context_no_pause(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FOUNDRY_SANDBOX", raising=False)
        orch, session = self._make_orchestrator_and_session(
            tmp_path,
            limits=SessionLimits(context_threshold_pct=85),
        )

        result = orch.compute_next_step(session, context_usage_pct=30)
        assert result.session.context.context_usage_pct == 30
        assert result.session.context.context_source == "caller"
        # Session should proceed (not paused for context)
        if result.session.status == SessionStatus.PAUSED:
            assert result.session.pause_reason != PauseReason.CONTEXT_LIMIT

    def test_step_counter_incremented_on_each_call(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FOUNDRY_SANDBOX", raising=False)
        orch, session = self._make_orchestrator_and_session(tmp_path)

        # First call with caller report
        orch.compute_next_step(session, context_usage_pct=20)
        # steps_since_last_report was reset to 0 by the report, then +1
        assert session.context.steps_since_last_report == 1

    def test_context_source_in_session_after_step(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FOUNDRY_SANDBOX", raising=False)
        orch, session = self._make_orchestrator_and_session(tmp_path)

        orch.compute_next_step(session, context_usage_pct=50)
        assert session.context.context_source == "caller"


# =============================================================================
# State Migration Tests
# =============================================================================


class TestStateMigration:
    """Verify v1 -> v2 migration adds context tracking fields."""

    def test_migrate_v1_to_v2(self):
        from foundry_mcp.core.autonomy.state_migrations import migrate_state

        v1_state = {
            "_schema_version": 1,
            "id": "test-session",
            "spec_id": "test-spec",
            "spec_structure_hash": "a" * 64,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "status": "running",
            "counters": {"tasks_completed": 0, "consecutive_errors": 0, "fidelity_review_cycles_in_active_phase": 0},
            "limits": {
                "max_tasks_per_session": 100,
                "max_consecutive_errors": 3,
                "context_threshold_pct": 85,
                "heartbeat_stale_minutes": 10,
                "heartbeat_grace_minutes": 5,
                "step_stale_minutes": 60,
                "max_fidelity_review_cycles_per_phase": 3,
            },
            "stop_conditions": {"stop_on_phase_completion": False, "auto_retry_fidelity_gate": True},
            "context": {"context_usage_pct": 0},
            "phase_gates": {},
            "completed_task_ids": [],
            "state_version": 1,
            "write_lock_enforced": True,
            "gate_policy": "strict",
        }

        migrated, warnings = migrate_state(v1_state)
        assert migrated["_schema_version"] == 2
        assert len(warnings) == 0

        # Check new context fields
        ctx = migrated["context"]
        assert ctx["context_source"] is None
        assert ctx["last_context_report_at"] is None
        assert ctx["last_context_report_pct"] is None
        assert ctx["consecutive_same_reports"] == 0
        assert ctx["steps_since_last_report"] == 0

        # Check new limits fields
        lim = migrated["limits"]
        assert lim["avg_pct_per_step"] == 3
        assert lim["context_staleness_threshold"] == 5
        assert lim["context_staleness_penalty_pct"] == 5

    def test_migrate_v1_preserves_existing_context(self):
        from foundry_mcp.core.autonomy.state_migrations import migrate_state

        v1_state = {
            "_schema_version": 1,
            "id": "test-session",
            "spec_id": "test-spec",
            "spec_structure_hash": "a" * 64,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "status": "running",
            "counters": {"tasks_completed": 5, "consecutive_errors": 0, "fidelity_review_cycles_in_active_phase": 0},
            "limits": {
                "max_tasks_per_session": 50,
                "max_consecutive_errors": 5,
                "context_threshold_pct": 90,
                "heartbeat_stale_minutes": 10,
                "heartbeat_grace_minutes": 5,
                "step_stale_minutes": 60,
                "max_fidelity_review_cycles_per_phase": 3,
            },
            "stop_conditions": {"stop_on_phase_completion": False, "auto_retry_fidelity_gate": True},
            "context": {"context_usage_pct": 45, "estimated_tokens_used": 5000},
            "phase_gates": {},
            "completed_task_ids": ["task-1", "task-2"],
            "state_version": 3,
            "write_lock_enforced": True,
            "gate_policy": "strict",
        }

        migrated, _ = migrate_state(v1_state)
        assert migrated["context"]["context_usage_pct"] == 45
        assert migrated["context"]["estimated_tokens_used"] == 5000
        assert migrated["counters"]["tasks_completed"] == 5
        assert migrated["limits"]["max_tasks_per_session"] == 50

    def test_v2_state_no_migration_needed(self):
        from foundry_mcp.core.autonomy.state_migrations import migrate_state, needs_migration

        v2_state = {
            "_schema_version": 2,
            "id": "test-session",
            "spec_id": "test-spec",
        }

        assert needs_migration(v2_state) is False
        migrated, warnings = migrate_state(v2_state)
        assert migrated["_schema_version"] == 2
        assert len(warnings) == 0

    def test_migrated_state_loads_as_model(self):
        from foundry_mcp.core.autonomy.models import AutonomousSessionState
        from foundry_mcp.core.autonomy.state_migrations import migrate_state

        v1_state = {
            "_schema_version": 1,
            "id": "test-session",
            "spec_id": "test-spec",
            "spec_structure_hash": "a" * 64,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "status": "running",
            "counters": {"tasks_completed": 0, "consecutive_errors": 0, "fidelity_review_cycles_in_active_phase": 0},
            "limits": {
                "max_tasks_per_session": 100,
                "max_consecutive_errors": 3,
                "context_threshold_pct": 85,
                "heartbeat_stale_minutes": 10,
                "heartbeat_grace_minutes": 5,
                "step_stale_minutes": 60,
                "max_fidelity_review_cycles_per_phase": 3,
            },
            "stop_conditions": {"stop_on_phase_completion": False, "auto_retry_fidelity_gate": True},
            "context": {"context_usage_pct": 0},
            "phase_gates": {},
            "completed_task_ids": [],
            "state_version": 1,
            "write_lock_enforced": True,
            "gate_policy": "strict",
        }

        migrated, _ = migrate_state(v1_state)
        session = AutonomousSessionState.model_validate(migrated)
        assert session.schema_version == 2
        assert session.context.context_source is None
        assert session.context.consecutive_same_reports == 0
        assert session.limits.avg_pct_per_step == 3
