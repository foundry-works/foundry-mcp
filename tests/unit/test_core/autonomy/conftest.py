"""Shared fixtures for autonomy unit tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

from foundry_mcp.core.autonomy.memory import AutonomyStorage
from foundry_mcp.core.autonomy.models.enums import (
    GatePolicy,
    PauseReason,
    SessionStatus,
)
from foundry_mcp.core.autonomy.models.gates import PhaseGateRecord
from foundry_mcp.core.autonomy.models.session_config import (
    SessionContext,
    SessionCounters,
    SessionLimits,
    StopConditions,
)
from foundry_mcp.core.autonomy.models.state import AutonomousSessionState
from foundry_mcp.core.autonomy.models.steps import LastStepIssued


def make_session(
    *,
    session_id: str = "test-session-001",
    spec_id: str = "test-spec-001",
    status: SessionStatus = SessionStatus.RUNNING,
    spec_structure_hash: str = "a" * 64,
    active_phase_id: Optional[str] = None,
    last_step_issued: Optional[LastStepIssued] = None,
    last_issued_response: Optional[Dict[str, Any]] = None,
    counters: Optional[SessionCounters] = None,
    limits: Optional[SessionLimits] = None,
    stop_conditions: Optional[StopConditions] = None,
    context: Optional[SessionContext] = None,
    gate_policy: GatePolicy = GatePolicy.STRICT,
    completed_task_ids: Optional[list] = None,
    phase_gates: Optional[Dict[str, PhaseGateRecord]] = None,
    pause_reason: Optional[PauseReason] = None,
    created_at: Optional[datetime] = None,
    updated_at: Optional[datetime] = None,
    **kwargs: Any,
) -> AutonomousSessionState:
    """Factory for creating test session states with sensible defaults."""
    now = datetime.now(timezone.utc)
    return AutonomousSessionState(
        **{"_schema_version": 3},
        id=session_id,
        spec_id=spec_id,
        spec_structure_hash=spec_structure_hash,
        status=status,
        created_at=created_at or now,
        updated_at=updated_at or now,
        active_phase_id=active_phase_id,
        last_step_issued=last_step_issued,
        last_issued_response=last_issued_response,
        counters=counters or SessionCounters(),
        limits=limits or SessionLimits(),
        stop_conditions=stop_conditions or StopConditions(),
        context=context or SessionContext(),
        gate_policy=gate_policy,
        completed_task_ids=completed_task_ids or [],
        phase_gates=phase_gates or {},
        pause_reason=pause_reason,
        required_phase_gates=kwargs.pop("required_phase_gates", {}),
        satisfied_gates=kwargs.pop("satisfied_gates", {}),
        **kwargs,
    )


def make_spec_data(
    *,
    spec_id: str = "test-spec-001",
    phases: Optional[list] = None,
) -> Dict[str, Any]:
    """Factory for creating test spec data."""
    if phases is None:
        phases = [
            {
                "id": "phase-1",
                "title": "Phase 1",
                "sequence_index": 0,
                "tasks": [
                    {"id": "task-1", "title": "Task 1", "type": "task", "status": "pending"},
                    {"id": "task-2", "title": "Task 2", "type": "task", "status": "pending"},
                    {"id": "verify-1", "title": "Verify 1", "type": "verify", "status": "pending"},
                ],
            },
            {
                "id": "phase-2",
                "title": "Phase 2",
                "sequence_index": 1,
                "tasks": [
                    {"id": "task-3", "title": "Task 3", "type": "task", "status": "pending"},
                ],
            },
        ]
    return {"spec_id": spec_id, "phases": phases}


@pytest.fixture
def session_factory():
    """Fixture providing the make_session factory."""
    return make_session


@pytest.fixture
def spec_factory():
    """Fixture providing the make_spec_data factory."""
    return make_spec_data


@pytest.fixture
def storage(tmp_path):
    """Create an AutonomyStorage backed by a temp directory."""
    return AutonomyStorage(storage_path=tmp_path / "sessions", workspace_path=tmp_path)


@pytest.fixture
def mock_config():
    """Create a mock ServerConfig."""
    config = MagicMock()
    config.workspace_path = None
    config.specs_dir = None
    config.feature_flags = {"autonomy_sessions": True}
    return config


@pytest.fixture
def sample_session():
    """Create a simple running session for quick tests."""
    return make_session()


@pytest.fixture
def sample_spec():
    """Create a simple spec for quick tests."""
    return make_spec_data()


@pytest.fixture
def workspace_with_spec(tmp_path):
    """Create a workspace directory with a spec file for integration-style tests."""
    specs_dir = tmp_path / "specs" / "active"
    specs_dir.mkdir(parents=True)

    spec_data = make_spec_data()
    spec_path = specs_dir / "test-spec-001.json"
    spec_path.write_text(json.dumps(spec_data, indent=2))

    return tmp_path, spec_data, spec_path
