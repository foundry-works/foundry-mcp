"""End-to-end unattended integration test for foundry-implement-v2."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pytest
from ulid import ULID

from foundry_mcp.config.server import ServerConfig
from foundry_mcp.core.authorization import set_server_role
from foundry_mcp.core.autonomy.memory import AutonomyStorage
from foundry_mcp.core.autonomy.models.enums import GateVerdict
from foundry_mcp.core.autonomy.models.gates import PendingGateEvidence
from foundry_mcp.core.autonomy.server_secret import compute_integrity_checksum
from foundry_mcp.skills.foundry_implement_v2 import StepExecutionResult, run_single_phase
from foundry_mcp.tools.unified.server import _dispatch_server_action
from foundry_mcp.tools.unified.spec import _dispatch_spec_action
from foundry_mcp.tools.unified.task import _dispatch_task_action


def _write_minimal_unattended_spec(workspace: Path, spec_id: str) -> None:
    specs_active = workspace / "specs" / "active"
    specs_active.mkdir(parents=True, exist_ok=True)
    spec_path = specs_active / f"{spec_id}.json"
    spec_payload = {
        "spec_id": spec_id,
        "title": "Unattended E2E",
        "description": "Minimal fixture for unattended phase-completion integration test",
        "phases": [
            {
                "id": "phase-1",
                "title": "Phase 1",
                "sequence_index": 0,
                "tasks": [
                    {
                        "id": "task-1",
                        "title": "Implement fixture task",
                        "type": "task",
                        "status": "pending",
                    }
                ],
            }
        ],
        "journal": [],
    }
    spec_path.write_text(json.dumps(spec_payload, indent=2), encoding="utf-8")


def _build_invoke(config: ServerConfig, call_log: list[tuple[str, dict[str, Any]]]) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    def _invoke(tool: str, payload: dict[str, Any]) -> dict[str, Any]:
        request_payload = dict(payload)
        call_log.append((tool, dict(request_payload)))
        action = request_payload.pop("action")
        if tool == "spec":
            return _dispatch_spec_action(action=action, payload=request_payload, config=config)
        if tool == "task":
            return _dispatch_task_action(action=action, payload=request_payload, config=config)
        if tool == "server":
            return _dispatch_server_action(action=action, payload=request_payload, config=config)
        raise AssertionError(f"Unsupported tool for integration invoke: {tool}")

    return _invoke


@pytest.mark.integration
def test_foundry_implement_v2_completes_single_phase_in_unattended_posture(tmp_path: Path) -> None:
    """Validates preflight -> session -> gate -> phase boundary stop in unattended posture."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    spec_id = "unattended-spec-001"
    _write_minimal_unattended_spec(workspace, spec_id)

    config = ServerConfig(
        workspace_roots=[workspace],
        specs_dir=workspace / "specs",
        feature_flags={
            "autonomy_sessions": True,
            "autonomy_fidelity_gates": True,
        },
    )
    config.apply_autonomy_posture_profile("unattended", source="test")
    config._validate_startup_configuration()
    assert config.autonomy_posture.profile == "unattended"
    assert config.autonomy_security.role == "autonomy_runner"
    assert config.autonomy_security.allow_lock_bypass is False
    assert config.autonomy_security.allow_gate_waiver is False
    assert config.autonomy_session_defaults.stop_on_phase_completion is True

    call_log: list[tuple[str, dict[str, Any]]] = []
    invoke = _build_invoke(config, call_log)

    storage = AutonomyStorage(workspace_path=workspace)

    def _run_fidelity_gate(step: dict[str, Any]) -> StepExecutionResult:
        session_id = storage.get_active_session(spec_id)
        assert session_id is not None
        session = storage.load(session_id)
        assert session is not None

        gate_attempt_id = f"gate_{ULID()}"
        now = datetime.now(timezone.utc)
        step_id = str(step["step_id"])
        phase_id = str(step["phase_id"])
        integrity_checksum = compute_integrity_checksum(
            gate_attempt_id=gate_attempt_id,
            step_id=step_id,
            phase_id=phase_id,
            verdict=GateVerdict.PASS.value,
        )
        session.pending_gate_evidence = PendingGateEvidence(
            gate_attempt_id=gate_attempt_id,
            step_id=step_id,
            phase_id=phase_id,
            verdict=GateVerdict.PASS,
            issued_at=now,
            integrity_checksum=integrity_checksum,
        )
        session.updated_at = now
        session.state_version += 1
        storage.save(session)
        return StepExecutionResult(
            outcome="success",
            gate_attempt_id=gate_attempt_id,
        )

    handlers = {
        "implement_task": lambda step: StepExecutionResult(
            outcome="success",
            note="implemented task",
            files_touched=("src/integration_fixture.txt",),
        ),
        "execute_verification": lambda step: StepExecutionResult(
            outcome="success",
            verification_command="echo verification",
            verification_exit_code=0,
            verification_output="verification passed",
        ),
        "run_fidelity_gate": _run_fidelity_gate,
        "address_fidelity_feedback": lambda step: StepExecutionResult(
            outcome="success",
            note="fidelity feedback addressed",
        ),
        "pause": lambda step: StepExecutionResult(outcome="skipped", note="pause step"),
        "complete_spec": lambda step: StepExecutionResult(
            outcome="success",
            note="spec complete",
        ),
    }

    original_role = "observer"
    try:
        set_server_role(config.autonomy_security.role)
        packet = run_single_phase(
            spec_id=spec_id,
            invoke=invoke,
            handlers=handlers,
            workspace=str(workspace),
            require_fidelity_gate=True,
        )
    finally:
        set_server_role(original_role)

    assert packet.spec_id == spec_id
    assert packet.loop_signal == "phase_complete"
    assert packet.final_status == "paused"
    assert packet.session_id is not None
    assert packet.last_step_id is not None

    # Confirm runtime-truth preflight happened via capabilities + role preflight list.
    assert any(
        tool == "server" and payload.get("action") == "capabilities"
        for tool, payload in call_log
    )
    assert any(
        tool == "task"
        and payload.get("action") == "session"
        and payload.get("command") == "list"
        for tool, payload in call_log
    )
