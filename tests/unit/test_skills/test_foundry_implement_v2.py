"""Tests for foundry-implement-v2 skill runtime contract helpers."""

from __future__ import annotations

import copy
from typing import Any, Sequence

import pytest

from foundry_mcp.skills.foundry_implement_v2 import (
    ActionShapeAdapter,
    FoundryImplementV2Error,
    StepExecutionResult,
    determine_exit,
    dispatch_step,
    report_step_result,
    run_single_phase,
    run_startup_preflight,
)


def _success(data: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "data": data, "error": None, "meta": {"version": "response-v2"}}


def _error(error_code: str, message: str = "error") -> dict[str, Any]:
    return {
        "success": False,
        "error": message,
        "data": {"error_code": error_code, "error_type": "validation"},
        "meta": {"version": "response-v2"},
    }


def _capabilities(
    autonomy_sessions: bool,
    autonomy_fidelity_gates: bool,
    *,
    posture_profile: str | None = None,
) -> dict[str, Any]:
    autonomy_runtime: dict[str, Any] = {
        "enabled_now": {
            "autonomy_sessions": autonomy_sessions,
            "autonomy_fidelity_gates": autonomy_fidelity_gates,
        }
    }
    if posture_profile is not None:
        autonomy_runtime["posture_profile"] = posture_profile

    return _success({"runtime": {"autonomy": autonomy_runtime}})


class _QueuedInvoker:
    def __init__(self, responses: Sequence[dict[str, Any]]) -> None:
        self._responses = [copy.deepcopy(item) for item in responses]
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, tool: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool, copy.deepcopy(payload)))
        if not self._responses:
            raise AssertionError("No queued response left for invoke()")
        return self._responses.pop(0)


def _default_step_handlers() -> dict[str, Any]:
    return {
        "implement_task": lambda step: StepExecutionResult(
            outcome="success",
            note="implemented",
            files_touched=("src/example.py",),
        ),
        "execute_verification": lambda step: StepExecutionResult(
            outcome="success",
            verification_command="pytest -q",
            verification_exit_code=0,
            verification_output="1 passed",
        ),
        "run_fidelity_gate": lambda step: StepExecutionResult(
            outcome="success",
            gate_attempt_id="gate-attempt-001",
        ),
        "address_fidelity_feedback": lambda step: StepExecutionResult(
            outcome="success",
            files_touched=("src/example.py",),
        ),
        "pause": lambda step: StepExecutionResult(outcome="skipped", note="pause requested"),
        "complete_spec": lambda step: StepExecutionResult(outcome="success", note="spec complete"),
    }


class TestStartupPreflight:
    def test_detects_canonical_action_shape(self) -> None:
        invoker = _QueuedInvoker(
            [
                _success({"spec_id": "spec-001"}),
                _success({"sessions": []}),
                _capabilities(True, True),
                _success({"sessions": []}),
            ]
        )

        result = run_startup_preflight(spec_id="spec-001", invoke=invoker)

        assert result.action_shape.mode == "canonical"
        assert invoker.calls[1][1]["action"] == "session"
        assert invoker.calls[1][1]["command"] == "list"

    def test_falls_back_to_legacy_action_shape(self) -> None:
        invoker = _QueuedInvoker(
            [
                _success({"spec_id": "spec-001"}),
                _error("INVALID_FORMAT"),
                _success({"sessions": []}),
                _capabilities(True, True),
                _success({"sessions": []}),
            ]
        )

        result = run_startup_preflight(spec_id="spec-001", invoke=invoker)

        assert result.action_shape.mode == "legacy"
        # Action-shape detection probes canonical first, then legacy.
        assert invoker.calls[1][1]["action"] == "session"
        assert invoker.calls[2][1]["action"] == "session-list"
        assert invoker.calls[4][1]["action"] == "session-list"

    def test_fails_fast_when_role_preflight_denied(self) -> None:
        invoker = _QueuedInvoker(
            [
                _success({"spec_id": "spec-001"}),
                _success({"sessions": []}),
                _capabilities(True, True),
                _error("AUTHORIZATION"),
            ]
        )

        with pytest.raises(FoundryImplementV2Error) as exc_info:
            run_startup_preflight(spec_id="spec-001", invoke=invoker)

        assert exc_info.value.code == "AUTHORIZATION"

    def test_fails_fast_when_debug_posture_detected(self) -> None:
        invoker = _QueuedInvoker(
            [
                _success({"spec_id": "spec-001"}),
                _success({"sessions": []}),
                _capabilities(True, True, posture_profile="debug"),
            ]
        )

        with pytest.raises(FoundryImplementV2Error) as exc_info:
            run_startup_preflight(spec_id="spec-001", invoke=invoker)

        assert exc_info.value.code == "POSTURE_UNSUPPORTED"
        assert exc_info.value.details["posture_profile"] == "debug"

    def test_allows_supervised_posture(self) -> None:
        invoker = _QueuedInvoker(
            [
                _success({"spec_id": "spec-001"}),
                _success({"sessions": []}),
                _capabilities(True, True, posture_profile="supervised"),
                _success({"sessions": []}),
            ]
        )

        result = run_startup_preflight(spec_id="spec-001", invoke=invoker)
        assert result.spec_id == "spec-001"


class TestStepDispatch:
    def test_implement_task_handler_includes_step_proof(self) -> None:
        handlers = _default_step_handlers()
        step = {
            "step_id": "step-implement-1",
            "type": "implement_task",
            "task_id": "task-1",
            "phase_id": "phase-1",
            "step_proof": "proof-123",
        }

        result = dispatch_step(step=step, handlers=handlers)

        assert result["step_type"] == "implement_task"
        assert result["task_id"] == "task-1"
        assert result["phase_id"] == "phase-1"
        assert result["step_proof"] == "proof-123"

    def test_execute_verification_constructs_receipt(self) -> None:
        handlers = _default_step_handlers()
        step = {
            "step_id": "step-verify-1",
            "type": "execute_verification",
            "task_id": "verify-1",
            "phase_id": "phase-1",
            "step_proof": "proof-456",
        }

        result = dispatch_step(step=step, handlers=handlers)

        receipt = result["verification_receipt"]
        assert result["step_type"] == "execute_verification"
        assert result["step_proof"] == "proof-456"
        assert receipt["step_id"] == "step-verify-1"
        assert receipt["exit_code"] == 0
        assert len(receipt["command_hash"]) == 64
        assert len(receipt["output_digest"]) == 64

    def test_run_fidelity_gate_requires_gate_attempt_id(self) -> None:
        handlers = _default_step_handlers()
        step = {
            "step_id": "step-gate-1",
            "type": "run_fidelity_gate",
            "phase_id": "phase-1",
            "step_proof": "proof-gate-1",
        }

        result = dispatch_step(step=step, handlers=handlers)
        assert result["gate_attempt_id"] == "gate-attempt-001"
        assert result["step_proof"] == "proof-gate-1"

    @pytest.mark.parametrize("step_type", ["address_fidelity_feedback", "pause", "complete_spec"])
    def test_remaining_step_handlers_dispatch(self, step_type: str) -> None:
        handlers = _default_step_handlers()
        step = {"step_id": f"step-{step_type}", "type": step_type}

        result = dispatch_step(step=step, handlers=handlers)

        assert result["step_type"] == step_type
        assert result["outcome"] in {"success", "skipped"}


class TestStepReportingTransport:
    def test_uses_session_step_report_when_payload_is_simple(self) -> None:
        invoker = _QueuedInvoker([_success({"session_id": "sess-1", "status": "running"})])
        result = report_step_result(
            invoke=invoker,
            action_shape=ActionShapeAdapter("canonical"),
            session_id="sess-1",
            last_step_result={
                "step_id": "step-1",
                "step_type": "pause",
                "outcome": "skipped",
                "note": "pause",
            },
        )

        assert result["success"] is True
        _, payload = invoker.calls[0]
        assert payload["action"] == "session-step"
        assert payload["command"] == "report"

    def test_uses_session_step_next_for_extended_payload(self) -> None:
        invoker = _QueuedInvoker([_success({"session_id": "sess-1", "status": "running"})])
        result = report_step_result(
            invoke=invoker,
            action_shape=ActionShapeAdapter("canonical"),
            session_id="sess-1",
            last_step_result={
                "step_id": "step-1",
                "step_type": "execute_verification",
                "outcome": "success",
                "verification_receipt": {"step_id": "step-1"},
            },
        )

        assert result["success"] is True
        _, payload = invoker.calls[0]
        assert payload["action"] == "session-step"
        assert payload["command"] == "next"
        assert "last_step_result" in payload


class TestExitDecisions:
    @pytest.mark.parametrize(
        ("loop_signal", "success_expected"),
        [
            ("phase_complete", True),
            ("spec_complete", True),
            ("paused_needs_attention", False),
            ("failed", False),
            ("blocked_runtime", False),
        ],
    )
    def test_loop_signal_exit_mapping(self, loop_signal: str, success_expected: bool) -> None:
        response = _success(
            {
                "session_id": "sess-1",
                "status": "paused",
                "pause_reason": "phase_complete" if loop_signal == "phase_complete" else "blocked",
                "loop_signal": loop_signal,
            }
        )

        decision = determine_exit(response)

        assert decision.should_stop is True
        assert decision.success is success_expected
        assert decision.loop_signal == loop_signal

    def test_runtime_block_error_maps_to_blocked_runtime(self) -> None:
        decision = determine_exit(_error("AUTHORIZATION"))
        assert decision.should_stop is True
        assert decision.success is False
        assert decision.loop_signal == "blocked_runtime"


class TestRunSinglePhase:
    def test_end_to_end_single_phase_completion(self) -> None:
        invoker = _QueuedInvoker(
            [
                # startup preflight
                _success({"spec_id": "spec-001"}),
                _success({"sessions": []}),
                _capabilities(True, True),
                _success({"sessions": []}),
                # session start
                _success({"session_id": "sess-001", "status": "running"}),
                # initial next step
                _success(
                    {
                        "session_id": "sess-001",
                        "status": "running",
                        "state_version": 1,
                        "next_step": {
                            "step_id": "step-1",
                            "type": "implement_task",
                            "task_id": "task-1",
                            "phase_id": "phase-1",
                            "step_proof": "proof-1",
                        },
                    }
                ),
                # report response returns phase boundary
                _success(
                    {
                        "session_id": "sess-001",
                        "status": "paused",
                        "pause_reason": "phase_complete",
                        "loop_signal": "phase_complete",
                        "active_phase_id": "phase-1",
                        "next_step": None,
                    }
                ),
            ]
        )

        packet = run_single_phase(
            spec_id="spec-001",
            invoke=invoker,
            handlers=_default_step_handlers(),
        )

        assert packet.spec_id == "spec-001"
        assert packet.session_id == "sess-001"
        assert packet.loop_signal == "phase_complete"
        assert packet.final_status == "paused"
        assert packet.active_phase_id == "phase-1"
        assert packet.last_step_id == "step-1"

        # Initial step pull and report both occur through session-step routing.
        task_calls = [call for call in invoker.calls if call[0] == "task"]
        assert any(call[1].get("action") == "session-step" for call in task_calls)

    def test_phase_by_phase_loop_reaches_spec_complete(self) -> None:
        invoker = _QueuedInvoker(
            [
                # run #1 preflight
                _success({"spec_id": "spec-001"}),
                _success({"sessions": []}),
                _capabilities(True, True, posture_profile="unattended"),
                _success({"sessions": []}),
                # run #1 start + one phase
                _success({"session_id": "sess-001", "status": "running"}),
                _success(
                    {
                        "session_id": "sess-001",
                        "status": "running",
                        "state_version": 1,
                        "next_step": {
                            "step_id": "step-1",
                            "type": "implement_task",
                            "task_id": "task-1",
                            "phase_id": "phase-1",
                            "step_proof": "proof-1",
                        },
                    }
                ),
                _success(
                    {
                        "session_id": "sess-001",
                        "status": "paused",
                        "pause_reason": "phase_complete",
                        "loop_signal": "phase_complete",
                        "active_phase_id": "phase-1",
                        "next_step": None,
                    }
                ),
                # run #2 preflight
                _success({"spec_id": "spec-001"}),
                _success({"sessions": []}),
                _capabilities(True, True, posture_profile="unattended"),
                _success({"sessions": []}),
                # run #2 start conflict -> reuse compatible session
                _error("SPEC_SESSION_EXISTS"),
                _success(
                    {
                        "sessions": [
                            {
                                "session_id": "sess-001",
                                "status": "paused",
                                "spec_id": "spec-001",
                            }
                        ]
                    }
                ),
                _success(
                    {
                        "session_id": "sess-001",
                        "status": "paused",
                        "stop_conditions": {"stop_on_phase_completion": True},
                        "write_lock_enforced": True,
                    }
                ),
                _success(
                    {
                        "session_id": "sess-001",
                        "status": "paused",
                        "pause_reason": "spec_complete",
                        "loop_signal": "spec_complete",
                        "active_phase_id": "phase-2",
                        "next_step": None,
                    }
                ),
            ]
        )

        first_packet = run_single_phase(
            spec_id="spec-001",
            invoke=invoker,
            handlers=_default_step_handlers(),
        )
        second_packet = run_single_phase(
            spec_id="spec-001",
            invoke=invoker,
            handlers=_default_step_handlers(),
        )

        assert first_packet.loop_signal == "phase_complete"
        assert second_packet.loop_signal == "spec_complete"
        assert second_packet.session_id == "sess-001"

    @pytest.mark.parametrize(
        ("next_response", "expected_loop_signal", "expected_status"),
        [
            (
                _success(
                    {
                        "session_id": "sess-001",
                        "status": "paused",
                        "pause_reason": "gate_failed",
                        "loop_signal": "paused_needs_attention",
                        "recommended_actions": [{"action": "review_gate_findings", "priority": "high"}],
                    }
                ),
                "paused_needs_attention",
                "paused",
            ),
            (
                _success(
                    {
                        "session_id": "sess-001",
                        "status": "failed",
                        "loop_signal": "failed",
                        "recommended_actions": [{"action": "inspect_session_failure", "priority": "high"}],
                    }
                ),
                "failed",
                "failed",
            ),
            (
                {
                    "success": False,
                    "error": "denied",
                    "data": {
                        "error_code": "AUTHORIZATION",
                        "recommended_actions": [{"action": "fix_role_assignment", "priority": "high"}],
                    },
                    "meta": {"version": "response-v2"},
                },
                "blocked_runtime",
                "failed",
            ),
        ],
    )
    def test_returns_escalation_packet_for_non_success_signals(
        self,
        next_response: dict[str, Any],
        expected_loop_signal: str,
        expected_status: str,
    ) -> None:
        invoker = _QueuedInvoker(
            [
                # startup preflight
                _success({"spec_id": "spec-001"}),
                _success({"sessions": []}),
                _capabilities(True, True, posture_profile="unattended"),
                _success({"sessions": []}),
                # session start
                _success({"session_id": "sess-001", "status": "running"}),
                # first step pull is terminal escalation
                next_response,
            ]
        )

        packet = run_single_phase(
            spec_id="spec-001",
            invoke=invoker,
            handlers=_default_step_handlers(),
        )

        assert packet.spec_id == "spec-001"
        assert packet.session_id == "sess-001"
        assert packet.loop_signal == expected_loop_signal
        assert packet.final_status == expected_status
        assert packet.details["response_success"] is bool(next_response.get("success"))
        assert isinstance(packet.details.get("recommended_actions"), list)
