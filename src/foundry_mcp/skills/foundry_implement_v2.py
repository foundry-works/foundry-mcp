"""Executable contract helpers for the foundry-implement-v2 skill."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, MutableMapping, Optional

from foundry_mcp.core.autonomy.models.verification import issue_verification_receipt

LoopSignalValue = Literal[
    "phase_complete",
    "spec_complete",
    "paused_needs_attention",
    "failed",
    "blocked_runtime",
]
StepOutcomeValue = Literal["success", "failure", "skipped"]
StepTypeValue = Literal[
    "implement_task",
    "execute_verification",
    "run_fidelity_gate",
    "address_fidelity_feedback",
    "pause",
    "complete_spec",
]
ActionShapeMode = Literal["canonical", "legacy"]

_SUCCESS_SIGNALS = {"phase_complete", "spec_complete"}
_ESCALATION_SIGNALS = {"paused_needs_attention", "failed", "blocked_runtime"}
_SUPPORTED_STEP_TYPES = {
    "implement_task",
    "execute_verification",
    "run_fidelity_gate",
    "address_fidelity_feedback",
    "pause",
    "complete_spec",
}
_SHAPE_DETECTION_NON_TERMINAL_ERRORS = {"AUTHORIZATION"}
_BLOCKED_RUNTIME_ERROR_CODES = {
    "AUTHORIZATION",
    "ERROR_REQUIRED_GATE_UNSATISFIED",
    "ERROR_GATE_AUDIT_FAILURE",
    "ERROR_GATE_INTEGRITY_CHECKSUM",
    "ERROR_INVALID_GATE_EVIDENCE",
}
_EXTENDED_REPORT_FIELDS = {
    "task_id",
    "phase_id",
    "gate_attempt_id",
    "step_proof",
    "verification_receipt",
}
_TERMINAL_SESSION_STATUSES = {"completed", "failed", "ended"}


# Error class (canonical definition in foundry_mcp.core.errors.execution)
from foundry_mcp.core.errors.execution import FoundryImplementV2Error  # noqa: E402


@dataclass(frozen=True)
class ActionShapeAdapter:
    """Builds task payloads for canonical or legacy session action shapes."""

    mode: ActionShapeMode

    def session_payload(self, command: str, **params: Any) -> dict[str, Any]:
        if self.mode == "canonical":
            return {"action": "session", "command": command, **params}
        return {"action": f"session-{command}", **params}

    def session_step_payload(self, command: str, **params: Any) -> dict[str, Any]:
        if self.mode == "canonical":
            return {"action": "session-step", "command": command, **params}
        return {"action": f"session-step-{command}", **params}


@dataclass(frozen=True)
class StartupPreflightResult:
    """Outcome of startup preflight checks."""

    spec_id: str
    action_shape: ActionShapeAdapter
    capabilities_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class StepExecutionResult:
    """Normalized result returned by a step handler."""

    outcome: StepOutcomeValue
    note: Optional[str] = None
    files_touched: tuple[str, ...] = ()
    task_id: Optional[str] = None
    phase_id: Optional[str] = None
    gate_attempt_id: Optional[str] = None
    verification_command: Optional[str] = None
    verification_exit_code: Optional[int] = None
    verification_output: Optional[str] = None


@dataclass(frozen=True)
class ExitDecision:
    """Deterministic continue/stop decision from a step response."""

    should_stop: bool
    success: bool
    loop_signal: Optional[LoopSignalValue]
    summary: str


@dataclass(frozen=True)
class ExitPacket:
    """Skill output contract for terminal exits."""

    spec_id: str
    session_id: Optional[str]
    final_status: str
    loop_signal: Optional[LoopSignalValue]
    pause_reason: Optional[str]
    active_phase_id: Optional[str]
    last_step_id: Optional[str]
    summary: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "session_id": self.session_id,
            "final_status": self.final_status,
            "loop_signal": self.loop_signal,
            "pause_reason": self.pause_reason,
            "active_phase_id": self.active_phase_id,
            "last_step_id": self.last_step_id,
            "summary": self.summary,
            "details": self.details,
        }


ToolInvoker = Callable[[str, dict[str, Any]], dict[str, Any]]
StepHandlers = Mapping[str, Callable[[Mapping[str, Any]], StepExecutionResult]]


def _as_error_code(response: Mapping[str, Any]) -> str:
    data = response.get("data")
    if not isinstance(data, Mapping):
        return ""
    value = data.get("error_code")
    if not isinstance(value, str):
        return ""
    return value.strip().upper()


def _as_loop_signal(response: Mapping[str, Any]) -> Optional[LoopSignalValue]:
    data = response.get("data")
    if not isinstance(data, Mapping):
        return None
    value = data.get("loop_signal")
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in _SUCCESS_SIGNALS or normalized in _ESCALATION_SIGNALS:
        return normalized  # type: ignore[return-value]
    return None


def _as_status(response: Mapping[str, Any]) -> str:
    data = response.get("data")
    if not isinstance(data, Mapping):
        return ""
    value = data.get("status")
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _as_pause_reason(response: Mapping[str, Any]) -> Optional[str]:
    data = response.get("data")
    if not isinstance(data, Mapping):
        return None
    value = data.get("pause_reason")
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _read_runtime_posture_profile(capabilities_response: Mapping[str, Any]) -> Optional[str]:
    data = capabilities_response.get("data")
    if not isinstance(data, Mapping):
        return None

    runtime = data.get("runtime")
    if not isinstance(runtime, Mapping):
        return None

    autonomy = runtime.get("autonomy")
    if not isinstance(autonomy, Mapping):
        return None

    profile = autonomy.get("posture_profile")
    if not isinstance(profile, str):
        return None

    normalized = profile.strip().lower()
    return normalized or None


def _with_workspace(payload: MutableMapping[str, Any], workspace: Optional[str]) -> dict[str, Any]:
    if workspace:
        payload["workspace"] = workspace
    return dict(payload)


def _is_shape_probe_success(response: Mapping[str, Any]) -> bool:
    if bool(response.get("success")):
        return True
    return _as_error_code(response) in _SHAPE_DETECTION_NON_TERMINAL_ERRORS


def detect_action_shape(
    invoke_task: Callable[[dict[str, Any]], dict[str, Any]],
) -> ActionShapeAdapter:
    """Detect canonical vs. legacy task action shape via runtime probes."""
    canonical = ActionShapeAdapter("canonical")
    canonical_probe = invoke_task(canonical.session_payload("list", limit=1))
    if _is_shape_probe_success(canonical_probe):
        return canonical

    legacy = ActionShapeAdapter("legacy")
    legacy_probe = invoke_task(legacy.session_payload("list", limit=1))
    if _is_shape_probe_success(legacy_probe):
        return legacy

    raise FoundryImplementV2Error(
        "ACTION_SHAPE_UNSUPPORTED",
        "Unable to detect compatible session action shape from runtime responses.",
        remediation=(
            "Verify task(session/session-step) routing and legacy aliases are enabled. "
            "Inspect server(action=capabilities) and task action-shape adapters."
        ),
        details={
            "canonical_error_code": _as_error_code(canonical_probe) or None,
            "legacy_error_code": _as_error_code(legacy_probe) or None,
        },
    )


def run_startup_preflight(
    *,
    spec_id: str,
    invoke: ToolInvoker,
    workspace: Optional[str] = None,
    require_fidelity_gate: bool = True,
) -> StartupPreflightResult:
    """Run startup preflight using runtime responses as source of truth."""
    spec_payload = _with_workspace({"action": "find", "spec_id": spec_id}, workspace)
    spec_response = invoke("spec", spec_payload)
    if not bool(spec_response.get("success")):
        raise FoundryImplementV2Error(
            "SPEC_RESOLUTION_FAILED",
            f"Spec preflight failed for '{spec_id}'.",
            remediation="Ensure the spec exists and the workspace path is correct.",
            details={"error_code": _as_error_code(spec_response) or None},
            response=spec_response,
        )

    def _invoke_task(payload: dict[str, Any]) -> dict[str, Any]:
        return invoke("task", _with_workspace(dict(payload), workspace))

    action_shape = detect_action_shape(_invoke_task)

    capabilities_response = invoke(
        "server",
        _with_workspace({"action": "capabilities"}, workspace),
    )
    if not bool(capabilities_response.get("success")):
        raise FoundryImplementV2Error(
            "CAPABILITIES_UNAVAILABLE",
            "Failed to read server capabilities during preflight.",
            remediation="Retry server(action=capabilities) and inspect server logs.",
            details={"error_code": _as_error_code(capabilities_response) or None},
            response=capabilities_response,
        )

    posture_profile = _read_runtime_posture_profile(capabilities_response)
    if posture_profile == "debug":
        raise FoundryImplementV2Error(
            "POSTURE_UNSUPPORTED",
            "Runtime posture 'debug' is not allowed for unattended foundry-implement-v2 execution.",
            remediation=(
                "Switch to autonomy_posture.profile='unattended' for headless runs, "
                "or execute steps manually in debug posture."
            ),
            details={"posture_profile": posture_profile},
            response=capabilities_response,
        )

    role_probe = _invoke_task(action_shape.session_payload("list", limit=1))
    if not bool(role_probe.get("success")):
        error_code = _as_error_code(role_probe)
        if error_code == "AUTHORIZATION":
            raise FoundryImplementV2Error(
                error_code,
                "Role preflight rejected autonomous session operations.",
                remediation=("Use a role allowed for session operations (for example autonomy_runner)."),
                details={"recommended_call": action_shape.session_payload("list", limit=1)},
                response=role_probe,
            )
        raise FoundryImplementV2Error(
            "ROLE_PREFLIGHT_FAILED",
            "Role preflight returned an unexpected non-success response.",
            remediation="Inspect response details and resolve runtime policy errors before retry.",
            details={"error_code": error_code or None},
            response=role_probe,
        )

    warnings: tuple[str, ...] = ()
    meta = capabilities_response.get("meta")
    if isinstance(meta, Mapping):
        raw_warnings = meta.get("warnings")
        if isinstance(raw_warnings, list):
            warnings = tuple(str(item) for item in raw_warnings)

    return StartupPreflightResult(
        spec_id=spec_id,
        action_shape=action_shape,
        capabilities_warnings=warnings,
    )


def build_last_step_result(
    step: Mapping[str, Any],
    handler_result: StepExecutionResult,
) -> dict[str, Any]:
    """Build last_step_result with receipt/proof semantics enforced."""
    step_id = step.get("step_id")
    step_type = step.get("type")
    if not isinstance(step_id, str) or not step_id:
        raise FoundryImplementV2Error(
            "STEP_PAYLOAD_INVALID",
            "next_step is missing step_id.",
            remediation="Request a fresh step with session-step-next.",
        )
    if not isinstance(step_type, str) or step_type not in _SUPPORTED_STEP_TYPES:
        raise FoundryImplementV2Error(
            "STEP_TYPE_UNSUPPORTED",
            f"Unsupported step type '{step_type}'.",
            remediation="Update the skill dispatch table to handle the emitted step type.",
        )

    result: dict[str, Any] = {
        "step_id": step_id,
        "step_type": step_type,
        "outcome": handler_result.outcome,
    }
    if handler_result.note:
        result["note"] = handler_result.note

    files_touched = tuple(handler_result.files_touched or ())
    if files_touched:
        result["files_touched"] = list(files_touched)

    task_id = handler_result.task_id or step.get("task_id")
    if isinstance(task_id, str) and task_id:
        result["task_id"] = task_id

    phase_id = handler_result.phase_id or step.get("phase_id")
    if isinstance(phase_id, str) and phase_id:
        result["phase_id"] = phase_id

    step_proof = step.get("step_proof")
    if isinstance(step_proof, str) and step_proof:
        result["step_proof"] = step_proof

    if step_type == "run_fidelity_gate":
        if not handler_result.gate_attempt_id:
            raise FoundryImplementV2Error(
                "GATE_ATTEMPT_REQUIRED",
                "run_fidelity_gate handler must return gate_attempt_id.",
                remediation="Capture gate_attempt_id from review(action=fidelity-gate).",
            )
        result["gate_attempt_id"] = handler_result.gate_attempt_id

    if step_type == "execute_verification" and handler_result.outcome == "success":
        command = handler_result.verification_command
        exit_code = handler_result.verification_exit_code
        output = handler_result.verification_output
        if not isinstance(command, str) or not command:
            raise FoundryImplementV2Error(
                "VERIFICATION_RECEIPT_REQUIRED",
                "execute_verification success requires verification command for receipt construction.",
            )
        if not isinstance(exit_code, int):
            raise FoundryImplementV2Error(
                "VERIFICATION_RECEIPT_REQUIRED",
                "execute_verification success requires integer verification_exit_code.",
            )
        if not isinstance(output, str):
            raise FoundryImplementV2Error(
                "VERIFICATION_RECEIPT_REQUIRED",
                "execute_verification success requires verification_output.",
            )
        receipt = issue_verification_receipt(
            step_id=step_id,
            command=command,
            exit_code=exit_code,
            output=output,
        )
        result["verification_receipt"] = receipt.model_dump(mode="json")

    return result


def dispatch_step(
    *,
    step: Mapping[str, Any],
    handlers: StepHandlers,
) -> dict[str, Any]:
    """Dispatch next_step by type and return last_step_result payload."""
    step_type = step.get("type")
    if not isinstance(step_type, str) or step_type not in _SUPPORTED_STEP_TYPES:
        raise FoundryImplementV2Error(
            "STEP_TYPE_UNSUPPORTED",
            f"Unsupported step type '{step_type}'.",
            remediation="Add a handler for this step type in foundry-implement-v2.",
        )
    handler = handlers.get(step_type)
    if handler is None:
        raise FoundryImplementV2Error(
            "HANDLER_MISSING",
            f"No handler is registered for step type '{step_type}'.",
            remediation="Register handlers for all six required step types.",
        )
    handler_result = handler(step)
    if not isinstance(handler_result, StepExecutionResult):
        raise FoundryImplementV2Error(
            "HANDLER_RESULT_INVALID",
            f"Handler for '{step_type}' must return StepExecutionResult.",
        )
    return build_last_step_result(step, handler_result)


def _can_use_report_alias(last_step_result: Mapping[str, Any]) -> bool:
    return not any(field in last_step_result for field in _EXTENDED_REPORT_FIELDS)


def report_step_result(
    *,
    invoke: ToolInvoker,
    action_shape: ActionShapeAdapter,
    session_id: str,
    last_step_result: Mapping[str, Any],
    workspace: Optional[str] = None,
) -> dict[str, Any]:
    """Report step outcome and fetch the next orchestration response."""
    if _can_use_report_alias(last_step_result):
        payload: dict[str, Any] = action_shape.session_step_payload(
            "report",
            session_id=session_id,
            step_id=last_step_result["step_id"],
            step_type=last_step_result["step_type"],
            outcome=last_step_result["outcome"],
        )
        if "note" in last_step_result:
            payload["note"] = last_step_result["note"]
        if "files_touched" in last_step_result:
            payload["files_touched"] = last_step_result["files_touched"]
    else:
        payload = action_shape.session_step_payload(
            "next",
            session_id=session_id,
            last_step_result=dict(last_step_result),
        )
    return invoke("task", _with_workspace(payload, workspace))


def determine_exit(response: Mapping[str, Any]) -> ExitDecision:
    """Determine deterministic continue/stop behavior from a step response."""
    loop_signal = _as_loop_signal(response)
    if loop_signal in _SUCCESS_SIGNALS:
        return ExitDecision(
            should_stop=True,
            success=True,
            loop_signal=loop_signal,
            summary=f"Stopped successfully on loop_signal={loop_signal}.",
        )
    if loop_signal in _ESCALATION_SIGNALS:
        return ExitDecision(
            should_stop=True,
            success=False,
            loop_signal=loop_signal,
            summary=f"Escalated on loop_signal={loop_signal}.",
        )

    status = _as_status(response)
    if not bool(response.get("success")):
        error_code = _as_error_code(response)
        if error_code in _BLOCKED_RUNTIME_ERROR_CODES:
            return ExitDecision(
                should_stop=True,
                success=False,
                loop_signal="blocked_runtime",
                summary=f"Escalated due to runtime block ({error_code or 'UNKNOWN'}).",
            )
        return ExitDecision(
            should_stop=True,
            success=False,
            loop_signal="failed",
            summary=f"Escalated due to step/session failure ({error_code or 'UNKNOWN'}).",
        )

    if status == "completed":
        return ExitDecision(
            should_stop=True,
            success=True,
            loop_signal="spec_complete",
            summary="Session completed; stopping successfully.",
        )
    if status == "paused":
        pause_reason = (_as_pause_reason(response) or "").lower()
        if pause_reason == "phase_complete":
            return ExitDecision(
                should_stop=True,
                success=True,
                loop_signal="phase_complete",
                summary="Single phase completed successfully.",
            )
        if pause_reason == "spec_complete":
            return ExitDecision(
                should_stop=True,
                success=True,
                loop_signal="spec_complete",
                summary="Spec completed successfully.",
            )
        return ExitDecision(
            should_stop=True,
            success=False,
            loop_signal="paused_needs_attention",
            summary=f"Session paused with actionable reason '{pause_reason or 'unknown'}'.",
        )
    if status in {"failed", "ended"}:
        return ExitDecision(
            should_stop=True,
            success=False,
            loop_signal="failed",
            summary=f"Session ended with status '{status}'.",
        )

    return ExitDecision(
        should_stop=False,
        success=False,
        loop_signal=None,
        summary="Continue loop execution.",
    )


def _extract_session_id(response: Mapping[str, Any]) -> Optional[str]:
    data = response.get("data")
    if not isinstance(data, Mapping):
        return None
    session_id = data.get("session_id")
    return session_id if isinstance(session_id, str) and session_id else None


def _extract_next_step(response: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    data = response.get("data")
    if not isinstance(data, Mapping):
        return None
    next_step = data.get("next_step")
    if not isinstance(next_step, Mapping):
        return None
    return next_step


def _build_exit_packet(
    *,
    spec_id: str,
    fallback_session_id: Optional[str],
    response: Mapping[str, Any],
    decision: ExitDecision,
    last_step_id: Optional[str],
) -> ExitPacket:
    data = response.get("data")
    payload = data if isinstance(data, Mapping) else {}
    final_status = payload.get("status")
    if not isinstance(final_status, str) or not final_status:
        final_status = "failed" if not decision.success else "completed"
    pause_reason = payload.get("pause_reason")
    if not isinstance(pause_reason, str):
        pause_reason = None

    active_phase_id = payload.get("active_phase_id")
    if not isinstance(active_phase_id, str):
        active_phase_id = None

    recommended_actions = payload.get("recommended_actions")
    serialized_recommended_actions: Optional[list[dict[str, Any]]] = None
    if isinstance(recommended_actions, list):
        collected: list[dict[str, Any]] = []
        for item in recommended_actions:
            if isinstance(item, Mapping):
                collected.append(dict(item))
        if collected:
            serialized_recommended_actions = collected

    details: dict[str, Any] = {
        "response_success": bool(response.get("success")),
        "error_code": _as_error_code(response) or None,
    }
    if serialized_recommended_actions is not None:
        details["recommended_actions"] = serialized_recommended_actions

    packet = ExitPacket(
        spec_id=spec_id,
        session_id=_extract_session_id(response) or fallback_session_id,
        final_status=final_status,
        loop_signal=decision.loop_signal,
        pause_reason=pause_reason,
        active_phase_id=active_phase_id,
        last_step_id=last_step_id,
        summary=decision.summary,
        details=details,
    )
    return packet


def _collect_non_terminal_sessions(list_response: Mapping[str, Any]) -> list[dict[str, Any]]:
    data = list_response.get("data")
    if not isinstance(data, Mapping):
        return []
    sessions = data.get("sessions")
    if not isinstance(sessions, list):
        return []
    collected: list[dict[str, Any]] = []
    for item in sessions:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("status", "")).lower()
        if status in _TERMINAL_SESSION_STATUSES:
            continue
        collected.append(dict(item))
    return collected


def _start_session_with_compatibility(
    *,
    invoke: ToolInvoker,
    action_shape: ActionShapeAdapter,
    spec_id: str,
    workspace: Optional[str],
    idempotency_key: Optional[str],
) -> str:
    start_payload = action_shape.session_payload(
        "start",
        spec_id=spec_id,
        gate_policy="strict",
        stop_on_phase_completion=True,
        auto_retry_fidelity_gate=True,
        enforce_autonomy_write_lock=True,
    )
    if idempotency_key:
        start_payload["idempotency_key"] = idempotency_key

    start_response = invoke("task", _with_workspace(start_payload, workspace))
    if bool(start_response.get("success")):
        session_id = _extract_session_id(start_response)
        if session_id:
            return session_id
        raise FoundryImplementV2Error(
            "SESSION_START_INVALID",
            "Session start succeeded but no session_id was returned.",
        )

    if _as_error_code(start_response) != "SPEC_SESSION_EXISTS":
        raise FoundryImplementV2Error(
            "SESSION_START_FAILED",
            "Failed to start autonomous session.",
            remediation="Resolve start error and retry.",
            details={"error_code": _as_error_code(start_response) or None},
            response=start_response,
        )

    list_payload = action_shape.session_payload("list", spec_id=spec_id, limit=5)
    list_response = invoke("task", _with_workspace(list_payload, workspace))
    if not bool(list_response.get("success")):
        raise FoundryImplementV2Error(
            "SESSION_REUSE_FAILED",
            "Active session exists, but compatible session lookup failed.",
            remediation="Run task(action=session, command=list, spec_id=...) and resolve conflicts.",
            response=list_response,
        )

    sessions = _collect_non_terminal_sessions(list_response)
    if len(sessions) != 1:
        raise FoundryImplementV2Error(
            "SESSION_REUSE_AMBIGUOUS",
            "Active sessions are ambiguous; refusing automatic reuse.",
            remediation="End or pause incompatible sessions, then retry.",
            details={"non_terminal_sessions": sessions},
            response=list_response,
        )

    candidate_id = sessions[0].get("session_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise FoundryImplementV2Error(
            "SESSION_REUSE_INVALID",
            "Session list returned a candidate without session_id.",
            response=list_response,
        )

    status_payload = action_shape.session_payload("status", session_id=candidate_id)
    status_response = invoke("task", _with_workspace(status_payload, workspace))
    if not bool(status_response.get("success")):
        raise FoundryImplementV2Error(
            "SESSION_REUSE_FAILED",
            "Failed to validate existing session compatibility.",
            response=status_response,
        )

    status_data = status_response.get("data")
    if not isinstance(status_data, Mapping):
        raise FoundryImplementV2Error(
            "SESSION_REUSE_INVALID",
            "Session status response is missing data payload.",
            response=status_response,
        )
    stop_conditions = status_data.get("stop_conditions")
    write_lock_enforced = bool(status_data.get("write_lock_enforced"))
    if not isinstance(stop_conditions, Mapping):
        stop_conditions = {}
    # Reject sessions paused at fidelity cycle limit — requires human review
    pause_reason = status_data.get("pause_reason")
    if isinstance(pause_reason, str) and pause_reason.strip().lower() == "fidelity_cycle_limit":
        raise FoundryImplementV2Error(
            "SESSION_REUSE_PAUSED_GATE_LIMIT",
            "Cannot reuse session paused due to fidelity cycle limit. "
            "The phase exhausted its fidelity review budget and needs human review.",
            remediation="Review gate evidence, then end the session and retry with a new one.",
            details={
                "session_id": candidate_id,
                "pause_reason": pause_reason,
                "fidelity_review_cycles": status_data.get("counters", {}).get("fidelity_review_cycles_in_active_phase"),
            },
            response=status_response,
        )

    compatible = bool(stop_conditions.get("stop_on_phase_completion")) and write_lock_enforced
    if not compatible:
        raise FoundryImplementV2Error(
            "SESSION_REUSE_INCOMPATIBLE",
            "Existing session is incompatible with one-phase unattended semantics.",
            remediation="End the existing session explicitly, then rerun foundry-implement-v2.",
            details={
                "session_id": candidate_id,
                "write_lock_enforced": write_lock_enforced,
                "stop_on_phase_completion": bool(stop_conditions.get("stop_on_phase_completion")),
            },
            response=status_response,
        )
    return candidate_id


def run_single_phase(
    *,
    spec_id: str,
    invoke: ToolInvoker,
    handlers: StepHandlers,
    workspace: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    require_fidelity_gate: bool = True,
    max_iterations: int = 200,
) -> ExitPacket:
    """Run one deterministic phase loop for foundry-implement-v2."""
    preflight = run_startup_preflight(
        spec_id=spec_id,
        invoke=invoke,
        workspace=workspace,
        require_fidelity_gate=require_fidelity_gate,
    )

    session_id = _start_session_with_compatibility(
        invoke=invoke,
        action_shape=preflight.action_shape,
        spec_id=spec_id,
        workspace=workspace,
        idempotency_key=idempotency_key,
    )

    next_response = invoke(
        "task",
        _with_workspace(
            preflight.action_shape.session_step_payload("next", session_id=session_id),
            workspace,
        ),
    )

    last_step_id: Optional[str] = None

    for _ in range(max_iterations):
        decision = determine_exit(next_response)
        if decision.should_stop:
            return _build_exit_packet(
                spec_id=spec_id,
                fallback_session_id=session_id,
                response=next_response,
                decision=decision,
                last_step_id=last_step_id,
            )

        step = _extract_next_step(next_response)
        if step is None:
            raise FoundryImplementV2Error(
                "STEP_MISSING",
                "Loop cannot continue because next_step is missing.",
                remediation="Use session-step-replay or inspect session status for drift.",
                response=dict(next_response),
            )
        step_id = step.get("step_id")
        if isinstance(step_id, str) and step_id:
            last_step_id = step_id

        last_step_result = dispatch_step(step=step, handlers=handlers)
        next_response = report_step_result(
            invoke=invoke,
            action_shape=preflight.action_shape,
            session_id=session_id,
            last_step_result=last_step_result,
            workspace=workspace,
        )

    raise FoundryImplementV2Error(
        "LOOP_LIMIT_EXCEEDED",
        f"Exceeded max_iterations={max_iterations} without terminal loop signal.",
        remediation="Inspect session status and increase limit only after triaging step churn.",
        details={"session_id": session_id, "max_iterations": max_iterations},
    )
