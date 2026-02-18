"""Session-step action handlers: next, report, replay.

This module provides handlers for session-step actions that drive
autonomous execution forward using the StepOrchestrator.

Phase B (current):
- next: Computes next step via orchestrator with replay-safe semantics
- report: Reports step outcome (Phase B)
- replay: Returns cached last response for safe retry (Phase B)

All session actions require an active autonomous session.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from foundry_mcp.config.server import ServerConfig
from foundry_mcp.core.autonomy.models.enums import (
    PhaseGateStatus,
    SessionStatus,
    StepOutcome,
    StepType,
)
from foundry_mcp.core.autonomy.models.responses import (
    NextStep,
    SessionStepResponseData,
)
from foundry_mcp.core.autonomy.models.steps import LastStepResult
from foundry_mcp.core.autonomy.orchestrator import (
    OrchestrationResult,
    StepOrchestrator,
    ERROR_STEP_RESULT_REQUIRED,
    ERROR_STEP_MISMATCH,
    ERROR_STEP_PROOF_MISSING,
    ERROR_STEP_PROOF_MISMATCH,
    ERROR_STEP_PROOF_CONFLICT,
    ERROR_STEP_PROOF_EXPIRED,
    ERROR_SPEC_REBASE_REQUIRED,
    ERROR_HEARTBEAT_STALE,
    ERROR_STEP_STALE,
    ERROR_ALL_TASKS_BLOCKED,
    ERROR_SESSION_UNRECOVERABLE,
    ERROR_INVALID_GATE_EVIDENCE,
    ERROR_GATE_BLOCKED,
    ERROR_REQUIRED_GATE_UNSATISFIED,
    ERROR_VERIFICATION_RECEIPT_MISSING,
    ERROR_VERIFICATION_RECEIPT_INVALID,
    ERROR_GATE_INTEGRITY_CHECKSUM,
    ERROR_GATE_AUDIT_FAILURE,
)
from foundry_mcp.core.responses.types import (
    ErrorCode,
    ErrorType,
)
from foundry_mcp.core.responses.builders import (
    error_response,
    success_response,
)
from foundry_mcp.core.spec import load_spec

from foundry_mcp.tools.unified.param_schema import AtLeastOne, Str, validate_payload
from foundry_mcp.tools.unified.task_handlers._helpers import (
    _get_storage,
    _request_id,
    _resolve_session,
    _session_not_found_response,
    _validate_context_usage_pct,
    attach_loop_metadata,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Declarative validation schemas
# ---------------------------------------------------------------------------

_STEP_REPORT_SCHEMA = {
    "spec_id": Str(),
    "session_id": Str(),
    "step_id": Str(required=True),
    "step_type": Str(required=True),
    "outcome": Str(required=True),
}


def _hash_last_step_result_payload(last_step_result: LastStepResult) -> str:
    """Create stable payload hash for step-proof replay detection."""
    payload = last_step_result.model_dump(mode="json", by_alias=True, exclude_none=True)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _restore_cached_proof_response(
    cached_response: Dict[str, Any],
    request_id: str,
) -> dict:
    """Restore cached response envelope for idempotent proof replay."""
    restored = json.loads(json.dumps(cached_response, default=str))
    meta = restored.get("meta")
    if isinstance(meta, dict):
        meta["request_id"] = request_id
    else:
        restored["meta"] = {"version": "response-v2", "request_id": request_id}
    return restored


def _persist_step_proof_response(
    *,
    storage: Any,
    session_id: str,
    step_proof: Optional[str],
    step_id: Optional[str],
    response: dict,
) -> None:
    """Persist response envelope for one-time proof token replay."""
    if not step_proof or not step_id:
        return
    try:
        storage.update_proof_record_response(
            session_id,
            step_proof,
            step_id=step_id,
            response=response,
        )
    except (OSError, ValueError, TimeoutError) as exc:
        logger.warning(
            "Failed to persist step-proof replay response for session %s step %s: %s",
            session_id,
            step_id,
            exc,
        )


_attach_loop_fields = attach_loop_metadata  # local alias for backward compat


def _map_orchestrator_error_to_response(
    error_code: Optional[str],
    error_message: str,
    request_id: str,
    session_id: Optional[str] = None,
    state_version: Optional[int] = None,
    missing_required_gates: Optional[List[str]] = None,
    gate_block: Optional[Dict[str, Any]] = None,
) -> dict:
    """Map orchestrator error codes to response format."""
    # Map orchestrator error codes to ErrorCode enum
    error_code_map = {
        ERROR_STEP_RESULT_REQUIRED: ErrorCode.VALIDATION_ERROR,
        ERROR_STEP_MISMATCH: ErrorCode.CONFLICT,
        ERROR_STEP_PROOF_MISSING: ErrorCode.MISSING_REQUIRED,
        ERROR_STEP_PROOF_MISMATCH: ErrorCode.CONFLICT,
        ERROR_STEP_PROOF_CONFLICT: ErrorCode.CONFLICT,
        ERROR_STEP_PROOF_EXPIRED: ErrorCode.CONFLICT,
        ERROR_SPEC_REBASE_REQUIRED: ErrorCode.CONFLICT,
        ERROR_HEARTBEAT_STALE: ErrorCode.UNAVAILABLE,
        ERROR_STEP_STALE: ErrorCode.UNAVAILABLE,
        ERROR_ALL_TASKS_BLOCKED: ErrorCode.RESOURCE_BUSY,
        ERROR_SESSION_UNRECOVERABLE: ErrorCode.INTERNAL_ERROR,
        ERROR_INVALID_GATE_EVIDENCE: ErrorCode.VALIDATION_ERROR,
        ERROR_GATE_BLOCKED: ErrorCode.VALIDATION_ERROR,
        ERROR_REQUIRED_GATE_UNSATISFIED: ErrorCode.FORBIDDEN,
        ERROR_VERIFICATION_RECEIPT_MISSING: ErrorCode.VALIDATION_ERROR,
        ERROR_VERIFICATION_RECEIPT_INVALID: ErrorCode.VALIDATION_ERROR,
        ERROR_GATE_INTEGRITY_CHECKSUM: ErrorCode.VALIDATION_ERROR,
        ERROR_GATE_AUDIT_FAILURE: ErrorCode.CONFLICT,
    }

    # Map to error types
    error_type_map = {
        ERROR_STEP_RESULT_REQUIRED: ErrorType.VALIDATION,
        ERROR_STEP_MISMATCH: ErrorType.CONFLICT,
        ERROR_STEP_PROOF_MISSING: ErrorType.VALIDATION,
        ERROR_STEP_PROOF_MISMATCH: ErrorType.CONFLICT,
        ERROR_STEP_PROOF_CONFLICT: ErrorType.CONFLICT,
        ERROR_STEP_PROOF_EXPIRED: ErrorType.CONFLICT,
        ERROR_SPEC_REBASE_REQUIRED: ErrorType.CONFLICT,
        ERROR_HEARTBEAT_STALE: ErrorType.UNAVAILABLE,
        ERROR_STEP_STALE: ErrorType.UNAVAILABLE,
        ERROR_ALL_TASKS_BLOCKED: ErrorType.UNAVAILABLE,
        ERROR_SESSION_UNRECOVERABLE: ErrorType.INTERNAL,
        ERROR_INVALID_GATE_EVIDENCE: ErrorType.VALIDATION,
        ERROR_GATE_BLOCKED: ErrorType.VALIDATION,
        ERROR_REQUIRED_GATE_UNSATISFIED: ErrorType.AUTHORIZATION,
        ERROR_VERIFICATION_RECEIPT_MISSING: ErrorType.VALIDATION,
        ERROR_VERIFICATION_RECEIPT_INVALID: ErrorType.VALIDATION,
        ERROR_GATE_INTEGRITY_CHECKSUM: ErrorType.VALIDATION,
        ERROR_GATE_AUDIT_FAILURE: ErrorType.CONFLICT,
    }

    mapped_code = error_code_map.get(error_code or "", ErrorCode.INTERNAL_ERROR)
    mapped_type = error_type_map.get(error_code or "", ErrorType.INTERNAL)

    details: Dict[str, Any] = {
        "error_code": error_code,
        "session_id": session_id,
    }
    if state_version is not None:
        details["state_version"] = state_version

    # Add remediation hints based on error type
    if error_code == ERROR_STEP_RESULT_REQUIRED:
        details["remediation"] = "Provide last_step_result with the outcome of the previous step"
    elif error_code == ERROR_STEP_MISMATCH:
        details["remediation"] = "Ensure step_id and step_type match the last issued step"
    elif error_code == ERROR_STEP_PROOF_MISSING:
        details["remediation"] = (
            "Include step_proof from the issued step in last_step_result for one-time proof validation."
        )
    elif error_code == ERROR_STEP_PROOF_MISMATCH:
        details["remediation"] = (
            "Use the latest issued step and matching step_proof token, or replay with the original proof payload."
        )
    elif error_code == ERROR_STEP_PROOF_CONFLICT:
        details["remediation"] = (
            "Resubmit exactly the same payload for this step_proof token. Different payloads are rejected."
        )
    elif error_code == ERROR_STEP_PROOF_EXPIRED:
        details["remediation"] = (
            "The proof replay grace window has expired. Request a fresh step via session-step-next."
        )
    elif error_code == ERROR_SPEC_REBASE_REQUIRED:
        details["remediation"] = "Use session-rebase to reconcile spec structure changes"
    elif error_code in (ERROR_HEARTBEAT_STALE, ERROR_STEP_STALE):
        details["remediation"] = "Session may need to be reset or heartbeat updated"
    elif error_code == ERROR_ALL_TASKS_BLOCKED:
        details["remediation"] = "Resolve task blockers to continue execution"
    elif error_code == ERROR_INVALID_GATE_EVIDENCE:
        details["remediation"] = "Provide valid gate evidence via fidelity-gate action"
    elif error_code == ERROR_VERIFICATION_RECEIPT_MISSING:
        details["remediation"] = (
            "Include verification_receipt in last_step_result when execute_verification "
            "reports outcome='success'."
        )
    elif error_code == ERROR_VERIFICATION_RECEIPT_INVALID:
        details["remediation"] = (
            "Use the server-issued verification receipt for the current step and "
            "resubmit with matching step_id and command hash."
        )
    elif error_code == ERROR_GATE_INTEGRITY_CHECKSUM:
        details["remediation"] = (
            "Re-run the gate step to obtain fresh evidence before reporting success."
        )
    elif error_code == ERROR_GATE_AUDIT_FAILURE:
        details["remediation"] = (
            "Gate audit failed due to evidence inconsistency. Re-run required gate "
            "checks or request maintainer intervention."
        )
    elif error_code == ERROR_GATE_BLOCKED:
        details["blocked_by_gate"] = True
        if missing_required_gates:
            details["missing_required_gates"] = missing_required_gates
        details["remediation"] = "Complete required phase gates before proceeding or request gate waiver from maintainer"
    elif error_code == ERROR_REQUIRED_GATE_UNSATISFIED:
        details["blocked_by_gate"] = True
        # Include gate_block details if available
        if gate_block:
            details["phase_id"] = gate_block.get("phase_id")
            details["gate_type"] = gate_block.get("gate_type")
            details["blocking_reason"] = gate_block.get("blocking_reason")
            recovery_action = gate_block.get("recovery_action")
            if recovery_action:
                details["recovery_action"] = recovery_action
                details["remediation"] = recovery_action.get("description", "Use gate-waiver to unblock")
            else:
                details["remediation"] = "Complete required phase gates before proceeding or request gate waiver from maintainer"
        else:
            details["remediation"] = "Complete required phase gates before proceeding or request gate waiver from maintainer"

    return _attach_loop_fields(
        asdict(
            error_response(
                error_message,
                error_code=mapped_code,
                error_type=mapped_type,
                request_id=request_id,
                details=details,
            )
        )
    )


def _build_next_step_response(
    result: OrchestrationResult,
    request_id: str,
) -> dict:
    """Build the response for session-step-next.

    Response format matches ADR:
    {
        "success": true,
        "data": {
            "session_id": "...",
            "status": "running" | "paused" | "completed" | "failed",
            "state_version": 7,
            "next_step": { ... } | null,
            "required_phase_gates": [...],
            "satisfied_gates": [...],
            "missing_required_gates": [...]
        },
        "error": null,
        "meta": {"version": "response-v2", "request_id": "..."}
    }
    """
    session = result.session

    # Compute gate invariant fields for observability
    required_phase_gates: list[str] = []
    satisfied_gates: list[str] = []
    missing_required_gates: list[str] = []

    for phase_id, gate_record in session.phase_gates.items():
        if gate_record.required:
            required_phase_gates.append(phase_id)
            if gate_record.status in (PhaseGateStatus.PASSED, PhaseGateStatus.WAIVED):
                satisfied_gates.append(phase_id)
            elif gate_record.status in (PhaseGateStatus.PENDING, PhaseGateStatus.FAILED):
                missing_required_gates.append(phase_id)

    # Build response data — loop_signal and recommended_actions are
    # attached by attach_loop_metadata() as a post-processing step.
    response_data = SessionStepResponseData(
        session_id=session.id,
        status=session.status,
        state_version=session.state_version,
        next_step=result.next_step,  # Pass the NextStep model directly
        required_phase_gates=required_phase_gates if required_phase_gates else None,
        satisfied_gates=satisfied_gates if satisfied_gates else None,
        missing_required_gates=missing_required_gates if missing_required_gates else None,
    )

    data = response_data.model_dump(mode="json", by_alias=True)
    # Inject pause_reason so attach_loop_metadata can derive loop_signal
    # from the response dict alone (SessionStepResponseData omits it).
    if session.pause_reason is not None:
        data["pause_reason"] = session.pause_reason.value

    return _attach_loop_fields(
        asdict(
            success_response(
                data=data,
                request_id=request_id,
            )
        )
    )


def _handle_session_step_next(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    session_id: Optional[str] = None,
    workspace: Optional[str] = None,
    last_step_result: Optional[Dict[str, Any]] = None,
    context_usage_pct: Optional[int] = None,
    heartbeat: Optional[bool] = None,
    **payload: Any,
) -> dict:
    """Handle session-step-next action.

    Gets the next step to execute in an autonomous session.

    This handler integrates with the StepOrchestrator to:
    - Validate feedback (last_step_result) on non-initial calls
    - Check for replay scenarios (return cached response)
    - Enforce pause guards and staleness checks
    - Determine the next step based on 18-step orchestration rules

    Args:
        config: Server configuration
        spec_id: Spec ID of the session
        workspace: Workspace path
        last_step_result: Result of previous step (required on non-initial calls)
            - step_id: ID of the completed step
            - step_type: Type of the step (implement_task, execute_verification, etc.)
            - task_id: Task ID if step involved a task
            - phase_id: Phase ID if step involved a phase
            - outcome: "success", "failure", or "skipped"
            - note: Optional note about the outcome
            - files_touched: Optional list of files modified
            - gate_attempt_id: Gate attempt ID for gate steps
            - step_proof: Optional one-time proof token for proof-enforced steps
            - verification_receipt: Optional receipt object for execute_verification success
        context_usage_pct: Caller-reported context usage percentage (0-100)
        heartbeat: If true, update heartbeat timestamp
        **payload: Additional parameters

    Returns:
        Response dict with:
        - session_id: Session identifier
        - status: Current session status
        - state_version: Monotonic state version
        - next_step: Next step object or null if terminal/paused
    """
    request_id = _request_id()

    # Validate context_usage_pct
    pct_err = _validate_context_usage_pct(context_usage_pct, "session-step-next", request_id)
    if pct_err:
        return _attach_loop_fields(pct_err)

    storage = _get_storage(config, workspace, request_id=request_id)
    if isinstance(storage, dict):
        return storage

    session, err = _resolve_session(storage, "session-step-next", request_id, session_id, spec_id)
    if err:
        return _attach_loop_fields(err)

    # Parse last_step_result if provided
    parsed_last_step_result: Optional[LastStepResult] = None
    consumed_step_proof: Optional[str] = None
    consumed_step_id: Optional[str] = None
    if last_step_result:
        try:
            # Convert outcome string to enum
            outcome_str = last_step_result.get("outcome", "success")
            outcome = StepOutcome(outcome_str)

            # step_type is required in LastStepResult
            step_type_str = last_step_result.get("step_type")
            if not step_type_str:
                return _attach_loop_fields(
                    asdict(
                        error_response(
                            "step_type is required in last_step_result",
                            error_code=ErrorCode.VALIDATION_ERROR,
                            error_type=ErrorType.VALIDATION,
                            request_id=request_id,
                            details={
                                "action": "session-step-next",
                                "field": "last_step_result.step_type",
                                "hint": "Provide step_type matching the last issued step",
                            },
                        )
                    )
                )
            step_type = StepType(step_type_str)

            parsed_last_step_result = LastStepResult(
                step_id=last_step_result.get("step_id", ""),
                step_type=step_type,
                task_id=last_step_result.get("task_id"),
                phase_id=last_step_result.get("phase_id"),
                outcome=outcome,
                note=last_step_result.get("note"),
                files_touched=last_step_result.get("files_touched"),
                gate_attempt_id=last_step_result.get("gate_attempt_id"),
                step_proof=last_step_result.get("step_proof"),
                verification_receipt=last_step_result.get("verification_receipt"),
            )
        except (ValueError, TypeError, ValidationError) as e:
            logger.warning("Failed to parse last_step_result: %s", e)
            return _attach_loop_fields(
                asdict(
                    error_response(
                        f"Invalid last_step_result format: {e}",
                        error_code=ErrorCode.VALIDATION_ERROR,
                        error_type=ErrorType.VALIDATION,
                        request_id=request_id,
                        details={
                            "action": "session-step-next",
                            "field": "last_step_result",
                            "hint": "Ensure outcome is one of: success, failure, skipped",
                        },
                    )
                )
            )

    if parsed_last_step_result is not None:
        expected_step_proof = (
            session.last_step_issued.step_proof
            if session.last_step_issued is not None
            else None
        )
        provided_step_proof = parsed_last_step_result.step_proof
        payload_hash = _hash_last_step_result_payload(parsed_last_step_result)

        # If a proof was provided for an older step, return cached response
        # (or deterministic conflict/expiry) before invoking orchestrator.
        if provided_step_proof and provided_step_proof != expected_step_proof:
            replay_record = storage.get_proof_record(
                session.id,
                provided_step_proof,
                include_expired=True,
            )
            if replay_record:
                if replay_record.payload_hash != payload_hash:
                    return _map_orchestrator_error_to_response(
                        error_code=ERROR_STEP_PROOF_CONFLICT,
                        error_message=(
                            "step_proof was already consumed with a different payload for this session."
                        ),
                        request_id=request_id,
                        session_id=session.id,
                        state_version=session.state_version,
                    )
                if replay_record.grace_expires_at <= datetime.now(timezone.utc):
                    return _map_orchestrator_error_to_response(
                        error_code=ERROR_STEP_PROOF_EXPIRED,
                        error_message=(
                            "step_proof replay window has expired; request a fresh step."
                        ),
                        request_id=request_id,
                        session_id=session.id,
                        state_version=session.state_version,
                    )
                if isinstance(replay_record.cached_response, dict):
                    return _attach_loop_fields(
                        _restore_cached_proof_response(
                            replay_record.cached_response,
                            request_id,
                        )
                    )
                return _map_orchestrator_error_to_response(
                    error_code=ERROR_STEP_PROOF_EXPIRED,
                    error_message=(
                        "step_proof was consumed but cached response is unavailable; request a fresh step."
                    ),
                    request_id=request_id,
                    session_id=session.id,
                    state_version=session.state_version,
                )

        if expected_step_proof:
            if not provided_step_proof:
                return _map_orchestrator_error_to_response(
                    error_code=ERROR_STEP_PROOF_MISSING,
                    error_message=(
                        "step_proof is required for this step. Include the one-time step_proof token "
                        "from the issued step in last_step_result."
                    ),
                    request_id=request_id,
                    session_id=session.id,
                    state_version=session.state_version,
                )
            if provided_step_proof != expected_step_proof:
                return _map_orchestrator_error_to_response(
                    error_code=ERROR_STEP_PROOF_MISMATCH,
                    error_message="step_proof does not match the currently issued step.",
                    request_id=request_id,
                    session_id=session.id,
                    state_version=session.state_version,
                )

            consumed, existing_record, proof_error = storage.consume_proof_with_lock(
                session.id,
                provided_step_proof,
                payload_hash,
                step_id=parsed_last_step_result.step_id,
            )
            if not consumed:
                mapped_code = (
                    ERROR_STEP_PROOF_EXPIRED
                    if proof_error == "PROOF_EXPIRED"
                    else ERROR_STEP_PROOF_CONFLICT
                )
                return _map_orchestrator_error_to_response(
                    error_code=mapped_code,
                    error_message=(
                        "step_proof replay window has expired; request a fresh step."
                        if mapped_code == ERROR_STEP_PROOF_EXPIRED
                        else "step_proof was already consumed with a different payload."
                    ),
                    request_id=request_id,
                    session_id=session.id,
                    state_version=session.state_version,
                )

            if existing_record is not None:
                if isinstance(existing_record.cached_response, dict):
                    return _attach_loop_fields(
                        _restore_cached_proof_response(
                            existing_record.cached_response,
                            request_id,
                        )
                    )
                return _map_orchestrator_error_to_response(
                    error_code=ERROR_STEP_PROOF_EXPIRED,
                    error_message=(
                        "step_proof already consumed but cached response is unavailable; request a fresh step."
                    ),
                    request_id=request_id,
                    session_id=session.id,
                    state_version=session.state_version,
                )

            consumed_step_proof = provided_step_proof
            consumed_step_id = parsed_last_step_result.step_id
    # Prepare heartbeat timestamp if requested
    heartbeat_at = None
    if heartbeat:
        heartbeat_at = datetime.now(timezone.utc)

    # Create orchestrator and compute next step
    orchestrator = StepOrchestrator(
        storage=storage,
        spec_loader=load_spec,
        workspace_path=Path(workspace) if workspace else Path.cwd(),
    )

    result = orchestrator.compute_next_step(
        session=session,
        last_step_result=parsed_last_step_result,
        context_usage_pct=context_usage_pct,
        heartbeat_at=heartbeat_at,
    )

    final_response: dict

    # Handle replay scenario — return cached SessionStepResponseData in envelope
    if result.replay_response is not None:
        logger.info("Returning cached replay response for session %s", session.id)
        final_response = _attach_loop_fields(
            asdict(
                success_response(
                    data=result.replay_response,
                    request_id=request_id,
                )
            )
        )
    else:
        # Persist session if needed
        if result.should_persist:
            try:
                storage.save(result.session)
                # Update active session pointer
                storage.set_active_session(result.session.spec_id, result.session.id)
            except (OSError, ValueError, TimeoutError) as e:
                logger.error("Failed to persist session %s: %s", session.id, e)
                final_response = _attach_loop_fields(
                    asdict(
                        error_response(
                            f"Failed to persist session state: {e}",
                            error_code=ErrorCode.INTERNAL_ERROR,
                            error_type=ErrorType.INTERNAL,
                            request_id=request_id,
                            details={
                                "session_id": session.id,
                                "state_version": session.state_version,
                            },
                        )
                    )
                )
            else:
                # Return error or success response
                if not result.success:
                    # Extract gate_block from warnings if present
                    gate_block = None
                    if result.warnings and isinstance(result.warnings, dict):
                        gate_block = result.warnings.get("gate_block")

                    final_response = _map_orchestrator_error_to_response(
                        error_code=result.error_code,
                        error_message=result.error_message or "Orchestration failed",
                        request_id=request_id,
                        session_id=session.id,
                        state_version=session.state_version,
                        gate_block=gate_block,
                    )
                else:
                    final_response = _build_next_step_response(result, request_id)
        else:
            if not result.success:
                gate_block = None
                if result.warnings and isinstance(result.warnings, dict):
                    gate_block = result.warnings.get("gate_block")
                final_response = _map_orchestrator_error_to_response(
                    error_code=result.error_code,
                    error_message=result.error_message or "Orchestration failed",
                    request_id=request_id,
                    session_id=session.id,
                    state_version=session.state_version,
                    gate_block=gate_block,
                )
            else:
                final_response = _build_next_step_response(result, request_id)

    _persist_step_proof_response(
        storage=storage,
        session_id=session.id,
        step_proof=consumed_step_proof,
        step_id=consumed_step_id,
        response=final_response,
    )
    return final_response


def _handle_session_step_report(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    session_id: Optional[str] = None,
    step_id: Optional[str] = None,
    step_type: Optional[str] = None,
    outcome: Optional[str] = None,
    note: Optional[str] = None,
    files_touched: Optional[list] = None,
    workspace: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Handle session-step-report action.

    Reports the outcome of a step execution.

    This is an alias for session-step-next with last_step_result.
    The report action was deprecated in favor of passing last_step_result
    directly to session-step-next for a more ergonomic API.

    Args:
        config: Server configuration
        spec_id: Spec ID of the session
        step_id: Step ID being reported
        outcome: Step outcome (success, failure, skipped)
        note: Optional note about the outcome
        files_touched: Optional list of files modified
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict with next step or error
    """
    request_id = _request_id()

    params = {"spec_id": spec_id, "session_id": session_id,
              "step_id": step_id, "step_type": step_type, "outcome": outcome}
    err = validate_payload(params, _STEP_REPORT_SCHEMA,
                           tool_name="task", action="session-step-report",
                           request_id=request_id,
                           cross_field_rules=[AtLeastOne(fields=("spec_id", "session_id"))])
    if err:
        return _attach_loop_fields(err)

    # Build last_step_result and delegate to next handler
    last_step_result: Dict[str, Any] = {
        "step_id": step_id,
        "step_type": step_type,
        "outcome": outcome,
    }
    if note:
        last_step_result["note"] = note
    if files_touched:
        last_step_result["files_touched"] = files_touched

    # Strip keys already consumed or built into last_step_result
    _CONSUMED_REPORT_KEYS = {"step_id", "step_type", "outcome", "note", "files_touched",
                              "spec_id", "session_id", "workspace", "last_step_result"}
    filtered = {k: v for k, v in payload.items() if k not in _CONSUMED_REPORT_KEYS}

    # Delegate to session-step-next
    return _handle_session_step_next(
        config=config,
        spec_id=spec_id,
        session_id=session_id,
        workspace=workspace,
        last_step_result=last_step_result,
        **filtered,
    )


def _handle_session_step_replay(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    session_id: Optional[str] = None,
    workspace: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Handle session-step-replay action.

    Replays the last issued response for safe retry.

    This returns the cached last_issued_response from the session state,
    enabling safe retry after network failures or timeouts without
    re-executing the step.

    Args:
        config: Server configuration
        spec_id: Spec ID of the session
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict with last issued response or error
    """
    request_id = _request_id()

    storage = _get_storage(config, workspace, request_id=request_id)
    if isinstance(storage, dict):
        return storage

    session, err = _resolve_session(storage, "session-step-replay", request_id, session_id, spec_id)
    if err:
        return _attach_loop_fields(err)

    # Check for cached response
    if not session.last_issued_response:
        return _attach_loop_fields(
            asdict(
                error_response(
                    "No cached response available for replay",
                    error_code=ErrorCode.NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    request_id=request_id,
                    details={
                        "action": "session-step-replay",
                        "session_id": session.id,
                        "hint": "Call session-step-next first to get a step to execute",
                    },
                )
            )
        )

    logger.info("Replaying cached response for session %s", session.id)
    return _attach_loop_fields(
        asdict(
            success_response(
                data=session.last_issued_response,
                request_id=request_id,
            )
        )
    )


def _handle_session_step_heartbeat(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    session_id: Optional[str] = None,
    context_usage_pct: Optional[int] = None,
    estimated_tokens_used: Optional[int] = None,
    workspace: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Handle session-step-heartbeat action.

    Updates session heartbeat and context metrics.
    ADR specifies heartbeat as a session-step command alongside next.

    Delegates to the session heartbeat handler for implementation.
    """
    from foundry_mcp.tools.unified.task_handlers.handlers_session import (
        _handle_session_heartbeat,
    )

    return _handle_session_heartbeat(
        config=config,
        spec_id=spec_id,
        session_id=session_id,
        context_usage_pct=context_usage_pct,
        estimated_tokens_used=estimated_tokens_used,
        workspace=workspace,
        **payload,
    )
