"""Unified verification tool with action routing."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.context import generate_correlation_id, get_correlation_id
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import audit_log, get_metrics, mcp_tool
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    sanitize_error_message,
    success_response,
)
from foundry_mcp.core.spec import find_specs_directory, load_spec, save_spec
from foundry_mcp.core.validation import add_verification, execute_verification
from foundry_mcp.tools.unified.router import (
    ActionDefinition,
    ActionRouter,
    ActionRouterError,
)

logger = logging.getLogger(__name__)
_metrics = get_metrics()

_ACTION_SUMMARY = {
    "add": "Persist verification results with optional dry-run preview",
    "execute": "Execute verification commands and optionally record results",
}


def _metric_name(action: str) -> str:
    return f"verification.{action}"


def _request_id() -> str:
    return get_correlation_id() or generate_correlation_id(prefix="verification")


def _validation_error(
    *,
    action: str,
    field: str,
    message: str,
    request_id: str,
    remediation: Optional[str] = None,
    code: ErrorCode = ErrorCode.VALIDATION_ERROR,
) -> dict:
    return asdict(
        error_response(
            f"Invalid field '{field}' for verification.{action}: {message}",
            error_code=code,
            error_type=ErrorType.VALIDATION,
            remediation=remediation,
            details={"field": field, "action": f"verification.{action}"},
            request_id=request_id,
        )
    )


def _handle_add(
    *,
    config: ServerConfig,  # noqa: ARG001 - reserved for future hooks
    spec_id: Optional[str] = None,
    verify_id: Optional[str] = None,
    result: Optional[str] = None,
    command: Optional[str] = None,
    output: Optional[str] = None,
    issues: Optional[str] = None,
    notes: Optional[str] = None,
    dry_run: bool = False,
    path: Optional[str] = None,
    **_: Any,
) -> dict:
    request_id = _request_id()
    action = "add"

    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            action=action,
            field="spec_id",
            message="Provide a non-empty spec_id",
            request_id=request_id,
            remediation='Use spec(action="list") to discover valid specification IDs',
            code=ErrorCode.MISSING_REQUIRED,
        )
    spec_id = spec_id.strip()

    if not isinstance(verify_id, str) or not verify_id.strip():
        return _validation_error(
            action=action,
            field="verify_id",
            message="Provide the verification node identifier (e.g., verify-1-1)",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    verify_id = verify_id.strip()

    if not isinstance(result, str) or not result.strip():
        return _validation_error(
            action=action,
            field="result",
            message="Provide the verification result (PASSED, FAILED, PARTIAL)",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    result_upper = result.strip().upper()
    if result_upper not in {"PASSED", "FAILED", "PARTIAL"}:
        return _validation_error(
            action=action,
            field="result",
            message="Result must be one of PASSED, FAILED, or PARTIAL",
            request_id=request_id,
            remediation="Use one of: PASSED, FAILED, PARTIAL",
        )

    if path is not None and not isinstance(path, str):
        return _validation_error(
            action=action,
            field="path",
            message="Workspace path must be a string",
            request_id=request_id,
        )
    if not isinstance(dry_run, bool):
        return _validation_error(
            action=action,
            field="dry_run",
            message="Expected a boolean value",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    metric_key = _metric_name(action)
    audit_log(
        "tool_invocation",
        tool="verification",
        action=action,
        spec_id=spec_id,
        verify_id=verify_id,
        result=result_upper,
        dry_run=dry_run,
    )

    specs_dir = find_specs_directory(path)
    if not specs_dir:
        _metrics.counter(metric_key, labels={"status": "specs_not_found"})
        return asdict(
            error_response(
                "Could not find specs directory",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Ensure you are in a project with a specs/ directory",
                request_id=request_id,
            )
        )

    spec_data = load_spec(spec_id, specs_dir)
    if not spec_data:
        _metrics.counter(metric_key, labels={"status": "spec_not_found"})
        return asdict(
            error_response(
                f"Specification '{spec_id}' not found",
                error_code=ErrorCode.SPEC_NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation='Verify the spec ID exists using spec(action="list")',
                request_id=request_id,
            )
        )

    if dry_run:
        hierarchy = spec_data.get("hierarchy", {})
        if not isinstance(hierarchy, dict) or hierarchy.get(verify_id) is None:
            _metrics.counter(metric_key, labels={"status": "verify_not_found"})
            return asdict(
                error_response(
                    f"Verification '{verify_id}' not found in spec",
                    error_code=ErrorCode.NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation="Verify the verification ID exists in the specification",
                    request_id=request_id,
                )
            )

        data: Dict[str, Any] = {
            "spec_id": spec_id,
            "verify_id": verify_id,
            "result": result_upper,
            "dry_run": True,
        }
        if command:
            data["command"] = command
        _metrics.counter(metric_key, labels={"status": "dry_run"})
        return asdict(
            success_response(
                data=data,
                request_id=request_id,
            )
        )

    try:
        success, error_msg = add_verification(
            spec_data=spec_data,
            verify_id=verify_id,
            result=result_upper,
            command=command,
            output=output,
            issues=issues,
            notes=notes,
        )
    except Exception as exc:
        logger.exception("Unexpected error adding verification")
        _metrics.counter(metric_key, labels={"status": "error"})
        return asdict(
            error_response(
                sanitize_error_message(exc, context="verification"),
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check logs for details",
                request_id=request_id,
            )
        )

    if not success:
        lowered = (error_msg or "").lower()
        if "not found" in lowered:
            code = ErrorCode.NOT_FOUND
            error_type = ErrorType.NOT_FOUND
        elif "already" in lowered or "duplicate" in lowered:
            code = ErrorCode.CONFLICT
            error_type = ErrorType.CONFLICT
        else:
            code = ErrorCode.VALIDATION_ERROR
            error_type = ErrorType.VALIDATION
        _metrics.counter(metric_key, labels={"status": error_type.value})
        return asdict(
            error_response(
                error_msg or "Failed to add verification",
                error_code=code,
                error_type=error_type,
                remediation="Check input parameters",
                request_id=request_id,
            )
        )

    if not save_spec(spec_id, spec_data, specs_dir):
        _metrics.counter(metric_key, labels={"status": "save_failed"})
        return asdict(
            error_response(
                "Failed to save specification",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check file permissions",
                request_id=request_id,
            )
        )

    _metrics.counter(metric_key, labels={"status": "success"})
    response_data = {
        "spec_id": spec_id,
        "verify_id": verify_id,
        "result": result_upper,
        "dry_run": False,
    }
    if command:
        response_data["command"] = command

    return asdict(
        success_response(
            data=response_data,
            request_id=request_id,
        )
    )


def _handle_execute(
    *,
    config: ServerConfig,  # noqa: ARG001 - reserved for future hooks
    spec_id: Optional[str] = None,
    verify_id: Optional[str] = None,
    record: bool = False,
    path: Optional[str] = None,
    **_: Any,
) -> dict:
    request_id = _request_id()
    action = "execute"

    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            action=action,
            field="spec_id",
            message="Provide a non-empty spec_id",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    spec_id = spec_id.strip()

    if not isinstance(verify_id, str) or not verify_id.strip():
        return _validation_error(
            action=action,
            field="verify_id",
            message="Provide the verification identifier",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )
    verify_id = verify_id.strip()

    if not isinstance(record, bool):
        return _validation_error(
            action=action,
            field="record",
            message="Expected a boolean value",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    if path is not None and not isinstance(path, str):
        return _validation_error(
            action=action,
            field="path",
            message="Workspace path must be a string",
            request_id=request_id,
        )

    metric_key = _metric_name(action)
    audit_log(
        "tool_invocation",
        tool="verification",
        action=action,
        spec_id=spec_id,
        verify_id=verify_id,
        record=record,
    )

    specs_dir = find_specs_directory(path)
    if not specs_dir:
        _metrics.counter(metric_key, labels={"status": "specs_not_found"})
        return asdict(
            error_response(
                "Could not find specs directory",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Ensure you are in a project with a specs/ directory",
                request_id=request_id,
            )
        )

    spec_data = load_spec(spec_id, specs_dir)
    if not spec_data:
        _metrics.counter(metric_key, labels={"status": "spec_not_found"})
        return asdict(
            error_response(
                f"Specification '{spec_id}' not found",
                error_code=ErrorCode.SPEC_NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation='Verify the spec ID exists using spec(action="list")',
                request_id=request_id,
            )
        )

    start_time = time.perf_counter()
    try:
        result_data = execute_verification(
            spec_data=spec_data,
            verify_id=verify_id,
            record=record,
            cwd=path,
        )
    except Exception as exc:
        logger.exception("Unexpected error executing verification")
        _metrics.counter(metric_key, labels={"status": "error"})
        return asdict(
            error_response(
                sanitize_error_message(exc, context="verification"),
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check logs for details",
                request_id=request_id,
            )
        )

    if record and result_data.get("recorded"):
        if not save_spec(spec_id, spec_data, specs_dir):
            result_data["recorded"] = False
            result_data["error"] = (
                result_data.get("error") or ""
            ) + "; Failed to save spec"

    if result_data.get("error") and not result_data.get("success"):
        error_msg = result_data["error"]
        lowered = error_msg.lower()
        if "not found" in lowered:
            code = ErrorCode.NOT_FOUND
            error_type = ErrorType.NOT_FOUND
        elif "no command" in lowered:
            code = ErrorCode.VALIDATION_ERROR
            error_type = ErrorType.VALIDATION
        else:
            code = ErrorCode.INTERNAL_ERROR
            error_type = ErrorType.INTERNAL
        _metrics.counter(metric_key, labels={"status": error_type.value})
        return asdict(
            error_response(
                error_msg if error_msg else "Failed to execute verification",
                error_code=code,
                error_type=error_type,
                remediation="Ensure the verification node has a valid command",
                request_id=request_id,
            )
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    response_data: Dict[str, Any] = {
        "spec_id": spec_id,
        "verify_id": verify_id,
        "result": result_data.get("result", "UNKNOWN"),
        "recorded": result_data.get("recorded", False),
    }
    if result_data.get("command"):
        response_data["command"] = result_data["command"]
    if result_data.get("output"):
        response_data["output"] = result_data["output"]
    if result_data.get("exit_code") is not None:
        response_data["exit_code"] = result_data["exit_code"]

    _metrics.counter(metric_key, labels={"status": "success"})
    return asdict(
        success_response(
            data=response_data,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )


_VERIFICATION_ROUTER = ActionRouter(
    tool_name="verification",
    actions=[
        ActionDefinition(
            name="add",
            handler=_handle_add,
            summary=_ACTION_SUMMARY["add"],
            aliases=("verification-add", "verification_add"),
        ),
        ActionDefinition(
            name="execute",
            handler=_handle_execute,
            summary=_ACTION_SUMMARY["execute"],
            aliases=("verification-execute", "verification_execute"),
        ),
    ],
)


def _dispatch_verification_action(
    *, action: str, payload: Dict[str, Any], config: ServerConfig
) -> dict:
    try:
        return _VERIFICATION_ROUTER.dispatch(action=action, config=config, **payload)
    except ActionRouterError as exc:
        request_id = _request_id()
        allowed = ", ".join(exc.allowed_actions)
        return asdict(
            error_response(
                f"Unsupported verification action '{action}'. Allowed actions: {allowed}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {allowed}",
                request_id=request_id,
            )
        )


def register_unified_verification_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the consolidated verification tool."""

    @canonical_tool(mcp, canonical_name="verification")
    @mcp_tool(tool_name="verification", emit_metrics=True, audit=True)
    def verification(  # noqa: PLR0913 - shared signature across actions
        action: str,
        spec_id: Optional[str] = None,
        verify_id: Optional[str] = None,
        result: Optional[str] = None,
        command: Optional[str] = None,
        output: Optional[str] = None,
        issues: Optional[str] = None,
        notes: Optional[str] = None,
        dry_run: bool = False,
        record: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        payload = {
            "spec_id": spec_id,
            "verify_id": verify_id,
            "result": result,
            "command": command,
            "output": output,
            "issues": issues,
            "notes": notes,
            "dry_run": dry_run,
            "record": record,
            "path": path,
        }
        return _dispatch_verification_action(
            action=action, payload=payload, config=config
        )

    logger.debug("Registered unified verification tool")


__all__ = [
    "register_unified_verification_tool",
]
