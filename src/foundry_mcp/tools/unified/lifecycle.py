"""Unified lifecycle tool backed by ActionRouter and lifecycle helpers."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import audit_log, get_metrics, mcp_tool
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    sanitize_error_message,
    success_response,
)
from foundry_mcp.core.lifecycle import (
    VALID_FOLDERS,
    MoveResult,
    LifecycleState,
    archive_spec,
    activate_spec,
    complete_spec,
    get_lifecycle_state,
    move_spec,
)
from foundry_mcp.tools.unified.common import (
    build_request_id,
    dispatch_with_standard_errors,
    make_metric_name,
    make_validation_error_fn,
    resolve_specs_dir,
)
from foundry_mcp.tools.unified.router import (
    ActionDefinition,
    ActionRouter,
)

# Import write-lock helpers for autonomy session protection.
try:
    from foundry_mcp.core.autonomy.write_lock import (
        check_autonomy_write_lock as _check_write_lock_impl,
        WriteLockStatus as _WriteLockStatus,
    )
    _WRITE_LOCK_AVAILABLE = True
except ImportError:
    _check_write_lock_impl = None  # type: ignore[misc,assignment]
    _WriteLockStatus = None  # type: ignore[misc,assignment]
    _WRITE_LOCK_AVAILABLE = False


def _check_autonomy_write_lock(
    spec_id: str,
    workspace: Optional[str],
    bypass_autonomy_lock: bool,
    bypass_reason: Optional[str],
    request_id: str,
    config: Optional[ServerConfig] = None,
) -> Optional[dict]:
    """Check autonomy write-lock and return error response if blocked."""
    if not _WRITE_LOCK_AVAILABLE or _check_write_lock_impl is None:
        return None

    allow_lock_bypass = False
    if config is not None:
        allow_lock_bypass = config.autonomy_security.allow_lock_bypass

    result = _check_write_lock_impl(
        spec_id=spec_id,
        workspace=workspace,
        bypass_flag=bypass_autonomy_lock,
        bypass_reason=bypass_reason,
        allow_lock_bypass=allow_lock_bypass,
    )

    if result.status == _WriteLockStatus.LOCKED:
        return asdict(error_response(
            result.message or "Autonomy write lock is active for this spec",
            error_code=ErrorCode.AUTONOMY_WRITE_LOCK_ACTIVE,
            error_type=ErrorType.CONFLICT,
            request_id=request_id,
            details={
                "session_id": result.session_id,
                "session_status": result.session_status,
                "hint": "Use bypass_autonomy_lock=true with bypass_reason to override",
            },
        ))

    return None

logger = logging.getLogger(__name__)
_metrics = get_metrics()

_ACTION_SUMMARY = {
    "move": "Move a specification between pending/active/completed/archived folders",
    "activate": "Activate a pending specification (moves to active/)",
    "complete": "Mark spec as successfully delivered (requires all tasks done, moves to completed/)",
    "archive": "Remove spec from workflow without completing (for cancelled/superseded specs, moves to archived/)",
    "state": "Inspect the current lifecycle state and progress",
}


def _metric_name(action: str) -> str:
    return make_metric_name("lifecycle", action)


def _request_id() -> str:
    return build_request_id("lifecycle")


_validation_error = make_validation_error_fn("lifecycle")


def _resolve_workspace_for_write_lock(
    config: ServerConfig,
    path: Optional[str],
) -> tuple[Optional[str], Optional[Path], Optional[dict]]:
    """Resolve canonical workspace root for write-lock checks.

    Lifecycle `path` may be a workspace, `specs/` directory, or spec file path.
    Write-lock storage expects the workspace root, so resolve specs first and
    then normalize to its parent directory.
    """
    specs_dir, specs_err = resolve_specs_dir(config, path)
    if specs_err or specs_dir is None:
        return None, specs_dir, specs_err
    return str(specs_dir.parent), specs_dir, None


def _classify_error(error_message: str) -> tuple[ErrorCode, ErrorType, str]:
    lowered = error_message.lower()
    if "not found" in lowered:
        return (
            ErrorCode.SPEC_NOT_FOUND,
            ErrorType.NOT_FOUND,
            'Verify the spec ID via spec(action="list")',
        )
    if "invalid folder" in lowered:
        return (
            ErrorCode.INVALID_FORMAT,
            ErrorType.VALIDATION,
            "Use one of the supported lifecycle folders",
        )
    if (
        "cannot move" in lowered
        or "cannot complete" in lowered
        or "already exists" in lowered
    ):
        return (
            ErrorCode.CONFLICT,
            ErrorType.CONFLICT,
            "Check the current lifecycle status and allowed transitions",
        )
    return (
        ErrorCode.INTERNAL_ERROR,
        ErrorType.INTERNAL,
        "Inspect server logs for additional context",
    )


def _move_result_response(
    *,
    action: str,
    result: MoveResult,
    request_id: str,
    elapsed_ms: float,
) -> dict:
    metric_labels = {"status": "success" if result.success else "error"}
    _metrics.counter(_metric_name(action), labels=metric_labels)

    if result.success:
        warnings: list[str] | None = None
        if result.old_path == result.new_path:
            warnings = [
                "Specification already resided in the requested folder; no file movement required",
            ]
        data = {
            "spec_id": result.spec_id,
            "from_folder": result.from_folder,
            "to_folder": result.to_folder,
            "old_path": result.old_path,
            "new_path": result.new_path,
        }
        return asdict(
            success_response(
                data=data,
                warnings=warnings,
                telemetry={"duration_ms": round(elapsed_ms, 2)},
                request_id=request_id,
            )
        )

    error_message = result.error or f"Failed to execute lifecycle.{action}"
    error_code, error_type, remediation = _classify_error(error_message)
    return asdict(
        error_response(
            error_message,
            error_code=error_code,
            error_type=error_type,
            remediation=remediation,
            details={
                "spec_id": result.spec_id,
                "from_folder": result.from_folder,
                "to_folder": result.to_folder,
            },
            request_id=request_id,
        )
    )


def _state_response(
    state: LifecycleState, *, request_id: str, elapsed_ms: float
) -> dict:
    return asdict(
        success_response(
            data={
                "spec_id": state.spec_id,
                "folder": state.folder,
                "status": state.status,
                "progress_percentage": state.progress_percentage,
                "total_tasks": state.total_tasks,
                "completed_tasks": state.completed_tasks,
                "can_complete": state.can_complete,
                "can_archive": state.can_archive,
            },
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )


def _handle_move(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    to_folder: Optional[str] = None,
    path: Optional[str] = None,
    force: Optional[bool] = None,  # Unused, accepted for router compatibility
    bypass_autonomy_lock: Optional[bool] = False,
    bypass_reason: Optional[str] = None,
) -> dict:
    action = "move"
    request_id = _request_id()

    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            action=action,
            field="spec_id",
            message="Provide a non-empty spec identifier",
            remediation='Call spec(action="list") to locate the correct spec_id',
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )

    if not isinstance(to_folder, str) or not to_folder.strip():
        return _validation_error(
            action=action,
            field="to_folder",
            message="Provide the destination folder",
            remediation="Use one of: pending, active, completed, archived",
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )

    normalized_folder = to_folder.strip().lower()
    if normalized_folder not in VALID_FOLDERS:
        return _validation_error(
            action=action,
            field="to_folder",
            message=f"Unsupported folder '{to_folder}'.",
            remediation="Use one of: pending, active, completed, archived",
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

    workspace_root, specs_dir, specs_err = _resolve_workspace_for_write_lock(config, path)
    if specs_err:
        return specs_err

    # Check autonomy write-lock before proceeding with protected mutation
    lock_error = _check_autonomy_write_lock(
        spec_id=spec_id.strip(),
        workspace=workspace_root,
        bypass_autonomy_lock=bool(bypass_autonomy_lock),
        bypass_reason=bypass_reason,
        request_id=request_id,
        config=config,
    )
    if lock_error:
        return lock_error

    audit_log(
        "tool_invocation",
        tool="lifecycle",
        action=action,
        spec_id=spec_id.strip(),
        to_folder=normalized_folder,
    )

    start = time.perf_counter()
    try:
        result = move_spec(spec_id.strip(), normalized_folder, specs_dir)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected error moving spec")
        _metrics.counter(_metric_name(action), labels={"status": "exception"})
        return asdict(
            error_response(
                sanitize_error_message(exc, context="lifecycle"),
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Inspect server logs for lifecycle move failures",
                request_id=request_id,
            )
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    return _move_result_response(
        action=action,
        result=result,
        request_id=request_id,
        elapsed_ms=elapsed_ms,
    )


def _handle_activate(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    to_folder: Optional[str] = None,  # Unused, accepted for router compatibility
    path: Optional[str] = None,
    force: Optional[bool] = None,  # Unused, accepted for router compatibility
    bypass_autonomy_lock: Optional[bool] = False,
    bypass_reason: Optional[str] = None,
) -> dict:
    action = "activate"
    request_id = _request_id()

    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            action=action,
            field="spec_id",
            message="Provide a non-empty spec identifier",
            remediation='Call spec(action="list") to locate the correct spec_id',
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )

    if path is not None and not isinstance(path, str):
        return _validation_error(
            action=action,
            field="path",
            message="Workspace path must be a string",
            request_id=request_id,
        )

    workspace_root, specs_dir, specs_err = _resolve_workspace_for_write_lock(config, path)
    if specs_err:
        return specs_err

    # Check autonomy write-lock before proceeding with protected mutation
    lock_error = _check_autonomy_write_lock(
        spec_id=spec_id.strip(),
        workspace=workspace_root,
        bypass_autonomy_lock=bool(bypass_autonomy_lock),
        bypass_reason=bypass_reason,
        request_id=request_id,
        config=config,
    )
    if lock_error:
        return lock_error

    audit_log(
        "tool_invocation",
        tool="lifecycle",
        action=action,
        spec_id=spec_id.strip(),
    )

    start = time.perf_counter()
    try:
        result = activate_spec(spec_id.strip(), specs_dir)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected error activating spec")
        _metrics.counter(_metric_name(action), labels={"status": "exception"})
        return asdict(
            error_response(
                sanitize_error_message(exc, context="lifecycle"),
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Inspect server logs for lifecycle activation failures",
                request_id=request_id,
            )
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    return _move_result_response(
        action=action,
        result=result,
        request_id=request_id,
        elapsed_ms=elapsed_ms,
    )


def _handle_complete(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    to_folder: Optional[str] = None,  # Unused, accepted for router compatibility
    force: Optional[bool] = False,
    path: Optional[str] = None,
    bypass_autonomy_lock: Optional[bool] = False,
    bypass_reason: Optional[str] = None,
) -> dict:
    action = "complete"
    request_id = _request_id()

    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            action=action,
            field="spec_id",
            message="Provide a non-empty spec identifier",
            remediation='Call spec(action="list") to locate the correct spec_id',
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )

    if force is not None and not isinstance(force, bool):
        return _validation_error(
            action=action,
            field="force",
            message="Force flag must be boolean",
            request_id=request_id,
        )

    if path is not None and not isinstance(path, str):
        return _validation_error(
            action=action,
            field="path",
            message="Workspace path must be a string",
            request_id=request_id,
        )

    workspace_root, specs_dir, specs_err = _resolve_workspace_for_write_lock(config, path)
    if specs_err:
        return specs_err

    # Check autonomy write-lock before proceeding with protected mutation
    lock_error = _check_autonomy_write_lock(
        spec_id=spec_id.strip(),
        workspace=workspace_root,
        bypass_autonomy_lock=bool(bypass_autonomy_lock),
        bypass_reason=bypass_reason,
        request_id=request_id,
        config=config,
    )
    if lock_error:
        return lock_error

    audit_log(
        "tool_invocation",
        tool="lifecycle",
        action=action,
        spec_id=spec_id.strip(),
        force=bool(force),
    )

    start = time.perf_counter()
    try:
        result = complete_spec(spec_id.strip(), specs_dir, force=bool(force))
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected error completing spec")
        _metrics.counter(_metric_name(action), labels={"status": "exception"})
        return asdict(
            error_response(
                sanitize_error_message(exc, context="lifecycle"),
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Inspect server logs for lifecycle completion failures",
                request_id=request_id,
            )
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    return _move_result_response(
        action=action,
        result=result,
        request_id=request_id,
        elapsed_ms=elapsed_ms,
    )


def _handle_archive(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    to_folder: Optional[str] = None,  # Unused, accepted for router compatibility
    path: Optional[str] = None,
    force: Optional[bool] = None,  # Unused, accepted for router compatibility
    bypass_autonomy_lock: Optional[bool] = False,
    bypass_reason: Optional[str] = None,
) -> dict:
    action = "archive"
    request_id = _request_id()

    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            action=action,
            field="spec_id",
            message="Provide a non-empty spec identifier",
            remediation='Call spec(action="list") to locate the correct spec_id',
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )

    if path is not None and not isinstance(path, str):
        return _validation_error(
            action=action,
            field="path",
            message="Workspace path must be a string",
            request_id=request_id,
        )

    workspace_root, specs_dir, specs_err = _resolve_workspace_for_write_lock(config, path)
    if specs_err:
        return specs_err

    # Check autonomy write-lock before proceeding with protected mutation
    lock_error = _check_autonomy_write_lock(
        spec_id=spec_id.strip(),
        workspace=workspace_root,
        bypass_autonomy_lock=bool(bypass_autonomy_lock),
        bypass_reason=bypass_reason,
        request_id=request_id,
        config=config,
    )
    if lock_error:
        return lock_error

    audit_log(
        "tool_invocation",
        tool="lifecycle",
        action=action,
        spec_id=spec_id.strip(),
    )

    start = time.perf_counter()
    try:
        result = archive_spec(spec_id.strip(), specs_dir)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected error archiving spec")
        _metrics.counter(_metric_name(action), labels={"status": "exception"})
        return asdict(
            error_response(
                sanitize_error_message(exc, context="lifecycle"),
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Inspect server logs for lifecycle archive failures",
                request_id=request_id,
            )
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    return _move_result_response(
        action=action,
        result=result,
        request_id=request_id,
        elapsed_ms=elapsed_ms,
    )


def _handle_state(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    to_folder: Optional[str] = None,  # Unused, accepted for router compatibility
    path: Optional[str] = None,
    force: Optional[bool] = None,  # Unused, accepted for router compatibility
) -> dict:
    action = "state"
    request_id = _request_id()

    if not isinstance(spec_id, str) or not spec_id.strip():
        return _validation_error(
            action=action,
            field="spec_id",
            message="Provide a non-empty spec identifier",
            remediation='Call spec(action="list") to locate the correct spec_id',
            request_id=request_id,
            code=ErrorCode.MISSING_REQUIRED,
        )

    if path is not None and not isinstance(path, str):
        return _validation_error(
            action=action,
            field="path",
            message="Workspace path must be a string",
            request_id=request_id,
        )

    specs_dir, specs_err = resolve_specs_dir(config, path)
    if specs_err:
        return specs_err

    start = time.perf_counter()
    try:
        state = get_lifecycle_state(spec_id.strip(), specs_dir)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected error fetching lifecycle state")
        _metrics.counter(_metric_name(action), labels={"status": "exception"})
        return asdict(
            error_response(
                sanitize_error_message(exc, context="lifecycle"),
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Inspect server logs for lifecycle state failures",
                request_id=request_id,
            )
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    if state is None:
        _metrics.counter(_metric_name(action), labels={"status": "not_found"})
        return asdict(
            error_response(
                f"Spec '{spec_id.strip()}' not found",
                error_code=ErrorCode.SPEC_NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation='Verify the spec exists via spec(action="list")',
                request_id=request_id,
            )
        )

    _metrics.counter(_metric_name(action), labels={"status": "success"})
    return _state_response(state, request_id=request_id, elapsed_ms=elapsed_ms)


_LIFECYCLE_ROUTER = ActionRouter(
    tool_name="lifecycle",
    actions=[
        ActionDefinition(
            name="move",
            handler=_handle_move,
            summary=_ACTION_SUMMARY["move"],
        ),
        ActionDefinition(
            name="activate",
            handler=_handle_activate,
            summary=_ACTION_SUMMARY["activate"],
        ),
        ActionDefinition(
            name="complete",
            handler=_handle_complete,
            summary=_ACTION_SUMMARY["complete"],
        ),
        ActionDefinition(
            name="archive",
            handler=_handle_archive,
            summary=_ACTION_SUMMARY["archive"],
        ),
        ActionDefinition(
            name="state",
            handler=_handle_state,
            summary=_ACTION_SUMMARY["state"],
        ),
    ],
)


def _dispatch_lifecycle_action(
    *, action: str, payload: Dict[str, Any], config: ServerConfig
) -> dict:
    return dispatch_with_standard_errors(
        _LIFECYCLE_ROUTER, "lifecycle", action, config=config, **payload
    )


def register_unified_lifecycle_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the consolidated lifecycle tool."""

    @canonical_tool(mcp, canonical_name="lifecycle")
    @mcp_tool(tool_name="lifecycle", emit_metrics=True, audit=True)
    def lifecycle(
        action: str,
        spec_id: Optional[str] = None,
        to_folder: Optional[str] = None,
        force: Optional[bool] = False,
        path: Optional[str] = None,
    ) -> dict:
        payload = {
            "spec_id": spec_id,
            "to_folder": to_folder,
            "force": force,
            "path": path,
        }
        return _dispatch_lifecycle_action(action=action, payload=payload, config=config)

    logger.debug("Registered unified lifecycle tool")


__all__ = [
    "register_unified_lifecycle_tool",
]
