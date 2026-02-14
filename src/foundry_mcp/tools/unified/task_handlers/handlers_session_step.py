"""Session-step action handlers: next, report, replay.

This module provides handlers for session-step actions that drive
autonomous execution forward.

Phase A (current):
- next: Stub returning FEATURE_DISABLED (requires Phase B orchestrator)
- report: Stub returning FEATURE_DISABLED (requires Phase B orchestrator)
- replay: Stub returning FEATURE_DISABLED (requires Phase B orchestrator)

All actions are feature-flag guarded by 'autonomy_sessions'.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.autonomy.memory import AutonomyStorage
from foundry_mcp.core.autonomy.models import SessionStatus
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
)

from foundry_mcp.tools.unified.task_handlers._helpers import (
    _request_id,
    _validation_error,
)


def _get_storage(config: ServerConfig, workspace: Optional[str] = None) -> AutonomyStorage:
    """Get AutonomyStorage instance."""
    ws_path = Path(workspace) if workspace else Path.cwd()
    return AutonomyStorage(workspace_path=ws_path)


def _session_not_found_response(action: str, request_id: str, spec_id: Optional[str] = None) -> dict:
    """Return session not found error response."""
    return asdict(error_response(
        "No active session found",
        error_code=ErrorCode.NOT_FOUND,
        error_type=ErrorType.NOT_FOUND,
        request_id=request_id,
        details={
            "action": action,
            "spec_id": spec_id,
            "hint": "Start a session with session-start action",
        },
    ))


def _handle_session_step_next(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    workspace: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Handle session-step-next action.

    Gets the next step to execute in an autonomous session.

    NOTE: This is a Phase B feature. Phase A returns FEATURE_DISABLED.

    Args:
        config: Server configuration
        spec_id: Spec ID of the session
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict with next step details or error
    """
    request_id = _request_id()

    if not spec_id:
        return _validation_error(
            action="session-step-next",
            field="spec_id",
            message="spec_id is required",
            request_id=request_id,
        )

    # Phase A: Return feature disabled with descriptive message
    # This feature requires the Phase B orchestrator which validates
    # the closed feedback loop between steps
    return asdict(error_response(
        "session-step-next requires Phase B orchestrator",
        error_code=ErrorCode.FEATURE_DISABLED,
        error_type=ErrorType.FEATURE_FLAG,
        request_id=request_id,
        details={
            "action": "session-step-next",
            "feature_flag": "autonomy_sessions",
            "phase": "A",
            "hint": "Use task(action=session-*) for lifecycle commands. Session-step commands require Phase B.",
        },
    ))


def _handle_session_step_report(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    step_id: Optional[str] = None,
    outcome: Optional[str] = None,
    note: Optional[str] = None,
    files_touched: Optional[list] = None,
    workspace: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Handle session-step-report action.

    Reports the outcome of a step execution.

    NOTE: This is a Phase B feature. Phase A returns FEATURE_DISABLED.

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
        Response dict with updated session state or error
    """
    request_id = _request_id()

    if not spec_id:
        return _validation_error(
            action="session-step-report",
            field="spec_id",
            message="spec_id is required",
            request_id=request_id,
        )

    if not step_id:
        return _validation_error(
            action="session-step-report",
            field="step_id",
            message="step_id is required",
            request_id=request_id,
        )

    if not outcome:
        return _validation_error(
            action="session-step-report",
            field="outcome",
            message="outcome is required",
            request_id=request_id,
        )

    # Phase A: Return feature disabled with descriptive message
    return asdict(error_response(
        "session-step-report requires Phase B orchestrator",
        error_code=ErrorCode.FEATURE_DISABLED,
        error_type=ErrorType.FEATURE_FLAG,
        request_id=request_id,
        details={
            "action": "session-step-report",
            "feature_flag": "autonomy_sessions",
            "phase": "A",
            "hint": "Use task(action=session-*) for lifecycle commands. Session-step commands require Phase B.",
        },
    ))


def _handle_session_step_replay(
    *,
    config: ServerConfig,
    spec_id: Optional[str] = None,
    workspace: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Handle session-step-replay action.

    Replays the last issued response for safe retry.

    NOTE: This is a Phase B feature. Phase A returns FEATURE_DISABLED.

    Args:
        config: Server configuration
        spec_id: Spec ID of the session
        workspace: Workspace path
        **payload: Additional parameters

    Returns:
        Response dict with last issued response or error
    """
    request_id = _request_id()

    if not spec_id:
        return _validation_error(
            action="session-step-replay",
            field="spec_id",
            message="spec_id is required",
            request_id=request_id,
        )

    # Phase A: Return feature disabled with descriptive message
    return asdict(error_response(
        "session-step-replay requires Phase B orchestrator",
        error_code=ErrorCode.FEATURE_DISABLED,
        error_type=ErrorType.FEATURE_FLAG,
        request_id=request_id,
        details={
            "action": "session-step-replay",
            "feature_flag": "autonomy_sessions",
            "phase": "A",
            "hint": "Use task(action=session-*) for lifecycle commands. Session-step commands require Phase B.",
        },
    ))
