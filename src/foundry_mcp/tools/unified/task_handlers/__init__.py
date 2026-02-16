"""Unified task router — split into domain-focused handler modules."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import mcp_tool
from foundry_mcp.tools.unified.common import (
    build_request_id,
    dispatch_with_standard_errors,
)
from foundry_mcp.tools.unified.router import ActionDefinition, ActionRouter

from foundry_mcp.tools.unified.task_handlers.handlers_lifecycle import (
    _handle_block,
    _handle_complete,
    _handle_list_blocked,
    _handle_start,
    _handle_unblock,
    _handle_update_status,
)
from foundry_mcp.tools.unified.task_handlers.handlers_batch import (
    _handle_complete_batch,
    _handle_fix_verification_types,
    _handle_metadata_batch,
    _handle_prepare_batch,
    _handle_reset_batch,
    _handle_start_batch,
)
from foundry_mcp.tools.unified.task_handlers.handlers_query import (
    _handle_check_deps,
    _handle_hierarchy,
    _handle_info,
    _handle_list,
    _handle_next,
    _handle_prepare,
    _handle_progress,
    _handle_query,
    _handle_session_config,
)
from foundry_mcp.tools.unified.task_handlers.handlers_mutation import (
    _handle_add,
    _handle_add_dependency,
    _handle_add_requirement,
    _handle_move,
    _handle_remove,
    _handle_remove_dependency,
    _handle_update_estimate,
    _handle_update_metadata,
)
from foundry_mcp.tools.unified.task_handlers.handlers_session import (
    _handle_gate_waiver,
    _handle_session_end,
    _handle_session_events,
    _handle_session_heartbeat,
    _handle_session_list,
    _handle_session_pause,
    _handle_session_rebase,
    _handle_session_reset,
    _handle_session_resume,
    _handle_session_start,
    _handle_session_status,
)
from foundry_mcp.tools.unified.task_handlers.handlers_session_step import (
    _handle_session_step_heartbeat,
    _handle_session_step_next,
    _handle_session_step_report,
    _handle_session_step_replay,
)
from foundry_mcp.tools.unified.task_handlers._helpers import (
    _attach_deprecation_metadata,
    _attach_session_step_loop_metadata,
    _emit_legacy_action_warning,
    _normalize_task_action_shape,
    _validation_error,
)

logger = logging.getLogger(__name__)


def _handle_session_action(
    *,
    command: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Canonical session action placeholder.

    Normal task dispatch rewrites action=session to concrete legacy actions
    before routing. This handler exists so discovery surfaces canonical action
    names even if callers bypass the normal dispatch adapter.
    """
    request_id = build_request_id("task")
    return _validation_error(
        action="session",
        field="command",
        message="command is required (start|status|pause|resume|rebase|end|list|reset)",
        request_id=request_id,
    )


def _handle_session_step_action(
    *,
    command: Optional[str] = None,
    **payload: Any,
) -> dict:
    """Canonical session-step action placeholder for discovery parity."""
    request_id = build_request_id("task")
    return _validation_error(
        action="session-step",
        field="command",
        message="command is required (next|report|replay|heartbeat)",
        request_id=request_id,
    )


_ACTION_DEFINITIONS = [
    ActionDefinition(name="prepare", handler=_handle_prepare, summary="Prepare next actionable task context"),
    ActionDefinition(name="prepare-batch", handler=_handle_prepare_batch, summary="Prepare multiple independent tasks for parallel execution"),
    ActionDefinition(name="start-batch", handler=_handle_start_batch, summary="Atomically start multiple tasks as in_progress"),
    ActionDefinition(name="complete-batch", handler=_handle_complete_batch, summary="Complete multiple tasks with partial failure support"),
    ActionDefinition(name="reset-batch", handler=_handle_reset_batch, summary="Reset stale or specified in_progress tasks to pending"),
    ActionDefinition(name="next", handler=_handle_next, summary="Return the next actionable task"),
    ActionDefinition(name="info", handler=_handle_info, summary="Fetch task metadata by ID"),
    ActionDefinition(name="check-deps", handler=_handle_check_deps, summary="Analyze task dependencies and blockers"),
    ActionDefinition(name="start", handler=_handle_start, summary="Start a task"),
    ActionDefinition(name="complete", handler=_handle_complete, summary="Complete a task"),
    ActionDefinition(name="update-status", handler=_handle_update_status, summary="Update task status"),
    ActionDefinition(name="block", handler=_handle_block, summary="Block a task"),
    ActionDefinition(name="unblock", handler=_handle_unblock, summary="Unblock a task"),
    ActionDefinition(name="list-blocked", handler=_handle_list_blocked, summary="List blocked tasks"),
    ActionDefinition(name="add", handler=_handle_add, summary="Add a task"),
    ActionDefinition(name="remove", handler=_handle_remove, summary="Remove a task"),
    ActionDefinition(name="move", handler=_handle_move, summary="Move task to new position or parent"),
    ActionDefinition(name="add-dependency", handler=_handle_add_dependency, summary="Add a dependency between two tasks"),
    ActionDefinition(name="remove-dependency", handler=_handle_remove_dependency, summary="Remove a dependency between two tasks"),
    ActionDefinition(name="add-requirement", handler=_handle_add_requirement, summary="Add a structured requirement to a task"),
    ActionDefinition(name="update-estimate", handler=_handle_update_estimate, summary="Update estimated effort"),
    ActionDefinition(name="update-metadata", handler=_handle_update_metadata, summary="Update task metadata fields"),
    ActionDefinition(name="metadata-batch", handler=_handle_metadata_batch, summary="Batch update metadata across multiple nodes matching filters"),
    ActionDefinition(name="fix-verification-types", handler=_handle_fix_verification_types, summary="Fix invalid/missing verification types across verify nodes"),
    ActionDefinition(name="progress", handler=_handle_progress, summary="Summarize completion metrics for a node"),
    ActionDefinition(name="list", handler=_handle_list, summary="List tasks with pagination and optional filters"),
    ActionDefinition(name="query", handler=_handle_query, summary="Query tasks by status or parent"),
    ActionDefinition(name="hierarchy", handler=_handle_hierarchy, summary="Return paginated hierarchy slices"),
    ActionDefinition(name="session-config", handler=_handle_session_config, summary="Get/set autonomous session configuration"),
    ActionDefinition(name="session", handler=_handle_session_action, summary="Canonical autonomous session lifecycle entrypoint (requires command)"),
    ActionDefinition(name="session-step", handler=_handle_session_step_action, summary="Canonical autonomous session-step entrypoint (requires command)"),
    # Session lifecycle actions (feature-flag guarded by autonomy_sessions)
    ActionDefinition(name="session-start", handler=_handle_session_start, summary="Start a new autonomous session for a spec"),
    ActionDefinition(name="session-pause", handler=_handle_session_pause, summary="Pause an active autonomous session"),
    ActionDefinition(name="session-resume", handler=_handle_session_resume, summary="Resume a paused autonomous session"),
    ActionDefinition(name="session-end", handler=_handle_session_end, summary="End an autonomous session (terminal state)"),
    ActionDefinition(name="session-status", handler=_handle_session_status, summary="Get current status of an autonomous session"),
    ActionDefinition(name="session-events", handler=_handle_session_events, summary="List journal-backed events for an autonomous session"),
    ActionDefinition(name="session-list", handler=_handle_session_list, summary="List autonomous sessions with optional filtering"),
    ActionDefinition(name="session-rebase", handler=_handle_session_rebase, summary="Rebase an active session to spec changes"),
    ActionDefinition(name="session-heartbeat", handler=_handle_session_heartbeat, summary="[Deprecated: use session-step-heartbeat] Update session heartbeat and context metrics"),
    ActionDefinition(name="session-reset", handler=_handle_session_reset, summary="Reset a failed session to allow retry"),
    ActionDefinition(name="gate-waiver", handler=_handle_gate_waiver, summary="Privileged gate waiver for required-gate invariant failures (maintainer only)"),
    # Session-step actions (feature-flag guarded by autonomy_sessions)
    ActionDefinition(name="session-step-next", handler=_handle_session_step_next, summary="Get the next step to execute in a session"),
    ActionDefinition(name="session-step-report", handler=_handle_session_step_report, summary="Report the outcome of a step execution"),
    ActionDefinition(name="session-step-replay", handler=_handle_session_step_replay, summary="Replay the last issued response for safe retry"),
    ActionDefinition(name="session-step-heartbeat", handler=_handle_session_step_heartbeat, summary="Update session heartbeat and context metrics (ADR session-step command)"),
]

_TASK_ROUTER = ActionRouter(tool_name="task", actions=_ACTION_DEFINITIONS)


def _dispatch_task_action(
    *, action: str, payload: Dict[str, Any], config: ServerConfig
) -> dict:
    request_id = payload.get("request_id")
    if not isinstance(request_id, str):
        request_id = build_request_id("task")
    normalized_action, normalized_payload, deprecation, error = _normalize_task_action_shape(
        action=action,
        payload=payload,
        request_id=request_id,
    )
    if error is not None:
        return error

    _emit_legacy_action_warning(normalized_action, deprecation)
    response = dispatch_with_standard_errors(
        _TASK_ROUTER, "task", normalized_action, config=config, **normalized_payload
    )
    response = _attach_deprecation_metadata(response, deprecation)
    return _attach_session_step_loop_metadata(normalized_action, response)


def register_unified_task_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the consolidated task tool."""

    @canonical_tool(
        mcp,
        canonical_name="task",
    )
    @mcp_tool(tool_name="task", emit_metrics=True, audit=True)
    def task(
        action: str,
        spec_id: Optional[str] = None,
        task_id: Optional[str] = None,
        task_ids: Optional[List[str]] = None,
        workspace: Optional[str] = None,
        status_filter: Optional[str] = None,
        include_completed: bool = True,
        node_id: str = "spec-root",
        include_phases: bool = True,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        parent: Optional[str] = None,
        status: Optional[str] = None,
        note: Optional[str] = None,
        completion_note: Optional[str] = None,
        reason: Optional[str] = None,
        blocker_type: str = "dependency",
        ticket: Optional[str] = None,
        resolution: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        acceptance_criteria: Optional[List[str]] = None,
        task_type: str = "task",
        estimated_hours: Optional[float] = None,
        position: Optional[int] = None,
        cascade: bool = False,
        complexity: Optional[str] = None,
        file_path: Optional[str] = None,
        task_category: Optional[str] = None,
        actual_hours: Optional[float] = None,
        status_note: Optional[str] = None,
        verification_type: Optional[str] = None,
        command: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
        max_depth: int = 2,
        include_metadata: bool = False,
        # metadata-batch specific parameters
        pattern: Optional[str] = None,
        node_type: Optional[str] = None,
        owners: Optional[List[str]] = None,
        labels: Optional[Dict[str, str]] = None,
        category: Optional[str] = None,
        parent_filter: Optional[str] = None,
        update_metadata: Optional[Dict[str, Any]] = None,
        # session-config specific parameters
        get: bool = False,
        auto_mode: Optional[bool] = None,
        # complete-batch specific parameters
        completions: Optional[List[Dict[str, Any]]] = None,
        # reset-batch specific parameters
        threshold_hours: Optional[float] = None,
        # session lifecycle parameters (autonomy_sessions feature flag)
        session_id: Optional[str] = None,
        force: bool = False,
        acknowledge_gate_review: Optional[bool] = None,
        acknowledged_gate_attempt_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        include_resume_context: bool = False,
        # session-start configuration parameters
        gate_policy: Optional[str] = None,
        max_tasks_per_session: Optional[int] = None,
        max_consecutive_errors: Optional[int] = None,
        context_threshold_pct: Optional[int] = None,
        stop_on_phase_completion: Optional[bool] = None,
        auto_retry_fidelity_gate: Optional[bool] = None,
        heartbeat_stale_minutes: Optional[int] = None,
        heartbeat_grace_minutes: Optional[int] = None,
        step_stale_minutes: Optional[int] = None,
        max_fidelity_review_cycles_per_phase: Optional[int] = None,
        enforce_autonomy_write_lock: Optional[bool] = None,
        estimated_tokens_used: Optional[int] = None,
        # session-step parameters
        step_id: Optional[str] = None,
        step_type: Optional[str] = None,
        outcome: Optional[str] = None,
        files_touched: Optional[List[str]] = None,
        context_usage_pct: Optional[int] = None,
        last_step_result: Optional[Dict[str, Any]] = None,
        heartbeat: Optional[bool] = None,
        # gate-waiver parameters
        reason_code: Optional[str] = None,
        reason_detail: Optional[str] = None,
    ) -> dict:
        payload = {
            "spec_id": spec_id,
            "task_id": task_id,
            "task_ids": task_ids,
            "workspace": workspace,
            "status_filter": status_filter,
            "include_completed": include_completed,
            "node_id": node_id,
            "include_phases": include_phases,
            "cursor": cursor,
            "limit": limit,
            "parent": parent,
            "status": status,
            "note": note,
            "completion_note": completion_note,
            "reason": reason,
            "blocker_type": blocker_type,
            "ticket": ticket,
            "resolution": resolution,
            "title": title,
            "description": description,
            "acceptance_criteria": acceptance_criteria,
            "task_type": task_type,
            "estimated_hours": estimated_hours,
            "position": position,
            "cascade": cascade,
            "complexity": complexity,
            "file_path": file_path,
            "task_category": task_category,
            "actual_hours": actual_hours,
            "status_note": status_note,
            "verification_type": verification_type,
            "command": command,
            "custom_metadata": custom_metadata,
            "dry_run": dry_run,
            "max_depth": max_depth,
            "include_metadata": include_metadata,
            # metadata-batch specific
            "pattern": pattern,
            "node_type": node_type,
            "owners": owners,
            "labels": labels,
            "category": category,
            "parent_filter": parent_filter,
            "update_metadata": update_metadata,
            # session-config specific
            "get": get,
            "auto_mode": auto_mode,
            # complete-batch specific
            "completions": completions,
            # reset-batch specific
            "threshold_hours": threshold_hours,
            # session lifecycle specific
            "session_id": session_id,
            "force": force,
            "acknowledge_gate_review": acknowledge_gate_review,
            "acknowledged_gate_attempt_id": acknowledged_gate_attempt_id,
            "idempotency_key": idempotency_key,
            "include_resume_context": include_resume_context,
            # session-start configuration
            "gate_policy": gate_policy,
            "max_tasks_per_session": max_tasks_per_session,
            "max_consecutive_errors": max_consecutive_errors,
            "context_threshold_pct": context_threshold_pct,
            "stop_on_phase_completion": stop_on_phase_completion,
            "auto_retry_fidelity_gate": auto_retry_fidelity_gate,
            "heartbeat_stale_minutes": heartbeat_stale_minutes,
            "heartbeat_grace_minutes": heartbeat_grace_minutes,
            "step_stale_minutes": step_stale_minutes,
            "max_fidelity_review_cycles_per_phase": max_fidelity_review_cycles_per_phase,
            "enforce_autonomy_write_lock": enforce_autonomy_write_lock,
            "estimated_tokens_used": estimated_tokens_used,
            # session-step specific
            "step_id": step_id,
            "step_type": step_type,
            "outcome": outcome,
            "files_touched": files_touched,
            "context_usage_pct": context_usage_pct,
            "last_step_result": last_step_result,
            "heartbeat": heartbeat,
            # gate-waiver specific
            "reason_code": reason_code,
            "reason_detail": reason_detail,
        }
        return _dispatch_task_action(action=action, payload=payload, config=config)

    logger.debug("Registered unified task tool")


__all__ = [
    "register_unified_task_tool",
]
