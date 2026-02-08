"""Unified task router — split into domain-focused handler modules."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import mcp_tool
from foundry_mcp.tools.unified.common import dispatch_with_standard_errors
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

logger = logging.getLogger(__name__)

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
]

_TASK_ROUTER = ActionRouter(tool_name="task", actions=_ACTION_DEFINITIONS)


def _dispatch_task_action(
    *, action: str, payload: Dict[str, Any], config: ServerConfig
) -> dict:
    return dispatch_with_standard_errors(
        _TASK_ROUTER, "task", action, config=config, payload=payload
    )


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
        }
        return _dispatch_task_action(action=action, payload=payload, config=config)

    logger.debug("Registered unified task tool")


__all__ = [
    "register_unified_task_tool",
]
