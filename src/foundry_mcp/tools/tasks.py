"""
Task operation tools for foundry-mcp.

Provides MCP tools for task discovery, status management, and progress tracking.
"""

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.spec import (
    find_specs_directory,
    load_spec,
    save_spec,
    get_node,
    update_node,
)
from foundry_mcp.core.task import (
    get_next_task,
    check_dependencies,
    prepare_task as core_prepare_task,
)
from foundry_mcp.core.progress import (
    get_progress_summary,
    update_parent_status,
    list_phases,
    get_status_icon,
)
from foundry_mcp.core.pagination import (
    encode_cursor,
    decode_cursor,
    paginated_response,
    normalize_page_size,
    CursorError,
)
from foundry_mcp.core.responses import (
    success_response,
    error_response,
    sanitize_error_message,
)
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.tools.unified.task import legacy_task_action

logger = logging.getLogger(__name__)


def register_task_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register task operation tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="task-prepare",
    )
    def task_prepare(
        spec_id: str, task_id: Optional[str] = None, workspace: Optional[str] = None
    ) -> dict:
        """
        Prepare complete context for task implementation.

        Combines task discovery, dependency checking, and context gathering.
        If no task_id provided, auto-discovers the next actionable task.

        Args:
            spec_id: Specification ID
            task_id: Optional task ID (auto-discovers if not provided)
            workspace: Optional workspace path

        Returns:
            JSON object with task data, dependencies, and context
        """
        return legacy_task_action(
            "prepare",
            config=config,
            spec_id=spec_id,
            task_id=task_id,
            workspace=workspace,
        )

    @canonical_tool(
        mcp,
        canonical_name="task-next",
    )
    def task_next(spec_id: str, workspace: Optional[str] = None) -> dict:
        """
        Find the next actionable task in a specification.

        Searches phases in order (in_progress first, then pending).
        Only returns unblocked tasks with pending status.

        Args:
            spec_id: Specification ID
            workspace: Optional workspace path

        Returns:
            JSON object with next task info or completion status
        """
        return legacy_task_action(
            "next",
            config=config,
            spec_id=spec_id,
            workspace=workspace,
        )

    @canonical_tool(
        mcp,
        canonical_name="task-info",
    )
    def task_info(spec_id: str, task_id: str, workspace: Optional[str] = None) -> dict:
        """
        Get detailed information about a specific task.

        Args:
            spec_id: Specification ID
            task_id: Task ID
            workspace: Optional workspace path

        Returns:
            JSON object with complete task information
        """
        return legacy_task_action(
            "info",
            config=config,
            spec_id=spec_id,
            task_id=task_id,
            workspace=workspace,
        )

    @canonical_tool(
        mcp,
        canonical_name="task-check-deps",
    )
    def task_check_deps(
        spec_id: str, task_id: str, workspace: Optional[str] = None
    ) -> dict:
        """
        Check dependency status for a task.

        Args:
            spec_id: Specification ID
            task_id: Task ID
            workspace: Optional workspace path

        Returns:
            JSON object with dependency analysis
        """
        return legacy_task_action(
            "check-deps",
            config=config,
            spec_id=spec_id,
            task_id=task_id,
            workspace=workspace,
        )

    @canonical_tool(
        mcp,
        canonical_name="task-update-status",
    )
    def task_update_status(
        spec_id: str,
        task_id: str,
        status: str,
        note: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> dict:
        """
        Update a task's status.

        Args:
            spec_id: Specification ID
            task_id: Task ID
            status: New status (pending, in_progress, completed, blocked)
            note: Optional note about the status change
            workspace: Optional workspace path

        Returns:
            JSON object with update result
        """
        return legacy_task_action(
            "update-status",
            config=config,
            spec_id=spec_id,
            task_id=task_id,
            status=status,
            note=note,
            workspace=workspace,
        )

    @canonical_tool(
        mcp,
        canonical_name="task-complete",
    )
    def task_complete(
        spec_id: str,
        task_id: str,
        completion_note: str,
        workspace: Optional[str] = None,
    ) -> dict:
        """
        Mark a task as completed with a completion note.

        Creates a journal entry documenting what was accomplished.

        Args:
            spec_id: Specification ID
            task_id: Task ID
            completion_note: Description of what was accomplished
            workspace: Optional workspace path

        Returns:
            JSON object with completion result
        """
        return legacy_task_action(
            "complete",
            config=config,
            spec_id=spec_id,
            task_id=task_id,
            completion_note=completion_note,
            workspace=workspace,
        )

    @canonical_tool(
        mcp,
        canonical_name="task-start",
    )
    def task_start(
        spec_id: str,
        task_id: str,
        note: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> dict:
        """
        Mark a task as in_progress (start working on it).

        Args:
            spec_id: Specification ID
            task_id: Task ID
            note: Optional note about starting the task
            workspace: Optional workspace path

        Returns:
            JSON object with result
        """
        return legacy_task_action(
            "start",
            config=config,
            spec_id=spec_id,
            task_id=task_id,
            note=note,
            workspace=workspace,
        )

    @canonical_tool(
        mcp,
        canonical_name="task-progress",
    )
    def task_progress(
        spec_id: str,
        node_id: str = "spec-root",
        include_phases: bool = True,
        workspace: Optional[str] = None,
    ) -> dict:
        """
        Get progress summary for a specification or node.

        Args:
            spec_id: Specification ID
            node_id: Node to get progress for (default: spec-root)
            include_phases: Include phase breakdown (default: True)
            workspace: Optional workspace path

        Returns:
            JSON object with progress information
        """
        return legacy_task_action(
            "progress",
            config=config,
            spec_id=spec_id,
            node_id=node_id,
            include_phases=include_phases,
            workspace=workspace,
        )

    @canonical_tool(
        mcp,
        canonical_name="task-list",
    )
    def task_list(
        spec_id: str,
        status_filter: Optional[str] = None,
        include_completed: bool = True,
        workspace: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> dict:
        """
        Get a flat list of all tasks in a specification.

        Args:
            spec_id: Specification ID
            status_filter: Optional filter by status (pending, in_progress, completed, blocked)
            include_completed: Whether to include completed tasks
            workspace: Optional workspace path
            limit: Number of tasks per page (default: 100, max: 1000)
            cursor: Pagination cursor from previous response

        Returns:
            JSON object with task list
        """
        return legacy_task_action(
            "list",
            config=config,
            spec_id=spec_id,
            status_filter=status_filter,
            include_completed=include_completed,
            workspace=workspace,
            limit=limit,
            cursor=cursor,
        )

    logger.debug(
        "Registered task tools: task-prepare/task-next/task-info/task-check-deps/"
        "task-update-status/task-complete/task-start/task-progress/task-list"
    )
