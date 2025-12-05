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
)
from foundry_mcp.core.responses import success_response, error_response, sanitize_error_message
from foundry_mcp.core.naming import canonical_tool

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
        try:
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return asdict(error_response("No specs directory found"))

            result = core_prepare_task(spec_id, specs_dir, task_id)
            return result

        except Exception as e:
            logger.error(f"Error preparing task: {e}")
            return asdict(error_response(sanitize_error_message(e, context="tasks")))

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
        try:
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return asdict(error_response("No specs directory found"))

            spec_data = load_spec(spec_id, specs_dir)
            if not spec_data:
                return asdict(error_response(f"Spec not found: {spec_id}"))

            next_task = get_next_task(spec_data)

            if next_task:
                task_id, task_data = next_task
                return asdict(
                    success_response(
                        found=True,
                        spec_id=spec_id,
                        task_id=task_id,
                        title=task_data.get("title", ""),
                        type=task_data.get("type", "task"),
                        status=task_data.get("status", "pending"),
                        metadata=task_data.get("metadata", {}),
                    )
                )
            else:
                # Check if spec is complete
                hierarchy = spec_data.get("hierarchy", {})
                all_tasks = [
                    node
                    for node in hierarchy.values()
                    if node.get("type") in ["task", "subtask", "verify"]
                ]
                completed = sum(1 for t in all_tasks if t.get("status") == "completed")
                pending = sum(1 for t in all_tasks if t.get("status") == "pending")

                if pending == 0 and completed > 0:
                    return asdict(
                        success_response(
                            found=False,
                            spec_id=spec_id,
                            spec_complete=True,
                            message="All tasks completed",
                        )
                    )
                else:
                    return asdict(
                        success_response(
                            found=False,
                            spec_id=spec_id,
                            spec_complete=False,
                            message="No actionable tasks (tasks may be blocked)",
                        )
                    )

        except Exception as e:
            logger.error(f"Error finding next task: {e}")
            return asdict(error_response(sanitize_error_message(e, context="tasks")))

    @canonical_tool(
        mcp,
        canonical_name="task-info",
    )
    def task_info(
        spec_id: str, task_id: str, workspace: Optional[str] = None
    ) -> dict:
        """
        Get detailed information about a specific task.

        Args:
            spec_id: Specification ID
            task_id: Task ID
            workspace: Optional workspace path

        Returns:
            JSON object with complete task information
        """
        try:
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return asdict(error_response("No specs directory found"))

            spec_data = load_spec(spec_id, specs_dir)
            if not spec_data:
                return asdict(error_response(f"Spec not found: {spec_id}"))

            task_data = get_node(spec_data, task_id)
            if not task_data:
                return asdict(error_response(f"Task not found: {task_id}"))

            return asdict(
                success_response(
                    spec_id=spec_id,
                    task_id=task_id,
                    title=task_data.get("title", ""),
                    type=task_data.get("type", "task"),
                    status=task_data.get("status", "pending"),
                    parent=task_data.get("parent"),
                    children=task_data.get("children", []),
                    metadata=task_data.get("metadata", {}),
                    dependencies=task_data.get("dependencies", {}),
                    completed_tasks=task_data.get("completed_tasks", 0),
                    total_tasks=task_data.get("total_tasks", 0),
                )
            )

        except Exception as e:
            logger.error(f"Error getting task info: {e}")
            return asdict(error_response(sanitize_error_message(e, context="tasks")))

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
        try:
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return asdict(error_response("No specs directory found"))

            spec_data = load_spec(spec_id, specs_dir)
            if not spec_data:
                return asdict(error_response(f"Spec not found: {spec_id}"))

            deps = check_dependencies(spec_data, task_id)
            deps["spec_id"] = spec_id
            return asdict(success_response(**deps))

        except Exception as e:
            logger.error(f"Error checking dependencies: {e}")
            return asdict(error_response(sanitize_error_message(e, context="tasks")))

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
        valid_statuses = ["pending", "in_progress", "completed", "blocked"]
        if status not in valid_statuses:
            return asdict(
                error_response(
                    f"Invalid status: {status}. Must be one of: {valid_statuses}"
                )
            )

        try:
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return asdict(error_response("No specs directory found"))

            spec_data = load_spec(spec_id, specs_dir)
            if not spec_data:
                return asdict(error_response(f"Spec not found: {spec_id}"))

            # Update task status
            updates = {"status": status}
            if status == "completed":
                updates["metadata"] = {
                    "completed_at": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                }
            elif status == "in_progress":
                updates["metadata"] = {
                    "started_at": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                }

            if not update_node(spec_data, task_id, updates):
                return asdict(error_response(f"Task not found: {task_id}"))

            # Update parent status chain
            update_parent_status(spec_data, task_id)

            # Add journal entry if note provided
            if note:
                journal = spec_data.setdefault("journal", [])
                journal.append(
                    {
                        "timestamp": datetime.now(timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "task_id": task_id,
                        "entry_type": "status_change",
                        "title": f"Status changed to {status}",
                        "content": note,
                        "author": "foundry-mcp",
                    }
                )

            # Save spec
            if not save_spec(spec_id, spec_data, specs_dir):
                return asdict(error_response("Failed to save spec"))

            return asdict(
                success_response(spec_id=spec_id, task_id=task_id, new_status=status)
            )

        except Exception as e:
            logger.error(f"Error updating status: {e}")
            return asdict(error_response(sanitize_error_message(e, context="tasks")))

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
        try:
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return asdict(error_response("No specs directory found"))

            spec_data = load_spec(spec_id, specs_dir)
            if not spec_data:
                return asdict(error_response(f"Spec not found: {spec_id}"))

            task_data = get_node(spec_data, task_id)
            if not task_data:
                return asdict(error_response(f"Task not found: {task_id}"))

            timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            # Update task status
            updates = {"status": "completed", "metadata": {"completed_at": timestamp}}

            if not update_node(spec_data, task_id, updates):
                return asdict(error_response(f"Failed to update task: {task_id}"))

            # Update parent status chain
            update_parent_status(spec_data, task_id)

            # Add completion journal entry
            journal = spec_data.setdefault("journal", [])
            journal.append(
                {
                    "timestamp": timestamp,
                    "task_id": task_id,
                    "entry_type": "status_change",
                    "title": f"Task Completed: {task_data.get('title', task_id)}",
                    "content": completion_note,
                    "author": "foundry-mcp",
                }
            )

            # Save spec
            if not save_spec(spec_id, spec_data, specs_dir):
                return asdict(error_response("Failed to save spec"))

            # Get updated progress
            progress = get_progress_summary(spec_data)

            return asdict(
                success_response(
                    spec_id=spec_id,
                    task_id=task_id,
                    completed_at=timestamp,
                    progress={
                        "completed_tasks": progress.get("completed_tasks", 0),
                        "total_tasks": progress.get("total_tasks", 0),
                        "percentage": progress.get("percentage", 0),
                    },
                )
            )

        except Exception as e:
            logger.error(f"Error completing task: {e}")
            return asdict(error_response(sanitize_error_message(e, context="tasks")))

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
        try:
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return asdict(error_response("No specs directory found"))

            spec_data = load_spec(spec_id, specs_dir)
            if not spec_data:
                return asdict(error_response(f"Spec not found: {spec_id}"))

            task_data = get_node(spec_data, task_id)
            if not task_data:
                return asdict(error_response(f"Task not found: {task_id}"))

            # Check dependencies before starting
            deps = check_dependencies(spec_data, task_id)
            if not deps.get("can_start", False):
                blockers = [
                    b.get("title", b.get("id", "")) for b in deps.get("blocked_by", [])
                ]
                return asdict(
                    error_response(f"Task is blocked by: {', '.join(blockers)}")
                )

            timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            # Update task status
            updates = {"status": "in_progress", "metadata": {"started_at": timestamp}}

            if not update_node(spec_data, task_id, updates):
                return asdict(error_response(f"Failed to update task: {task_id}"))

            # Update parent status chain
            update_parent_status(spec_data, task_id)

            # Add journal entry if note provided
            if note:
                journal = spec_data.setdefault("journal", [])
                journal.append(
                    {
                        "timestamp": timestamp,
                        "task_id": task_id,
                        "entry_type": "status_change",
                        "title": f"Task Started: {task_data.get('title', task_id)}",
                        "content": note,
                        "author": "foundry-mcp",
                    }
                )

            # Save spec
            if not save_spec(spec_id, spec_data, specs_dir):
                return asdict(error_response("Failed to save spec"))

            return asdict(
                success_response(
                    spec_id=spec_id,
                    task_id=task_id,
                    started_at=timestamp,
                    title=task_data.get("title", ""),
                    type=task_data.get("type", "task"),
                )
            )

        except Exception as e:
            logger.error(f"Error starting task: {e}")
            return asdict(error_response(sanitize_error_message(e, context="tasks")))

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
        try:
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return asdict(error_response("No specs directory found"))

            spec_data = load_spec(spec_id, specs_dir)
            if not spec_data:
                return asdict(error_response(f"Spec not found: {spec_id}"))

            progress = get_progress_summary(spec_data, node_id)

            if include_phases:
                progress["phases"] = list_phases(spec_data)

            return asdict(success_response(**progress))

        except Exception as e:
            logger.error(f"Error getting progress: {e}")
            return asdict(error_response(sanitize_error_message(e, context="tasks")))

    logger.debug(
        "Registered task tools: task-prepare/task-next/task-info/task-check-deps/"
        "task-update-status/task-complete/task-start/task-progress"
    )
