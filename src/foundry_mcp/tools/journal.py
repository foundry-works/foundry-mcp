"""
Journal tools for foundry-mcp.

Provides MCP tools for journal entries and blocker management.
"""

import logging
from dataclasses import asdict
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.observability import mcp_tool
from foundry_mcp.core.spec import (
    find_specs_directory,
    find_spec_file,
    load_spec,
    save_spec,
)
from foundry_mcp.core.journal import (
    add_journal_entry,
    get_journal_entries,
    mark_blocked,
    unblock,
    get_blocker_info,
    list_blocked_tasks,
    find_unjournaled_tasks,
)
from foundry_mcp.core.responses import success_response, error_response

logger = logging.getLogger(__name__)


def register_journal_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register journal tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @mcp.tool()
    @mcp_tool(tool_name="foundry_add_journal")
    def foundry_add_journal(
        spec_id: str,
        title: str,
        content: str,
        entry_type: str = "note",
        task_id: Optional[str] = None,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Add a journal entry to a specification.

        Journal entries document decisions, progress, blockers, and notes.

        Args:
            spec_id: Specification ID
            title: Entry title
            content: Entry content
            entry_type: Type of entry (status_change, deviation, blocker, decision, note)
            task_id: Optional associated task ID
            workspace: Optional workspace path

        Returns:
            JSON object with created entry details
        """
        valid_types = ["status_change", "deviation", "blocker", "decision", "note"]
        if entry_type not in valid_types:
            return asdict(error_response(f"Invalid entry_type: {entry_type}. Must be one of: {valid_types}"))

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

            # Add journal entry
            entry = add_journal_entry(
                spec_data,
                title=title,
                content=content,
                entry_type=entry_type,
                task_id=task_id,
                author="foundry-mcp",
            )

            # Save spec
            if not save_spec(spec_id, spec_data, specs_dir):
                return asdict(error_response("Failed to save spec"))

            return asdict(success_response(
                spec_id=spec_id,
                entry={
                    "timestamp": entry.timestamp,
                    "entry_type": entry.entry_type,
                    "title": entry.title,
                    "task_id": entry.task_id,
                }
            ))

        except Exception as e:
            logger.error(f"Error adding journal entry: {e}")
            return asdict(error_response(str(e)))

    @mcp.tool()
    @mcp_tool(tool_name="foundry_get_journal")
    def foundry_get_journal(
        spec_id: str,
        task_id: Optional[str] = None,
        entry_type: Optional[str] = None,
        limit: int = 10,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Get journal entries from a specification.

        Args:
            spec_id: Specification ID
            task_id: Optional filter by task ID
            entry_type: Optional filter by entry type
            limit: Maximum entries to return (default: 10)
            workspace: Optional workspace path

        Returns:
            JSON object with journal entries
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

            entries = get_journal_entries(
                spec_data,
                task_id=task_id,
                entry_type=entry_type,
                limit=limit,
            )

            return asdict(success_response(
                spec_id=spec_id,
                count=len(entries),
                entries=[
                    {
                        "timestamp": e.timestamp,
                        "entry_type": e.entry_type,
                        "title": e.title,
                        "content": e.content,
                        "author": e.author,
                        "task_id": e.task_id,
                    }
                    for e in entries
                ]
            ))

        except Exception as e:
            logger.error(f"Error getting journal entries: {e}")
            return asdict(error_response(str(e)))

    @mcp.tool()
    @mcp_tool(tool_name="foundry_mark_blocked")
    def foundry_mark_blocked(
        spec_id: str,
        task_id: str,
        reason: str,
        blocker_type: str = "dependency",
        ticket: Optional[str] = None,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Mark a task as blocked.

        Records blocker metadata including type, description, and optional ticket.

        Args:
            spec_id: Specification ID
            task_id: Task to mark as blocked
            reason: Description of the blocker
            blocker_type: Type of blocker (dependency, technical, resource, decision)
            ticket: Optional ticket/issue reference
            workspace: Optional workspace path

        Returns:
            JSON object with block result
        """
        valid_types = ["dependency", "technical", "resource", "decision"]
        if blocker_type not in valid_types:
            return asdict(error_response(f"Invalid blocker_type: {blocker_type}. Must be one of: {valid_types}"))

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

            # Mark task as blocked
            if not mark_blocked(spec_data, task_id, reason, blocker_type, ticket):
                return asdict(error_response(f"Task not found: {task_id}"))

            # Add journal entry for blocker
            add_journal_entry(
                spec_data,
                title=f"Task Blocked: {task_id}",
                content=f"Blocker ({blocker_type}): {reason}" + (f" [Ticket: {ticket}]" if ticket else ""),
                entry_type="blocker",
                task_id=task_id,
                author="foundry-mcp",
            )

            # Save spec
            if not save_spec(spec_id, spec_data, specs_dir):
                return asdict(error_response("Failed to save spec"))

            return asdict(success_response(
                spec_id=spec_id,
                task_id=task_id,
                blocker_type=blocker_type,
                reason=reason,
                ticket=ticket
            ))

        except Exception as e:
            logger.error(f"Error marking task blocked: {e}")
            return asdict(error_response(str(e)))

    @mcp.tool()
    @mcp_tool(tool_name="foundry_unblock")
    def foundry_unblock(
        spec_id: str,
        task_id: str,
        resolution: Optional[str] = None,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Unblock a task.

        Moves blocker info to resolved_blockers and sets status to pending.

        Args:
            spec_id: Specification ID
            task_id: Task to unblock
            resolution: Optional description of how blocker was resolved
            workspace: Optional workspace path

        Returns:
            JSON object with unblock result
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

            # Get blocker info before unblocking
            blocker = get_blocker_info(spec_data, task_id)
            if not blocker:
                return asdict(error_response(f"Task {task_id} is not blocked"))

            # Unblock the task
            if not unblock(spec_data, task_id, resolution):
                return asdict(error_response(f"Failed to unblock task: {task_id}"))

            # Add journal entry for resolution
            add_journal_entry(
                spec_data,
                title=f"Task Unblocked: {task_id}",
                content=f"Resolved: {resolution or 'Blocker resolved'}",
                entry_type="note",
                task_id=task_id,
                author="foundry-mcp",
            )

            # Save spec
            if not save_spec(spec_id, spec_data, specs_dir):
                return asdict(error_response("Failed to save spec"))

            return asdict(success_response(
                spec_id=spec_id,
                task_id=task_id,
                previous_blocker={
                    "type": blocker.blocker_type,
                    "description": blocker.description,
                },
                resolution=resolution or "Blocker resolved",
                new_status="pending"
            ))

        except Exception as e:
            logger.error(f"Error unblocking task: {e}")
            return asdict(error_response(str(e)))

    @mcp.tool()
    @mcp_tool(tool_name="foundry_list_blocked")
    def foundry_list_blocked(
        spec_id: str,
        workspace: Optional[str] = None
    ) -> dict:
        """
        List all blocked tasks in a specification.

        Args:
            spec_id: Specification ID
            workspace: Optional workspace path

        Returns:
            JSON object with list of blocked tasks
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

            blocked = list_blocked_tasks(spec_data)

            return asdict(success_response(
                spec_id=spec_id,
                count=len(blocked),
                blocked_tasks=blocked
            ))

        except Exception as e:
            logger.error(f"Error listing blocked tasks: {e}")
            return asdict(error_response(str(e)))

    @mcp.tool()
    @mcp_tool(tool_name="foundry_unjournaled_tasks")
    def foundry_unjournaled_tasks(
        spec_id: str,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Find completed tasks that need journal entries.

        Args:
            spec_id: Specification ID
            workspace: Optional workspace path

        Returns:
            JSON object with list of unjournaled tasks
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

            unjournaled = find_unjournaled_tasks(spec_data)

            return asdict(success_response(
                spec_id=spec_id,
                count=len(unjournaled),
                unjournaled_tasks=unjournaled
            ))

        except Exception as e:
            logger.error(f"Error finding unjournaled tasks: {e}")
            return asdict(error_response(str(e)))

    logger.debug("Registered journal tools: foundry_add_journal, foundry_get_journal, "
                 "foundry_mark_blocked, foundry_unblock, foundry_list_blocked, "
                 "foundry_unjournaled_tasks")
