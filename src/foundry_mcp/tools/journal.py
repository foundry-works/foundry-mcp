"""
Journal tools for foundry-mcp.

Provides MCP tools for journal entries and blocker management.
"""

import logging
from dataclasses import asdict
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.pagination import (
    encode_cursor,
    decode_cursor,
    CursorError,
    normalize_page_size,
)
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
from foundry_mcp.core.responses import success_response, error_response, sanitize_error_message
from foundry_mcp.core.naming import canonical_tool

logger = logging.getLogger(__name__)


def register_journal_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register journal tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="journal-add",
    )
    def journal_add(
        spec_id: str,
        title: str,
        content: str,
        entry_type: str = "note",
        task_id: Optional[str] = None,
        workspace: Optional[str] = None,
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
            return asdict(
                error_response(
                    f"Invalid entry_type: {entry_type}. Must be one of: {valid_types}"
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

            return asdict(
                success_response(
                    spec_id=spec_id,
                    entry={
                        "timestamp": entry.timestamp,
                        "entry_type": entry.entry_type,
                        "title": entry.title,
                        "task_id": entry.task_id,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Error adding journal entry: {e}")
            return asdict(error_response(sanitize_error_message(e, context="journal")))

    @canonical_tool(
        mcp,
        canonical_name="journal-list",
    )
    def journal_list(
        spec_id: str,
        task_id: Optional[str] = None,
        entry_type: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        workspace: Optional[str] = None,
    ) -> dict:
        """
        Get journal entries from a specification with optional pagination.

        Args:
            spec_id: Specification ID
            task_id: Optional filter by task ID
            entry_type: Optional filter by entry type
            cursor: Pagination cursor from previous response
            limit: Number of entries per page (default: 100, max: 1000)
            workspace: Optional workspace path

        Returns:
            JSON object with journal entries and pagination metadata
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

            # Normalize page size
            page_size = normalize_page_size(limit)

            # Decode cursor if provided
            start_after_ts = None
            if cursor:
                try:
                    cursor_data = decode_cursor(cursor)
                    start_after_ts = cursor_data.get("last_ts")
                except CursorError:
                    return asdict(error_response("Invalid pagination cursor"))

            # Get all matching entries (no limit from core function)
            all_entries = get_journal_entries(
                spec_data,
                task_id=task_id,
                entry_type=entry_type,
                limit=None,
            )

            # Sort by timestamp for consistent pagination (newest first)
            all_entries.sort(key=lambda e: e.timestamp, reverse=True)

            # Apply cursor-based pagination
            if start_after_ts:
                start_index = 0
                for i, entry in enumerate(all_entries):
                    if entry.timestamp == start_after_ts:
                        start_index = i + 1
                        break
                all_entries = all_entries[start_index:]

            # Fetch one extra to detect has_more
            page_entries = all_entries[: page_size + 1]
            has_more = len(page_entries) > page_size
            if has_more:
                page_entries = page_entries[:page_size]

            # Build next cursor
            next_cursor = None
            if has_more and page_entries:
                next_cursor = encode_cursor({"last_ts": page_entries[-1].timestamp})

            return asdict(
                success_response(
                    data={
                        "spec_id": spec_id,
                        "count": len(page_entries),
                        "entries": [
                            {
                                "timestamp": e.timestamp,
                                "entry_type": e.entry_type,
                                "title": e.title,
                                "content": e.content,
                                "author": e.author,
                                "task_id": e.task_id,
                            }
                            for e in page_entries
                        ],
                    },
                    pagination={
                        "cursor": next_cursor,
                        "has_more": has_more,
                        "page_size": page_size,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Error getting journal entries: {e}")
            return asdict(error_response(sanitize_error_message(e, context="journal")))

    @canonical_tool(
        mcp,
        canonical_name="task-block",
    )
    def task_block(
        spec_id: str,
        task_id: str,
        reason: str,
        blocker_type: str = "dependency",
        ticket: Optional[str] = None,
        workspace: Optional[str] = None,
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
            return asdict(
                error_response(
                    f"Invalid blocker_type: {blocker_type}. Must be one of: {valid_types}"
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

            # Mark task as blocked
            if not mark_blocked(spec_data, task_id, reason, blocker_type, ticket):
                return asdict(error_response(f"Task not found: {task_id}"))

            # Add journal entry for blocker
            add_journal_entry(
                spec_data,
                title=f"Task Blocked: {task_id}",
                content=f"Blocker ({blocker_type}): {reason}"
                + (f" [Ticket: {ticket}]" if ticket else ""),
                entry_type="blocker",
                task_id=task_id,
                author="foundry-mcp",
            )

            # Save spec
            if not save_spec(spec_id, spec_data, specs_dir):
                return asdict(error_response("Failed to save spec"))

            return asdict(
                success_response(
                    spec_id=spec_id,
                    task_id=task_id,
                    blocker_type=blocker_type,
                    reason=reason,
                    ticket=ticket,
                )
            )

        except Exception as e:
            logger.error(f"Error marking task blocked: {e}")
            return asdict(error_response(sanitize_error_message(e, context="journal")))

    @canonical_tool(
        mcp,
        canonical_name="task-unblock",
    )
    def task_unblock(
        spec_id: str,
        task_id: str,
        resolution: Optional[str] = None,
        workspace: Optional[str] = None,
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

            return asdict(
                success_response(
                    spec_id=spec_id,
                    task_id=task_id,
                    previous_blocker={
                        "type": blocker.blocker_type,
                        "description": blocker.description,
                    },
                    resolution=resolution or "Blocker resolved",
                    new_status="pending",
                )
            )

        except Exception as e:
            logger.error(f"Error unblocking task: {e}")
            return asdict(error_response(sanitize_error_message(e, context="journal")))

    @canonical_tool(
        mcp,
        canonical_name="task-list-blocked",
    )
    def task_list_blocked(
        spec_id: str,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        workspace: Optional[str] = None,
    ) -> dict:
        """
        List all blocked tasks in a specification with optional pagination.

        Args:
            spec_id: Specification ID
            cursor: Pagination cursor from previous response
            limit: Number of tasks per page (default: 100, max: 1000)
            workspace: Optional workspace path

        Returns:
            JSON object with list of blocked tasks and pagination metadata
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

            # Normalize page size
            page_size = normalize_page_size(limit)

            # Decode cursor if provided
            start_after_id = None
            if cursor:
                try:
                    cursor_data = decode_cursor(cursor)
                    start_after_id = cursor_data.get("last_id")
                except CursorError:
                    return asdict(error_response("Invalid pagination cursor"))

            all_blocked = list_blocked_tasks(spec_data)

            # Sort by task_id for consistent pagination
            all_blocked.sort(key=lambda t: t.get("task_id", ""))

            # Apply cursor-based pagination
            if start_after_id:
                start_index = 0
                for i, task in enumerate(all_blocked):
                    if task.get("task_id") == start_after_id:
                        start_index = i + 1
                        break
                all_blocked = all_blocked[start_index:]

            # Fetch one extra to detect has_more
            page_tasks = all_blocked[: page_size + 1]
            has_more = len(page_tasks) > page_size
            if has_more:
                page_tasks = page_tasks[:page_size]

            # Build next cursor
            next_cursor = None
            if has_more and page_tasks:
                next_cursor = encode_cursor({"last_id": page_tasks[-1].get("task_id")})

            return asdict(
                success_response(
                    data={
                        "spec_id": spec_id,
                        "count": len(page_tasks),
                        "blocked_tasks": page_tasks,
                    },
                    pagination={
                        "cursor": next_cursor,
                        "has_more": has_more,
                        "page_size": page_size,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Error listing blocked tasks: {e}")
            return asdict(error_response(sanitize_error_message(e, context="journal")))

    @canonical_tool(
        mcp,
        canonical_name="journal-list-unjournaled",
    )
    def journal_list_unjournaled(
        spec_id: str,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        workspace: Optional[str] = None,
    ) -> dict:
        """
        Find completed tasks that need journal entries with optional pagination.

        Args:
            spec_id: Specification ID
            cursor: Pagination cursor from previous response
            limit: Number of tasks per page (default: 100, max: 1000)
            workspace: Optional workspace path

        Returns:
            JSON object with list of unjournaled tasks and pagination metadata
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

            # Normalize page size
            page_size = normalize_page_size(limit)

            # Decode cursor if provided
            start_after_id = None
            if cursor:
                try:
                    cursor_data = decode_cursor(cursor)
                    start_after_id = cursor_data.get("last_id")
                except CursorError:
                    return asdict(error_response("Invalid pagination cursor"))

            all_unjournaled = find_unjournaled_tasks(spec_data)

            # Sort by task_id for consistent pagination
            all_unjournaled.sort(key=lambda t: t.get("task_id", ""))

            # Apply cursor-based pagination
            if start_after_id:
                start_index = 0
                for i, task in enumerate(all_unjournaled):
                    if task.get("task_id") == start_after_id:
                        start_index = i + 1
                        break
                all_unjournaled = all_unjournaled[start_index:]

            # Fetch one extra to detect has_more
            page_tasks = all_unjournaled[: page_size + 1]
            has_more = len(page_tasks) > page_size
            if has_more:
                page_tasks = page_tasks[:page_size]

            # Build next cursor
            next_cursor = None
            if has_more and page_tasks:
                next_cursor = encode_cursor({"last_id": page_tasks[-1].get("task_id")})

            return asdict(
                success_response(
                    data={
                        "spec_id": spec_id,
                        "count": len(page_tasks),
                        "unjournaled_tasks": page_tasks,
                    },
                    pagination={
                        "cursor": next_cursor,
                        "has_more": has_more,
                        "page_size": page_size,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Error finding unjournaled tasks: {e}")
            return asdict(error_response(sanitize_error_message(e, context="journal")))

    logger.debug(
        "Registered journal tools: journal-add/journal-list/task-block/task-unblock/task-list-blocked/journal-list-unjournaled"
    )
