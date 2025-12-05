"""
Query tools for foundry-mcp.

Provides MCP tools for finding and listing SDD specifications.
"""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional, List, Dict, Any

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.pagination import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    encode_cursor,
    decode_cursor,
    CursorError,
    normalize_page_size,
)
from foundry_mcp.core.responses import success_response, error_response, sanitize_error_message
from foundry_mcp.core.spec import (
    find_specs_directory,
    find_spec_file,
    list_specs,
    load_spec,
)
from foundry_mcp.core.naming import canonical_tool

logger = logging.getLogger(__name__)


# JSON Schema definitions for tool inputs/outputs

FIND_SPECS_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "spec_id": {"type": "string", "description": "Specification ID to find"},
        "workspace": {
            "type": "string",
            "description": "Optional workspace path to search in",
        },
    },
    "required": ["spec_id"],
}

FIND_SPECS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "spec_id": {"type": "string"},
        "path": {"type": "string"},
        "status_folder": {"type": "string"},
        "error": {"type": "string"},
    },
}

LIST_SPECS_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["active", "pending", "completed", "archived", "all"],
            "description": "Filter by status folder",
            "default": "all",
        },
        "workspace": {"type": "string", "description": "Optional workspace path"},
        "include_progress": {
            "type": "boolean",
            "description": "Include task progress information",
            "default": True,
        },
    },
}

LIST_SPECS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "specs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "spec_id": {"type": "string"},
                    "title": {"type": "string"},
                    "status": {"type": "string"},
                    "total_tasks": {"type": "integer"},
                    "completed_tasks": {"type": "integer"},
                    "progress_percentage": {"type": "integer"},
                },
            },
        },
        "count": {"type": "integer"},
        "error": {"type": "string"},
    },
}


def register_query_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register query tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="spec-find",
    )
    def spec_find(
        spec_id: str,
        workspace: Optional[str] = None,
    ) -> dict:
        """
        Find a specification file by ID.

        Searches across all status folders (active, pending, completed, archived)
        to locate a specification file.

        Args:
            spec_id: Specification ID to find
            workspace: Optional workspace path to search in

        Returns:
            JSON object with found status, path, and status folder
        """
        try:
            # Determine specs directory
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return asdict(error_response("No specs directory found"))

            # Find the spec file
            spec_file = find_spec_file(spec_id, specs_dir)

            if spec_file:
                # Determine status folder from path
                status_folder = spec_file.parent.name

                return asdict(
                    success_response(
                        found=True,
                        spec_id=spec_id,
                        path=str(spec_file),
                        status_folder=status_folder,
                    )
                )
            else:
                return asdict(success_response(found=False, spec_id=spec_id))

        except Exception as e:
            logger.error(f"Error finding spec {spec_id}: {e}")
            return asdict(error_response(sanitize_error_message(e, context="spec queries")))

    @canonical_tool(
        mcp,
        canonical_name="spec-list",
    )
    def spec_list(
        status: str = "all",
        workspace: Optional[str] = None,
        include_progress: bool = True,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict:
        """
        List all specifications with optional filtering and pagination.

        Returns a list of specifications with their IDs, titles, status,
        and optionally progress information. Supports cursor-based pagination.

        Args:
            status: Filter by status folder (active, pending, completed, archived, or all)
            workspace: Optional workspace path
            include_progress: Whether to include task progress (default: True)
            cursor: Pagination cursor from previous response
            limit: Number of specs per page (default: 100, max: 1000)

        Returns:
            JSON object with list of specs, count, and pagination metadata
        """
        try:
            # Determine specs directory
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return asdict(error_response("No specs directory found"))

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

            # List all specs (sorted by spec_id)
            filter_status = None if status == "all" else status
            all_specs = list_specs(specs_dir=specs_dir, status=filter_status)

            # Sort by spec_id for consistent pagination
            all_specs.sort(key=lambda s: s.get("spec_id", ""))

            # Apply cursor-based pagination
            if start_after_id:
                # Find index of cursor position
                start_index = 0
                for i, spec in enumerate(all_specs):
                    if spec.get("spec_id") == start_after_id:
                        start_index = i + 1
                        break
                all_specs = all_specs[start_index:]

            # Fetch one extra to detect has_more
            specs = all_specs[: page_size + 1]
            has_more = len(specs) > page_size
            if has_more:
                specs = specs[:page_size]

            # Optionally strip progress info
            if not include_progress:
                specs = [
                    {
                        "spec_id": s["spec_id"],
                        "title": s["title"],
                        "status": s["status"],
                    }
                    for s in specs
                ]

            # Build next cursor
            next_cursor = None
            if has_more and specs:
                next_cursor = encode_cursor({"last_id": specs[-1]["spec_id"]})

            return asdict(
                success_response(
                    data={"specs": specs, "count": len(specs)},
                    pagination={
                        "cursor": next_cursor,
                        "has_more": has_more,
                        "page_size": page_size,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Error listing specs: {e}")
            return asdict(error_response(sanitize_error_message(e, context="spec queries")))

    @canonical_tool(
        mcp,
        canonical_name="task-query",
    )
    def task_query(
        spec_id: str,
        status: Optional[str] = None,
        parent: Optional[str] = None,
        workspace: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict:
        """
        Query tasks within a specification with optional filtering and pagination.

        Filter tasks by status and/or parent task ID. Supports cursor-based pagination.

        Args:
            spec_id: Specification ID
            status: Filter by task status (pending, in_progress, completed, blocked)
            parent: Filter by parent task ID
            workspace: Optional workspace path
            cursor: Pagination cursor from previous response
            limit: Number of tasks per page (default: 100, max: 1000)

        Returns:
            JSON object with matching tasks and pagination metadata
        """
        try:
            # Determine specs directory
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

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

            hierarchy = spec_data.get("hierarchy", {})

            # Filter tasks
            tasks = []
            for task_id, task_data in hierarchy.items():
                # Apply status filter
                if status and task_data.get("status") != status:
                    continue

                # Apply parent filter
                if parent and task_data.get("parent") != parent:
                    continue

                tasks.append(
                    {
                        "task_id": task_id,
                        "title": task_data.get("title", task_id),
                        "status": task_data.get("status", "unknown"),
                        "type": task_data.get("type", "task"),
                        "parent": task_data.get("parent"),
                    }
                )

            # Sort by task_id for consistent pagination
            tasks.sort(key=lambda t: t["task_id"])

            # Apply cursor-based pagination
            if start_after_id:
                # Find index of cursor position
                start_index = 0
                for i, task in enumerate(tasks):
                    if task["task_id"] == start_after_id:
                        start_index = i + 1
                        break
                tasks = tasks[start_index:]

            # Fetch one extra to detect has_more
            page_tasks = tasks[: page_size + 1]
            has_more = len(page_tasks) > page_size
            if has_more:
                page_tasks = page_tasks[:page_size]

            # Build next cursor
            next_cursor = None
            if has_more and page_tasks:
                next_cursor = encode_cursor({"last_id": page_tasks[-1]["task_id"]})

            return asdict(
                success_response(
                    data={
                        "spec_id": spec_id,
                        "tasks": page_tasks,
                        "count": len(page_tasks),
                    },
                    pagination={
                        "cursor": next_cursor,
                        "has_more": has_more,
                        "page_size": page_size,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Error querying tasks in {spec_id}: {e}")
            return asdict(error_response(sanitize_error_message(e, context="spec queries")))

    logger.debug("Registered query tools: spec-find/spec-list/task-query")
