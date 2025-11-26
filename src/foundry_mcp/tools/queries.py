"""
Query tools for foundry-mcp.

Provides MCP tools for finding and listing SDD specifications.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.observability import mcp_tool
from foundry_mcp.core.spec import (
    find_specs_directory,
    find_spec_file,
    list_specs,
    load_spec,
)

logger = logging.getLogger(__name__)


# JSON Schema definitions for tool inputs/outputs

FIND_SPECS_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "spec_id": {
            "type": "string",
            "description": "Specification ID to find"
        },
        "workspace": {
            "type": "string",
            "description": "Optional workspace path to search in"
        }
    },
    "required": ["spec_id"]
}

FIND_SPECS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "spec_id": {"type": "string"},
        "path": {"type": "string"},
        "status_folder": {"type": "string"},
        "error": {"type": "string"}
    }
}

LIST_SPECS_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["active", "pending", "completed", "archived", "all"],
            "description": "Filter by status folder",
            "default": "all"
        },
        "workspace": {
            "type": "string",
            "description": "Optional workspace path"
        },
        "include_progress": {
            "type": "boolean",
            "description": "Include task progress information",
            "default": True
        }
    }
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
                    "progress_percentage": {"type": "integer"}
                }
            }
        },
        "count": {"type": "integer"},
        "error": {"type": "string"}
    }
}


def register_query_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register query tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @mcp.tool()
    @mcp_tool(tool_name="foundry_find_specs")
    def foundry_find_specs(
        spec_id: str,
        workspace: Optional[str] = None
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
                return {
                    "success": False,
                    "data": {},
                    "error": "No specs directory found"
                }

            # Find the spec file
            spec_file = find_spec_file(spec_id, specs_dir)

            if spec_file:
                # Determine status folder from path
                status_folder = spec_file.parent.name

                return {
                    "success": True,
                    "data": {
                        "found": True,
                        "spec_id": spec_id,
                        "path": str(spec_file),
                        "status_folder": status_folder
                    },
                    "error": None
                }
            else:
                return {
                    "success": True,
                    "data": {
                        "found": False,
                        "spec_id": spec_id
                    },
                    "error": None
                }

        except Exception as e:
            logger.error(f"Error finding spec {spec_id}: {e}")
            return {
                "success": False,
                "data": {},
                "error": str(e)
            }

    @mcp.tool()
    @mcp_tool(tool_name="foundry_list_specs")
    def foundry_list_specs(
        status: str = "all",
        workspace: Optional[str] = None,
        include_progress: bool = True
    ) -> dict:
        """
        List all specifications with optional filtering.

        Returns a list of specifications with their IDs, titles, status,
        and optionally progress information.

        Args:
            status: Filter by status folder (active, pending, completed, archived, or all)
            workspace: Optional workspace path
            include_progress: Whether to include task progress (default: True)

        Returns:
            JSON object with list of specs and count
        """
        try:
            # Determine specs directory
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return {
                    "success": False,
                    "data": {},
                    "error": "No specs directory found"
                }

            # List specs
            filter_status = None if status == "all" else status
            specs = list_specs(specs_dir=specs_dir, status=filter_status)

            # Optionally strip progress info
            if not include_progress:
                specs = [
                    {
                        "spec_id": s["spec_id"],
                        "title": s["title"],
                        "status": s["status"]
                    }
                    for s in specs
                ]

            return {
                "success": True,
                "data": {
                    "specs": specs,
                    "count": len(specs)
                },
                "error": None
            }

        except Exception as e:
            logger.error(f"Error listing specs: {e}")
            return {
                "success": False,
                "data": {},
                "error": str(e)
            }

    @mcp.tool()
    @mcp_tool(tool_name="foundry_query_tasks")
    def foundry_query_tasks(
        spec_id: str,
        status: Optional[str] = None,
        parent: Optional[str] = None,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Query tasks within a specification.

        Filter tasks by status and/or parent task ID.

        Args:
            spec_id: Specification ID
            status: Filter by task status (pending, in_progress, completed, blocked)
            parent: Filter by parent task ID
            workspace: Optional workspace path

        Returns:
            JSON object with matching tasks
        """
        try:
            # Determine specs directory
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            spec_data = load_spec(spec_id, specs_dir)

            if not spec_data:
                return {
                    "tasks": [],
                    "count": 0,
                    "error": f"Spec not found: {spec_id}"
                }

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

                tasks.append({
                    "task_id": task_id,
                    "title": task_data.get("title", task_id),
                    "status": task_data.get("status", "unknown"),
                    "type": task_data.get("type", "task"),
                    "parent": task_data.get("parent"),
                })

            return {
                "spec_id": spec_id,
                "tasks": tasks,
                "count": len(tasks)
            }

        except Exception as e:
            logger.error(f"Error querying tasks in {spec_id}: {e}")
            return {
                "tasks": [],
                "count": 0,
                "error": str(e)
            }

    logger.debug("Registered query tools: foundry_find_specs, foundry_list_specs, foundry_query_tasks")
