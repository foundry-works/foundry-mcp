"""
Lifecycle tools for foundry-mcp.

Provides MCP tools for spec lifecycle management.
"""

import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.observability import mcp_tool
from foundry_mcp.core.spec import find_specs_directory
from foundry_mcp.core.lifecycle import (
    move_spec,
    activate_spec,
    complete_spec,
    archive_spec,
    get_lifecycle_state,
    list_specs_by_folder,
    get_folder_for_spec,
    VALID_FOLDERS,
)

logger = logging.getLogger(__name__)


def register_lifecycle_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register lifecycle tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @mcp.tool()
    @mcp_tool(tool_name="foundry_move_spec")
    def foundry_move_spec(
        spec_id: str,
        to_folder: str,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Move a specification between status folders.

        Moves spec between pending, active, completed, and archived folders
        with transition validation.

        Args:
            spec_id: Specification ID
            to_folder: Target folder (pending, active, completed, archived)
            workspace: Optional workspace path

        Returns:
            JSON object with move result
        """
        try:
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return {
                    "success": False,
                    "error": "No specs directory found"
                }

            result = move_spec(spec_id, to_folder, specs_dir)

            return {
                "success": result.success,
                "spec_id": result.spec_id,
                "from_folder": result.from_folder,
                "to_folder": result.to_folder,
                "old_path": result.old_path,
                "new_path": result.new_path,
                "error": result.error,
            }

        except Exception as e:
            logger.error(f"Error moving spec: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    @mcp.tool()
    @mcp_tool(tool_name="foundry_activate_spec")
    def foundry_activate_spec(
        spec_id: str,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Activate a specification (move from pending to active).

        Args:
            spec_id: Specification ID
            workspace: Optional workspace path

        Returns:
            JSON object with activation result
        """
        try:
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return {
                    "success": False,
                    "error": "No specs directory found"
                }

            result = activate_spec(spec_id, specs_dir)

            return {
                "success": result.success,
                "spec_id": result.spec_id,
                "from_folder": result.from_folder,
                "to_folder": result.to_folder,
                "new_path": result.new_path,
                "error": result.error,
            }

        except Exception as e:
            logger.error(f"Error activating spec: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    @mcp.tool()
    @mcp_tool(tool_name="foundry_complete_spec")
    def foundry_complete_spec(
        spec_id: str,
        force: bool = False,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Mark a specification as completed.

        Moves spec to completed folder. By default, validates that
        all tasks are complete before allowing the move.

        Args:
            spec_id: Specification ID
            force: Force completion even with incomplete tasks
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
                return {
                    "success": False,
                    "error": "No specs directory found"
                }

            result = complete_spec(spec_id, specs_dir, force=force)

            return {
                "success": result.success,
                "spec_id": result.spec_id,
                "from_folder": result.from_folder,
                "to_folder": result.to_folder,
                "new_path": result.new_path,
                "error": result.error,
            }

        except Exception as e:
            logger.error(f"Error completing spec: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    @mcp.tool()
    @mcp_tool(tool_name="foundry_archive_spec")
    def foundry_archive_spec(
        spec_id: str,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Archive a specification.

        Moves spec to archived folder for long-term storage.

        Args:
            spec_id: Specification ID
            workspace: Optional workspace path

        Returns:
            JSON object with archive result
        """
        try:
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return {
                    "success": False,
                    "error": "No specs directory found"
                }

            result = archive_spec(spec_id, specs_dir)

            return {
                "success": result.success,
                "spec_id": result.spec_id,
                "from_folder": result.from_folder,
                "to_folder": result.to_folder,
                "new_path": result.new_path,
                "error": result.error,
            }

        except Exception as e:
            logger.error(f"Error archiving spec: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    @mcp.tool()
    @mcp_tool(tool_name="foundry_lifecycle_state")
    def foundry_lifecycle_state(
        spec_id: str,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Get the current lifecycle state of a specification.

        Returns folder location, status, progress, and transition eligibility.

        Args:
            spec_id: Specification ID
            workspace: Optional workspace path

        Returns:
            JSON object with lifecycle state
        """
        try:
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return {
                    "success": False,
                    "error": "No specs directory found"
                }

            state = get_lifecycle_state(spec_id, specs_dir)

            if not state:
                return {
                    "success": False,
                    "error": f"Spec not found: {spec_id}"
                }

            return {
                "success": True,
                "spec_id": state.spec_id,
                "folder": state.folder,
                "status": state.status,
                "progress_percentage": state.progress_percentage,
                "total_tasks": state.total_tasks,
                "completed_tasks": state.completed_tasks,
                "can_complete": state.can_complete,
                "can_archive": state.can_archive,
            }

        except Exception as e:
            logger.error(f"Error getting lifecycle state: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    @mcp.tool()
    @mcp_tool(tool_name="foundry_list_specs_by_folder")
    def foundry_list_specs_by_folder(
        folder: Optional[str] = None,
        workspace: Optional[str] = None
    ) -> dict:
        """
        List specifications organized by folder.

        Args:
            folder: Optional filter to specific folder (pending, active, completed, archived)
            workspace: Optional workspace path

        Returns:
            JSON object with specs organized by folder
        """
        try:
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return {
                    "success": False,
                    "error": "No specs directory found"
                }

            if folder and folder not in VALID_FOLDERS:
                return {
                    "success": False,
                    "error": f"Invalid folder: {folder}. Must be one of: {list(VALID_FOLDERS)}"
                }

            result = list_specs_by_folder(specs_dir, folder)

            # Calculate totals
            total_specs = sum(len(specs) for specs in result.values())

            return {
                "success": True,
                "total_specs": total_specs,
                "folders": result,
            }

        except Exception as e:
            logger.error(f"Error listing specs by folder: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    logger.debug("Registered lifecycle tools: foundry_move_spec, foundry_activate_spec, "
                 "foundry_complete_spec, foundry_archive_spec, foundry_lifecycle_state, "
                 "foundry_list_specs_by_folder")
