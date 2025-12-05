"""
Lifecycle tools for foundry-mcp.

Provides MCP tools for spec lifecycle management.
"""

import logging
from dataclasses import asdict
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response, sanitize_error_message
from foundry_mcp.core.pagination import (
    encode_cursor,
    decode_cursor,
    paginated_response,
    normalize_page_size,
    CursorError,
)
from foundry_mcp.core.naming import canonical_tool
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

    @canonical_tool(
        mcp,
        canonical_name="spec-lifecycle-move",
    )
    def spec_lifecycle_move(
        spec_id: str, to_folder: str, workspace: Optional[str] = None
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
                return asdict(error_response("No specs directory found"))

            result = move_spec(spec_id, to_folder, specs_dir)

            if not result.success:
                return asdict(error_response(result.error or "Failed to move spec"))

            return asdict(
                success_response(
                    spec_id=result.spec_id,
                    from_folder=result.from_folder,
                    to_folder=result.to_folder,
                    old_path=result.old_path,
                    new_path=result.new_path,
                )
            )

        except Exception as e:
            logger.error(f"Error moving spec: {e}")
            return asdict(error_response(sanitize_error_message(e, context="spec lifecycle")))

    @canonical_tool(
        mcp,
        canonical_name="spec-lifecycle-activate",
    )
    def spec_lifecycle_activate(spec_id: str, workspace: Optional[str] = None) -> dict:
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
                return asdict(error_response("No specs directory found"))

            result = activate_spec(spec_id, specs_dir)

            if not result.success:
                return asdict(error_response(result.error or "Failed to activate spec"))

            return asdict(
                success_response(
                    spec_id=result.spec_id,
                    from_folder=result.from_folder,
                    to_folder=result.to_folder,
                    new_path=result.new_path,
                )
            )

        except Exception as e:
            logger.error(f"Error activating spec: {e}")
            return asdict(error_response(sanitize_error_message(e, context="spec lifecycle")))

    @canonical_tool(
        mcp,
        canonical_name="spec-lifecycle-complete",
    )
    def spec_lifecycle_complete(
        spec_id: str, force: bool = False, workspace: Optional[str] = None
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
                return asdict(error_response("No specs directory found"))

            result = complete_spec(spec_id, specs_dir, force=force)

            if not result.success:
                return asdict(error_response(result.error or "Failed to complete spec"))

            return asdict(
                success_response(
                    spec_id=result.spec_id,
                    from_folder=result.from_folder,
                    to_folder=result.to_folder,
                    new_path=result.new_path,
                )
            )

        except Exception as e:
            logger.error(f"Error completing spec: {e}")
            return asdict(error_response(sanitize_error_message(e, context="spec lifecycle")))

    @canonical_tool(
        mcp,
        canonical_name="spec-lifecycle-archive",
    )
    def spec_lifecycle_archive(spec_id: str, workspace: Optional[str] = None) -> dict:
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
                return asdict(error_response("No specs directory found"))

            result = archive_spec(spec_id, specs_dir)

            if not result.success:
                return asdict(error_response(result.error or "Failed to archive spec"))

            return asdict(
                success_response(
                    spec_id=result.spec_id,
                    from_folder=result.from_folder,
                    to_folder=result.to_folder,
                    new_path=result.new_path,
                )
            )

        except Exception as e:
            logger.error(f"Error archiving spec: {e}")
            return asdict(error_response(sanitize_error_message(e, context="spec lifecycle")))

    @canonical_tool(
        mcp,
        canonical_name="spec-lifecycle-state",
    )
    def spec_lifecycle_state(spec_id: str, workspace: Optional[str] = None) -> dict:
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
                return asdict(error_response("No specs directory found"))

            state = get_lifecycle_state(spec_id, specs_dir)

            if not state:
                return asdict(error_response(f"Spec not found: {spec_id}"))

            return asdict(
                success_response(
                    spec_id=state.spec_id,
                    folder=state.folder,
                    status=state.status,
                    progress_percentage=state.progress_percentage,
                    total_tasks=state.total_tasks,
                    completed_tasks=state.completed_tasks,
                    can_complete=state.can_complete,
                    can_archive=state.can_archive,
                )
            )

        except Exception as e:
            logger.error(f"Error getting lifecycle state: {e}")
            return asdict(error_response(sanitize_error_message(e, context="spec lifecycle")))

    @canonical_tool(
        mcp,
        canonical_name="spec-list-by-folder",
    )
    def spec_list_by_folder(
        folder: Optional[str] = None,
        workspace: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> dict:
        """
        List specifications organized by folder.

        Args:
            folder: Optional filter to specific folder (pending, active, completed, archived)
            workspace: Optional workspace path
            limit: Number of specs per page (default: 100, max: 1000)
            cursor: Pagination cursor from previous response

        Returns:
            JSON object with specs organized by folder
        """
        try:
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return asdict(error_response("No specs directory found"))

            if folder and folder not in VALID_FOLDERS:
                return asdict(
                    error_response(
                        f"Invalid folder: {folder}. Must be one of: {list(VALID_FOLDERS)}"
                    )
                )

            # Normalize page size
            page_size = normalize_page_size(limit)

            # Decode cursor if provided
            start_after_id = None
            if cursor:
                try:
                    cursor_data = decode_cursor(cursor)
                    start_after_id = cursor_data.get("last_id")
                except CursorError as e:
                    return asdict(
                        error_response(
                            f"Invalid cursor: {e.reason}",
                            code="INVALID_CURSOR",
                            details={"cursor": cursor},
                        )
                    )

            result = list_specs_by_folder(specs_dir, folder)

            # Flatten specs with folder info for pagination
            all_specs = []
            for folder_name, specs in result.items():
                for spec in specs:
                    all_specs.append({**spec, "folder": folder_name})

            # Sort for consistent pagination
            all_specs.sort(key=lambda s: s.get("spec_id", ""))
            total_count = len(all_specs)

            # Find starting position from cursor
            start_index = 0
            if start_after_id:
                for i, spec in enumerate(all_specs):
                    if spec.get("spec_id") == start_after_id:
                        start_index = i + 1
                        break

            # Get page of specs (fetch one extra to detect has_more)
            page_specs = all_specs[start_index : start_index + page_size + 1]
            has_more = len(page_specs) > page_size
            if has_more:
                page_specs = page_specs[:page_size]

            # Build next cursor if more pages exist
            next_cursor = None
            if has_more and page_specs:
                next_cursor = encode_cursor({"last_id": page_specs[-1].get("spec_id")})

            return paginated_response(
                data={
                    "specs": page_specs,
                    "filter": {"folder": folder} if folder else None,
                },
                cursor=next_cursor,
                has_more=has_more,
                page_size=page_size,
                total_count=total_count,
            )

        except Exception as e:
            logger.error(f"Error listing specs by folder: {e}")
            return asdict(error_response(sanitize_error_message(e, context="spec lifecycle")))

    logger.debug(
        "Registered lifecycle tools: spec-lifecycle-move/spec-lifecycle-activate/"
        "spec-lifecycle-complete/spec-lifecycle-archive/spec-lifecycle-state/"
        "spec-list-by-folder"
    )
