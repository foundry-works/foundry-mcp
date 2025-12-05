"""
Rendering tools for foundry-mcp.

Provides MCP tools for spec rendering and visualization.
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
from foundry_mcp.core.spec import (
    find_specs_directory,
    load_spec,
)
from foundry_mcp.core.rendering import (
    render_spec_to_markdown,
    render_progress_bar,
    render_task_list,
    get_status_icon,
    RenderOptions,
)

logger = logging.getLogger(__name__)


def register_rendering_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register rendering tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="spec-render",
    )
    def spec_render(
        spec_id: str,
        mode: str = "basic",
        include_journal: bool = False,
        max_depth: int = 0,
        workspace: Optional[str] = None,
    ) -> dict:
        """
        Render a specification to human-readable markdown.

        Generates formatted documentation with progress visualization,
        task hierarchy, and optional journal summary.

        Args:
            spec_id: Specification ID
            mode: Rendering mode ("basic" or "enhanced")
            include_journal: Whether to include recent journal entries
            max_depth: Maximum depth to render (0 = unlimited)
            workspace: Optional workspace path

        Returns:
            JSON object with markdown content and render metadata
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

            options = RenderOptions(
                mode=mode,
                include_metadata=True,
                include_progress=True,
                include_dependencies=True,
                include_journal=include_journal,
                max_depth=max_depth,
            )

            result = render_spec_to_markdown(spec_data, options)

            return asdict(
                success_response(
                    spec_id=result.spec_id,
                    title=result.title,
                    markdown=result.markdown,
                    total_sections=result.total_sections,
                    total_tasks=result.total_tasks,
                    completed_tasks=result.completed_tasks,
                    progress_percentage=(
                        result.completed_tasks / result.total_tasks * 100
                        if result.total_tasks > 0
                        else 0
                    ),
                )
            )

        except Exception as e:
            logger.error(f"Error rendering spec: {e}")
            return asdict(error_response(sanitize_error_message(e, context="spec rendering")))

    @canonical_tool(
        mcp,
        canonical_name="spec-render-progress",
    )
    def spec_render_progress(
        spec_id: str, bar_width: int = 20, workspace: Optional[str] = None
    ) -> dict:
        """
        Get a visual progress summary for a specification.

        Returns ASCII progress bars for the spec and each phase.

        Args:
            spec_id: Specification ID
            bar_width: Width of progress bars in characters
            workspace: Optional workspace path

        Returns:
            JSON object with progress visualization
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

            hierarchy = spec_data.get("hierarchy", {})
            root = hierarchy.get("spec-root", {})
            metadata = spec_data.get("metadata", {})

            total_tasks = root.get("total_tasks", 0)
            completed_tasks = root.get("completed_tasks", 0)

            # Overall progress
            overall_bar = render_progress_bar(completed_tasks, total_tasks, bar_width)
            overall_icon = get_status_icon(root.get("status", "pending"))

            # Phase progress
            phases = []
            for phase_id in root.get("children", []):
                phase = hierarchy.get(phase_id, {})
                phase_total = phase.get("total_tasks", 0)
                phase_completed = phase.get("completed_tasks", 0)
                phase_status = phase.get("status", "pending")
                phase_icon = get_status_icon(phase_status)
                phase_bar = render_progress_bar(phase_completed, phase_total, bar_width)

                phases.append(
                    {
                        "id": phase_id,
                        "title": phase.get("title", "Untitled"),
                        "status": phase_status,
                        "icon": phase_icon,
                        "progress_bar": phase_bar,
                        "completed": phase_completed,
                        "total": phase_total,
                    }
                )

            return asdict(
                success_response(
                    spec_id=spec_id,
                    title=metadata.get("title") or root.get("title", "Untitled"),
                    overall={
                        "status": root.get("status", "pending"),
                        "icon": overall_icon,
                        "progress_bar": overall_bar,
                        "completed": completed_tasks,
                        "total": total_tasks,
                    },
                    phases=phases,
                )
            )

        except Exception as e:
            logger.error(f"Error rendering progress: {e}")
            return asdict(error_response(sanitize_error_message(e, context="spec rendering")))

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
        try:
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
                except CursorError as e:
                    return asdict(
                        error_response(
                            f"Invalid cursor: {e.reason}",
                            code="INVALID_CURSOR",
                            details={"cursor": cursor},
                        )
                    )

            spec_data = load_spec(spec_id, specs_dir)
            if not spec_data:
                return asdict(error_response(f"Spec not found: {spec_id}"))

            # Extract and filter task data
            hierarchy = spec_data.get("hierarchy", {})
            tasks = []

            for node_id, node in hierarchy.items():
                node_type = node.get("type", "")
                if node_type not in ("task", "subtask", "verify"):
                    continue

                status = node.get("status", "pending")

                if status_filter and status != status_filter:
                    continue

                if not include_completed and status == "completed":
                    continue

                tasks.append(
                    {
                        "id": node_id,
                        "title": node.get("title", "Untitled"),
                        "type": node_type,
                        "status": status,
                        "icon": get_status_icon(status),
                        "file_path": node.get("metadata", {}).get("file_path"),
                        "parent": node.get("parent"),
                    }
                )

            # Sort for consistent pagination
            tasks.sort(key=lambda t: t.get("id", ""))
            total_count = len(tasks)

            # Find starting position from cursor
            start_index = 0
            if start_after_id:
                for i, task in enumerate(tasks):
                    if task.get("id") == start_after_id:
                        start_index = i + 1
                        break

            # Get page of tasks (fetch one extra to detect has_more)
            page_tasks = tasks[start_index : start_index + page_size + 1]
            has_more = len(page_tasks) > page_size
            if has_more:
                page_tasks = page_tasks[:page_size]

            # Build next cursor if more pages exist
            next_cursor = None
            if has_more and page_tasks:
                next_cursor = encode_cursor({"last_id": page_tasks[-1].get("id")})

            return paginated_response(
                data={
                    "spec_id": spec_id,
                    "tasks": page_tasks,
                    "filters": {
                        "status_filter": status_filter,
                        "include_completed": include_completed,
                    },
                },
                cursor=next_cursor,
                has_more=has_more,
                page_size=page_size,
                total_count=total_count,
            )

        except Exception as e:
            logger.error(f"Error listing tasks: {e}")
            return asdict(error_response(sanitize_error_message(e, context="spec rendering")))

    logger.debug(
        "Registered rendering tools: spec-render/spec-render-progress/task-list"
    )
