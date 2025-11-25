"""
FastMCP server for foundry-mcp.

Provides MCP tools and resources for SDD spec management.
"""

import logging
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import get_config, ServerConfig
from foundry_mcp.core.observability import mcp_tool, audit_log, get_metrics
from foundry_mcp.core.spec import (
    load_spec,
    save_spec,
    list_specs,
    find_specs_directory,
)
from foundry_mcp.tools.queries import register_query_tools
from foundry_mcp.tools.tasks import register_task_tools

logger = logging.getLogger(__name__)


def create_server(config: Optional[ServerConfig] = None) -> FastMCP:
    """
    Create and configure the FastMCP server instance.

    Args:
        config: Optional server configuration. If not provided, loads from environment.

    Returns:
        Configured FastMCP server instance
    """
    if config is None:
        config = get_config()

    # Setup logging
    config.setup_logging()

    # Create FastMCP server
    mcp = FastMCP(
        name=config.server_name,
        version=config.server_version,
    )

    # Register tools
    _register_tools(mcp, config)
    register_query_tools(mcp, config)
    register_task_tools(mcp, config)

    # Register resources
    _register_resources(mcp, config)

    logger.info(f"Server created: {config.server_name} v{config.server_version}")

    return mcp


def _register_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """Register all MCP tools with the server."""

    @mcp.tool()
    @mcp_tool(tool_name="list_specs")
    def tool_list_specs(status: str = "all") -> str:
        """
        List all specification files with optional filtering.

        Args:
            status: Filter by status (active, pending, completed, archived, or all)

        Returns:
            JSON string with list of specs and their progress
        """
        import json

        specs_dir = config.specs_dir or find_specs_directory()
        if not specs_dir:
            return json.dumps({"error": "No specs directory found"})

        filter_status = None if status == "all" else status
        specs = list_specs(specs_dir=specs_dir, status=filter_status)

        return json.dumps({"specs": specs, "count": len(specs)})

    @mcp.tool()
    @mcp_tool(tool_name="get_spec")
    def tool_get_spec(spec_id: str) -> str:
        """
        Get a specification by ID.

        Args:
            spec_id: Specification ID to retrieve

        Returns:
            JSON string with spec data or error
        """
        import json

        specs_dir = config.specs_dir or find_specs_directory()
        spec_data = load_spec(spec_id, specs_dir)

        if spec_data is None:
            return json.dumps({"error": f"Spec not found: {spec_id}"})

        # Return summary for large specs
        hierarchy = spec_data.get("hierarchy", {})
        total_tasks = len(hierarchy)
        completed_tasks = sum(
            1 for task in hierarchy.values()
            if task.get("status") == "completed"
        )

        return json.dumps({
            "spec_id": spec_data.get("spec_id", spec_id),
            "title": spec_data.get("metadata", {}).get("title", spec_data.get("title", "Untitled")),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "progress_percentage": int((completed_tasks / total_tasks * 100)) if total_tasks > 0 else 0,
        })

    @mcp.tool()
    @mcp_tool(tool_name="get_spec_hierarchy")
    def tool_get_spec_hierarchy(spec_id: str) -> str:
        """
        Get the full hierarchy of a specification.

        Args:
            spec_id: Specification ID

        Returns:
            JSON string with hierarchy data
        """
        import json

        specs_dir = config.specs_dir or find_specs_directory()
        spec_data = load_spec(spec_id, specs_dir)

        if spec_data is None:
            return json.dumps({"error": f"Spec not found: {spec_id}"})

        return json.dumps({
            "spec_id": spec_id,
            "hierarchy": spec_data.get("hierarchy", {}),
        })

    @mcp.tool()
    @mcp_tool(tool_name="get_task")
    def tool_get_task(spec_id: str, task_id: str) -> str:
        """
        Get a specific task from a specification.

        Args:
            spec_id: Specification ID
            task_id: Task ID within the specification

        Returns:
            JSON string with task data
        """
        import json

        specs_dir = config.specs_dir or find_specs_directory()
        spec_data = load_spec(spec_id, specs_dir)

        if spec_data is None:
            return json.dumps({"error": f"Spec not found: {spec_id}"})

        hierarchy = spec_data.get("hierarchy", {})
        task = hierarchy.get(task_id)

        if task is None:
            return json.dumps({"error": f"Task not found: {task_id}"})

        return json.dumps({
            "spec_id": spec_id,
            "task_id": task_id,
            "task": task,
        })

    logger.debug("Registered tools: list_specs, get_spec, get_spec_hierarchy, get_task")


def _register_resources(mcp: FastMCP, config: ServerConfig) -> None:
    """Register all MCP resources with the server."""

    @mcp.resource("specs://list")
    def resource_specs_list() -> str:
        """List all specifications as a resource."""
        import json

        specs_dir = config.specs_dir or find_specs_directory()
        if not specs_dir:
            return json.dumps({"error": "No specs directory found"})

        specs = list_specs(specs_dir=specs_dir)
        return json.dumps({"specs": specs})

    @mcp.resource("specs://{spec_id}")
    def resource_spec(spec_id: str) -> str:
        """Get a specification by ID as a resource."""
        import json

        specs_dir = config.specs_dir or find_specs_directory()
        spec_data = load_spec(spec_id, specs_dir)

        if spec_data is None:
            return json.dumps({"error": f"Spec not found: {spec_id}"})

        return json.dumps(spec_data)

    logger.debug("Registered resources: specs://list, specs://{spec_id}")


def main() -> None:
    """Main entry point for the foundry-mcp server."""
    try:
        config = get_config()
        server = create_server(config)

        logger.info(f"Starting {config.server_name} v{config.server_version}")
        audit_log("tool_invocation", tool="server_start", version=config.server_version)

        # Run the server
        server.run()

    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        audit_log("tool_invocation", tool="server_error", error=str(e), success=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
