"""
FastMCP server for foundry-mcp.

Provides MCP tools and resources for SDD spec management.
"""

import logging
import sys
from dataclasses import asdict
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import get_config, ServerConfig
from foundry_mcp.core.observability import audit_log, get_metrics
from foundry_mcp.core.responses import success_response, error_response
from foundry_mcp.core.discovery import get_capabilities, get_tool_registry
from foundry_mcp.core.feature_flags import get_flag_service
from foundry_mcp.core.naming import canonical_tool

# Response contract version flag
# v2 uses standardized {success, data, error} format
RESPONSE_CONTRACT_V2 = True

# Server capabilities
SERVER_CAPABILITIES = {
    "response_contract_v2": RESPONSE_CONTRACT_V2,
    "schema_version": "2.0.0",
}
from foundry_mcp.core.spec import (
    load_spec,
    save_spec,
    list_specs,
    find_specs_directory,
)
from foundry_mcp.tools.queries import register_query_tools
from foundry_mcp.tools.tasks import register_task_tools
from foundry_mcp.tools.validation import register_validation_tools
from foundry_mcp.tools.journal import register_journal_tools
from foundry_mcp.tools.rendering import register_rendering_tools
from foundry_mcp.tools.lifecycle import register_lifecycle_tools
from foundry_mcp.tools.docs import register_docs_tools
from foundry_mcp.tools.testing import register_testing_tools
from foundry_mcp.tools.discovery import register_discovery_tools
from foundry_mcp.tools.environment import register_environment_tools
from foundry_mcp.resources.specs import register_spec_resources
from foundry_mcp.prompts.workflows import register_workflow_prompts

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
    )

    # Register tools
    _register_tools(mcp, config)
    register_query_tools(mcp, config)
    register_task_tools(mcp, config)
    register_validation_tools(mcp, config)
    register_journal_tools(mcp, config)
    register_rendering_tools(mcp, config)
    register_lifecycle_tools(mcp, config)
    register_docs_tools(mcp, config)
    register_testing_tools(mcp, config)
    register_discovery_tools(mcp, config)
    register_environment_tools(mcp, config)

    # Register resources
    _register_resources(mcp, config)
    register_spec_resources(mcp, config)

    # Register prompts
    register_workflow_prompts(mcp, config)

    logger.info(f"Server created: {config.server_name} v{config.server_version}")

    return mcp


def _register_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """Register all MCP tools with the server."""

    @canonical_tool(
        mcp,
        canonical_name="sdd-server-capabilities",
    )
    def sdd_server_capabilities() -> dict:
        """
        Get server capabilities, feature flags, and contract version.

        Returns comprehensive information about the server's response format,
        supported features, and active feature flags.

        WHEN TO USE:
        - Client initialization and feature detection
        - Checking available server features
        - Understanding response format and versioning

        Returns:
            JSON object with server capabilities, feature flags, and tool stats
        """
        metrics = get_metrics()
        metrics.counter("response.v2", labels={"tool": "sdd-server-capabilities"})

        # Get discovery capabilities
        discovery_caps = get_capabilities()

        # Get tool registry stats
        registry = get_tool_registry()
        tool_stats = registry.get_stats()

        # Get active feature flags summary
        flag_service = get_flag_service()
        flags_summary = {
            "enabled_count": len(
                [f for f in flag_service.list_flags() if flag_service.is_enabled(f)]
            ),
            "total_count": len(flag_service.list_flags()),
        }

        return asdict(
            success_response(
                server_name=config.server_name,
                server_version=config.server_version,
                capabilities={
                    **SERVER_CAPABILITIES,
                    **discovery_caps.get("capabilities", {}),
                },
                feature_flags=flags_summary,
                tools=tool_stats,
                schema_version=discovery_caps.get("schema_version", "1.0.0"),
                api_version=discovery_caps.get("api_version", "2024-11-01"),
            )
        )

    @canonical_tool(
        mcp,
        canonical_name="spec-list-basic",
    )
    def spec_list_basic(status: str = "all") -> dict:
        """
        List all specification files with optional filtering.

        Args:
            status: Filter by status (active, pending, completed, archived, or all)

        Returns:
            Dict with list of specs and their progress
        """
        specs_dir = config.specs_dir or find_specs_directory()
        if not specs_dir:
            return asdict(error_response("No specs directory found"))

        filter_status = None if status == "all" else status
        specs = list_specs(specs_dir=specs_dir, status=filter_status)

        return asdict(success_response(specs=specs, count=len(specs)))

    @canonical_tool(
        mcp,
        canonical_name="spec-get",
    )
    def spec_get(spec_id: str) -> dict:
        """
        Get a specification by ID.

        Args:
            spec_id: Specification ID to retrieve

        Returns:
            Dict with spec data or error
        """
        specs_dir = config.specs_dir or find_specs_directory()
        spec_data = load_spec(spec_id, specs_dir)

        if spec_data is None:
            return asdict(error_response(f"Spec not found: {spec_id}"))

        # Return summary for large specs
        hierarchy = spec_data.get("hierarchy", {})
        total_tasks = len(hierarchy)
        completed_tasks = sum(
            1 for task in hierarchy.values() if task.get("status") == "completed"
        )

        return asdict(
            success_response(
                spec_id=spec_data.get("spec_id", spec_id),
                title=spec_data.get("metadata", {}).get(
                    "title", spec_data.get("title", "Untitled")
                ),
                total_tasks=total_tasks,
                completed_tasks=completed_tasks,
                progress_percentage=int((completed_tasks / total_tasks * 100))
                if total_tasks > 0
                else 0,
            )
        )

    @canonical_tool(
        mcp,
        canonical_name="spec-get-hierarchy",
    )
    def spec_get_hierarchy(spec_id: str) -> dict:
        """
        Get the full hierarchy of a specification.

        Args:
            spec_id: Specification ID

        Returns:
            Dict with hierarchy data
        """
        specs_dir = config.specs_dir or find_specs_directory()
        spec_data = load_spec(spec_id, specs_dir)

        if spec_data is None:
            return asdict(error_response(f"Spec not found: {spec_id}"))

        return asdict(
            success_response(
                spec_id=spec_id,
                hierarchy=spec_data.get("hierarchy", {}),
            )
        )

    @canonical_tool(
        mcp,
        canonical_name="task-get",
    )
    def task_get(spec_id: str, task_id: str) -> dict:
        """
        Get a specific task from a specification.

        Args:
            spec_id: Specification ID
            task_id: Task ID within the specification

        Returns:
            Dict with task data
        """
        specs_dir = config.specs_dir or find_specs_directory()
        spec_data = load_spec(spec_id, specs_dir)

        if spec_data is None:
            return asdict(error_response(f"Spec not found: {spec_id}"))

        hierarchy = spec_data.get("hierarchy", {})
        task = hierarchy.get(task_id)

        if task is None:
            return asdict(error_response(f"Task not found: {task_id}"))

        return asdict(
            success_response(
                spec_id=spec_id,
                task_id=task_id,
                task=task,
            )
        )

    logger.debug(
        "Registered tools: sdd-server-capabilities, spec-list-basic, spec-get, spec-get-hierarchy, task-get"
    )


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
