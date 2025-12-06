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
from foundry_mcp.core.observability import audit_log, get_metrics, get_observability_manager
from foundry_mcp.core.responses import success_response, error_response
from foundry_mcp.core.pagination import (
    encode_cursor,
    decode_cursor,
    paginated_response,
    normalize_page_size,
    CursorError,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
)
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
from foundry_mcp.tools.health import register_health_tools
from foundry_mcp.tools.spec_helpers import register_spec_helper_tools
from foundry_mcp.tools.authoring import register_authoring_tools
from foundry_mcp.tools.analysis import register_analysis_tools
from foundry_mcp.tools.mutations import register_mutation_tools
from foundry_mcp.tools.reporting import register_reporting_tools
from foundry_mcp.tools.utilities import register_utility_tools
from foundry_mcp.tools.context import register_context_tools
from foundry_mcp.tools.review import register_review_tools
from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools
from foundry_mcp.tools.documentation import register_documentation_tools
from foundry_mcp.tools.providers import register_provider_tools
from foundry_mcp.tools.errors import register_error_tools
from foundry_mcp.tools.metrics import register_metrics_tools
from foundry_mcp.resources.specs import register_spec_resources
from foundry_mcp.prompts.workflows import register_workflow_prompts

logger = logging.getLogger(__name__)


def _init_observability(config: ServerConfig) -> None:
    """Initialize the observability stack from server configuration.

    Initializes OpenTelemetry tracing and Prometheus metrics based on
    the configuration. Gracefully handles missing optional dependencies.

    Args:
        config: ServerConfig with observability settings
    """
    obs_config = config.observability

    # Skip initialization if observability is disabled
    if not obs_config.enabled:
        logger.debug("Observability disabled in configuration")
        return

    # Initialize via ObservabilityManager
    manager = get_observability_manager()
    manager.initialize(obs_config)

    # Log initialization status
    tracing_status = "enabled" if manager.is_tracing_enabled() else "disabled"
    metrics_status = "enabled" if manager.is_metrics_enabled() else "disabled"
    logger.info(
        f"Observability initialized: tracing={tracing_status}, metrics={metrics_status}"
    )


def _init_error_collection(config: ServerConfig) -> None:
    """Initialize the error collection infrastructure.

    Sets up the ErrorCollector with the configured storage backend
    and retention settings.

    Args:
        config: ServerConfig with error collection settings
    """
    err_config = config.error_collection

    # Skip initialization if error collection is disabled
    if not err_config.enabled:
        logger.debug("Error collection disabled in configuration")
        return

    try:
        from foundry_mcp.core.error_collection import get_error_collector
        from foundry_mcp.core.error_store import get_error_store

        # Initialize the error store with configured path
        storage_path = err_config.get_storage_path()
        store = get_error_store(storage_path)

        # Initialize the collector with the store
        collector = get_error_collector()
        collector.initialize(store, err_config)

        logger.info(f"Error collection initialized: storage_path={storage_path}")
    except Exception as e:
        # Don't fail server startup due to error collection issues
        logger.warning(f"Failed to initialize error collection: {e}")


def _init_metrics_persistence(config: ServerConfig) -> None:
    """Initialize the metrics persistence infrastructure.

    Sets up the MetricsPersistenceCollector with the configured storage
    backend and persistence settings. Runs retention cleanup on startup.

    Args:
        config: ServerConfig with metrics persistence settings
    """
    metrics_config = config.metrics_persistence

    # Skip initialization if metrics persistence is disabled
    if not metrics_config.enabled:
        logger.debug("Metrics persistence disabled in configuration")
        return

    try:
        from foundry_mcp.core.metrics_persistence import initialize_metrics_persistence
        from foundry_mcp.core.metrics_store import get_metrics_store

        collector = initialize_metrics_persistence(metrics_config)

        if collector is not None:
            storage_path = metrics_config.get_storage_path()

            # Run retention cleanup on startup
            store = get_metrics_store(storage_path)
            deleted_count = store.cleanup(
                retention_days=metrics_config.retention_days,
                max_records=metrics_config.max_records,
            )

            if deleted_count > 0:
                logger.info(f"Metrics cleanup: removed {deleted_count} old records")

            logger.info(f"Metrics persistence initialized: storage_path={storage_path}")
    except Exception as e:
        # Don't fail server startup due to metrics persistence issues
        logger.warning(f"Failed to initialize metrics persistence: {e}")


# Dashboard is now decoupled from MCP server - run separately via:
#   python -m foundry_mcp.dashboard
# This avoids any interference with MCP stdio transport.


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

    # Initialize observability (OTel + Prometheus)
    _init_observability(config)

    # Initialize error collection infrastructure
    _init_error_collection(config)

    # Initialize metrics persistence infrastructure
    _init_metrics_persistence(config)

    # Note: Dashboard is now decoupled - run separately via:
    #   python -m foundry_mcp.dashboard

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
    register_health_tools(mcp, config)
    register_spec_helper_tools(mcp, config)
    register_authoring_tools(mcp, config)
    register_analysis_tools(mcp, config)
    register_mutation_tools(mcp, config)
    register_reporting_tools(mcp, config)
    register_utility_tools(mcp, config)
    register_context_tools(mcp, config)
    register_review_tools(mcp, config)
    register_pr_workflow_tools(mcp, config)
    register_documentation_tools(mcp, config)
    register_provider_tools(mcp, config)
    register_error_tools(mcp, config)
    register_metrics_tools(mcp, config)

    # Register resources
    _register_resources(mcp, config)
    register_spec_resources(mcp, config)

    # Register prompts
    register_workflow_prompts(mcp, config)

    logger.info(f"Server created: {config.server_name} v{config.server_version}")

    return mcp


def _filter_hierarchy(
    hierarchy: dict,
    max_depth: int,
    include_metadata: bool,
    current_depth: int = 0,
) -> dict:
    """Filter hierarchy to reduce response size.

    Args:
        hierarchy: Full hierarchy dict
        max_depth: Maximum depth (0=unlimited)
        include_metadata: Whether to include full metadata
        current_depth: Current traversal depth

    Returns:
        Filtered hierarchy dict
    """
    result = {}

    for node_id, node_data in hierarchy.items():
        # Calculate node depth from ID (spec-root=0, phase-N=1, task-N-N=2, etc.)
        node_depth = node_id.count("-") if node_id != "spec-root" else 0

        # Skip nodes beyond max_depth
        if max_depth > 0 and node_depth > max_depth:
            continue

        # Build filtered node
        filtered_node = {
            "type": node_data.get("type"),
            "title": node_data.get("title"),
            "status": node_data.get("status"),
        }

        # Include children refs (but they'll be filtered by depth)
        if "children" in node_data:
            filtered_node["children"] = node_data["children"]

        # Include parent ref
        if "parent" in node_data:
            filtered_node["parent"] = node_data["parent"]

        # Optionally include full metadata
        if include_metadata:
            if "metadata" in node_data:
                filtered_node["metadata"] = node_data["metadata"]
            if "dependencies" in node_data:
                filtered_node["dependencies"] = node_data["dependencies"]

        result[node_id] = filtered_node

    return result


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
    def spec_get_hierarchy(
        spec_id: str,
        max_depth: int = 2,
        include_metadata: bool = False,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> dict:
        """
        Get the full hierarchy of a specification.

        Args:
            spec_id: Specification ID
            max_depth: Maximum depth to traverse (0=unlimited, default=2 for phases+tasks)
            include_metadata: Include full metadata for each node (default=False for compact output)
            limit: Number of nodes per page (default: 100, max: 1000)
            cursor: Pagination cursor from previous response

        Returns:
            Dict with hierarchy data
        """
        # Input validation per MCP best practices
        if max_depth < 0 or max_depth > 10:
            return asdict(
                error_response(
                    "max_depth must be between 0 and 10",
                    code="VALIDATION_ERROR",
                    details={"field": "max_depth", "received": max_depth, "valid_range": "0-10"},
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

        specs_dir = config.specs_dir or find_specs_directory()
        spec_data = load_spec(spec_id, specs_dir)

        if spec_data is None:
            return asdict(error_response(f"Spec not found: {spec_id}"))

        full_hierarchy = spec_data.get("hierarchy", {})
        full_node_count = len(full_hierarchy)

        # Apply depth and metadata filtering for compact output
        if max_depth > 0 or not include_metadata:
            filtered_hierarchy = _filter_hierarchy(full_hierarchy, max_depth, include_metadata)
        else:
            filtered_hierarchy = full_hierarchy

        filtered_count = len(filtered_hierarchy)

        # Sort node IDs for consistent pagination order
        sorted_ids = sorted(filtered_hierarchy.keys())

        # Find starting position from cursor
        start_index = 0
        if start_after_id:
            try:
                start_index = sorted_ids.index(start_after_id) + 1
            except ValueError:
                # Cursor points to non-existent node (maybe filtered out)
                start_index = 0

        # Get page of nodes (fetch one extra to detect has_more)
        page_ids = sorted_ids[start_index : start_index + page_size + 1]
        has_more = len(page_ids) > page_size
        if has_more:
            page_ids = page_ids[:page_size]

        # Build paginated hierarchy
        hierarchy = {node_id: filtered_hierarchy[node_id] for node_id in page_ids}

        # Build next cursor if more pages exist
        next_cursor = None
        if has_more and page_ids:
            next_cursor = encode_cursor({"last_id": page_ids[-1]})

        return paginated_response(
            data={
                "spec_id": spec_id,
                "hierarchy": hierarchy,
                "node_count": len(hierarchy),
                "total_nodes": filtered_count,
                "filters_applied": {
                    "max_depth": max_depth,
                    "include_metadata": include_metadata,
                },
            },
            cursor=next_cursor,
            has_more=has_more,
            page_size=page_size,
            total_count=filtered_count,
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
        # Shutdown observability to flush pending traces/metrics
        get_observability_manager().shutdown()
        sys.exit(0)
    except BaseException as e:
        # Log detailed error info, especially for ExceptionGroups
        logger.error(f"Server error: {type(e).__name__}: {e}")
        if hasattr(e, 'exceptions'):
            # Handle ExceptionGroup/TaskGroup
            for i, sub_exc in enumerate(e.exceptions):
                logger.error(f"  Sub-exception {i}: {type(sub_exc).__name__}: {sub_exc}")
                import traceback
                tb_str = ''.join(traceback.format_exception(type(sub_exc), sub_exc, sub_exc.__traceback__))
                logger.error(f"  Traceback:\n{tb_str}")
        audit_log("tool_invocation", tool="server_error", error=str(e), success=False)
        # Shutdown observability to flush pending traces/metrics
        get_observability_manager().shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
