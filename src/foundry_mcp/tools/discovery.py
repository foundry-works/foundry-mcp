"""
Discovery tools for foundry-mcp.

Provides MCP tools for tool discovery, capability negotiation,
and metadata introspection.
"""

import logging
from dataclasses import asdict
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.pagination import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    encode_cursor,
    decode_cursor,
    CursorError,
    paginated_response,
)
from foundry_mcp.core.discovery import (
    get_tool_registry,
    get_capabilities,
    negotiate_capabilities,
)

logger = logging.getLogger(__name__)


def register_discovery_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register discovery tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="tool-list",
    )
    def tool_list(
        category: Optional[str] = None,
        tag: Optional[str] = None,
        include_deprecated: bool = False,
        cursor: Optional[str] = None,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> dict:
        """
        List available tools with filtering and pagination.

        Returns a paginated list of tools with optional filtering by
        category, tag, and deprecation status.

        WHEN TO USE:
        - Discovering what tools are available
        - Finding tools by category or tag
        - Browsing tool capabilities before invocation

        Args:
            category: Filter by category (e.g., "specs", "tasks", "docs")
            tag: Filter by semantic tag (e.g., "read", "write", "query")
            include_deprecated: Include deprecated tools (default: False)
            cursor: Pagination cursor from previous response
            limit: Number of tools to return (default: 100, max: 1000)

        Returns:
            JSON object with tools list and pagination metadata
        """
        try:
            # Validate and clamp limit
            limit = min(max(1, limit), MAX_PAGE_SIZE)

            # Get registry and list tools
            registry = get_tool_registry()
            all_tools = registry.list_tools(
                category=category,
                tag=tag,
                include_deprecated=include_deprecated,
            )

            # Handle cursor-based pagination
            start_idx = 0
            if cursor:
                try:
                    cursor_data = decode_cursor(cursor)
                    start_idx = cursor_data.get("offset", 0)
                except CursorError as e:
                    return asdict(
                        error_response(
                            f"Invalid cursor: {e}",
                            error_code="INVALID_CURSOR",
                            error_type="validation",
                        )
                    )

            # Paginate results
            end_idx = start_idx + limit
            paginated_tools = all_tools[start_idx:end_idx]
            has_more = end_idx < len(all_tools)

            # Generate next cursor if there are more results
            next_cursor = None
            if has_more:
                next_cursor = encode_cursor({"offset": end_idx})

            # Get categories for response
            categories = registry.list_categories()

            return paginated_response(
                data={
                    "tools": paginated_tools,
                    "categories": [c["name"] for c in categories],
                    "filters_applied": {
                        "category": category,
                        "tag": tag,
                        "include_deprecated": include_deprecated,
                    },
                },
                cursor=next_cursor,
                has_more=has_more,
                page_size=limit,
                total_count=len(all_tools),
            )

        except Exception as e:
            logger.exception("Error listing tools")
            return asdict(error_response(f"Failed to list tools: {e}"))

    @canonical_tool(
        mcp,
        canonical_name="tool-get-schema",
    )
    def tool_get_schema(tool_name: str) -> dict:
        """
        Get detailed schema for a specific tool.

        Returns full JSON Schema, examples, rate limits, and related tools
        for tool introspection and validation.

        WHEN TO USE:
        - Understanding tool parameters before invocation
        - Getting example inputs and outputs
        - Checking rate limits and constraints
        - Finding related tools

        Args:
            tool_name: Name of the tool to get schema for

        Returns:
            JSON object with tool schema and metadata
        """
        try:
            if not tool_name:
                return asdict(
                    error_response(
                        "tool_name is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a tool name to get its schema",
                    )
                )

            registry = get_tool_registry()
            schema = registry.get_tool_schema(tool_name)

            if schema is None:
                return asdict(
                    error_response(
                        f"Tool '{tool_name}' not found",
                        error_code="NOT_FOUND",
                        error_type="not_found",
                        remediation="Use tool-list to see available tools",
                    )
                )

            return asdict(success_response(data=schema))

        except Exception as e:
            logger.exception(f"Error getting tool schema for {tool_name}")
            return asdict(error_response(f"Failed to get tool schema: {e}"))

    @canonical_tool(
        mcp,
        canonical_name="capability-get",
    )
    def capability_get() -> dict:
        """
        Get server capabilities for client negotiation.

        Returns information about supported features including response
        version, streaming support, batch operations, pagination, and
        feature flags.

        WHEN TO USE:
        - Understanding server capabilities before making requests
        - Checking if specific features are supported
        - Client initialization and feature detection

        Returns:
            JSON object with server capabilities and version info
        """
        try:
            capabilities = get_capabilities()
            return asdict(success_response(data=capabilities))

        except Exception as e:
            logger.exception("Error getting capabilities")
            return asdict(error_response(f"Failed to get capabilities: {e}"))

    @canonical_tool(
        mcp,
        canonical_name="capability-negotiate",
    )
    def capability_negotiate(
        requested_version: Optional[str] = None,
        requested_features: Optional[str] = None,
    ) -> dict:
        """
        Negotiate capabilities with the server.

        Allows clients to request specific response versions and features.
        Server responds with what it can actually provide and any warnings
        about unsupported requests.

        Args:
            requested_version: Desired response version (e.g., "response-v2")
            requested_features: Comma-separated list of requested features
                               (e.g., "streaming,batch,pagination")

        Returns:
            JSON object with negotiated capabilities and warnings
        """
        try:
            # Parse features from comma-separated string
            features_list = None
            if requested_features:
                features_list = [f.strip() for f in requested_features.split(",")]

            result = negotiate_capabilities(
                requested_version=requested_version,
                requested_features=features_list,
            )

            return asdict(success_response(data=result))

        except Exception as e:
            logger.exception("Error negotiating capabilities")
            return asdict(error_response(f"Failed to negotiate capabilities: {e}"))

    @canonical_tool(
        mcp,
        canonical_name="tool-list-categories",
    )
    def tool_list_categories() -> dict:
        """
        List tool categories with tool counts.

        Returns all available tool categories and the number of
        (non-deprecated) tools in each category.

        WHEN TO USE:
        - Understanding how tools are organized
        - Filtering tools by category
        - Getting an overview of available functionality

        Returns:
            JSON object with categories and tool counts
        """
        try:
            registry = get_tool_registry()
            categories = registry.list_categories()
            stats = registry.get_stats()

            return asdict(
                success_response(
                    data={
                        "categories": categories,
                        "stats": stats,
                    }
                )
            )

        except Exception as e:
            logger.exception("Error listing categories")
            return asdict(error_response(f"Failed to list categories: {e}"))
