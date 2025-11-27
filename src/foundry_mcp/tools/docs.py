"""
Documentation tools for foundry-mcp.

Provides MCP tools for querying codebase documentation.
"""

import logging
from dataclasses import asdict
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.observability import mcp_tool
from foundry_mcp.core.docs import DocsQuery
from foundry_mcp.core.pagination import (
    encode_cursor,
    decode_cursor,
    CursorError,
    normalize_page_size,
)
from foundry_mcp.core.responses import success_response, error_response

logger = logging.getLogger(__name__)


def register_docs_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register documentation tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    def _get_query(workspace: Optional[str] = None) -> DocsQuery:
        """Get a DocsQuery instance for the given workspace."""
        from pathlib import Path
        ws = Path(workspace) if workspace else (config.specs_dir.parent if config.specs_dir else None)
        return DocsQuery(workspace=ws)

    @mcp.tool()
    @mcp_tool(tool_name="foundry_find_class")
    def foundry_find_class(
        name: str,
        exact: bool = True,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Find a class by name in codebase documentation with optional pagination.

        Searches loaded codebase.json for class definitions matching
        the given name.

        Args:
            name: Class name to search for
            exact: If True, exact match; if False, substring match
            cursor: Pagination cursor from previous response
            limit: Number of results per page (default: 100, max: 1000)
            workspace: Optional workspace path (defaults to config)

        Returns:
            JSON object with matching classes and pagination metadata
        """
        try:
            query = _get_query(workspace)
            if not query.load():
                return asdict(error_response("Documentation not loaded. Run 'sdd doc generate' first."))

            result = query.find_class(name, exact)

            if not result.success:
                return asdict(error_response(result.error))

            # Normalize page size
            page_size = normalize_page_size(limit)

            # Decode cursor if provided
            start_after_name = None
            if cursor:
                try:
                    cursor_data = decode_cursor(cursor)
                    start_after_name = cursor_data.get("last_name")
                except CursorError:
                    return asdict(error_response("Invalid pagination cursor"))

            # Sort results by name for consistent pagination
            all_results = sorted(result.results, key=lambda r: r.name)

            # Apply cursor-based pagination
            if start_after_name:
                start_index = 0
                for i, r in enumerate(all_results):
                    if r.name == start_after_name:
                        start_index = i + 1
                        break
                all_results = all_results[start_index:]

            # Fetch one extra to detect has_more
            page_results = all_results[: page_size + 1]
            has_more = len(page_results) > page_size
            if has_more:
                page_results = page_results[:page_size]

            # Build next cursor
            next_cursor = None
            if has_more and page_results:
                next_cursor = encode_cursor({"last_name": page_results[-1].name})

            return asdict(success_response(
                data={
                    "query_type": result.query_type,
                    "count": len(page_results),
                    "results": [
                        {
                            "name": r.name,
                            "file_path": r.file_path,
                            "line_number": r.line_number,
                            "data": r.data,
                        }
                        for r in page_results
                    ]
                },
                pagination={
                    "cursor": next_cursor,
                    "has_more": has_more,
                    "page_size": page_size,
                }
            ))

        except Exception as e:
            logger.error(f"Error finding class: {e}")
            return asdict(error_response(str(e)))

    @mcp.tool()
    @mcp_tool(tool_name="foundry_find_function")
    def foundry_find_function(
        name: str,
        exact: bool = True,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Find a function by name in codebase documentation with optional pagination.

        Searches loaded codebase.json for function definitions matching
        the given name.

        Args:
            name: Function name to search for
            exact: If True, exact match; if False, substring match
            cursor: Pagination cursor from previous response
            limit: Number of results per page (default: 100, max: 1000)
            workspace: Optional workspace path (defaults to config)

        Returns:
            JSON object with matching functions and pagination metadata
        """
        try:
            query = _get_query(workspace)
            if not query.load():
                return asdict(error_response("Documentation not loaded. Run 'sdd doc generate' first."))

            result = query.find_function(name, exact)

            if not result.success:
                return asdict(error_response(result.error))

            # Normalize page size
            page_size = normalize_page_size(limit)

            # Decode cursor if provided
            start_after_name = None
            if cursor:
                try:
                    cursor_data = decode_cursor(cursor)
                    start_after_name = cursor_data.get("last_name")
                except CursorError:
                    return asdict(error_response("Invalid pagination cursor"))

            # Sort results by name for consistent pagination
            all_results = sorted(result.results, key=lambda r: r.name)

            # Apply cursor-based pagination
            if start_after_name:
                start_index = 0
                for i, r in enumerate(all_results):
                    if r.name == start_after_name:
                        start_index = i + 1
                        break
                all_results = all_results[start_index:]

            # Fetch one extra to detect has_more
            page_results = all_results[: page_size + 1]
            has_more = len(page_results) > page_size
            if has_more:
                page_results = page_results[:page_size]

            # Build next cursor
            next_cursor = None
            if has_more and page_results:
                next_cursor = encode_cursor({"last_name": page_results[-1].name})

            return asdict(success_response(
                data={
                    "query_type": result.query_type,
                    "count": len(page_results),
                    "results": [
                        {
                            "name": r.name,
                            "file_path": r.file_path,
                            "line_number": r.line_number,
                            "data": r.data,
                        }
                        for r in page_results
                    ]
                },
                pagination={
                    "cursor": next_cursor,
                    "has_more": has_more,
                    "page_size": page_size,
                }
            ))

        except Exception as e:
            logger.error(f"Error finding function: {e}")
            return asdict(error_response(str(e)))

    @mcp.tool()
    @mcp_tool(tool_name="foundry_trace_calls")
    def foundry_trace_calls(
        function_name: str,
        direction: str = "both",
        max_depth: int = 3,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Trace function calls in the call graph.

        Follows caller/callee relationships to show function dependencies.

        Args:
            function_name: Function to trace from
            direction: "callers" (who calls this), "callees" (what this calls), or "both"
            max_depth: Maximum traversal depth (default: 3)
            workspace: Optional workspace path (defaults to config)

        Returns:
            JSON object with call graph entries
        """
        try:
            query = _get_query(workspace)
            if not query.load():
                return asdict(error_response("Documentation not loaded. Run 'sdd doc generate' first."))

            result = query.trace_calls(function_name, direction, max_depth)

            if not result.success:
                return asdict(error_response(result.error))

            return asdict(success_response(
                query_type=result.query_type,
                count=result.count,
                results=[
                    {
                        "caller": entry.caller,
                        "callee": entry.callee,
                        "caller_file": entry.caller_file,
                        "callee_file": entry.callee_file,
                    }
                    for entry in result.results
                ],
                metadata=result.metadata
            ))

        except Exception as e:
            logger.error(f"Error tracing calls: {e}")
            return asdict(error_response(str(e)))

    @mcp.tool()
    @mcp_tool(tool_name="foundry_impact_analysis")
    def foundry_impact_analysis(
        target: str,
        target_type: str = "auto",
        max_depth: int = 3,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Analyze impact of changing a class or function.

        Identifies direct and indirect impacts of modifying the target,
        including affected files and an impact score.

        Args:
            target: Name of class or function to analyze
            target_type: "class", "function", or "auto" (detect from name)
            max_depth: Maximum depth for impact propagation (default: 3)
            workspace: Optional workspace path (defaults to config)

        Returns:
            JSON object with impact analysis
        """
        try:
            query = _get_query(workspace)
            if not query.load():
                return asdict(error_response("Documentation not loaded. Run 'sdd doc generate' first."))

            result = query.impact_analysis(target, target_type, max_depth)

            if result.success and result.results:
                impact = result.results[0]
                return asdict(success_response(
                    query_type=result.query_type,
                    target=impact.target,
                    target_type=impact.target_type,
                    impact_score=impact.impact_score,
                    direct_impacts=impact.direct_impacts,
                    indirect_impacts=impact.indirect_impacts,
                    affected_files=impact.affected_files,
                    metadata=result.metadata
                ))
            else:
                return asdict(error_response(result.error or "No impact analysis available"))

        except Exception as e:
            logger.error(f"Error analyzing impact: {e}")
            return asdict(error_response(str(e)))

    @mcp.tool()
    @mcp_tool(tool_name="foundry_get_callers")
    def foundry_get_callers(
        function_name: str,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Get functions that call the specified function.

        Args:
            function_name: Function to find callers for
            workspace: Optional workspace path (defaults to config)

        Returns:
            JSON object with caller functions
        """
        try:
            query = _get_query(workspace)
            if not query.load():
                return asdict(error_response("Documentation not loaded. Run 'sdd doc generate' first."))

            result = query.get_callers(function_name)

            if not result.success:
                return asdict(error_response(result.error))

            return asdict(success_response(
                query_type=result.query_type,
                count=result.count,
                callers=[r.name for r in result.results],
                results=[
                    {
                        "name": r.name,
                        "file_path": r.file_path,
                    }
                    for r in result.results
                ],
                metadata=result.metadata
            ))

        except Exception as e:
            logger.error(f"Error getting callers: {e}")
            return asdict(error_response(str(e)))

    @mcp.tool()
    @mcp_tool(tool_name="foundry_get_callees")
    def foundry_get_callees(
        function_name: str,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Get functions called by the specified function.

        Args:
            function_name: Function to find callees for
            workspace: Optional workspace path (defaults to config)

        Returns:
            JSON object with callee functions
        """
        try:
            query = _get_query(workspace)
            if not query.load():
                return asdict(error_response("Documentation not loaded. Run 'sdd doc generate' first."))

            result = query.get_callees(function_name)

            if not result.success:
                return asdict(error_response(result.error))

            return asdict(success_response(
                query_type=result.query_type,
                count=result.count,
                callees=[r.name for r in result.results],
                results=[
                    {
                        "name": r.name,
                        "file_path": r.file_path,
                    }
                    for r in result.results
                ],
                metadata=result.metadata
            ))

        except Exception as e:
            logger.error(f"Error getting callees: {e}")
            return asdict(error_response(str(e)))

    @mcp.tool()
    @mcp_tool(tool_name="foundry_docs_stats")
    def foundry_docs_stats(
        workspace: Optional[str] = None
    ) -> dict:
        """
        Get documentation statistics.

        Returns counts of classes, functions, files, and dependencies
        in the loaded documentation.

        Args:
            workspace: Optional workspace path (defaults to config)

        Returns:
            JSON object with documentation statistics
        """
        try:
            query = _get_query(workspace)
            if not query.load():
                return asdict(error_response("Documentation not loaded. Run 'sdd doc generate' first."))

            result = query.get_stats()

            if result.success and result.results:
                stats = result.results[0]
                return asdict(success_response(stats=stats))
            else:
                return asdict(error_response(result.error or "No stats available"))

        except Exception as e:
            logger.error(f"Error getting docs stats: {e}")
            return asdict(error_response(str(e)))

    logger.debug("Registered docs tools: foundry_find_class, foundry_find_function, "
                 "foundry_trace_calls, foundry_impact_analysis, foundry_get_callers, "
                 "foundry_get_callees, foundry_docs_stats")
