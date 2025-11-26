"""
Testing tools for foundry-mcp.

Provides MCP tools for running and discovering tests.
"""

import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.observability import mcp_tool
from foundry_mcp.core.testing import (
    TestRunner,
    get_presets,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)


def register_testing_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register testing tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    def _get_runner(workspace: Optional[str] = None) -> TestRunner:
        """Get a TestRunner instance for the given workspace."""
        from pathlib import Path
        ws = Path(workspace) if workspace else (config.specs_dir.parent if config.specs_dir else None)
        return TestRunner(workspace=ws)

    @mcp.tool()
    @mcp_tool(tool_name="foundry_run_tests")
    def foundry_run_tests(
        target: Optional[str] = None,
        preset: Optional[str] = None,
        timeout: int = 300,
        verbose: bool = True,
        fail_fast: bool = False,
        markers: Optional[str] = None,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Run tests using pytest.

        Executes tests with configurable options including presets,
        markers, and timeout.

        Args:
            target: Test target (file, directory, or test name pattern)
            preset: Use a preset configuration (quick, full, unit, integration, smoke)
            timeout: Timeout in seconds (default: 300)
            verbose: Enable verbose output (default: True)
            fail_fast: Stop on first failure (default: False)
            markers: Pytest markers expression (e.g., "not slow")
            workspace: Optional workspace path (defaults to config)

        Returns:
            JSON object with test results and schema_version
        """
        try:
            runner = _get_runner(workspace)
            result = runner.run_tests(
                target=target,
                preset=preset,
                timeout=timeout,
                verbose=verbose,
                fail_fast=fail_fast,
                markers=markers,
            )

            return {
                "success": result.success,
                "schema_version": result.schema_version,
                "execution_id": result.execution_id,
                "timestamp": result.timestamp,
                "summary": {
                    "total": result.total,
                    "passed": result.passed,
                    "failed": result.failed,
                    "skipped": result.skipped,
                    "errors": result.errors,
                },
                "tests": [
                    {
                        "name": t.name,
                        "outcome": t.outcome,
                        "duration": t.duration,
                        "message": t.message,
                    }
                    for t in result.tests
                ],
                "command": result.command,
                "duration": result.duration,
                "error": result.error,
                "metadata": result.metadata,
            }

        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return {
                "success": False,
                "schema_version": SCHEMA_VERSION,
                "error": str(e),
            }

    @mcp.tool()
    @mcp_tool(tool_name="foundry_discover_tests")
    def foundry_discover_tests(
        target: Optional[str] = None,
        pattern: str = "test_*.py",
        workspace: Optional[str] = None
    ) -> dict:
        """
        Discover tests without running them.

        Collects test information including names, files, and markers.

        Args:
            target: Directory or file to search
            pattern: File pattern for test files (default: test_*.py)
            workspace: Optional workspace path (defaults to config)

        Returns:
            JSON object with discovered tests and schema_version
        """
        try:
            runner = _get_runner(workspace)
            result = runner.discover_tests(target=target, pattern=pattern)

            return {
                "success": result.success,
                "schema_version": result.schema_version,
                "timestamp": result.timestamp,
                "total": result.total,
                "test_files": result.test_files,
                "tests": [
                    {
                        "name": t.name,
                        "file_path": t.file_path,
                        "line_number": t.line_number,
                        "markers": t.markers,
                    }
                    for t in result.tests
                ],
                "error": result.error,
                "metadata": result.metadata,
            }

        except Exception as e:
            logger.error(f"Error discovering tests: {e}")
            return {
                "success": False,
                "schema_version": SCHEMA_VERSION,
                "error": str(e),
            }

    @mcp.tool()
    @mcp_tool(tool_name="foundry_test_presets")
    def foundry_test_presets() -> dict:
        """
        Get available test presets.

        Lists configured presets with their settings (timeout, markers, etc.).

        Returns:
            JSON object with preset configurations and schema_version
        """
        try:
            presets = get_presets()

            return {
                "success": True,
                "schema_version": SCHEMA_VERSION,
                "presets": presets,
                "available": list(presets.keys()),
            }

        except Exception as e:
            logger.error(f"Error getting presets: {e}")
            return {
                "success": False,
                "schema_version": SCHEMA_VERSION,
                "error": str(e),
            }

    @mcp.tool()
    @mcp_tool(tool_name="foundry_run_quick_tests")
    def foundry_run_quick_tests(
        target: Optional[str] = None,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Run quick tests (preset: quick).

        Fast test run with fail_fast enabled and slow tests excluded.

        Args:
            target: Test target (file, directory, or test name pattern)
            workspace: Optional workspace path (defaults to config)

        Returns:
            JSON object with test results and schema_version
        """
        try:
            runner = _get_runner(workspace)
            result = runner.run_tests(target=target, preset="quick")

            return {
                "success": result.success,
                "schema_version": result.schema_version,
                "execution_id": result.execution_id,
                "summary": {
                    "total": result.total,
                    "passed": result.passed,
                    "failed": result.failed,
                    "skipped": result.skipped,
                },
                "error": result.error,
            }

        except Exception as e:
            logger.error(f"Error running quick tests: {e}")
            return {
                "success": False,
                "schema_version": SCHEMA_VERSION,
                "error": str(e),
            }

    @mcp.tool()
    @mcp_tool(tool_name="foundry_run_unit_tests")
    def foundry_run_unit_tests(
        target: Optional[str] = None,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Run unit tests (preset: unit).

        Runs tests marked with 'unit' marker.

        Args:
            target: Test target (file, directory, or test name pattern)
            workspace: Optional workspace path (defaults to config)

        Returns:
            JSON object with test results and schema_version
        """
        try:
            runner = _get_runner(workspace)
            result = runner.run_tests(target=target, preset="unit")

            return {
                "success": result.success,
                "schema_version": result.schema_version,
                "execution_id": result.execution_id,
                "summary": {
                    "total": result.total,
                    "passed": result.passed,
                    "failed": result.failed,
                    "skipped": result.skipped,
                },
                "error": result.error,
            }

        except Exception as e:
            logger.error(f"Error running unit tests: {e}")
            return {
                "success": False,
                "schema_version": SCHEMA_VERSION,
                "error": str(e),
            }

    logger.debug("Registered testing tools: foundry_run_tests, foundry_discover_tests, "
                 "foundry_test_presets, foundry_run_quick_tests, foundry_run_unit_tests")
