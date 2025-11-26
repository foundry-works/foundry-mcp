"""
Testing tools for foundry-mcp.

Provides MCP tools for running and discovering tests.
"""

import logging
from dataclasses import asdict
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.observability import mcp_tool
from foundry_mcp.core.testing import TestRunner, get_presets
from foundry_mcp.core.responses import success_response, error_response

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
            JSON object with test results
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

            if not result.success:
                return asdict(error_response(result.error))

            return asdict(success_response(
                execution_id=result.execution_id,
                timestamp=result.timestamp,
                summary={
                    "total": result.total,
                    "passed": result.passed,
                    "failed": result.failed,
                    "skipped": result.skipped,
                    "errors": result.errors,
                },
                tests=[
                    {
                        "name": t.name,
                        "outcome": t.outcome,
                        "duration": t.duration,
                        "message": t.message,
                    }
                    for t in result.tests
                ],
                command=result.command,
                duration=result.duration,
                metadata=result.metadata
            ))

        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return asdict(error_response(str(e)))

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
            JSON object with discovered tests
        """
        try:
            runner = _get_runner(workspace)
            result = runner.discover_tests(target=target, pattern=pattern)

            if not result.success:
                return asdict(error_response(result.error))

            return asdict(success_response(
                timestamp=result.timestamp,
                total=result.total,
                test_files=result.test_files,
                tests=[
                    {
                        "name": t.name,
                        "file_path": t.file_path,
                        "line_number": t.line_number,
                        "markers": t.markers,
                    }
                    for t in result.tests
                ],
                metadata=result.metadata
            ))

        except Exception as e:
            logger.error(f"Error discovering tests: {e}")
            return asdict(error_response(str(e)))

    @mcp.tool()
    @mcp_tool(tool_name="foundry_test_presets")
    def foundry_test_presets() -> dict:
        """
        Get available test presets.

        Lists configured presets with their settings (timeout, markers, etc.).

        Returns:
            JSON object with preset configurations
        """
        try:
            presets = get_presets()

            return asdict(success_response(
                presets=presets,
                available=list(presets.keys())
            ))

        except Exception as e:
            logger.error(f"Error getting presets: {e}")
            return asdict(error_response(str(e)))

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
            JSON object with test results
        """
        try:
            runner = _get_runner(workspace)
            result = runner.run_tests(target=target, preset="quick")

            if not result.success:
                return asdict(error_response(result.error))

            return asdict(success_response(
                execution_id=result.execution_id,
                summary={
                    "total": result.total,
                    "passed": result.passed,
                    "failed": result.failed,
                    "skipped": result.skipped,
                }
            ))

        except Exception as e:
            logger.error(f"Error running quick tests: {e}")
            return asdict(error_response(str(e)))

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
            JSON object with test results
        """
        try:
            runner = _get_runner(workspace)
            result = runner.run_tests(target=target, preset="unit")

            if not result.success:
                return asdict(error_response(result.error))

            return asdict(success_response(
                execution_id=result.execution_id,
                summary={
                    "total": result.total,
                    "passed": result.passed,
                    "failed": result.failed,
                    "skipped": result.skipped,
                }
            ))

        except Exception as e:
            logger.error(f"Error running unit tests: {e}")
            return asdict(error_response(str(e)))

    logger.debug("Registered testing tools: foundry_run_tests, foundry_discover_tests, "
                 "foundry_test_presets, foundry_run_quick_tests, foundry_run_unit_tests")
