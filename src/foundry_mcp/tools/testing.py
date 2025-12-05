"""
Testing tools for foundry-mcp.

Provides MCP tools for running and discovering tests.
"""

import logging
from dataclasses import asdict
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.testing import TestRunner, get_presets
from foundry_mcp.core.responses import success_response, error_response, sanitize_error_message
from foundry_mcp.core.naming import canonical_tool

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

        ws = (
            Path(workspace)
            if workspace
            else (config.specs_dir.parent if config.specs_dir else None)
        )
        return TestRunner(workspace=ws)

    @canonical_tool(
        mcp,
        canonical_name="test-run",
    )
    def test_run(
        target: Optional[str] = None,
        preset: Optional[str] = None,
        timeout: int = 300,
        verbose: bool = True,
        fail_fast: bool = False,
        markers: Optional[str] = None,
        workspace: Optional[str] = None,
        include_passed: bool = False,
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
            include_passed: Include passed tests in response (default: False for concise output)

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

            # Only return error for actual errors (not test failures)
            # Test failures are reported in the success response with tests_passed=False
            if result.error:
                return asdict(error_response(result.error))

            # Filter tests for concise response - only failures/errors by default
            # This keeps responses small for LLM context windows
            if include_passed:
                filtered_tests = result.tests
            else:
                filtered_tests = [
                    t for t in result.tests
                    if t.outcome in ("failed", "error")
                ]

            return asdict(
                success_response(
                    execution_id=result.execution_id,
                    timestamp=result.timestamp,
                    tests_passed=result.success,  # True if all tests passed
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
                        for t in filtered_tests
                    ],
                    filtered=not include_passed,  # Indicate if results were filtered
                    command=result.command,
                    duration=result.duration,
                    metadata=result.metadata,
                )
            )

        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return asdict(error_response(sanitize_error_message(e, context="testing")))

    @canonical_tool(
        mcp,
        canonical_name="test-discover",
    )
    def test_discover(
        target: Optional[str] = None,
        pattern: str = "test_*.py",
        workspace: Optional[str] = None,
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

            # Only return error for actual errors
            if result.error:
                return asdict(error_response(result.error))

            return asdict(
                success_response(
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
                    metadata=result.metadata,
                )
            )

        except Exception as e:
            logger.error(f"Error discovering tests: {e}")
            return asdict(error_response(sanitize_error_message(e, context="testing")))

    @canonical_tool(
        mcp,
        canonical_name="test-presets",
    )
    def test_presets() -> dict:
        """
        Get available test presets.

        Lists configured presets with their settings (timeout, markers, etc.).

        Returns:
            JSON object with preset configurations
        """
        try:
            presets = get_presets()

            return asdict(
                success_response(presets=presets, available=list(presets.keys()))
            )

        except Exception as e:
            logger.error(f"Error getting presets: {e}")
            return asdict(error_response(sanitize_error_message(e, context="testing")))

    @canonical_tool(
        mcp,
        canonical_name="test-run-quick",
    )
    def test_run_quick(
        target: Optional[str] = None, workspace: Optional[str] = None
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

            # Only return error for actual errors (not test failures)
            if result.error:
                return asdict(error_response(result.error))

            return asdict(
                success_response(
                    execution_id=result.execution_id,
                    tests_passed=result.success,
                    summary={
                        "total": result.total,
                        "passed": result.passed,
                        "failed": result.failed,
                        "skipped": result.skipped,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Error running quick tests: {e}")
            return asdict(error_response(sanitize_error_message(e, context="testing")))

    @canonical_tool(
        mcp,
        canonical_name="test-run-unit",
    )
    def test_run_unit(
        target: Optional[str] = None, workspace: Optional[str] = None
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

            # Only return error for actual errors (not test failures)
            if result.error:
                return asdict(error_response(result.error))

            return asdict(
                success_response(
                    execution_id=result.execution_id,
                    tests_passed=result.success,
                    summary={
                        "total": result.total,
                        "passed": result.passed,
                        "failed": result.failed,
                        "skipped": result.skipped,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Error running unit tests: {e}")
            return asdict(error_response(sanitize_error_message(e, context="testing")))

    logger.debug(
        "Registered testing tools: test-run/test-discover/test-presets/test-run-quick/"
        "test-run-unit"
    )
