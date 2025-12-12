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
from foundry_mcp.core.responses import (
    success_response,
    error_response,
    sanitize_error_message,
)
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.tools.unified.test import legacy_test_action, list_test_presets

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
            return legacy_test_action(
                "run",
                config=config,
                target=target,
                preset=preset,
                timeout=timeout,
                verbose=verbose,
                fail_fast=fail_fast,
                markers=markers,
                workspace=workspace,
                include_passed=include_passed,
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
            return legacy_test_action(
                "discover",
                config=config,
                target=target,
                pattern=pattern,
                workspace=workspace,
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
            return list_test_presets()
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
            result = legacy_test_action(
                "run",
                config=config,
                target=target,
                preset="quick",
                workspace=workspace,
            )
            if not result.get("success"):
                return result
            data = result.get("data", {})
            return asdict(
                success_response(
                    execution_id=data.get("execution_id"),
                    tests_passed=data.get("tests_passed"),
                    summary=data.get("summary"),
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
            result = legacy_test_action(
                "run",
                config=config,
                target=target,
                preset="unit",
                workspace=workspace,
            )
            if not result.get("success"):
                return result
            data = result.get("data", {})
            return asdict(
                success_response(
                    execution_id=data.get("execution_id"),
                    tests_passed=data.get("tests_passed"),
                    summary=data.get("summary"),
                )
            )
        except Exception as e:
            logger.error(f"Error running unit tests: {e}")
            return asdict(error_response(sanitize_error_message(e, context="testing")))

    logger.debug(
        "Registered testing tools: test-run/test-discover/test-presets/test-run-quick/"
        "test-run-unit"
    )
