"""Unified test tool with action routing.

Provides the unified `test(action=...)` entry point.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.context import generate_correlation_id, get_correlation_id
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import get_metrics, mcp_tool
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
)
from foundry_mcp.core.testing import TestRunner, get_presets
from foundry_mcp.tools.unified.router import (
    ActionDefinition,
    ActionRouter,
    ActionRouterError,
)

logger = logging.getLogger(__name__)
_metrics = get_metrics()


def _request_id() -> str:
    return get_correlation_id() or generate_correlation_id(prefix="test")


def _metric(action: str) -> str:
    return f"unified_tools.test.{action.replace('-', '_')}"


def _get_runner(config: ServerConfig, workspace: Optional[str]) -> TestRunner:
    ws: Optional[Path] = None
    if workspace:
        ws = Path(workspace)
    elif config.specs_dir is not None:
        ws = config.specs_dir.parent
    return TestRunner(workspace=ws)


def _validation_error(
    *, message: str, request_id: str, remediation: Optional[str] = None
) -> dict:
    return asdict(
        error_response(
            message,
            error_code=ErrorCode.VALIDATION_ERROR,
            error_type=ErrorType.VALIDATION,
            remediation=remediation,
            request_id=request_id,
        )
    )


def _handle_run(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()

    preset = payload.get("preset")
    if preset is not None and not isinstance(preset, str):
        return _validation_error(
            message="preset must be a string",
            request_id=request_id,
            remediation="Use preset=quick|unit|full",
        )

    if isinstance(preset, str):
        presets = get_presets()
        if preset not in presets:
            return _validation_error(
                message=f"Unknown preset: {preset}",
                request_id=request_id,
                remediation=f"Use one of: {', '.join(sorted(presets))}",
            )

    target = payload.get("target")
    if target is not None and not isinstance(target, str):
        return _validation_error(
            message="target must be a string",
            request_id=request_id,
            remediation="Provide a pytest target like tests/unit or tests/test_file.py",
        )

    timeout = payload.get("timeout", 300)
    if timeout is not None:
        try:
            timeout_int = int(timeout)
        except (TypeError, ValueError):
            return _validation_error(
                message="timeout must be an integer",
                request_id=request_id,
                remediation="Provide a timeout in seconds",
            )
        if timeout_int <= 0:
            return _validation_error(
                message="timeout must be > 0",
                request_id=request_id,
                remediation="Provide a timeout in seconds",
            )
        timeout = timeout_int

    verbose_value = payload.get("verbose", True)
    if verbose_value is not None and not isinstance(verbose_value, bool):
        return _validation_error(
            message="verbose must be a boolean",
            request_id=request_id,
            remediation="Provide verbose=true|false",
        )
    verbose = verbose_value if isinstance(verbose_value, bool) else True

    fail_fast_value = payload.get("fail_fast", False)
    if fail_fast_value is not None and not isinstance(fail_fast_value, bool):
        return _validation_error(
            message="fail_fast must be a boolean",
            request_id=request_id,
            remediation="Provide fail_fast=true|false",
        )
    fail_fast = fail_fast_value if isinstance(fail_fast_value, bool) else False
    markers = payload.get("markers")
    if markers is not None and not isinstance(markers, str):
        return _validation_error(
            message="markers must be a string",
            request_id=request_id,
            remediation="Provide a pytest markers expression like 'not slow'",
        )

    workspace = payload.get("workspace")
    if workspace is not None and not isinstance(workspace, str):
        return _validation_error(
            message="workspace must be a string",
            request_id=request_id,
            remediation="Provide an absolute path to the workspace",
        )

    include_passed_value = payload.get("include_passed", False)
    if include_passed_value is not None and not isinstance(include_passed_value, bool):
        return _validation_error(
            message="include_passed must be a boolean",
            request_id=request_id,
            remediation="Provide include_passed=true|false",
        )
    include_passed = (
        include_passed_value if isinstance(include_passed_value, bool) else False
    )

    runner = _get_runner(config, workspace)

    start = time.perf_counter()
    result = runner.run_tests(
        target=target,
        preset=preset,
        timeout=timeout,
        verbose=verbose,
        fail_fast=fail_fast,
        markers=markers,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    _metrics.timer(_metric("run") + ".duration_ms", elapsed_ms)
    _metrics.counter(
        _metric("run"), labels={"status": "success" if result.success else "failure"}
    )

    if result.error:
        return asdict(
            error_response(
                result.error,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                request_id=request_id,
            )
        )

    filtered_tests = (
        result.tests
        if include_passed
        else [t for t in result.tests if t.outcome in ("failed", "error")]
    )

    return asdict(
        success_response(
            execution_id=result.execution_id,
            timestamp=result.timestamp,
            tests_passed=result.success,
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
            filtered=not include_passed,
            command=result.command,
            duration=result.duration,
            metadata=dict(result.metadata or {}),
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )


def _handle_discover(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()

    target = payload.get("target")
    if target is not None and not isinstance(target, str):
        return _validation_error(
            message="target must be a string",
            request_id=request_id,
            remediation="Provide a test directory or file to search",
        )

    pattern = payload.get("pattern", "test_*.py")
    if not isinstance(pattern, str) or not pattern:
        return _validation_error(
            message="pattern must be a non-empty string",
            request_id=request_id,
            remediation="Provide a file glob pattern like test_*.py",
        )

    workspace = payload.get("workspace")
    if workspace is not None and not isinstance(workspace, str):
        return _validation_error(
            message="workspace must be a string",
            request_id=request_id,
            remediation="Provide an absolute path to the workspace",
        )

    runner = _get_runner(config, workspace)

    start = time.perf_counter()
    result = runner.discover_tests(target=target, pattern=pattern)
    elapsed_ms = (time.perf_counter() - start) * 1000

    _metrics.timer(_metric("discover") + ".duration_ms", elapsed_ms)
    _metrics.counter(
        _metric("discover"),
        labels={"status": "success" if result.success else "failure"},
    )

    if result.error:
        return asdict(
            error_response(
                result.error,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                request_id=request_id,
            )
        )

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
            metadata=dict(result.metadata or {}),
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )


_ACTION_SUMMARY = {
    "run": "Execute pytest using a preset or target.",
    "discover": "Discover pytest tests without executing.",
}


def _build_router() -> ActionRouter:
    actions = [
        ActionDefinition(
            name="run", handler=_handle_run, summary=_ACTION_SUMMARY["run"]
        ),
        ActionDefinition(
            name="discover",
            handler=_handle_discover,
            summary=_ACTION_SUMMARY["discover"],
        ),
    ]
    return ActionRouter(tool_name="test", actions=actions)


_TEST_ROUTER = _build_router()


def _dispatch_test_action(
    *, action: str, payload: Dict[str, Any], config: ServerConfig
) -> dict:
    try:
        return _TEST_ROUTER.dispatch(action, config=config, payload=payload)
    except ActionRouterError as exc:
        allowed = ", ".join(exc.allowed_actions)
        request_id = _request_id()
        return asdict(
            error_response(
                f"Unsupported test action '{action}'. Allowed actions: {allowed}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {allowed}",
                request_id=request_id,
            )
        )


def register_unified_test_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the consolidated test tool."""

    @canonical_tool(mcp, canonical_name="test")
    @mcp_tool(tool_name="test", emit_metrics=True, audit=True)
    def test(
        action: str,
        target: Optional[str] = None,
        preset: Optional[str] = None,
        timeout: int = 300,
        verbose: bool = True,
        fail_fast: bool = False,
        markers: Optional[str] = None,
        pattern: str = "test_*.py",
        workspace: Optional[str] = None,
        include_passed: bool = False,
    ) -> dict:
        payload: Dict[str, Any] = {
            "target": target,
            "preset": preset,
            "timeout": timeout,
            "verbose": verbose,
            "fail_fast": fail_fast,
            "markers": markers,
            "pattern": pattern,
            "workspace": workspace,
            "include_passed": include_passed,
        }
        return _dispatch_test_action(action=action, payload=payload, config=config)

    logger.debug("Registered unified test tool")


__all__ = [
    "register_unified_test_tool",
]
