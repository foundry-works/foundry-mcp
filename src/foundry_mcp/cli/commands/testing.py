"""Testing commands for SDD CLI.

Provides commands for running and managing tests including:
- Running pytest with presets
- Discovering tests
- Checking test toolchain
- AI consultation for test failures
"""

import json
import subprocess
import time
from typing import Optional

import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error, emit_success
from foundry_mcp.cli.registry import get_context
from foundry_mcp.cli.resilience import (
    FAST_TIMEOUT,
    MEDIUM_TIMEOUT,
    SLOW_TIMEOUT,
    handle_keyboard_interrupt,
    with_sync_timeout,
)

logger = get_cli_logger()

# Default timeout for test operations
TEST_TIMEOUT = 300  # 5 minutes


@click.group("test")
def test_group() -> None:
    """Test runner commands."""
    pass


@test_group.command("run")
@click.argument("target", required=False)
@click.option(
    "--preset",
    type=click.Choice(["quick", "full", "unit", "integration", "smoke"]),
    help="Use a preset configuration.",
)
@click.option(
    "--timeout",
    type=int,
    default=TEST_TIMEOUT,
    help="Timeout in seconds.",
)
@click.option(
    "--verbose/--quiet",
    default=True,
    help="Enable verbose output.",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    help="Stop on first failure.",
)
@click.option(
    "--markers",
    help="Pytest markers expression (e.g., 'not slow').",
)
@click.option(
    "--coverage/--no-coverage",
    default=False,
    help="Enable coverage reporting via pytest-cov.",
)
@click.option(
    "--parallel",
    "-n",
    type=int,
    default=None,
    help="Run tests in parallel with N workers (requires pytest-xdist).",
)
@click.pass_context
@cli_command("test-run")
@handle_keyboard_interrupt()
@with_sync_timeout(TEST_TIMEOUT, "Test run timed out")
def test_run_cmd(
    ctx: click.Context,
    target: Optional[str],
    preset: Optional[str],
    timeout: int,
    verbose: bool,
    fail_fast: bool,
    markers: Optional[str],
    coverage: bool,
    parallel: Optional[int],
) -> None:
    """Run tests using pytest.

    TARGET is the test target (file, directory, or test name pattern).
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    # Build pytest command
    cmd = ["pytest"]

    if target:
        cmd.append(target)

    if verbose:
        cmd.append("-v")

    if fail_fast:
        cmd.append("-x")

    if markers:
        cmd.extend(["-m", markers])

    # Apply preset configurations
    if preset == "quick":
        cmd.extend(["-x", "-m", "not slow"])
    elif preset == "unit":
        cmd.extend(["-m", "unit"])
    elif preset == "integration":
        cmd.extend(["-m", "integration"])
    elif preset == "smoke":
        cmd.extend(["-m", "smoke", "-x"])

    # Coverage support
    if coverage:
        cmd.extend(["--cov", "--cov-report=term-missing"])

    # Parallel execution support (requires pytest-xdist)
    if parallel is not None and parallel > 0:
        cmd.extend(["-n", str(parallel)])

    # Add JSON output format
    cmd.extend(["--tb=short", "-q"])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cli_ctx.specs_dir.parent) if cli_ctx.specs_dir else None,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Parse pytest output
        passed = 0
        failed = 0
        skipped = 0
        errors = 0

        for line in result.stdout.split("\n"):
            if "passed" in line:
                try:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "passed":
                            passed = int(parts[i - 1])
                        elif p == "failed":
                            failed = int(parts[i - 1])
                        elif p == "skipped":
                            skipped = int(parts[i - 1])
                        elif p == "error" or p == "errors":
                            errors = int(parts[i - 1])
                except (ValueError, IndexError):
                    pass

        emit_success({
            "target": target,
            "preset": preset,
            "exit_code": result.returncode,
            "passed": result.returncode == 0,
            "summary": {
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "errors": errors,
                "total": passed + failed + skipped + errors,
            },
            "stdout": result.stdout,
            "stderr": result.stderr if result.returncode != 0 else None,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

    except subprocess.TimeoutExpired:
        emit_error(
            f"Test run timed out after {timeout}s",
            code="TIMEOUT",
            error_type="internal",
            remediation="Try a smaller test target or increase timeout with --timeout",
            details={"target": target, "timeout_seconds": timeout},
        )
    except FileNotFoundError:
        emit_error(
            "pytest not found",
            code="PYTEST_NOT_FOUND",
            error_type="internal",
            remediation="Install pytest: pip install pytest",
            details={"hint": "Install pytest: pip install pytest"},
        )


@test_group.command("discover")
@click.argument("target", required=False)
@click.option(
    "--pattern",
    default="test_*.py",
    help="File pattern for test files.",
)
@click.option(
    "--list",
    "list_only",
    is_flag=True,
    default=True,
    help="List tests without running (default behavior).",
)
@click.pass_context
@cli_command("test-discover")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Test discovery timed out")
def test_discover_cmd(
    ctx: click.Context,
    target: Optional[str],
    pattern: str,
    list_only: bool,
) -> None:
    """Discover tests without running them.

    TARGET is the directory or file to search.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    # Build pytest collect command
    cmd = ["pytest", "--collect-only", "-q"]

    if target:
        cmd.append(target)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(cli_ctx.specs_dir.parent) if cli_ctx.specs_dir else None,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Parse collected tests
        tests = []
        for line in result.stdout.split("\n"):
            line = line.strip()
            if "::" in line and not line.startswith("<"):
                tests.append(line)

        emit_success({
            "target": target,
            "pattern": pattern,
            "tests": tests,
            "total_count": len(tests),
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

    except subprocess.TimeoutExpired:
        emit_error(
            "Test discovery timed out",
            code="TIMEOUT",
            error_type="internal",
            remediation="Try a smaller target directory or check for slow fixtures",
            details={"target": target},
        )
    except FileNotFoundError:
        emit_error(
            "pytest not found",
            code="PYTEST_NOT_FOUND",
            error_type="internal",
            remediation="Install pytest: pip install pytest",
            details={"hint": "Install pytest: pip install pytest"},
        )


@test_group.command("presets")
@click.pass_context
@cli_command("test-presets")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Presets lookup timed out")
def test_presets_cmd(ctx: click.Context) -> None:
    """Get available test presets."""
    start_time = time.perf_counter()

    presets = {
        "quick": {
            "description": "Fast test run with fail_fast and slow tests excluded",
            "markers": "not slow",
            "fail_fast": True,
            "timeout": 60,
        },
        "full": {
            "description": "Complete test suite",
            "markers": None,
            "fail_fast": False,
            "timeout": 300,
        },
        "unit": {
            "description": "Unit tests only",
            "markers": "unit",
            "fail_fast": False,
            "timeout": 120,
        },
        "integration": {
            "description": "Integration tests only",
            "markers": "integration",
            "fail_fast": False,
            "timeout": 300,
        },
        "smoke": {
            "description": "Smoke tests for quick validation",
            "markers": "smoke",
            "fail_fast": True,
            "timeout": 30,
        },
    }

    duration_ms = (time.perf_counter() - start_time) * 1000

    emit_success({
        "presets": presets,
        "default_preset": "quick",
        "telemetry": {"duration_ms": round(duration_ms, 2)},
    })


@test_group.command("check-tools")
@click.pass_context
@cli_command("test-check-tools")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Tool check timed out")
def test_check_tools_cmd(ctx: click.Context) -> None:
    """Check test toolchain availability."""
    start_time = time.perf_counter()

    tools = {}

    # Check pytest
    try:
        result = subprocess.run(
            ["pytest", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        tools["pytest"] = {
            "available": result.returncode == 0,
            "version": result.stdout.split("\n")[0].strip() if result.returncode == 0 else None,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        tools["pytest"] = {"available": False, "version": None}

    # Check coverage
    try:
        result = subprocess.run(
            ["coverage", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        tools["coverage"] = {
            "available": result.returncode == 0,
            "version": result.stdout.split("\n")[0].strip() if result.returncode == 0 else None,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        tools["coverage"] = {"available": False, "version": None}

    # Check pytest-cov
    try:
        result = subprocess.run(
            ["python", "-c", "import pytest_cov; print(pytest_cov.__version__)"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        tools["pytest-cov"] = {
            "available": result.returncode == 0,
            "version": result.stdout.strip() if result.returncode == 0 else None,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        tools["pytest-cov"] = {"available": False, "version": None}

    duration_ms = (time.perf_counter() - start_time) * 1000

    all_available = all(t.get("available", False) for t in tools.values())
    recommendations = []

    if not tools.get("pytest", {}).get("available"):
        recommendations.append("Install pytest: pip install pytest")
    if not tools.get("coverage", {}).get("available"):
        recommendations.append("Install coverage: pip install coverage")
    if not tools.get("pytest-cov", {}).get("available"):
        recommendations.append("Install pytest-cov: pip install pytest-cov")

    emit_success({
        "tools": tools,
        "all_available": all_available,
        "recommendations": recommendations,
        "telemetry": {"duration_ms": round(duration_ms, 2)},
    })


@test_group.command("quick")
@click.argument("target", required=False)
@click.pass_context
@cli_command("test-quick")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Quick tests timed out")
def test_quick_cmd(ctx: click.Context, target: Optional[str]) -> None:
    """Run quick tests (preset: quick)."""
    ctx.invoke(test_run_cmd, target=target, preset="quick")


@test_group.command("unit")
@click.argument("target", required=False)
@click.pass_context
@cli_command("test-unit")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Unit tests timed out")
def test_unit_cmd(ctx: click.Context, target: Optional[str]) -> None:
    """Run unit tests (preset: unit)."""
    ctx.invoke(test_run_cmd, target=target, preset="unit")


# Consultation timeout (longer for AI analysis)
CONSULT_TIMEOUT = 300


@test_group.command("consult")
@click.argument("pattern", required=False)
@click.option(
    "--issue",
    required=True,
    help="Description of the test failure or issue to analyze.",
)
@click.option(
    "--tools",
    help="Comma-separated list of AI tools to use (e.g., 'gemini,cursor-agent').",
)
@click.option(
    "--model",
    help="Specific LLM model to use for analysis.",
)
@click.pass_context
@cli_command("test-consult")
@handle_keyboard_interrupt()
@with_sync_timeout(CONSULT_TIMEOUT, "Test consultation timed out")
def test_consult_cmd(
    ctx: click.Context,
    pattern: Optional[str],
    issue: str,
    tools: Optional[str],
    model: Optional[str],
) -> None:
    """Consult AI about test failures or issues.

    PATTERN is an optional test pattern to filter tests (e.g., 'test_auth*').

    Example:
        sdd test consult --issue "test_login is flaky and fails intermittently"
        sdd test consult test_api --issue "assertion error on line 42"
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    # Build consultation prompt
    prompt_parts = [f"Test issue: {issue}"]
    if pattern:
        prompt_parts.append(f"Test pattern: {pattern}")

    # Check for recent test output to include as context
    test_context = None
    try:
        # Run a quick test discovery to get context
        if pattern:
            discover_result = subprocess.run(
                ["pytest", "--collect-only", "-q", pattern],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(cli_ctx.specs_dir.parent) if cli_ctx.specs_dir else None,
            )
            if discover_result.returncode == 0:
                test_context = discover_result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Build the consultation command
    cmd = ["sdd", "consult", "--json"]
    cmd.extend(["--prompt", " | ".join(prompt_parts)])

    if tools:
        cmd.extend(["--tools", tools])
    if model:
        cmd.extend(["--model", model])
    if cli_ctx.specs_dir:
        cmd.extend(["--path", str(cli_ctx.specs_dir.parent)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=CONSULT_TIMEOUT,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        if result.returncode == 0:
            try:
                response_data = json.loads(result.stdout)
                emit_success({
                    "pattern": pattern,
                    "issue": issue,
                    "tools_used": tools.split(",") if tools else ["default"],
                    "model": model,
                    "response": response_data,
                    "test_context": test_context[:500] if test_context else None,
                    "telemetry": {"duration_ms": round(duration_ms, 2)},
                })
            except json.JSONDecodeError:
                emit_success({
                    "pattern": pattern,
                    "issue": issue,
                    "tools_used": tools.split(",") if tools else ["default"],
                    "model": model,
                    "response": {"raw_output": result.stdout},
                    "test_context": test_context[:500] if test_context else None,
                    "telemetry": {"duration_ms": round(duration_ms, 2)},
                })
        else:
            emit_error(
                "Test consultation failed",
                code="CONSULT_FAILED",
                error_type="internal",
                remediation="Check AI tool availability and API configuration",
                details={
                    "pattern": pattern,
                    "issue": issue,
                    "stderr": result.stderr[:500] if result.stderr else None,
                },
            )

    except subprocess.TimeoutExpired:
        emit_error(
            f"Test consultation timed out after {CONSULT_TIMEOUT}s",
            code="TIMEOUT",
            error_type="internal",
            remediation="Try a more specific issue description or check AI service status",
            details={"pattern": pattern, "issue": issue, "timeout_seconds": CONSULT_TIMEOUT},
        )
    except FileNotFoundError:
        emit_error(
            "sdd command not found",
            code="SDD_NOT_FOUND",
            error_type="internal",
            remediation="Ensure sdd is installed and in PATH",
            details={"hint": "Run: pip install foundry-sdd"},
        )


# Top-level alias
@click.command("run-tests")
@click.argument("target", required=False)
@click.option("--preset", type=click.Choice(["quick", "full", "unit", "integration", "smoke"]))
@click.option("--fail-fast", is_flag=True)
@click.pass_context
@cli_command("run-tests-alias")
@handle_keyboard_interrupt()
@with_sync_timeout(TEST_TIMEOUT, "Tests timed out")
def run_tests_alias_cmd(
    ctx: click.Context,
    target: Optional[str],
    preset: Optional[str],
    fail_fast: bool,
) -> None:
    """Run tests (alias for test run)."""
    ctx.invoke(test_run_cmd, target=target, preset=preset, fail_fast=fail_fast)
