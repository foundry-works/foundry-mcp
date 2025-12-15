"""Development utility commands for SDD CLI.

Provides commands for skills development including:
- Documentation generation helpers
- Installation helpers
- Server start helpers
- Development tooling
"""

import subprocess
import time
from pathlib import Path
from typing import Optional

import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error, emit_success
from foundry_mcp.cli.registry import get_context
from foundry_mcp.cli.resilience import (
    FAST_TIMEOUT,
    MEDIUM_TIMEOUT,
    handle_keyboard_interrupt,
    with_sync_timeout,
)

logger = get_cli_logger()


@click.group("dev")
def dev_group() -> None:
    """Development utility commands."""
    pass


@dev_group.command("gendocs")
@click.option(
    "--output-dir",
    default="docs",
    help="Output directory for generated documentation.",
)
@click.option(
    "--format",
    "doc_format",
    type=click.Choice(["markdown", "html", "json"]),
    default="markdown",
    help="Output format for documentation.",
)
@click.option(
    "--include-private",
    is_flag=True,
    help="Include private/internal APIs in documentation.",
)
@click.pass_context
@cli_command("gendocs")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Documentation generation timed out")
def dev_gendocs_cmd(
    ctx: click.Context,
    output_dir: str,
    doc_format: str,
    include_private: bool,
) -> None:
    """Generate documentation from source code.

    Scans the codebase and generates API documentation.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    # Determine project root
    project_root = cli_ctx.specs_dir.parent if cli_ctx.specs_dir else Path.cwd()
    output_path = project_root / output_dir

    # Create output directory if needed
    output_path.mkdir(parents=True, exist_ok=True)

    # Check for common doc generators
    generators = []

    # Check pdoc
    try:
        result = subprocess.run(
            ["pdoc", "--version"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            generators.append("pdoc")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check sphinx
    try:
        result = subprocess.run(
            ["sphinx-build", "--version"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            generators.append("sphinx")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    duration_ms = (time.perf_counter() - start_time) * 1000

    if not generators:
        emit_success(
            {
                "status": "no_generator",
                "output_dir": str(output_path),
                "format": doc_format,
                "available_generators": [],
                "recommendations": [
                    "Install pdoc: pip install pdoc",
                    "Or install sphinx: pip install sphinx",
                ],
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            }
        )
        return

    emit_success(
        {
            "status": "ready",
            "output_dir": str(output_path),
            "format": doc_format,
            "available_generators": generators,
            "include_private": include_private,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        }
    )


@dev_group.command("install")
@click.option(
    "--dev",
    is_flag=True,
    help="Install with development dependencies.",
)
@click.option(
    "--editable/--no-editable",
    default=True,
    help="Install in editable mode (default: true).",
)
@click.option(
    "--extras",
    help="Comma-separated list of extras to install.",
)
@click.pass_context
@cli_command("install")
@handle_keyboard_interrupt()
@with_sync_timeout(300, "Installation timed out")
def dev_install_cmd(
    ctx: click.Context,
    dev: bool,
    editable: bool,
    extras: Optional[str],
) -> None:
    """Install the package for development.

    Installs the current package with optional dev dependencies.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    # Determine project root
    project_root = cli_ctx.specs_dir.parent if cli_ctx.specs_dir else Path.cwd()

    # Check for pyproject.toml or setup.py
    has_pyproject = (project_root / "pyproject.toml").exists()
    has_setup = (project_root / "setup.py").exists()

    if not has_pyproject and not has_setup:
        emit_error(
            "No Python project found",
            code="NO_PROJECT",
            error_type="validation",
            remediation="Ensure pyproject.toml or setup.py exists in the project root",
            details={
                "hint": "Ensure pyproject.toml or setup.py exists",
                "project_root": str(project_root),
            },
        )
        return

    # Build pip install command
    cmd = ["pip", "install"]

    if editable:
        cmd.append("-e")

    # Build package specifier
    package_spec = "."
    if extras:
        package_spec = f".[{extras}]"
    if dev:
        if extras:
            package_spec = f".[dev,{extras}]"
        else:
            package_spec = ".[dev]"

    cmd.append(package_spec)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(project_root),
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        if result.returncode != 0:
            emit_error(
                "Installation failed",
                code="INSTALL_FAILED",
                error_type="internal",
                remediation="Check the error output and ensure all dependencies are available",
                details={
                    "exit_code": result.returncode,
                    "stderr": result.stderr[:500] if result.stderr else None,
                },
            )
            return

        emit_success(
            {
                "status": "installed",
                "editable": editable,
                "dev": dev,
                "extras": extras,
                "project_root": str(project_root),
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            }
        )

    except subprocess.TimeoutExpired:
        emit_error(
            "Installation timed out",
            code="TIMEOUT",
            error_type="internal",
            remediation="Try again with a faster network or fewer dependencies",
            details={"timeout_seconds": 300},
        )
    except FileNotFoundError:
        emit_error(
            "pip not found",
            code="PIP_NOT_FOUND",
            error_type="internal",
            remediation="Ensure pip is installed and in PATH",
            details={"hint": "Ensure pip is installed and in PATH"},
        )


@dev_group.command("start")
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Port to run the development server on.",
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind the server to.",
)
@click.option(
    "--reload/--no-reload",
    default=True,
    help="Enable auto-reload on file changes.",
)
@click.pass_context
@cli_command("start")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Server check timed out")
def dev_start_cmd(
    ctx: click.Context,
    port: int,
    host: str,
    reload: bool,
) -> None:
    """Start a development server.

    Checks for common development server configurations and
    provides instructions for starting.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    # Determine project root
    project_root = cli_ctx.specs_dir.parent if cli_ctx.specs_dir else Path.cwd()

    # Check for various server configurations
    server_configs = []

    # Check for uvicorn (FastAPI/Starlette)
    try:
        result = subprocess.run(
            ["uvicorn", "--version"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            server_configs.append(
                {
                    "name": "uvicorn",
                    "command": f"uvicorn main:app --host {host} --port {port}"
                    + (" --reload" if reload else ""),
                }
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check for flask
    try:
        result = subprocess.run(
            ["flask", "--version"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            server_configs.append(
                {
                    "name": "flask",
                    "command": f"flask run --host {host} --port {port}"
                    + (" --reload" if reload else ""),
                }
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check for MCP server config
    mcp_config = project_root / "mcp.json"
    if mcp_config.exists():
        server_configs.append(
            {
                "name": "mcp",
                "command": f"python -m foundry_mcp.server --port {port}",
            }
        )

    duration_ms = (time.perf_counter() - start_time) * 1000

    emit_success(
        {
            "host": host,
            "port": port,
            "reload": reload,
            "available_servers": server_configs,
            "project_root": str(project_root),
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        }
    )


@dev_group.command("check")
@click.pass_context
@cli_command("check")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Environment check timed out")
def dev_check_cmd(ctx: click.Context) -> None:
    """Check development environment status.

    Verifies that all development tools are available.
    """
    start_time = time.perf_counter()
    get_context(ctx)

    tools = {}

    # Check Python
    try:
        result = subprocess.run(
            ["python", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        tools["python"] = {
            "available": result.returncode == 0,
            "version": result.stdout.strip() if result.returncode == 0 else None,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        tools["python"] = {"available": False}

    # Check pip
    try:
        result = subprocess.run(
            ["pip", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        tools["pip"] = {
            "available": result.returncode == 0,
            "version": result.stdout.split()[1] if result.returncode == 0 else None,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        tools["pip"] = {"available": False}

    # Check git
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        tools["git"] = {
            "available": result.returncode == 0,
            "version": result.stdout.strip() if result.returncode == 0 else None,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        tools["git"] = {"available": False}

    # Check node (optional)
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        tools["node"] = {
            "available": result.returncode == 0,
            "version": result.stdout.strip() if result.returncode == 0 else None,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        tools["node"] = {"available": False}

    duration_ms = (time.perf_counter() - start_time) * 1000

    all_required = all(
        tools.get(t, {}).get("available", False) for t in ["python", "pip", "git"]
    )

    emit_success(
        {
            "tools": tools,
            "all_required_available": all_required,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        }
    )
