"""Review commands for SDD CLI.

Provides commands for LLM-powered spec review and fidelity checking.
Integrates review templates and fidelity hooks for quality assurance.
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
    SLOW_TIMEOUT,
    MEDIUM_TIMEOUT,
    with_sync_timeout,
    handle_keyboard_interrupt,
)

logger = get_cli_logger()

# Review types supported
REVIEW_TYPES = ["quick", "full", "security", "feasibility"]

# External review tools
REVIEW_TOOLS = ["cursor-agent", "gemini", "codex"]

# Fidelity review timeout (longer for AI consultation)
FIDELITY_TIMEOUT = 600


@click.group("review")
def review_group() -> None:
    """Spec review and fidelity checking commands."""
    pass


@review_group.command("spec")
@click.argument("spec_id")
@click.option(
    "--type",
    "review_type",
    type=click.Choice(REVIEW_TYPES),
    default="quick",
    help="Type of review to perform.",
)
@click.option(
    "--tools",
    help="Comma-separated list of review tools to use.",
)
@click.option(
    "--model",
    help="LLM model to use for review.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be reviewed without executing.",
)
@click.pass_context
@cli_command("review-spec")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Review timed out")
def review_spec_cmd(
    ctx: click.Context,
    spec_id: str,
    review_type: str,
    tools: Optional[str],
    model: Optional[str],
    dry_run: bool,
) -> None:
    """Run an LLM-powered review on a specification.

    SPEC_ID is the specification identifier.

    Review types:
    - quick: Fast structural review (no LLM required)
    - full: Comprehensive review with LLM analysis
    - security: Security-focused analysis
    - feasibility: Implementation feasibility assessment
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)
    specs_dir = cli_ctx.specs_dir

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set SDD_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set SDD_SPECS_DIR"},
        )

    # Get LLM status
    llm_status = _get_llm_status()

    # Check if LLM is required but not configured
    if review_type != "quick" and not llm_status.get("configured"):
        emit_error(
            f"Review type '{review_type}' requires LLM configuration",
            code="LLM_NOT_CONFIGURED",
            error_type="validation",
            remediation="Set FOUNDRY_MCP_LLM_API_KEY or use --type quick for structural review",
            details={
                "review_type": review_type,
                "hint": "Set FOUNDRY_MCP_LLM_API_KEY or use --type quick",
            },
        )

    # Dry run mode
    if dry_run:
        cmd_preview = ["sdd", "review", spec_id, "--type", review_type, "--json"]
        if tools:
            cmd_preview.extend(["--tools", tools])
        if model:
            cmd_preview.extend(["--model", model])

        emit_success({
            "spec_id": spec_id,
            "review_type": review_type,
            "dry_run": True,
            "command": " ".join(cmd_preview),
            "llm_status": llm_status,
            "message": "Dry run - no review executed",
        })
        return

    # Build and execute review command
    cmd = ["sdd", "review", spec_id, "--type", review_type, "--json"]
    if tools:
        cmd.extend(["--tools", tools])
    if model:
        cmd.extend(["--model", model])
    if specs_dir:
        cmd.extend(["--path", str(specs_dir.parent)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SLOW_TIMEOUT,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Review failed"
            emit_error(
                f"Review failed: {error_msg}",
                code="REVIEW_FAILED",
                error_type="internal",
                remediation="Check that the spec exists and LLM is properly configured",
                details={
                    "spec_id": spec_id,
                    "review_type": review_type,
                    "exit_code": result.returncode,
                },
            )

        # Parse review output
        try:
            review_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            review_data = {"raw_output": result.stdout}

        emit_success({
            "spec_id": spec_id,
            "review_type": review_type,
            "llm_status": llm_status,
            **review_data,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

    except subprocess.TimeoutExpired:
        emit_error(
            f"Review timed out after {SLOW_TIMEOUT}s",
            code="TIMEOUT",
            error_type="internal",
            remediation="Consider using --type quick for faster review or reviewing smaller scope",
            details={
                "spec_id": spec_id,
                "review_type": review_type,
                "timeout_seconds": SLOW_TIMEOUT,
            },
        )
    except FileNotFoundError:
        emit_error(
            "SDD CLI not found",
            code="CLI_NOT_FOUND",
            error_type="internal",
            remediation="Ensure 'sdd' is installed and in PATH",
            details={"hint": "Ensure 'sdd' is installed and in PATH"},
        )


@review_group.command("tools")
@click.pass_context
@cli_command("review-tools")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Review tools lookup timed out")
def review_tools_cmd(ctx: click.Context) -> None:
    """List available review tools and their status."""
    start_time = time.perf_counter()

    llm_status = _get_llm_status()

    # Check tool availability
    tools_info = []
    for tool in REVIEW_TOOLS:
        try:
            result = subprocess.run(
                [tool, "--version"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            available = result.returncode == 0
            version = result.stdout.strip() if available else None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            available = False
            version = None

        tools_info.append({
            "name": tool,
            "available": available,
            "version": version,
        })

    duration_ms = (time.perf_counter() - start_time) * 1000

    emit_success({
        "tools": tools_info,
        "llm_status": llm_status,
        "review_types": REVIEW_TYPES,
        "telemetry": {"duration_ms": round(duration_ms, 2)},
    })


@review_group.command("plan-tools")
@click.pass_context
@cli_command("review-plan-tools")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Plan tools lookup timed out")
def review_plan_tools_cmd(ctx: click.Context) -> None:
    """List available plan review toolchains."""
    start_time = time.perf_counter()

    llm_status = _get_llm_status()

    # Define plan review toolchains
    plan_tools = [
        {
            "name": "quick-review",
            "description": "Fast structural review for basic validation",
            "capabilities": ["structure", "syntax", "basic_quality"],
            "llm_required": False,
            "estimated_time": "< 10 seconds",
        },
        {
            "name": "full-review",
            "description": "Comprehensive review with LLM analysis",
            "capabilities": ["structure", "quality", "feasibility", "suggestions"],
            "llm_required": True,
            "estimated_time": "30-60 seconds",
        },
        {
            "name": "security-review",
            "description": "Security-focused analysis of plan",
            "capabilities": ["security", "trust_boundaries", "data_flow"],
            "llm_required": True,
            "estimated_time": "30-60 seconds",
        },
        {
            "name": "feasibility-review",
            "description": "Implementation feasibility assessment",
            "capabilities": ["complexity", "dependencies", "risk"],
            "llm_required": True,
            "estimated_time": "30-60 seconds",
        },
    ]

    # Add availability status
    available_tools = []
    for tool in plan_tools:
        tool_info = tool.copy()
        if tool["llm_required"] and not llm_status.get("configured"):
            tool_info["status"] = "unavailable"
            tool_info["reason"] = "LLM not configured"
        else:
            tool_info["status"] = "available"
        available_tools.append(tool_info)

    # Build recommendations
    if llm_status.get("configured"):
        recommendations = [
            "Use 'full-review' for comprehensive plan analysis",
            "Run 'security-review' before implementation of sensitive features",
            "Use 'feasibility-review' for complex or risky plans",
        ]
    else:
        recommendations = [
            "Use 'quick-review' for basic validation (no LLM required)",
            "Configure LLM to unlock full review capabilities",
            "Set FOUNDRY_MCP_LLM_API_KEY or provider-specific env var",
        ]

    duration_ms = (time.perf_counter() - start_time) * 1000

    emit_success({
        "plan_tools": available_tools,
        "llm_status": llm_status,
        "recommendations": recommendations,
        "telemetry": {"duration_ms": round(duration_ms, 2)},
    })


@review_group.command("fidelity")
@click.argument("spec_id")
@click.option(
    "--task",
    "task_id",
    help="Review specific task implementation.",
)
@click.option(
    "--phase",
    "phase_id",
    help="Review entire phase implementation.",
)
@click.option(
    "--files",
    multiple=True,
    help="Review specific file(s) only.",
)
@click.option(
    "--incremental",
    is_flag=True,
    help="Only review changed files since last run.",
)
@click.option(
    "--base-branch",
    default="main",
    help="Base branch for git diff.",
)
@click.pass_context
@cli_command("review-fidelity")
@handle_keyboard_interrupt()
@with_sync_timeout(FIDELITY_TIMEOUT, "Fidelity review timed out")
def review_fidelity_cmd(
    ctx: click.Context,
    spec_id: str,
    task_id: Optional[str],
    phase_id: Optional[str],
    files: tuple,
    incremental: bool,
    base_branch: str,
) -> None:
    """Compare implementation against specification.

    SPEC_ID is the specification identifier.

    Performs a fidelity review to verify that code implementation
    matches the specification requirements.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)
    specs_dir = cli_ctx.specs_dir

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set SDD_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set SDD_SPECS_DIR"},
        )

    # Validate mutually exclusive options
    if task_id and phase_id:
        emit_error(
            "Cannot specify both --task and --phase",
            code="INVALID_OPTIONS",
            error_type="validation",
            remediation="Use either --task or --phase, not both",
            details={"hint": "Use either --task or --phase, not both"},
        )

    # Get LLM status
    llm_status = _get_llm_status()

    # Build command
    cmd = ["sdd", "fidelity-review", spec_id, "--json"]

    if task_id:
        cmd.extend(["--task", task_id])
    if phase_id:
        cmd.extend(["--phase", phase_id])
    for file in files:
        cmd.extend(["--file", file])
    if incremental:
        cmd.append("--incremental")
    if base_branch != "main":
        cmd.extend(["--base-branch", base_branch])
    if specs_dir:
        cmd.extend(["--path", str(specs_dir.parent)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FIDELITY_TIMEOUT,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Fidelity review failed"
            emit_error(
                f"Fidelity review failed: {error_msg}",
                code="FIDELITY_FAILED",
                error_type="internal",
                remediation="Check that the spec exists and files are accessible",
                details={
                    "spec_id": spec_id,
                    "exit_code": result.returncode,
                },
            )

        # Parse review output
        try:
            review_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            review_data = {"raw_output": result.stdout}

        # Determine scope
        scope = "spec"
        if task_id:
            scope = f"task:{task_id}"
        elif phase_id:
            scope = f"phase:{phase_id}"
        elif files:
            scope = f"files:{len(files)}"

        emit_success({
            "spec_id": spec_id,
            "scope": scope,
            "llm_status": llm_status,
            **review_data,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

    except subprocess.TimeoutExpired:
        emit_error(
            f"Fidelity review timed out after {FIDELITY_TIMEOUT}s",
            code="TIMEOUT",
            error_type="internal",
            remediation="Consider reviewing a smaller scope (single task or phase)",
            details={
                "spec_id": spec_id,
                "timeout_seconds": FIDELITY_TIMEOUT,
                "hint": "Consider reviewing a smaller scope (single task or phase)",
            },
        )
    except FileNotFoundError:
        emit_error(
            "SDD CLI not found",
            code="CLI_NOT_FOUND",
            error_type="internal",
            remediation="Ensure 'sdd' is installed and in PATH",
            details={"hint": "Ensure 'sdd' is installed and in PATH"},
        )


def _get_llm_status() -> dict:
    """Get LLM configuration status."""
    try:
        from foundry_mcp.core.llm_config import get_llm_config

        config = get_llm_config()
        return {
            "configured": config.get_api_key() is not None,
            "provider": config.provider.value,
            "model": config.get_model(),
        }
    except ImportError:
        return {"configured": False, "error": "LLM config not available"}
    except Exception as e:
        return {"configured": False, "error": str(e)}


# Top-level aliases
@click.command("review-spec")
@click.argument("spec_id")
@click.option(
    "--type",
    "review_type",
    type=click.Choice(REVIEW_TYPES),
    default="quick",
    help="Type of review to perform.",
)
@click.pass_context
@cli_command("review-spec-alias")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Review timed out")
def review_spec_alias_cmd(
    ctx: click.Context,
    spec_id: str,
    review_type: str,
) -> None:
    """Run an LLM-powered review on a specification (alias)."""
    # Delegate to main command
    ctx.invoke(review_spec_cmd, spec_id=spec_id, review_type=review_type)
