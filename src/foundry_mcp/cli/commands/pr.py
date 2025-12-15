"""PR workflow commands for SDD CLI.

Provides commands for creating GitHub PRs with SDD spec context
and AI-enhanced PR descriptions.
"""

import json
import subprocess
import time
from typing import Any, Dict, Optional, cast

import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error, emit_success
from foundry_mcp.cli.registry import get_context
from foundry_mcp.cli.resilience import (
    FAST_TIMEOUT,
    MEDIUM_TIMEOUT,
    with_sync_timeout,
    handle_keyboard_interrupt,
)

logger = get_cli_logger()

# Default timeout for PR operations
PR_TIMEOUT = 120


@click.group("pr")
def pr_group() -> None:
    """Pull request workflow commands."""
    pass


@pr_group.command("create")
@click.argument("spec_id")
@click.option(
    "--title",
    help="PR title (default: auto-generated from spec).",
)
@click.option(
    "--base",
    "base_branch",
    default="main",
    help="Base branch for PR.",
)
@click.option(
    "--include-journals/--no-journals",
    default=True,
    help="Include journal entries in PR description.",
)
@click.option(
    "--include-diffs/--no-diffs",
    default=True,
    help="Include git diffs in LLM context.",
)
@click.option(
    "--model",
    help="LLM model for description generation.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview PR content without creating.",
)
@click.pass_context
@cli_command("create")
@handle_keyboard_interrupt()
@with_sync_timeout(PR_TIMEOUT, "PR creation timed out")
def pr_create_cmd(
    ctx: click.Context,
    spec_id: str,
    title: Optional[str],
    base_branch: str,
    include_journals: bool,
    include_diffs: bool,
    model: Optional[str],
    dry_run: bool,
) -> None:
    """Create a GitHub PR with AI-enhanced description from SDD spec.

    SPEC_ID is the specification identifier.

    Uses spec context (tasks, journals, progress) to generate
    comprehensive PR descriptions. Requires GitHub CLI (gh).
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
        return

    # Get LLM status
    llm_status = _get_llm_status()

    # Build command
    cmd = ["sdd", "create-pr", spec_id, "--json"]

    if title:
        cmd.extend(["--title", title])
    cmd.extend(["--base", base_branch])

    if include_journals:
        cmd.append("--include-journals")
    if include_diffs:
        cmd.append("--include-diffs")
    if model:
        cmd.extend(["--model", model])
    if specs_dir:
        cmd.extend(["--path", str(specs_dir.parent)])
    if dry_run:
        cmd.append("--dry-run")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=PR_TIMEOUT,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "PR creation failed"
            emit_error(
                f"PR creation failed: {error_msg}",
                code="PR_FAILED",
                error_type="internal",
                remediation="Ensure GitHub CLI is authenticated and repository is properly configured",
                details={
                    "spec_id": spec_id,
                    "exit_code": result.returncode,
                },
            )
            return

        # Parse output
        try:
            pr_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            pr_data = {"raw_output": result.stdout}

        emit_success(
            {
                "spec_id": spec_id,
                "dry_run": dry_run,
                "llm_status": llm_status,
                **pr_data,
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            }
        )

    except subprocess.TimeoutExpired:
        emit_error(
            f"PR creation timed out after {PR_TIMEOUT}s",
            code="TIMEOUT",
            error_type="internal",
            remediation="Check network connectivity and GitHub API status",
            details={
                "spec_id": spec_id,
                "timeout_seconds": PR_TIMEOUT,
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


@pr_group.command("context")
@click.argument("spec_id")
@click.option(
    "--include-tasks/--no-tasks",
    default=True,
    help="Include completed task summaries.",
)
@click.option(
    "--include-journals/--no-journals",
    default=True,
    help="Include recent journal entries.",
)
@click.option(
    "--include-progress/--no-progress",
    default=True,
    help="Include phase/task progress stats.",
)
@click.pass_context
@cli_command("context")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "PR context lookup timed out")
def pr_context_cmd(
    ctx: click.Context,
    spec_id: str,
    include_tasks: bool,
    include_journals: bool,
    include_progress: bool,
) -> None:
    """Get specification context for PR description generation.

    SPEC_ID is the specification identifier.

    Retrieves completed tasks, journal entries, and progress
    to help craft meaningful PR descriptions.
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
        return

    # Use prepare_review_context from core/review for consistent context gathering
    from foundry_mcp.core.review import ReviewContext, prepare_review_context

    review_ctx = prepare_review_context(
        spec_id=spec_id,
        specs_dir=specs_dir,
        include_tasks=include_tasks,
        include_journals=include_journals,
        max_journal_entries=5,
    )

    if review_ctx is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs list",
            details={"spec_id": spec_id},
        )
        return

    review_ctx = cast(ReviewContext, review_ctx)

    context: Dict[str, Any] = {
        "spec_id": spec_id,
        "title": review_ctx.title,
    }

    # Get progress stats from context
    if include_progress:
        context["progress"] = {
            "total_tasks": review_ctx.progress.get("total_tasks", 0),
            "completed_tasks": review_ctx.progress.get("completed_tasks", 0),
            "percentage": review_ctx.progress.get("percentage", 0),
        }

    # Get completed tasks from context
    if include_tasks:
        context["completed_tasks"] = review_ctx.completed_tasks

    # Get journal entries from context
    if include_journals:
        context["journals"] = review_ctx.journal_entries

    duration_ms = (time.perf_counter() - start_time) * 1000

    emit_success(
        {
            **context,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        }
    )


@pr_group.command("status")
@click.pass_context
@cli_command("status")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "PR status check timed out")
def pr_status_cmd(ctx: click.Context) -> None:
    """Check prerequisites for PR creation."""
    start_time = time.perf_counter()

    # Check GitHub CLI
    gh_available = False
    gh_version = None
    try:
        result = subprocess.run(
            ["gh", "--version"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        gh_available = result.returncode == 0
        if gh_available:
            gh_version = result.stdout.split("\n")[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check if authenticated
    gh_authenticated = False
    if gh_available:
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            gh_authenticated = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Check LLM status
    llm_status = _get_llm_status()

    # Check git status
    git_clean = False
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        git_clean = result.returncode == 0 and not result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    duration_ms = (time.perf_counter() - start_time) * 1000

    # Build status
    prerequisites = {
        "github_cli": {
            "available": gh_available,
            "version": gh_version,
            "authenticated": gh_authenticated,
        },
        "git": {
            "clean": git_clean,
        },
        "llm": llm_status,
    }

    ready = gh_available and gh_authenticated
    recommendations = []

    if not gh_available:
        recommendations.append("Install GitHub CLI: https://cli.github.com/")
    elif not gh_authenticated:
        recommendations.append("Authenticate with: gh auth login")
    if not git_clean:
        recommendations.append("Commit or stash uncommitted changes")
    if not llm_status.get("configured"):
        recommendations.append("Configure LLM for enhanced PR descriptions")

    emit_success(
        {
            "ready": ready,
            "prerequisites": prerequisites,
            "recommendations": recommendations,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        }
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
