"""Rendering commands for SDD CLI.

Provides commands for generating human-readable markdown from specs.
Feature flags gate access to enhanced rendering modes.
"""

import time
from typing import Optional

import click

from foundry_mcp.cli.flags import get_cli_flags
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
from foundry_mcp.core.feature_flags import FlagState
from foundry_mcp.core.rate_limit import check_rate_limit
from foundry_mcp.core.spec import load_spec, find_spec_file
from foundry_mcp.core.rendering import (
    RenderOptions,
    render_spec_to_markdown,
    render_progress_bar,
    render_task_list,
)


# Register CLI feature flags for rendering
def _register_render_flags() -> None:
    """Register render-related feature flags."""
    registry = get_cli_flags()
    registry.register_cli_flag(
        name="enhanced_render",
        description="Enable enhanced rendering mode with AI-generated insights",
        default_enabled=False,
        state=FlagState.BETA,
    )


# Register flags on module load
_register_render_flags()


@click.group("render")
def render_group() -> None:
    """Spec rendering and documentation commands."""
    pass


@render_group.command("spec")
@click.argument("spec_id")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["basic", "enhanced"]),
    default=None,
    help="Rendering mode.",
)
@click.option(
    "--enhancement-level",
    type=click.Choice(["basic", "enhanced"]),
    default=None,
    help="Alias for --mode (rendering enhancement level).",
)
@click.option(
    "--include-journal/--no-journal",
    default=False,
    help="Include journal entries.",
)
@click.option(
    "--max-depth",
    type=int,
    default=0,
    help="Maximum depth to render (0 = unlimited).",
)
@click.option(
    "--phase",
    "phases",
    multiple=True,
    help="Filter to specific phase IDs.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Write markdown to file instead of JSON output.",
)
@click.option(
    "--path",
    type=click.Path(exists=True),
    help="Project root path (overrides default spec location).",
)
@click.option(
    "--enable-feature",
    "enable_features",
    multiple=True,
    help="Enable feature flag(s) for this command.",
)
@click.pass_context
@cli_command("render-spec")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Rendering timed out")
def render_spec_cmd(
    ctx: click.Context,
    spec_id: str,
    mode: Optional[str],
    enhancement_level: Optional[str],
    include_journal: bool,
    max_depth: int,
    phases: tuple,
    output: Optional[str],
    path: Optional[str],
    enable_features: tuple,
) -> None:
    """Render a specification to markdown.

    SPEC_ID is the specification identifier.

    Generates human-readable documentation from the spec structure.
    Enhanced mode requires --enable-feature enhanced_render flag.
    """
    # Apply feature flag overrides
    from foundry_mcp.cli.flags import apply_cli_flag_overrides
    if enable_features:
        apply_cli_flag_overrides(enable=list(enable_features))
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    # Resolve render mode (--mode takes precedence over --enhancement-level)
    render_mode = mode or enhancement_level or "basic"

    # Use --path if provided, otherwise fall back to context
    from pathlib import Path
    if path:
        specs_dir = Path(path) / "specs"
        if not specs_dir.exists():
            specs_dir = Path(path)
    else:
        specs_dir = cli_ctx.specs_dir

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set SDD_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set SDD_SPECS_DIR"},
        )

    # Check rate limit for render operations
    rate_result = check_rate_limit("render_spec")
    if not rate_result.allowed:
        emit_error(
            "Rate limit exceeded for render operations",
            code="RATE_LIMITED",
            error_type="validation",
            remediation=f"Wait {round(rate_result.reset_in, 0)} seconds before retrying",
            details={
                "reset_in_seconds": round(rate_result.reset_in, 2),
                "limit": rate_result.limit,
            },
        )

    # Check feature flag for enhanced mode
    if render_mode == "enhanced":
        flags = get_cli_flags()
        if not flags.is_enabled("enhanced_render"):
            emit_error(
                "Enhanced render mode requires feature flag",
                code="FEATURE_DISABLED",
                error_type="validation",
                remediation="Use --enable-feature enhanced_render to enable",
                details={
                    "flag": "enhanced_render",
                    "hint": "Use --enable-feature enhanced_render to enable",
                    "state": "beta",
                },
            )

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs list",
            details={"spec_id": spec_id},
        )

    # Build render options
    options = RenderOptions(
        mode=render_mode,
        include_journal=include_journal,
        max_depth=max_depth,
        phase_filter=list(phases) if phases else None,
    )

    # Render
    result = render_spec_to_markdown(spec_data, options)

    duration_ms = (time.perf_counter() - start_time) * 1000

    # Handle output to file
    if output:
        output_path = Path(output)
        output_path.write_text(result.markdown)
        emit_success({
            "spec_id": result.spec_id,
            "title": result.title,
            "mode": render_mode,
            "output_file": str(output_path),
            "stats": {
                "total_sections": result.total_sections,
                "total_tasks": result.total_tasks,
                "completed_tasks": result.completed_tasks,
            },
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })
    else:
        emit_success({
            "spec_id": result.spec_id,
            "title": result.title,
            "mode": render_mode,
            "markdown": result.markdown,
            "stats": {
                "total_sections": result.total_sections,
                "total_tasks": result.total_tasks,
                "completed_tasks": result.completed_tasks,
            },
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })


@render_group.command("progress")
@click.argument("spec_id")
@click.option(
    "--bar-width",
    type=int,
    default=20,
    help="Width of progress bars in characters.",
)
@click.pass_context
@cli_command("render-progress")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Progress render timed out")
def render_progress_cmd(
    ctx: click.Context,
    spec_id: str,
    bar_width: int,
) -> None:
    """Render progress visualization for a specification.

    SPEC_ID is the specification identifier.

    Returns ASCII progress bars for the spec and each phase.
    """
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

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs list",
            details={"spec_id": spec_id},
        )

    hierarchy = spec_data.get("hierarchy", {})
    root = hierarchy.get("spec-root", {})

    # Overall progress
    total = root.get("total_tasks", 0)
    completed = root.get("completed_tasks", 0)
    overall_bar = render_progress_bar(completed, total, bar_width)

    # Phase progress
    phases = []
    phase_ids = root.get("children", [])
    for phase_id in phase_ids:
        phase = hierarchy.get(phase_id, {})
        if phase.get("type") != "phase":
            continue

        phase_total = phase.get("total_tasks", 0)
        phase_completed = phase.get("completed_tasks", 0)
        phase_bar = render_progress_bar(phase_completed, phase_total, bar_width)

        phases.append({
            "id": phase_id,
            "title": phase.get("title", "Untitled"),
            "status": phase.get("status", "pending"),
            "completed": phase_completed,
            "total": phase_total,
            "progress_bar": phase_bar,
        })

    emit_success({
        "spec_id": spec_id,
        "overall": {
            "completed": completed,
            "total": total,
            "progress_bar": overall_bar,
        },
        "phases": phases,
    })


@render_group.command("tasks")
@click.argument("spec_id")
@click.option(
    "--status",
    type=click.Choice(["pending", "in_progress", "completed", "blocked"]),
    help="Filter by task status.",
)
@click.option(
    "--include-completed/--no-completed",
    default=True,
    help="Include completed tasks.",
)
@click.pass_context
@cli_command("render-tasks")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Task render timed out")
def render_tasks_cmd(
    ctx: click.Context,
    spec_id: str,
    status: Optional[str],
    include_completed: bool,
) -> None:
    """Render a flat list of tasks.

    SPEC_ID is the specification identifier.

    Returns all tasks in markdown list format.
    """
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

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs list",
            details={"spec_id": spec_id},
        )

    # Render task list
    markdown = render_task_list(
        spec_data,
        status_filter=status,
        include_completed=include_completed,
    )

    # Count tasks for stats
    hierarchy = spec_data.get("hierarchy", {})
    task_count = 0
    for node in hierarchy.values():
        if node.get("type") in ("task", "subtask", "verify"):
            node_status = node.get("status", "pending")
            if status and node_status != status:
                continue
            if not include_completed and node_status == "completed":
                continue
            task_count += 1

    emit_success({
        "spec_id": spec_id,
        "task_count": task_count,
        "status_filter": status,
        "include_completed": include_completed,
        "markdown": markdown,
    })


# Top-level render command (alias for spec)
@click.command("render")
@click.argument("spec_id")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["basic", "enhanced"]),
    default=None,
    help="Rendering mode.",
)
@click.option(
    "--enhancement-level",
    type=click.Choice(["basic", "enhanced"]),
    default=None,
    help="Alias for --mode (rendering enhancement level).",
)
@click.option(
    "--include-journal/--no-journal",
    default=False,
    help="Include journal entries.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Write markdown to file instead of JSON output.",
)
@click.option(
    "--path",
    type=click.Path(exists=True),
    help="Project root path (overrides default spec location).",
)
@click.option(
    "--enable-feature",
    "enable_features",
    multiple=True,
    help="Enable feature flag(s) for this command.",
)
@click.pass_context
@cli_command("render-alias")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Rendering timed out")
def render_cmd(
    ctx: click.Context,
    spec_id: str,
    mode: Optional[str],
    enhancement_level: Optional[str],
    include_journal: bool,
    output: Optional[str],
    path: Optional[str],
    enable_features: tuple,
) -> None:
    """Render a specification to markdown.

    SPEC_ID is the specification identifier.
    Enhanced mode requires --enable-feature enhanced_render flag.
    """
    # Apply feature flag overrides
    from foundry_mcp.cli.flags import apply_cli_flag_overrides
    if enable_features:
        apply_cli_flag_overrides(enable=list(enable_features))

    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    # Resolve render mode (--mode takes precedence over --enhancement-level)
    render_mode = mode or enhancement_level or "basic"

    # Use --path if provided, otherwise fall back to context
    from pathlib import Path
    if path:
        specs_dir = Path(path) / "specs"
        if not specs_dir.exists():
            specs_dir = Path(path)
    else:
        specs_dir = cli_ctx.specs_dir

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set SDD_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set SDD_SPECS_DIR"},
        )

    # Check rate limit for render operations
    rate_result = check_rate_limit("render_spec")
    if not rate_result.allowed:
        emit_error(
            "Rate limit exceeded for render operations",
            code="RATE_LIMITED",
            error_type="validation",
            remediation=f"Wait {round(rate_result.reset_in, 0)} seconds before retrying",
            details={
                "reset_in_seconds": round(rate_result.reset_in, 2),
                "limit": rate_result.limit,
            },
        )

    # Check feature flag for enhanced mode
    if render_mode == "enhanced":
        flags = get_cli_flags()
        if not flags.is_enabled("enhanced_render"):
            emit_error(
                "Enhanced render mode requires feature flag",
                code="FEATURE_DISABLED",
                error_type="validation",
                remediation="Use --enable-feature enhanced_render to enable",
                details={
                    "flag": "enhanced_render",
                    "hint": "Use --enable-feature enhanced_render to enable",
                    "state": "beta",
                },
            )

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs list",
            details={"spec_id": spec_id},
        )

    # Build render options
    options = RenderOptions(
        mode=render_mode,
        include_journal=include_journal,
    )

    # Render
    result = render_spec_to_markdown(spec_data, options)

    duration_ms = (time.perf_counter() - start_time) * 1000

    # Handle output to file
    if output:
        output_path = Path(output)
        output_path.write_text(result.markdown)
        emit_success({
            "spec_id": result.spec_id,
            "title": result.title,
            "mode": render_mode,
            "output_file": str(output_path),
            "stats": {
                "total_sections": result.total_sections,
                "total_tasks": result.total_tasks,
                "completed_tasks": result.completed_tasks,
            },
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })
    else:
        emit_success({
            "spec_id": result.spec_id,
            "title": result.title,
            "mode": render_mode,
            "markdown": result.markdown,
            "stats": {
                "total_sections": result.total_sections,
                "total_tasks": result.total_tasks,
                "completed_tasks": result.completed_tasks,
            },
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })
