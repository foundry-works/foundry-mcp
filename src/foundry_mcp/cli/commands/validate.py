"""Validation commands for SDD CLI.

Provides commands for spec validation, auto-fix, statistics, and reporting.
"""

import time
from typing import Optional

import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error, emit_success
from foundry_mcp.cli.registry import get_context
from foundry_mcp.cli.resilience import (
    MEDIUM_TIMEOUT,
    with_sync_timeout,
    handle_keyboard_interrupt,
)

logger = get_cli_logger()
from foundry_mcp.core.spec import load_spec, find_spec_file
from foundry_mcp.core.validation import (
    apply_fixes,
    calculate_stats,
    get_fix_actions,
    validate_spec,
)


@click.group("validate")
def validate_group() -> None:
    """Spec validation and fix commands."""
    pass


@validate_group.command("check")
@click.argument("spec_id")
@click.pass_context
@cli_command("check")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Validation check timed out")
def validate_check_cmd(ctx: click.Context, spec_id: str) -> None:
    """Validate a specification and report diagnostics.

    SPEC_ID is the specification identifier.
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
        return

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
        return

    # Run validation
    result = validate_spec(spec_data)

    # Format diagnostics for output
    diagnostics = []
    for diag in result.diagnostics:
        diagnostics.append(
            {
                "code": diag.code,
                "message": diag.message,
                "severity": diag.severity,
                "category": diag.category,
                "location": diag.location,
                "suggested_fix": diag.suggested_fix,
                "auto_fixable": diag.auto_fixable,
            }
        )

    emit_success(
        {
            "spec_id": result.spec_id,
            "is_valid": result.is_valid,
            "error_count": result.error_count,
            "warning_count": result.warning_count,
            "info_count": result.info_count,
            "diagnostics": diagnostics,
        }
    )


@validate_group.command("fix")
@click.argument("spec_id")
@click.option("--dry-run", is_flag=True, help="Preview fixes without applying.")
@click.option("--no-backup", is_flag=True, help="Skip creating backup file.")
@click.pass_context
@cli_command("fix")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Validation fix timed out")
def validate_fix_cmd(
    ctx: click.Context,
    spec_id: str,
    dry_run: bool,
    no_backup: bool,
) -> None:
    """Apply auto-fixes to a specification.

    SPEC_ID is the specification identifier.
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
        return

    # Find spec path
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs list",
            details={"spec_id": spec_id},
        )
        return

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Failed to load specification: {spec_id}",
            code="INTERNAL_ERROR",
            error_type="internal",
            remediation="Check that the spec file is valid JSON",
            details={"spec_id": spec_id},
        )
        return

    # Validate to get diagnostics
    result = validate_spec(spec_data)

    # Generate fix actions
    actions = get_fix_actions(result, spec_data)

    if not actions:
        emit_success(
            {
                "spec_id": spec_id,
                "applied_count": 0,
                "skipped_count": 0,
                "message": "No auto-fixable issues found",
            }
        )
        return

    # Apply fixes
    report = apply_fixes(
        actions,
        str(spec_path),
        dry_run=dry_run,
        create_backup=not no_backup,
    )

    # Format applied/skipped actions
    applied = [
        {"id": a.id, "description": a.description, "category": a.category}
        for a in report.applied_actions
    ]
    skipped = [
        {"id": a.id, "description": a.description, "category": a.category}
        for a in report.skipped_actions
    ]

    emit_success(
        {
            "spec_id": spec_id,
            "dry_run": dry_run,
            "applied_count": len(applied),
            "skipped_count": len(skipped),
            "applied_actions": applied,
            "skipped_actions": skipped,
            "backup_path": report.backup_path,
        }
    )


@validate_group.command("stats")
@click.argument("spec_id")
@click.pass_context
@cli_command("stats")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Statistics calculation timed out")
def validate_stats_cmd(ctx: click.Context, spec_id: str) -> None:
    """Get statistics for a specification.

    SPEC_ID is the specification identifier.
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
        return

    # Find spec path
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs list",
            details={"spec_id": spec_id},
        )
        return

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Failed to load specification: {spec_id}",
            code="INTERNAL_ERROR",
            error_type="internal",
            remediation="Check that the spec file is valid JSON",
            details={"spec_id": spec_id},
        )
        return

    # Calculate stats
    stats = calculate_stats(spec_data, str(spec_path))

    emit_success(
        {
            "spec_id": stats.spec_id,
            "title": stats.title,
            "version": stats.version,
            "status": stats.status,
            "totals": stats.totals,
            "status_counts": stats.status_counts,
            "max_depth": stats.max_depth,
            "avg_tasks_per_phase": stats.avg_tasks_per_phase,
            "verification_coverage": round(stats.verification_coverage * 100, 1),
            "progress": round(stats.progress * 100, 1),
            "file_size_kb": round(stats.file_size_kb, 2),
        }
    )


@validate_group.command("report")
@click.argument("spec_id")
@click.option(
    "--sections",
    "-s",
    default="all",
    help="Sections to include: validation,stats,health,all",
)
@click.pass_context
@cli_command("report")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Report generation timed out")
def validate_report_cmd(
    ctx: click.Context,
    spec_id: str,
    sections: str,
) -> None:
    """Generate a comprehensive report for a specification.

    SPEC_ID is the specification identifier.

    Combines validation, statistics, and health assessment into
    a single report suitable for review and documentation.
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

    # Find spec path
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs list",
            details={"spec_id": spec_id},
        )
        return

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Failed to load specification: {spec_id}",
            code="INTERNAL_ERROR",
            error_type="internal",
            remediation="Check that the spec file is valid JSON",
            details={"spec_id": spec_id},
        )
        return

    # Parse sections
    requested_sections = set()
    if sections.lower() == "all":
        requested_sections = {"validation", "stats", "health"}
    else:
        for s in sections.lower().split(","):
            s = s.strip()
            if s in ("validation", "stats", "health"):
                requested_sections.add(s)

    output: dict = {
        "spec_id": spec_id,
        "sections": list(requested_sections),
    }

    # Validation section
    if "validation" in requested_sections or "health" in requested_sections:
        result = validate_spec(spec_data)
        diagnostics = []
        for diag in result.diagnostics:
            diagnostics.append(
                {
                    "code": diag.code,
                    "message": diag.message,
                    "severity": diag.severity,
                    "category": diag.category,
                    "location": diag.location,
                    "suggested_fix": diag.suggested_fix,
                    "auto_fixable": diag.auto_fixable,
                }
            )

        if "validation" in requested_sections:
            output["validation"] = {
                "is_valid": result.is_valid,
                "error_count": result.error_count,
                "warning_count": result.warning_count,
                "info_count": result.info_count,
                "diagnostics": diagnostics,
            }

    # Stats section
    if "stats" in requested_sections or "health" in requested_sections:
        stats = calculate_stats(spec_data, str(spec_path))

        if "stats" in requested_sections:
            output["statistics"] = {
                "title": stats.title,
                "version": stats.version,
                "status": stats.status,
                "totals": stats.totals,
                "status_counts": stats.status_counts,
                "max_depth": stats.max_depth,
                "avg_tasks_per_phase": stats.avg_tasks_per_phase,
                "verification_coverage": round(stats.verification_coverage * 100, 1),
                "progress": stats.progress,
                "file_size_kb": round(stats.file_size_kb, 2),
            }

    # Health section
    if "health" in requested_sections:
        health_score = 100
        health_issues = []

        # Validation impact
        if "validation" in output:
            validation = output["validation"]
            if not validation["is_valid"]:
                error_count = validation["error_count"]
                health_issues.append(f"Validation errors: {error_count}")
                health_score -= min(30, error_count * 10)
            if validation["warning_count"] > 5:
                health_issues.append(
                    f"High warning count: {validation['warning_count']}"
                )
                health_score -= min(20, validation["warning_count"] * 2)

        # Stats impact
        if "stats" in output:
            statistics = output["statistics"]
            if statistics["verification_coverage"] < 50:
                health_issues.append(
                    f"Low verification coverage: {statistics['verification_coverage']}%"
                )
                health_score -= 10

        health_score = max(0, health_score)

        if health_score >= 80:
            health_status = "healthy"
        elif health_score >= 50:
            health_status = "needs_attention"
        else:
            health_status = "critical"

        output["health"] = {
            "score": health_score,
            "status": health_status,
            "issues": health_issues,
        }

    # Summary
    output["summary"] = {
        "spec_id": spec_id,
        "is_valid": output.get("validation", {}).get("is_valid", True),
        "error_count": output.get("validation", {}).get("error_count", 0),
        "warning_count": output.get("validation", {}).get("warning_count", 0),
        "health_score": output.get("health", {}).get("score", 100),
    }

    duration_ms = (time.perf_counter() - start_time) * 1000
    output["telemetry"] = {"duration_ms": round(duration_ms, 2)}

    emit_success(output)


@validate_group.command("analyze-deps")
@click.argument("spec_id")
@click.option(
    "--bottleneck-threshold",
    "-t",
    type=int,
    default=3,
    help="Minimum tasks blocked to flag as bottleneck.",
)
@click.option(
    "--limit",
    type=int,
    default=100,
    help="Maximum items to return per section.",
)
@click.pass_context
@cli_command("analyze-deps")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Dependency analysis timed out")
def validate_analyze_deps_cmd(
    ctx: click.Context,
    spec_id: str,
    bottleneck_threshold: int,
    limit: int,
) -> None:
    """Analyze dependency graph health for a specification.

    SPEC_ID is the specification identifier.

    Identifies blocking tasks, bottlenecks, circular dependencies,
    and the critical path for task completion.
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
        return

    hierarchy = spec_data.get("hierarchy", {})

    # Collect all dependency relationships
    all_deps = []
    blocks_count: dict = {}  # task_id -> count of tasks it blocks
    blocked_by_map: dict = {}  # task_id -> list of blockers

    for node_id, node in hierarchy.items():
        if node.get("type") not in ["task", "subtask", "verify", "phase"]:
            continue

        deps = node.get("dependencies", {})
        blocked_by = deps.get("blocked_by", [])
        blocks = deps.get("blocks", [])

        blocked_by_map[node_id] = blocked_by

        for blocker_id in blocked_by:
            all_deps.append(
                {
                    "from": blocker_id,
                    "to": node_id,
                    "type": "blocks",
                }
            )

        # Count how many tasks each node blocks
        for blocked_id in blocks:
            blocks_count[node_id] = blocks_count.get(node_id, 0) + 1

        # Also count from blocked_by relationships
        for blocker_id in blocked_by:
            blocks_count[blocker_id] = blocks_count.get(blocker_id, 0) + 1

    # Find bottlenecks (tasks blocking many others)
    bottlenecks = []
    for task_id, count in sorted(blocks_count.items(), key=lambda x: -x[1]):
        if count >= bottleneck_threshold:
            task = hierarchy.get(task_id, {})
            bottlenecks.append(
                {
                    "id": task_id,
                    "title": task.get("title", ""),
                    "status": task.get("status", ""),
                    "blocks_count": count,
                }
            )
        if len(bottlenecks) >= limit:
            break

    # Detect circular dependencies using DFS
    circular_deps = []
    visited = set()
    rec_stack = set()

    def detect_cycle(node_id: str, path: list) -> Optional[list]:
        if node_id in rec_stack:
            # Found a cycle
            cycle_start = path.index(node_id)
            return path[cycle_start:] + [node_id]
        if node_id in visited:
            return None

        visited.add(node_id)
        rec_stack.add(node_id)
        path.append(node_id)

        for blocker_id in blocked_by_map.get(node_id, []):
            cycle = detect_cycle(blocker_id, path[:])
            if cycle:
                return cycle

        rec_stack.remove(node_id)
        return None

    for node_id in hierarchy:
        if node_id not in visited:
            cycle = detect_cycle(node_id, [])
            if cycle and cycle not in circular_deps:
                circular_deps.append(cycle)
                if len(circular_deps) >= limit:
                    break

    # Calculate critical path (longest dependency chain)
    def get_chain_length(node_id: str, memo: dict) -> int:
        if node_id in memo:
            return memo[node_id]
        blockers = blocked_by_map.get(node_id, [])
        if not blockers:
            memo[node_id] = 1
            return 1
        max_blocker = max(get_chain_length(b, memo) for b in blockers)
        memo[node_id] = max_blocker + 1
        return memo[node_id]

    chain_lengths: dict = {}
    for node_id in hierarchy:
        if hierarchy.get(node_id, {}).get("type") in ["task", "subtask", "verify"]:
            try:
                get_chain_length(node_id, chain_lengths)
            except RecursionError:
                # Circular dependency detected
                pass

    # Find critical path
    critical_path = []
    if chain_lengths:
        max_length = max(chain_lengths.values())
        for node_id, length in sorted(chain_lengths.items(), key=lambda x: -x[1]):
            if length == max_length:
                task = hierarchy.get(node_id, {})
                critical_path.append(
                    {
                        "id": node_id,
                        "title": task.get("title", ""),
                        "status": task.get("status", ""),
                        "chain_length": length,
                    }
                )
            if len(critical_path) >= limit:
                break

    duration_ms = (time.perf_counter() - start_time) * 1000

    emit_success(
        {
            "spec_id": spec_id,
            "dependency_count": len(all_deps),
            "bottlenecks": bottlenecks,
            "bottleneck_threshold": bottleneck_threshold,
            "circular_deps": circular_deps,
            "has_circular_deps": len(circular_deps) > 0,
            "critical_path": critical_path,
            "max_chain_length": max(chain_lengths.values()) if chain_lengths else 0,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        }
    )


# Top-level validate command (alias for check)
@click.command("validate")
@click.argument("spec_id")
@click.option(
    "--fix", "auto_fix", is_flag=True, help="Auto-fix issues after validation."
)
@click.option(
    "--dry-run", is_flag=True, help="Preview fixes without applying (requires --fix)."
)
@click.option(
    "--preview", is_flag=True, help="Show summary only (counts and issue codes)."
)
@click.option(
    "--diff",
    "show_diff",
    is_flag=True,
    help="Show unified diff of changes (requires --fix).",
)
@click.option(
    "--select", "select_codes", help="Only fix selected issue codes (comma-separated)."
)
@click.pass_context
@cli_command("validate")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Validation timed out")
def validate_cmd(
    ctx: click.Context,
    spec_id: str,
    auto_fix: bool,
    dry_run: bool,
    preview: bool,
    show_diff: bool,
    select_codes: Optional[str],
) -> None:
    """Validate a specification and optionally apply fixes.

    SPEC_ID is the specification identifier.
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
        return

    # Find spec path
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs list",
            details={"spec_id": spec_id},
        )
        return

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Failed to load specification: {spec_id}",
            code="INTERNAL_ERROR",
            error_type="internal",
            remediation="Check that the spec file is valid JSON",
            details={"spec_id": spec_id},
        )
        return

    # Run validation
    result = validate_spec(spec_data)

    # Parse select codes if provided
    selected_codes = None
    if select_codes:
        selected_codes = set(code.strip() for code in select_codes.split(","))

    # Format diagnostics (filtered by select if provided)
    diagnostics = []
    for diag in result.diagnostics:
        if selected_codes and diag.code not in selected_codes:
            continue
        if preview:
            # Preview mode: only include code and severity
            diagnostics.append(
                {
                    "code": diag.code,
                    "severity": diag.severity,
                    "auto_fixable": diag.auto_fixable,
                }
            )
        else:
            diagnostics.append(
                {
                    "code": diag.code,
                    "message": diag.message,
                    "severity": diag.severity,
                    "category": diag.category,
                    "location": diag.location,
                    "suggested_fix": diag.suggested_fix,
                    "auto_fixable": diag.auto_fixable,
                }
            )

    output: dict = {
        "spec_id": result.spec_id,
        "is_valid": result.is_valid,
        "error_count": result.error_count,
        "warning_count": result.warning_count,
        "info_count": result.info_count,
        "preview": preview,
    }

    if not preview:
        output["diagnostics"] = diagnostics
    else:
        # Preview mode: group by code
        code_summary: dict = {}
        for diag in diagnostics:
            code = diag["code"]
            if code not in code_summary:
                code_summary[code] = {
                    "count": 0,
                    "severity": diag["severity"],
                    "auto_fixable": diag["auto_fixable"],
                }
            code_summary[code]["count"] += 1
        output["issue_summary"] = code_summary

    # Apply fixes if requested
    if auto_fix:
        actions = get_fix_actions(result, spec_data)

        # Filter actions by selected codes
        if selected_codes:
            actions = [
                a
                for a in actions
                if a.id in selected_codes
                or any(code in a.id for code in selected_codes)
            ]

        if actions:
            # Read original content for diff
            original_content = None
            if show_diff and spec_path:
                with open(spec_path) as f:
                    original_content = f.read()

            report = apply_fixes(
                actions,
                str(spec_path),
                dry_run=dry_run,
                create_backup=True,
            )
            output["fix_applied"] = not dry_run
            output["fix_dry_run"] = dry_run
            output["fixes_count"] = len(report.applied_actions)
            output["fixes"] = [
                {"id": a.id, "description": a.description}
                for a in report.applied_actions
            ]
            output["backup_path"] = report.backup_path

            # Generate diff if requested
            if show_diff and spec_path and original_content and not dry_run:
                import difflib

                with open(spec_path) as f:
                    new_content = f.read()
                diff_lines = list(
                    difflib.unified_diff(
                        original_content.splitlines(keepends=True),
                        new_content.splitlines(keepends=True),
                        fromfile=f"{spec_id} (before)",
                        tofile=f"{spec_id} (after)",
                    )
                )
                output["diff"] = "".join(diff_lines)
        else:
            output["fix_applied"] = False
            output["fixes_count"] = 0
            output["fixes"] = []

    emit_success(output)


# Top-level fix command (alias for validate fix)
@click.command("fix")
@click.argument("spec_id")
@click.option("--dry-run", is_flag=True, help="Preview fixes without applying.")
@click.option("--no-backup", is_flag=True, help="Skip creating backup file.")
@click.option("--diff", "show_diff", is_flag=True, help="Show unified diff of changes.")
@click.option(
    "--select", "select_codes", help="Only fix selected issue codes (comma-separated)."
)
@click.pass_context
@cli_command("fix")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Fix operation timed out")
def fix_cmd(
    ctx: click.Context,
    spec_id: str,
    dry_run: bool,
    no_backup: bool,
    show_diff: bool,
    select_codes: Optional[str],
) -> None:
    """Apply auto-fixes to a specification.

    SPEC_ID is the specification identifier.

    This is a top-level alias for `sdd validate fix`.
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
        return

    # Find spec path
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs list",
            details={"spec_id": spec_id},
        )
        return

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Failed to load specification: {spec_id}",
            code="INTERNAL_ERROR",
            error_type="internal",
            remediation="Check that the spec file is valid JSON",
            details={"spec_id": spec_id},
        )
        return

    # Validate to get diagnostics
    result = validate_spec(spec_data)

    # Generate fix actions
    actions = get_fix_actions(result, spec_data)

    # Parse and filter by selected codes
    if select_codes:
        selected_codes = set(code.strip() for code in select_codes.split(","))
        actions = [
            a
            for a in actions
            if a.id in selected_codes or any(code in a.id for code in selected_codes)
        ]

    if not actions:
        emit_success(
            {
                "spec_id": spec_id,
                "applied_count": 0,
                "skipped_count": 0,
                "message": "No auto-fixable issues found"
                + (" matching selection" if select_codes else ""),
            }
        )
        return

    # Read original content for diff
    original_content = None
    if show_diff and spec_path:
        with open(spec_path) as f:
            original_content = f.read()

    # Apply fixes
    report = apply_fixes(
        actions,
        str(spec_path),
        dry_run=dry_run,
        create_backup=not no_backup,
    )

    # Format applied/skipped actions
    applied = [
        {"id": a.id, "description": a.description, "category": a.category}
        for a in report.applied_actions
    ]
    skipped = [
        {"id": a.id, "description": a.description, "category": a.category}
        for a in report.skipped_actions
    ]

    output: dict = {
        "spec_id": spec_id,
        "dry_run": dry_run,
        "applied_count": len(applied),
        "skipped_count": len(skipped),
        "applied_actions": applied,
        "skipped_actions": skipped,
        "backup_path": report.backup_path,
    }

    # Generate diff if requested
    if show_diff and spec_path and original_content and not dry_run:
        import difflib

        with open(spec_path) as f:
            new_content = f.read()
        diff_lines = list(
            difflib.unified_diff(
                original_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"{spec_id} (before)",
                tofile=f"{spec_id} (after)",
            )
        )
        output["diff"] = "".join(diff_lines)

    emit_success(output)
