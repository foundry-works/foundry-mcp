"""
Reporting tools for foundry-mcp.

Provides MCP tools for generating human-readable validation and analysis reports
for specifications, including validation summaries, progress reports, and health checks.
"""

import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.spec import (
    find_specs_directory,
    find_spec_file,
    load_spec,
)
from foundry_mcp.core.validation import (
    validate_spec,
    calculate_stats,
)
from foundry_mcp.core.responses import success_response, error_response, sanitize_error_message
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.resilience import (
    CircuitBreaker,
    CircuitBreakerError,
)
from foundry_mcp.core.observability import (
    get_metrics,
    get_audit_logger,
    mcp_tool,
)

logger = logging.getLogger(__name__)

# Circuit breaker for report generation operations
_report_breaker = CircuitBreaker(
    name="reporting",
    failure_threshold=5,
    recovery_timeout=30.0,
)


def _format_severity_indicator(severity: str) -> str:
    """Format severity level for display."""
    indicators = {
        "error": "[ERROR]",
        "warning": "[WARN]",
        "info": "[INFO]",
    }
    return indicators.get(severity.lower(), f"[{severity.upper()}]")


def _format_diagnostic(diag: Dict[str, Any]) -> str:
    """Format a single diagnostic for human-readable output."""
    indicator = _format_severity_indicator(diag.get("severity", "info"))
    code = diag.get("code", "UNKNOWN")
    message = diag.get("message", "No message")
    location = diag.get("location", "")

    line = f"  {indicator} {code}: {message}"
    if location:
        line += f" (at {location})"

    fix = diag.get("suggested_fix")
    if fix:
        line += f"\n    -> Fix: {fix}"

    return line


def _generate_validation_section(
    validation_result: Dict[str, Any],
) -> List[str]:
    """Generate the validation section of the report."""
    lines = []
    lines.append("## Validation Results")
    lines.append("")

    is_valid = validation_result.get("is_valid", False)
    error_count = validation_result.get("error_count", 0)
    warning_count = validation_result.get("warning_count", 0)
    info_count = validation_result.get("info_count", 0)

    status = "PASSED" if is_valid else "FAILED"
    status_emoji = "✓" if is_valid else "✗"

    lines.append(f"Status: {status_emoji} {status}")
    lines.append(f"- Errors: {error_count}")
    lines.append(f"- Warnings: {warning_count}")
    lines.append(f"- Info: {info_count}")
    lines.append("")

    diagnostics = validation_result.get("diagnostics", [])
    if diagnostics:
        lines.append("### Diagnostics")
        lines.append("")

        # Group by severity
        errors = [d for d in diagnostics if d.get("severity") == "error"]
        warnings = [d for d in diagnostics if d.get("severity") == "warning"]
        infos = [d for d in diagnostics if d.get("severity") == "info"]

        for diag in errors + warnings + infos:
            lines.append(_format_diagnostic(diag))

        lines.append("")

    return lines


def _generate_stats_section(stats: Dict[str, Any]) -> List[str]:
    """Generate the statistics section of the report."""
    lines = []
    lines.append("## Specification Statistics")
    lines.append("")

    # Basic info
    lines.append(f"Title: {stats.get('title', 'Unknown')}")
    lines.append(f"Version: {stats.get('version', 'Unknown')}")
    lines.append(f"Status: {stats.get('status', 'Unknown')}")
    lines.append("")

    # Progress
    progress = stats.get("progress", {})
    completed = progress.get("completed", 0)
    total = progress.get("total", 0)
    percentage = progress.get("percentage", 0)

    lines.append("### Progress")
    lines.append(f"- Tasks Completed: {completed}/{total} ({percentage}%)")

    # Progress bar
    bar_width = 20
    filled = int(bar_width * percentage / 100) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    lines.append(f"- Progress: [{bar}] {percentage}%")
    lines.append("")

    # Task status breakdown
    status_counts = stats.get("status_counts", {})
    if status_counts:
        lines.append("### Task Status Breakdown")
        for status, count in sorted(status_counts.items()):
            lines.append(f"- {status}: {count}")
        lines.append("")

    # Totals
    totals = stats.get("totals", {})
    if totals:
        lines.append("### Totals")
        lines.append(f"- Phases: {totals.get('phases', 0)}")
        lines.append(f"- Tasks: {totals.get('tasks', 0)}")
        lines.append(f"- Subtasks: {totals.get('subtasks', 0)}")
        lines.append("")

    # Metrics
    max_depth = stats.get("max_depth")
    avg_tasks = stats.get("avg_tasks_per_phase")
    verification = stats.get("verification_coverage")
    file_size = stats.get("file_size_kb")

    if any([max_depth, avg_tasks, verification, file_size]):
        lines.append("### Metrics")
        if max_depth is not None:
            lines.append(f"- Max Nesting Depth: {max_depth}")
        if avg_tasks is not None:
            lines.append(f"- Avg Tasks per Phase: {avg_tasks:.1f}")
        if verification is not None:
            lines.append(f"- Verification Coverage: {verification}%")
        if file_size is not None:
            lines.append(f"- File Size: {file_size:.1f} KB")
        lines.append("")

    return lines


def _generate_health_section(
    validation_result: Dict[str, Any],
    stats: Dict[str, Any],
) -> List[str]:
    """Generate the health assessment section of the report."""
    lines = []
    lines.append("## Health Assessment")
    lines.append("")

    health_issues = []
    health_score = 100

    # Check validation status
    if not validation_result.get("is_valid", False):
        error_count = validation_result.get("error_count", 0)
        health_issues.append(f"Validation errors: {error_count}")
        health_score -= min(30, error_count * 10)

    # Check progress
    progress = stats.get("progress", {})
    percentage = progress.get("percentage", 0)
    if percentage < 25:
        health_issues.append("Low progress (< 25%)")

    # Check for warnings
    warning_count = validation_result.get("warning_count", 0)
    if warning_count > 5:
        health_issues.append(f"High warning count: {warning_count}")
        health_score -= min(20, warning_count * 2)

    # Check verification coverage
    verification = stats.get("verification_coverage", 0)
    if verification is not None and verification < 50:
        health_issues.append(f"Low verification coverage: {verification}%")
        health_score -= 10

    health_score = max(0, health_score)

    if health_score >= 80:
        health_status = "HEALTHY"
        health_emoji = "✓"
    elif health_score >= 50:
        health_status = "NEEDS ATTENTION"
        health_emoji = "⚠"
    else:
        health_status = "CRITICAL"
        health_emoji = "✗"

    lines.append(f"Health Score: {health_emoji} {health_score}/100 ({health_status})")
    lines.append("")

    if health_issues:
        lines.append("### Issues Detected")
        for issue in health_issues:
            lines.append(f"- {issue}")
        lines.append("")
    else:
        lines.append("No health issues detected.")
        lines.append("")

    return lines


def register_reporting_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register reporting tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """
    metrics = get_metrics()
    audit = get_audit_logger()

    @canonical_tool(
        mcp,
        canonical_name="spec-report",
    )
    @mcp_tool(tool_name="spec-report", emit_metrics=True, audit=True)
    def spec_report(
        spec_id: str,
        format: str = "markdown",
        sections: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> dict:
        """
        Generate a comprehensive human-readable report for a specification.

        Combines validation results, statistics, and health assessment into
        a single report suitable for review and documentation.

        Args:
            spec_id: Specification ID to report on
            format: Output format ('markdown', 'summary', 'json')
            sections: Comma-separated list of sections to include
                     ('validation', 'stats', 'health', 'all'). Default: 'all'
            workspace: Optional workspace path

        Returns:
            JSON object with:
            - report: The formatted report content
            - format: The output format used
            - sections: Sections included in the report
            - summary: Quick summary of key metrics
        """
        start_time = time.perf_counter()

        try:
            # Circuit breaker check
            if not _report_breaker.can_execute():
                status = _report_breaker.get_status()
                metrics.counter(
                    "report.circuit_breaker_open",
                    labels={"tool": "spec-report"},
                )
                return asdict(
                    error_response(
                        "Report generation temporarily unavailable",
                        data={
                            "retry_after_seconds": status.get("retry_after_seconds"),
                            "breaker_state": status.get("state"),
                        },
                    )
                )

            # Resolve specs directory
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                _report_breaker.record_failure()
                return asdict(error_response("No specs directory found"))

            # Load spec
            spec_data = load_spec(spec_id, specs_dir)
            if not spec_data:
                return asdict(error_response(f"Spec not found: {spec_id}"))

            spec_path = find_spec_file(spec_id, specs_dir)

            # Parse sections
            requested_sections = set()
            if sections is None or sections.lower() == "all":
                requested_sections = {"validation", "stats", "health"}
            else:
                for s in sections.lower().split(","):
                    s = s.strip()
                    if s in ("validation", "stats", "health"):
                        requested_sections.add(s)

            # Gather data
            validation_result = {}
            stats_result = {}

            if "validation" in requested_sections or "health" in requested_sections:
                result = validate_spec(spec_data)
                validation_result = {
                    "is_valid": result.is_valid,
                    "error_count": result.error_count,
                    "warning_count": result.warning_count,
                    "info_count": result.info_count,
                    "diagnostics": [
                        {
                            "code": d.code,
                            "message": d.message,
                            "severity": d.severity,
                            "category": d.category,
                            "location": d.location,
                            "suggested_fix": d.suggested_fix,
                            "auto_fixable": d.auto_fixable,
                        }
                        for d in result.diagnostics
                    ],
                }

            if "stats" in requested_sections or "health" in requested_sections:
                stats = calculate_stats(spec_data, str(spec_path) if spec_path else None)
                stats_result = {
                    "title": stats.title,
                    "version": stats.version,
                    "status": stats.status,
                    "totals": stats.totals,
                    "status_counts": stats.status_counts,
                    "max_depth": stats.max_depth,
                    "avg_tasks_per_phase": stats.avg_tasks_per_phase,
                    "verification_coverage": stats.verification_coverage,
                    "progress": stats.progress,
                    "file_size_kb": stats.file_size_kb,
                }

            # Generate report based on format
            if format.lower() == "json":
                # Return raw data
                report_data = {
                    "spec_id": spec_id,
                }
                if "validation" in requested_sections:
                    report_data["validation"] = validation_result
                if "stats" in requested_sections:
                    report_data["statistics"] = stats_result
                if "health" in requested_sections:
                    # Calculate health score
                    health_score = 100
                    if not validation_result.get("is_valid", True):
                        health_score -= min(30, validation_result.get("error_count", 0) * 10)
                    if validation_result.get("warning_count", 0) > 5:
                        health_score -= min(20, validation_result.get("warning_count", 0) * 2)
                    report_data["health"] = {
                        "score": max(0, health_score),
                        "status": "healthy" if health_score >= 80 else (
                            "needs_attention" if health_score >= 50 else "critical"
                        ),
                    }

                _report_breaker.record_success()
                duration_ms = (time.perf_counter() - start_time) * 1000

                metrics.timer(
                    "report.generation_time",
                    duration_ms,
                    labels={"spec_id": spec_id, "format": "json"},
                )

                return asdict(
                    success_response(
                        report=report_data,
                        format="json",
                        sections=list(requested_sections),
                        telemetry={"duration_ms": round(duration_ms, 2)},
                    )
                )

            # Generate markdown or summary format
            lines = []
            lines.append(f"# Spec Report: {spec_id}")
            lines.append("")
            lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
            lines.append("")

            if "validation" in requested_sections:
                lines.extend(_generate_validation_section(validation_result))

            if "stats" in requested_sections:
                lines.extend(_generate_stats_section(stats_result))

            if "health" in requested_sections:
                lines.extend(_generate_health_section(validation_result, stats_result))

            report_content = "\n".join(lines)

            # Generate summary
            summary = {
                "spec_id": spec_id,
                "is_valid": validation_result.get("is_valid", True),
                "error_count": validation_result.get("error_count", 0),
                "warning_count": validation_result.get("warning_count", 0),
            }

            if stats_result:
                progress = stats_result.get("progress", {})
                summary["progress_percentage"] = progress.get("percentage", 0)
                summary["tasks_completed"] = progress.get("completed", 0)
                summary["tasks_total"] = progress.get("total", 0)

            _report_breaker.record_success()
            duration_ms = (time.perf_counter() - start_time) * 1000

            metrics.timer(
                "report.generation_time",
                duration_ms,
                labels={"spec_id": spec_id, "format": format},
            )

            return asdict(
                success_response(
                    report=report_content,
                    format=format.lower(),
                    sections=list(requested_sections),
                    summary=summary,
                    telemetry={"duration_ms": round(duration_ms, 2)},
                )
            )

        except CircuitBreakerError as e:
            logger.warning(f"Circuit breaker open for reporting: {e}")
            return asdict(
                error_response(
                    "Report generation temporarily unavailable",
                    data={"retry_after_seconds": e.retry_after},
                )
            )
        except Exception as e:
            _report_breaker.record_failure()
            logger.error(f"Error generating report: {e}")
            return asdict(error_response(sanitize_error_message(e, context="reporting")))

    @canonical_tool(
        mcp,
        canonical_name="spec-report-summary",
    )
    @mcp_tool(tool_name="spec-report-summary", emit_metrics=True, audit=True)
    def spec_report_summary(
        spec_id: str,
        workspace: Optional[str] = None,
    ) -> dict:
        """
        Generate a quick summary report for a specification.

        Provides a condensed view of validation status and progress,
        suitable for dashboards and quick checks.

        Args:
            spec_id: Specification ID to summarize
            workspace: Optional workspace path

        Returns:
            JSON object with key metrics and status indicators
        """
        start_time = time.perf_counter()

        try:
            # Resolve specs directory
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return asdict(error_response("No specs directory found"))

            # Load spec
            spec_data = load_spec(spec_id, specs_dir)
            if not spec_data:
                return asdict(error_response(f"Spec not found: {spec_id}"))

            spec_path = find_spec_file(spec_id, specs_dir)

            # Validate
            result = validate_spec(spec_data)

            # Get stats
            stats = calculate_stats(spec_data, str(spec_path) if spec_path else None)

            # Calculate health score
            health_score = 100
            if not result.is_valid:
                health_score -= min(30, result.error_count * 10)
            if result.warning_count > 5:
                health_score -= min(20, result.warning_count * 2)
            health_score = max(0, health_score)

            duration_ms = (time.perf_counter() - start_time) * 1000

            return asdict(
                success_response(
                    spec_id=spec_id,
                    title=stats.title,
                    status=stats.status,
                    validation={
                        "is_valid": result.is_valid,
                        "errors": result.error_count,
                        "warnings": result.warning_count,
                    },
                    progress={
                        "completed": stats.progress.get("completed", 0),
                        "total": stats.progress.get("total", 0),
                        "percentage": stats.progress.get("percentage", 0),
                    },
                    health={
                        "score": health_score,
                        "status": "healthy" if health_score >= 80 else (
                            "needs_attention" if health_score >= 50 else "critical"
                        ),
                    },
                    telemetry={"duration_ms": round(duration_ms, 2)},
                )
            )

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return asdict(error_response(sanitize_error_message(e, context="reporting")))

    logger.debug("Registered reporting tools: spec-report/spec-report-summary")
