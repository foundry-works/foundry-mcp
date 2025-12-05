"""
Planning tools for foundry-mcp.

Provides MCP tools for task planning and execution utilities,
including plan formatting, phase management, time reporting,
and spec state reconciliation. Uses direct Python API calls to core modules.
"""

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response, sanitize_error_message
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import audit_log, get_metrics
from foundry_mcp.core.spec import find_specs_directory, find_spec_file, load_spec
from foundry_mcp.core.progress import get_progress_summary, list_phases

logger = logging.getLogger(__name__)

# Metrics singleton for planning tools
_metrics = get_metrics()


def register_planning_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register planning tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="plan-format",
    )
    def plan_format(
        spec_id: str,
        task_id: str,
        path: Optional[str] = None,
    ) -> dict:
        """
        Pretty-print a task plan for sharing and review.

        Generates a human-readable formatted output of a task plan,
        suitable for sharing with team members or including in documentation.

        WHEN TO USE:
        - Sharing task plans with team members
        - Generating documentation from task plans
        - Reviewing task structure before implementation
        - Creating readable summaries of complex tasks

        Args:
            spec_id: Specification ID containing the task
            task_id: Task ID to format
            path: Project root path (default: current directory)

        Returns:
            JSON object with formatted plan:
            - formatted: Human-readable plan text
            - task_id: The task ID that was formatted
            - spec_id: The specification ID
        """
        tool_name = "plan_format"
        start_time = time.perf_counter()

        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            if not task_id:
                return asdict(error_response(
                    "task_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a task_id parameter",
                ))

            # Resolve workspace path
            ws_path = Path(path) if path else Path.cwd()

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="plan-format",
                action="format_plan",
                spec_id=spec_id,
                task_id=task_id,
            )

            # Find and load spec
            specs_dir = find_specs_directory(ws_path)
            if not specs_dir:
                return asdict(error_response(
                    f"Specs directory not found in {ws_path}",
                    error_code="NOT_FOUND",
                    error_type="planning",
                    data={"spec_id": spec_id, "workspace": str(ws_path)},
                    remediation="Ensure specs/ directory exists in workspace",
                ))

            spec_file = find_spec_file(spec_id, specs_dir)
            if not spec_file:
                return asdict(error_response(
                    f"Specification '{spec_id}' not found",
                    error_code="NOT_FOUND",
                    error_type="planning",
                    remediation="Verify the spec ID exists using spec-list",
                ))

            spec_data = load_spec(spec_file)
            if not spec_data:
                return asdict(error_response(
                    f"Failed to load spec '{spec_id}'",
                    error_code="LOAD_ERROR",
                    error_type="planning",
                    data={"spec_id": spec_id, "spec_file": str(spec_file)},
                ))

            # Get task from hierarchy
            hierarchy = spec_data.get("hierarchy", {})
            task = hierarchy.get(task_id)
            if not task:
                return asdict(error_response(
                    f"Task '{task_id}' not found in spec '{spec_id}'",
                    error_code="NOT_FOUND",
                    error_type="planning",
                    remediation="Verify the task ID exists in the spec",
                ))

            # Format the task plan directly from task data
            task_title = task.get("title", "Untitled Task")
            task_status = task.get("status", "pending")
            task_description = task.get("description", "")
            metadata = task.get("metadata", {})

            # Build formatted output
            lines = [
                f"# Task: {task_title}",
                f"**ID:** {task_id}",
                f"**Status:** {task_status}",
            ]
            if task_description:
                lines.append(f"**Description:** {task_description}")
            if metadata.get("estimated_hours"):
                lines.append(f"**Estimated Hours:** {metadata['estimated_hours']}")
            if metadata.get("actual_hours"):
                lines.append(f"**Actual Hours:** {metadata['actual_hours']}")
            if metadata.get("file_path"):
                lines.append(f"**File:** {metadata['file_path']}")

            # Include children if any
            children = task.get("children", [])
            if children:
                lines.append("\n## Subtasks:")
                for child_id in children:
                    child = hierarchy.get(child_id, {})
                    child_title = child.get("title", child_id)
                    child_status = child.get("status", "pending")
                    lines.append(f"- [{child_status}] {child_title} ({child_id})")

            formatted = "\n".join(lines)

            duration_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer(f"planning.{tool_name}.duration_ms", duration_ms)
            _metrics.counter(f"planning.{tool_name}", labels={"status": "success"})

            return asdict(success_response(
                spec_id=spec_id,
                task_id=task_id,
                title=task_title,
                status=task_status,
                formatted=formatted,
                duration_ms=round(duration_ms, 2),
            ))

        except Exception as e:
            logger.exception(f"Unexpected error in {tool_name}")
            _metrics.counter(f"planning.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                sanitize_error_message(e, context="planning"),
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="phase-list",
    )
    def phase_list(
        spec_id: str,
        path: Optional[str] = None,
    ) -> dict:
        """
        Enumerate all phases in a specification.

        Returns all phases in a specification with their completion status
        and progress.

        WHEN TO USE:
        - Getting an overview of spec structure
        - Checking phase completion status
        - Planning which phase to work on next
        - Understanding spec organization

        Args:
            spec_id: Specification ID to enumerate phases for
            path: Project root path (default: current directory)

        Returns:
            JSON object with phase list:
            - phases: Array of phase objects with id, title, status, progress
            - total_phases: Total number of phases
            - completed_phases: Number of completed phases
        """
        tool_name = "phase_list"
        start_time = time.perf_counter()

        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            # Resolve workspace path
            ws_path = Path(path) if path else Path.cwd()

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="phase-list",
                action="list_phases",
                spec_id=spec_id,
            )

            # Find and load spec
            specs_dir = find_specs_directory(ws_path)
            if not specs_dir:
                return asdict(error_response(
                    f"Specs directory not found in {ws_path}",
                    error_code="NOT_FOUND",
                    error_type="planning",
                    data={"spec_id": spec_id, "workspace": str(ws_path)},
                    remediation="Ensure specs/ directory exists in workspace",
                ))

            spec_file = find_spec_file(spec_id, specs_dir)
            if not spec_file:
                return asdict(error_response(
                    f"Specification '{spec_id}' not found",
                    error_code="NOT_FOUND",
                    error_type="planning",
                    remediation="Verify the spec ID exists using spec-list",
                ))

            spec_data = load_spec(spec_file)
            if not spec_data:
                return asdict(error_response(
                    f"Failed to load spec '{spec_id}'",
                    error_code="LOAD_ERROR",
                    error_type="planning",
                    data={"spec_id": spec_id, "spec_file": str(spec_file)},
                ))

            # Use list_phases from core
            phases = list_phases(spec_data)
            completed_count = sum(1 for p in phases if p.get("status") == "completed")

            duration_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer(f"planning.{tool_name}.duration_ms", duration_ms)
            _metrics.counter(f"planning.{tool_name}", labels={"status": "success"})

            return asdict(success_response(
                spec_id=spec_id,
                phases=phases,
                total_phases=len(phases),
                completed_phases=completed_count,
                duration_ms=round(duration_ms, 2),
            ))

        except Exception as e:
            logger.exception(f"Unexpected error in {tool_name}")
            _metrics.counter(f"planning.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                sanitize_error_message(e, context="planning"),
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="phase-check-complete",
    )
    def phase_check_complete(
        spec_id: str,
        phase_id: Optional[str] = None,
        task_id: Optional[str] = None,
        path: Optional[str] = None,
    ) -> dict:
        """
        Verify completion readiness for a phase or spec.

        Checks whether all tasks in a phase or the entire spec are completed
        and ready for sign-off.

        WHEN TO USE:
        - Verifying phase completion before moving to next phase
        - Checking if a spec is ready for final review
        - Validating that all tasks in a scope are done
        - Pre-merge verification of spec completion

        Args:
            spec_id: Specification ID to check
            phase_id: Optional phase ID to limit scope (mutually exclusive with task_id)
            task_id: Optional task ID to limit scope (mutually exclusive with phase_id)
            path: Project root path (default: current directory)

        Returns:
            JSON object with completion status:
            - is_complete: Boolean indicating if scope is fully complete
            - scope: The scope checked (spec, phase, or task)
            - total_tasks: Total tasks in scope
            - completed_tasks: Number of completed tasks
            - pending_tasks: Array of pending task IDs (if any)
            - blocked_tasks: Array of blocked task IDs (if any)
        """
        tool_name = "phase_check_complete"
        start_time = time.perf_counter()

        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            # Validate mutual exclusivity
            if phase_id and task_id:
                return asdict(error_response(
                    "phase_id and task_id are mutually exclusive",
                    error_code="INVALID_PARAMS",
                    error_type="validation",
                    remediation="Provide either phase_id or task_id, not both",
                ))

            # Resolve workspace path
            ws_path = Path(path) if path else Path.cwd()

            # Determine scope for logging
            scope = "spec"
            scope_id = spec_id
            if phase_id:
                scope = "phase"
                scope_id = phase_id
            elif task_id:
                scope = "task"
                scope_id = task_id

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="phase-check-complete",
                action="check_complete",
                spec_id=spec_id,
                scope=scope,
                scope_id=scope_id,
            )

            # Find and load spec
            specs_dir = find_specs_directory(ws_path)
            if not specs_dir:
                return asdict(error_response(
                    f"Specs directory not found in {ws_path}",
                    error_code="NOT_FOUND",
                    error_type="planning",
                    data={"spec_id": spec_id, "workspace": str(ws_path)},
                    remediation="Ensure specs/ directory exists in workspace",
                ))

            spec_file = find_spec_file(spec_id, specs_dir)
            if not spec_file:
                return asdict(error_response(
                    f"Specification '{spec_id}' not found",
                    error_code="NOT_FOUND",
                    error_type="planning",
                    remediation="Verify the spec ID exists using spec-list",
                ))

            spec_data = load_spec(spec_file)
            if not spec_data:
                return asdict(error_response(
                    f"Failed to load spec '{spec_id}'",
                    error_code="LOAD_ERROR",
                    error_type="planning",
                    data={"spec_id": spec_id, "spec_file": str(spec_file)},
                ))

            # Get progress for scope using get_progress_summary
            root_id = phase_id or task_id or "spec-root"
            progress = get_progress_summary(spec_data, node_id=root_id)

            total = progress.get("total_tasks", 0)
            completed = progress.get("completed_tasks", 0)
            is_complete = total > 0 and total == completed

            # Find pending and blocked tasks
            hierarchy = spec_data.get("hierarchy", {})
            pending_tasks: List[str] = []
            blocked_tasks: List[str] = []

            def collect_task_status(node_id: str) -> None:
                node = hierarchy.get(node_id, {})
                node_type = node.get("type", "")
                if node_type in ("task", "subtask"):
                    status = node.get("status", "")
                    if status == "pending":
                        pending_tasks.append(node_id)
                    elif status == "blocked":
                        blocked_tasks.append(node_id)
                for child_id in node.get("children", []):
                    collect_task_status(child_id)

            collect_task_status(root_id)

            duration_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer(f"planning.{tool_name}.duration_ms", duration_ms)
            _metrics.counter(f"planning.{tool_name}", labels={"status": "success", "scope": scope})

            # Craft appropriate message
            if is_complete:
                message = f"{scope.title()} is complete ({completed}/{total} tasks)"
            else:
                remaining = total - completed
                message = f"{scope.title()} incomplete: {remaining} tasks remaining"

            result_data: Dict[str, Any] = {
                "spec_id": spec_id,
                "scope": scope,
                "is_complete": is_complete,
                "total_tasks": total,
                "completed_tasks": completed,
                "pending_tasks": pending_tasks,
                "blocked_tasks": blocked_tasks,
                "duration_ms": round(duration_ms, 2),
            }

            if phase_id:
                result_data["phase_id"] = phase_id
            elif task_id:
                result_data["task_id"] = task_id

            return asdict(success_response(
                message=message,
                **result_data,
            ))

        except Exception as e:
            logger.exception(f"Unexpected error in {tool_name}")
            _metrics.counter(f"planning.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                sanitize_error_message(e, context="planning"),
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="phase-report-time",
    )
    def phase_report_time(
        spec_id: str,
        phase_id: str,
        path: Optional[str] = None,
    ) -> dict:
        """
        Summarize time tracking metrics for a phase.

        Aggregates estimated and actual hours for all tasks in a phase.

        WHEN TO USE:
        - Reviewing time spent on a phase
        - Comparing estimated vs actual hours
        - Planning future phases based on historical data
        - Generating time reports for stakeholders

        Args:
            spec_id: Specification ID containing the phase
            phase_id: Phase ID to report time for
            path: Project root path (default: current directory)

        Returns:
            JSON object with time metrics:
            - estimated_hours: Total estimated hours for phase
            - actual_hours: Total actual hours spent
            - variance_hours: Difference (actual - estimated)
            - variance_percent: Percentage variance
            - task_count: Number of tasks in phase
            - completed_count: Number of completed tasks
        """
        tool_name = "phase_report_time"
        start_time = time.perf_counter()

        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            if not phase_id:
                return asdict(error_response(
                    "phase_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a phase_id parameter",
                ))

            # Resolve workspace path
            ws_path = Path(path) if path else Path.cwd()

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="phase-report-time",
                action="report_phase_time",
                spec_id=spec_id,
                phase_id=phase_id,
            )

            # Find and load spec
            specs_dir = find_specs_directory(ws_path)
            if not specs_dir:
                return asdict(error_response(
                    f"Specs directory not found in {ws_path}",
                    error_code="NOT_FOUND",
                    error_type="planning",
                    data={"spec_id": spec_id, "workspace": str(ws_path)},
                ))

            spec_file = find_spec_file(spec_id, specs_dir)
            if not spec_file:
                return asdict(error_response(
                    f"Specification '{spec_id}' not found",
                    error_code="NOT_FOUND",
                    error_type="planning",
                ))

            spec_data = load_spec(spec_file)
            if not spec_data:
                return asdict(error_response(
                    f"Failed to load spec '{spec_id}'",
                    error_code="LOAD_ERROR",
                    error_type="planning",
                ))

            # Get phase data
            hierarchy = spec_data.get("hierarchy", {})
            phase = hierarchy.get(phase_id)
            if not phase:
                return asdict(error_response(
                    f"Phase '{phase_id}' not found in spec '{spec_id}'",
                    error_code="NOT_FOUND",
                    error_type="planning",
                ))

            # Calculate time metrics by traversing phase tasks
            estimated_hours = 0.0
            actual_hours = 0.0
            task_count = 0
            completed_count = 0

            def collect_task_times(node_id: str) -> None:
                nonlocal estimated_hours, actual_hours, task_count, completed_count
                node = hierarchy.get(node_id, {})
                node_type = node.get("type", "")
                if node_type in ("task", "subtask"):
                    task_count += 1
                    metadata = node.get("metadata", {})
                    estimated_hours += float(metadata.get("estimated_hours", 0))
                    actual_hours += float(metadata.get("actual_hours", 0))
                    if node.get("status") == "completed":
                        completed_count += 1
                for child_id in node.get("children", []):
                    collect_task_times(child_id)

            collect_task_times(phase_id)

            variance = actual_hours - estimated_hours
            variance_pct = (variance / estimated_hours * 100) if estimated_hours > 0 else 0

            duration_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer(f"planning.{tool_name}.duration_ms", duration_ms)
            _metrics.counter(f"planning.{tool_name}", labels={"status": "success"})

            return asdict(success_response(
                spec_id=spec_id,
                phase_id=phase_id,
                phase_title=phase.get("title", ""),
                estimated_hours=round(estimated_hours, 2),
                actual_hours=round(actual_hours, 2),
                variance_hours=round(variance, 2),
                variance_percent=round(variance_pct, 1),
                task_count=task_count,
                completed_count=completed_count,
                duration_ms=round(duration_ms, 2),
            ))

        except Exception as e:
            logger.exception(f"Unexpected error in {tool_name}")
            _metrics.counter(f"planning.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                sanitize_error_message(e, context="planning"),
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="spec-reconcile-state",
    )
    def spec_reconcile_state(
        spec_id: str,
        dry_run: bool = True,
        path: Optional[str] = None,
    ) -> dict:
        """
        Compare filesystem state against spec state to detect drift.

        Detects file changes that are not reflected in the specification,
        helping identify when implementations have drifted from the planned work.

        WHEN TO USE:
        - Before resuming work on a spec after a break
        - After manual file edits outside the SDD workflow
        - Verifying spec accurately reflects current codebase state
        - Detecting untracked changes before marking tasks complete

        Args:
            spec_id: Specification ID to reconcile
            dry_run: If True (default), only report drift without making changes
            path: Project root path (default: current directory)

        Returns:
            JSON object with reconciliation results:
            - has_drift: Boolean indicating if drift was detected
            - modified_files: Files changed since spec last updated
            - new_files: Files created but not in spec
            - missing_files: Files in spec but not on filesystem
            - recommendations: Suggested actions to resolve drift
        """
        # State reconciliation requires filesystem diff analysis and git integration.
        # This functionality is not available as a direct core API.
        # Use the sdd-toolkit:sdd-fidelity-review skill for drift detection.
        return asdict(
            error_response(
                "State reconciliation requires filesystem diff analysis and git integration. "
                "Use the sdd-toolkit:sdd-fidelity-review skill for drift detection.",
                error_code="NOT_IMPLEMENTED",
                error_type="unavailable",
                data={
                    "spec_id": spec_id,
                    "dry_run": dry_run,
                    "alternative": "sdd-toolkit:sdd-fidelity-review skill",
                    "feature_status": "requires_git_integration",
                },
                remediation="Use the sdd-toolkit:sdd-fidelity-review skill which provides "
                "comprehensive drift detection with AI-powered analysis.",
            )
        )

    @canonical_tool(
        mcp,
        canonical_name="plan-report-time",
    )
    def plan_report_time(
        spec_id: str,
        path: Optional[str] = None,
    ) -> dict:
        """
        Generate a comprehensive time tracking summary for a spec.

        Aggregates time metrics across all phases and tasks in a specification.

        WHEN TO USE:
        - Generating project status reports
        - Reviewing overall time spent vs estimated
        - Planning future sprints based on velocity
        - Identifying phases that took longer than expected

        Args:
            spec_id: Specification ID to report time for
            path: Project root path (default: current directory)

        Returns:
            JSON object with time metrics:
            - total_estimated_hours: Sum of all estimated hours
            - total_actual_hours: Sum of all actual hours
            - total_variance_hours: Overall variance (actual - estimated)
            - total_variance_percent: Overall percentage variance
            - phases: Array of per-phase time summaries
            - completion_rate: Percentage of tasks completed
        """
        tool_name = "plan_report_time"
        start_time = time.perf_counter()

        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            # Resolve workspace path
            ws_path = Path(path) if path else Path.cwd()

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="plan-report-time",
                action="report_time",
                spec_id=spec_id,
            )

            # Find and load spec
            specs_dir = find_specs_directory(ws_path)
            if not specs_dir:
                return asdict(error_response(
                    f"Specs directory not found in {ws_path}",
                    error_code="NOT_FOUND",
                    error_type="planning",
                ))

            spec_file = find_spec_file(spec_id, specs_dir)
            if not spec_file:
                return asdict(error_response(
                    f"Specification '{spec_id}' not found",
                    error_code="NOT_FOUND",
                    error_type="planning",
                ))

            spec_data = load_spec(spec_file)
            if not spec_data:
                return asdict(error_response(
                    f"Failed to load spec '{spec_id}'",
                    error_code="LOAD_ERROR",
                    error_type="planning",
                ))

            hierarchy = spec_data.get("hierarchy", {})

            # Calculate time metrics across all phases
            total_estimated = 0.0
            total_actual = 0.0
            total_tasks = 0
            completed_tasks = 0
            phase_summaries: List[Dict[str, Any]] = []

            # Get phases using list_phases
            phases = list_phases(spec_data)

            for phase_info in phases:
                phase_id = phase_info.get("id", "")
                phase = hierarchy.get(phase_id, {})

                # Calculate phase time metrics
                phase_estimated = 0.0
                phase_actual = 0.0
                phase_task_count = 0
                phase_completed = 0

                def collect_phase_times(node_id: str) -> None:
                    nonlocal phase_estimated, phase_actual, phase_task_count, phase_completed
                    node = hierarchy.get(node_id, {})
                    node_type = node.get("type", "")
                    if node_type in ("task", "subtask"):
                        phase_task_count += 1
                        metadata = node.get("metadata", {})
                        phase_estimated += float(metadata.get("estimated_hours", 0))
                        phase_actual += float(metadata.get("actual_hours", 0))
                        if node.get("status") == "completed":
                            phase_completed += 1
                    for child_id in node.get("children", []):
                        collect_phase_times(child_id)

                collect_phase_times(phase_id)

                phase_summaries.append({
                    "phase_id": phase_id,
                    "title": phase.get("title", ""),
                    "estimated_hours": round(phase_estimated, 2),
                    "actual_hours": round(phase_actual, 2),
                    "task_count": phase_task_count,
                    "completed_count": phase_completed,
                })

                total_estimated += phase_estimated
                total_actual += phase_actual
                total_tasks += phase_task_count
                completed_tasks += phase_completed

            variance = total_actual - total_estimated
            variance_pct = (variance / total_estimated * 100) if total_estimated > 0 else 0
            completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

            duration_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer(f"planning.{tool_name}.duration_ms", duration_ms)
            _metrics.counter(f"planning.{tool_name}", labels={"status": "success"})

            return asdict(success_response(
                spec_id=spec_id,
                spec_title=spec_data.get("metadata", {}).get("title", ""),
                total_estimated_hours=round(total_estimated, 2),
                total_actual_hours=round(total_actual, 2),
                total_variance_hours=round(variance, 2),
                total_variance_percent=round(variance_pct, 1),
                phases=phase_summaries,
                total_tasks=total_tasks,
                completed_tasks=completed_tasks,
                completion_rate=round(completion_rate, 1),
                duration_ms=round(duration_ms, 2),
            ))

        except Exception as e:
            logger.exception(f"Unexpected error in {tool_name}")
            _metrics.counter(f"planning.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                sanitize_error_message(e, context="planning"),
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="spec-audit",
    )
    def spec_audit(
        spec_id: str,
        path: Optional[str] = None,
    ) -> dict:
        """
        Run comprehensive quality audits on a specification.

        Performs higher-level quality checks beyond basic validation,
        including best practice adherence, completeness, and consistency checks.

        WHEN TO USE:
        - Before marking a spec as ready for review
        - Identifying quality issues in spec structure
        - Ensuring specs follow best practices
        - Pre-merge quality gates

        Args:
            spec_id: Specification ID to audit
            path: Project root path (default: current directory)

        Returns:
            JSON object with audit results:
            - passed: Boolean indicating if audit passed
            - score: Overall quality score (0-100)
            - findings: Array of audit findings with severity
            - recommendations: Suggested improvements
            - categories: Breakdown by audit category
        """
        # Comprehensive spec auditing requires complex quality checks
        # including best practice analysis and AI-powered recommendations.
        # Use the sdd-toolkit:sdd-plan-review skill for comprehensive audits.
        return asdict(
            error_response(
                "Comprehensive spec auditing requires complex quality checks "
                "and AI-powered analysis. Use the sdd-toolkit:sdd-plan-review skill.",
                error_code="NOT_IMPLEMENTED",
                error_type="unavailable",
                data={
                    "spec_id": spec_id,
                    "alternative": "sdd-toolkit:sdd-plan-review skill",
                    "feature_status": "requires_ai_analysis",
                },
                remediation="Use the sdd-toolkit:sdd-plan-review skill which provides "
                "multi-model AI consultation and comprehensive quality analysis. "
                "For basic validation, use the spec-validate MCP tool.",
            )
        )
