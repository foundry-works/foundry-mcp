"""
Non-LLM review logic for SDD specifications.

Provides structural review capabilities that don't require AI/LLM integration.
For LLM-powered reviews, use the sdd-toolkit:sdd-plan-review skill.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from foundry_mcp.core.spec import load_spec, find_specs_directory
from foundry_mcp.core.validation import (
    ValidationResult,
    validate_spec,
    calculate_stats,
    SpecStats,
)
from foundry_mcp.core.progress import get_progress_summary, list_phases


# Review types that don't require LLM
QUICK_REVIEW_TYPES = ["quick"]

# Review types that require LLM
LLM_REQUIRED_TYPES = ["full", "security", "feasibility"]


@dataclass
class ReviewFinding:
    """
    A single review finding from structural analysis.
    """
    code: str  # Finding code (e.g., "EMPTY_PHASE", "MISSING_ESTIMATES")
    message: str  # Human-readable description
    severity: str  # "error", "warning", "info"
    category: str  # "structure", "quality", "completeness", "metadata"
    location: Optional[str] = None  # Node ID or path where issue occurred
    suggestion: Optional[str] = None  # Suggested fix


@dataclass
class QuickReviewResult:
    """
    Result of a quick (non-LLM) structural review.
    """
    spec_id: str
    title: str
    review_type: str = "quick"
    is_valid: bool = True
    findings: List[ReviewFinding] = field(default_factory=list)
    stats: Optional[SpecStats] = None
    progress: Optional[Dict[str, Any]] = None
    phases: Optional[List[Dict[str, Any]]] = None
    summary: str = ""
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0


@dataclass
class ReviewContext:
    """
    Context for review operations.

    Provides spec data, progress, and other context needed for reviews.
    """
    spec_id: str
    spec_data: Dict[str, Any]
    title: str
    progress: Dict[str, Any]
    phases: List[Dict[str, Any]]
    stats: SpecStats
    validation: ValidationResult
    completed_tasks: List[Dict[str, Any]]
    journal_entries: List[Dict[str, Any]]


def get_llm_status() -> Dict[str, Any]:
    """
    Get LLM configuration status.

    Returns:
        Dict with configured, provider, and model info
    """
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


def prepare_review_context(
    spec_id: str,
    specs_dir: Optional[Path] = None,
    include_tasks: bool = True,
    include_journals: bool = True,
    max_journal_entries: int = 10,
) -> Optional[ReviewContext]:
    """
    Prepare context for review operations.

    Gathers spec data, validation results, progress, and other context
    needed for both quick and LLM-powered reviews.

    Args:
        spec_id: Specification ID to review
        specs_dir: Optional specs directory (auto-discovered if not provided)
        include_tasks: Include completed task summaries
        include_journals: Include recent journal entries
        max_journal_entries: Maximum journal entries to include

    Returns:
        ReviewContext with all gathered data, or None if spec not found
    """
    # Discover specs directory if not provided
    if specs_dir is None:
        specs_dir = find_specs_directory()
        if specs_dir is None:
            return None

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None

    title = spec_data.get("title", "Untitled")
    hierarchy = spec_data.get("hierarchy", {})

    # Get validation results
    validation = validate_spec(spec_data)

    # Get stats
    stats = calculate_stats(spec_data)

    # Get progress
    progress = get_progress_summary(spec_data)

    # Get phases
    phases = list_phases(spec_data)

    # Get completed tasks
    completed_tasks = []
    if include_tasks:
        for node_id, node in hierarchy.items():
            if node.get("type") in ("task", "subtask", "verify"):
                if node.get("status") == "completed":
                    completed_tasks.append({
                        "id": node_id,
                        "title": node.get("title", "Untitled"),
                        "type": node.get("type"),
                        "parent": node.get("parent"),
                    })

    # Get journal entries
    journal_entries = []
    if include_journals:
        journal = spec_data.get("journal", [])
        # Get most recent entries
        journal_entries = journal[-max_journal_entries:] if journal else []

    return ReviewContext(
        spec_id=spec_id,
        spec_data=spec_data,
        title=title,
        progress=progress,
        phases=phases,
        stats=stats,
        validation=validation,
        completed_tasks=completed_tasks,
        journal_entries=journal_entries,
    )


def quick_review(
    spec_id: str,
    specs_dir: Optional[Path] = None,
) -> QuickReviewResult:
    """
    Perform a quick structural review of a specification.

    This is a non-LLM review that checks:
    - Spec structure and validation
    - Progress and phase organization
    - Task completeness and estimates
    - Common quality issues

    Args:
        spec_id: Specification ID to review
        specs_dir: Optional specs directory

    Returns:
        QuickReviewResult with findings and stats
    """
    # Get review context
    context = prepare_review_context(
        spec_id=spec_id,
        specs_dir=specs_dir,
        include_tasks=True,
        include_journals=False,
    )

    if context is None:
        return QuickReviewResult(
            spec_id=spec_id,
            title="Unknown",
            is_valid=False,
            summary="Specification not found",
            findings=[
                ReviewFinding(
                    code="SPEC_NOT_FOUND",
                    message=f"Specification '{spec_id}' not found",
                    severity="error",
                    category="structure",
                )
            ],
            error_count=1,
        )

    findings: List[ReviewFinding] = []

    # Convert validation diagnostics to findings
    for diag in context.validation.diagnostics:
        findings.append(
            ReviewFinding(
                code=diag.code,
                message=diag.message,
                severity=diag.severity,
                category=diag.category,
                location=diag.location,
                suggestion=diag.suggested_fix,
            )
        )

    # Check for empty phases (phases with no children in the hierarchy)
    hierarchy = context.spec_data.get("hierarchy", {})
    for node_id, node in hierarchy.items():
        if node.get("type") == "phase":
            children = node.get("children", [])
            if len(children) == 0:
                findings.append(
                    ReviewFinding(
                        code="EMPTY_PHASE",
                        message=f"Phase '{node.get('title', node_id)}' has no tasks",
                        severity="warning",
                        category="structure",
                        location=node_id,
                        suggestion="Add tasks to this phase or remove it",
                    )
                )

    # Check for missing estimates
    tasks_without_estimates = 0
    for node_id, node in hierarchy.items():
        if node.get("type") in ("task", "subtask"):
            metadata = node.get("metadata", {})
            if metadata.get("estimated_hours") is None:
                tasks_without_estimates += 1

    if tasks_without_estimates > 0:
        findings.append(
            ReviewFinding(
                code="MISSING_ESTIMATES",
                message=f"{tasks_without_estimates} task(s) are missing time estimates",
                severity="info",
                category="completeness",
                suggestion="Add estimated_hours to task metadata for better planning",
            )
        )

    # Check for tasks without file paths
    tasks_without_files = 0
    for node_id, node in hierarchy.items():
        if node.get("type") in ("task", "subtask"):
            metadata = node.get("metadata", {})
            if not metadata.get("file_path"):
                tasks_without_files += 1

    if tasks_without_files > 0:
        findings.append(
            ReviewFinding(
                code="MISSING_FILE_PATHS",
                message=f"{tasks_without_files} task(s) are missing file path references",
                severity="info",
                category="completeness",
                suggestion="Add file_path to task metadata for better navigation",
            )
        )

    # Check for blocked tasks
    blocked_count = 0
    for node_id, node in hierarchy.items():
        if node.get("status") == "blocked":
            blocked_count += 1

    if blocked_count > 0:
        findings.append(
            ReviewFinding(
                code="BLOCKED_TASKS",
                message=f"{blocked_count} task(s) are currently blocked",
                severity="warning",
                category="quality",
                suggestion="Resolve blockers to continue progress",
            )
        )

    # Check overall progress
    progress = context.progress
    percentage = progress.get("percentage", 0)

    # Count severities
    error_count = sum(1 for f in findings if f.severity == "error")
    warning_count = sum(1 for f in findings if f.severity == "warning")
    info_count = sum(1 for f in findings if f.severity == "info")

    # Build summary
    summary_parts = [
        f"Reviewed '{context.title}' ({spec_id})",
        f"Progress: {percentage:.0f}% ({progress.get('completed_tasks', 0)}/{progress.get('total_tasks', 0)} tasks)",
    ]

    if error_count > 0:
        summary_parts.append(f"{error_count} error(s)")
    if warning_count > 0:
        summary_parts.append(f"{warning_count} warning(s)")
    if info_count > 0:
        summary_parts.append(f"{info_count} info finding(s)")

    if error_count == 0 and warning_count == 0:
        summary_parts.append("No critical issues found")

    return QuickReviewResult(
        spec_id=spec_id,
        title=context.title,
        review_type="quick",
        is_valid=context.validation.is_valid,
        findings=findings,
        stats=context.stats,
        progress=context.progress,
        phases=context.phases,
        summary=". ".join(summary_parts) + ".",
        error_count=error_count,
        warning_count=warning_count,
        info_count=info_count,
    )


def review_type_requires_llm(review_type: str) -> bool:
    """
    Check if a review type requires LLM.

    Args:
        review_type: Review type to check

    Returns:
        True if LLM is required, False for quick reviews
    """
    return review_type in LLM_REQUIRED_TYPES
