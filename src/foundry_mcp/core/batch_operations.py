"""
Batch operations for parallel task execution in SDD workflows.

Provides functions to identify independent tasks that can run in parallel
and manage batch state transitions.

Security Note:
    This module validates file paths to ensure they are within the project root.
    See docs/mcp_best_practices/08-security-trust-boundaries.md for guidance.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from foundry_mcp.core.spec import load_spec, find_specs_directory
from foundry_mcp.core.task import is_unblocked

# Constants for batch operations
DEFAULT_MAX_TASKS = 3
"""Default maximum number of tasks to return in a batch."""

MAX_RETRY_COUNT = 3
"""Maximum retry count before excluding a task from batch selection."""


def _get_active_phases(spec_data: Dict[str, Any]) -> List[str]:
    """
    Get phases that are eligible for task selection.

    Returns phases in priority order: in_progress first, then pending.
    Phases with status 'completed' or 'blocked' are excluded.

    Args:
        spec_data: Loaded spec data dictionary

    Returns:
        List of phase IDs in priority order
    """
    hierarchy = spec_data.get("hierarchy", {})
    spec_root = hierarchy.get("spec-root", {})
    phase_order = spec_root.get("children", [])

    active_phases: List[str] = []

    # First pass: in_progress phases (highest priority)
    for phase_id in phase_order:
        phase = hierarchy.get(phase_id, {})
        if phase.get("type") == "phase" and phase.get("status") == "in_progress":
            active_phases.append(phase_id)

    # Second pass: pending phases
    for phase_id in phase_order:
        phase = hierarchy.get(phase_id, {})
        if phase.get("type") == "phase" and phase.get("status") == "pending":
            active_phases.append(phase_id)

    return active_phases


def _is_path_ancestor(parent_path: str, child_path: str) -> bool:
    """
    Check if parent_path is an ancestor directory of child_path.

    Args:
        parent_path: Potential parent directory path
        child_path: Potential child path

    Returns:
        True if parent_path is an ancestor of child_path
    """
    # Normalize paths
    parent_abs = os.path.abspath(parent_path)
    child_abs = os.path.abspath(child_path)

    # Ensure parent ends with separator for proper prefix matching
    if not parent_abs.endswith(os.sep):
        parent_abs += os.sep

    return child_abs.startswith(parent_abs)


def _paths_conflict(path_a: Optional[str], path_b: Optional[str]) -> bool:
    """
    Check if two file paths conflict for parallel execution.

    Conflicts occur when:
    - Both paths are the same file
    - One path is an ancestor directory of the other

    Args:
        path_a: First file path (may be None)
        path_b: Second file path (may be None)

    Returns:
        True if paths conflict and tasks cannot run in parallel
    """
    if not path_a or not path_b:
        # If either path is missing, handled by barrier logic elsewhere
        return False

    abs_a = os.path.abspath(path_a)
    abs_b = os.path.abspath(path_b)

    # Same file
    if abs_a == abs_b:
        return True

    # Check ancestry in both directions
    if _is_path_ancestor(abs_a, abs_b) or _is_path_ancestor(abs_b, abs_a):
        return True

    return False


def _is_within_project_root(file_path: str, project_root: Optional[Path] = None) -> bool:
    """
    Validate that a file path is within the project root.

    Security measure to prevent path traversal attacks.

    Args:
        file_path: Path to validate
        project_root: Project root directory (auto-detected if None)

    Returns:
        True if path is within project root
    """
    if project_root is None:
        specs_dir = find_specs_directory()
        if specs_dir:
            # Project root is typically parent of specs directory
            project_root = specs_dir.parent
        else:
            # Fall back to current working directory
            project_root = Path.cwd()

    abs_path = os.path.abspath(file_path)
    abs_root = os.path.abspath(project_root)

    # Ensure root ends with separator
    if not abs_root.endswith(os.sep):
        abs_root += os.sep

    return abs_path.startswith(abs_root) or abs_path == abs_root.rstrip(os.sep)


def _get_task_file_path(task_data: Dict[str, Any]) -> Optional[str]:
    """
    Extract file_path from task metadata.

    Args:
        task_data: Task data dictionary

    Returns:
        File path string or None if not set
    """
    return task_data.get("metadata", {}).get("file_path")


def _get_retry_count(task_data: Dict[str, Any]) -> int:
    """
    Get retry count from task metadata.

    Args:
        task_data: Task data dictionary

    Returns:
        Retry count (0 if not set)
    """
    return task_data.get("metadata", {}).get("retry_count", 0)


def _has_direct_dependency(
    hierarchy: Dict[str, Any],
    task_a_id: str,
    task_a_data: Dict[str, Any],
    task_b_id: str,
    task_b_data: Dict[str, Any],
) -> bool:
    """
    Check if two tasks have a direct dependency relationship.

    Args:
        hierarchy: Spec hierarchy dictionary
        task_a_id: First task ID
        task_a_data: First task data
        task_b_id: Second task ID
        task_b_data: Second task data

    Returns:
        True if one task blocks or is blocked by the other
    """
    a_deps = task_a_data.get("dependencies", {})
    b_deps = task_b_data.get("dependencies", {})

    # Check if A blocks B or B blocks A
    if task_b_id in a_deps.get("blocks", []):
        return True
    if task_a_id in b_deps.get("blocks", []):
        return True

    # Check blocked_by relationships
    if task_b_id in a_deps.get("blocked_by", []):
        return True
    if task_a_id in b_deps.get("blocked_by", []):
        return True

    return False


def _is_in_active_phase(
    spec_data: Dict[str, Any],
    task_id: str,
    task_data: Dict[str, Any],
    active_phases: List[str],
) -> bool:
    """
    Check if task belongs to one of the active phases.

    Args:
        spec_data: Loaded spec data
        task_id: Task identifier
        task_data: Task data dictionary
        active_phases: List of active phase IDs

    Returns:
        True if task is in an active phase
    """
    hierarchy = spec_data.get("hierarchy", {})

    # Walk up parent chain to find phase
    current = task_data
    while current:
        parent_id = current.get("parent")
        if not parent_id:
            return False

        if parent_id in active_phases:
            return True

        parent = hierarchy.get(parent_id)
        if not parent:
            return False

        if parent.get("type") == "phase":
            # Found phase but it's not in active list
            return parent_id in active_phases

        current = parent

    return False


def get_independent_tasks(
    spec_id: str,
    max_tasks: int = DEFAULT_MAX_TASKS,
    specs_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> Tuple[List[Tuple[str, Dict[str, Any]]], Optional[str]]:
    """
    Find multiple independent tasks that can be executed in parallel.

    Independent tasks have:
    - No dependency relationships between them (blocks/blocked_by)
    - Different file paths (or no path conflicts via ancestry)
    - No tasks without file_path (those are EXCLUSIVE BARRIERS)

    Tasks are filtered to:
    - Status = pending (not failed, in_progress, or completed)
    - Not blocked by incomplete dependencies
    - Not exceeding retry threshold
    - Within active phases only
    - Leaf tasks preferred (no children)

    Security:
    - All file paths are validated to be within project root

    Args:
        spec_id: Specification identifier
        max_tasks: Maximum number of tasks to return (default 3)
        specs_dir: Optional specs directory path
        project_root: Optional project root for path validation

    Returns:
        Tuple of:
        - List of (task_id, task_data) tuples for independent tasks
        - Error message string if operation failed, None on success

    Note:
        Independence is file-based only. Tasks may have logical coupling
        that this function cannot detect. The caller should be aware of
        this limitation.
    """
    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if not spec_data:
        return [], f"Spec '{spec_id}' not found"

    hierarchy = spec_data.get("hierarchy", {})
    if not hierarchy:
        return [], f"Spec '{spec_id}' has no hierarchy"

    # Get active phases
    active_phases = _get_active_phases(spec_data)
    if not active_phases:
        # No error - spec may be complete or all phases blocked
        # Caller should check spec_complete flag
        return [], None

    # Collect candidate tasks
    candidates: List[Tuple[str, Dict[str, Any]]] = []

    for task_id, task_data in hierarchy.items():
        # Must be a task type (not phase, spec, etc.)
        if task_data.get("type") not in ("task", "subtask", "verify"):
            continue

        # Must be pending status
        if task_data.get("status") != "pending":
            continue

        # Skip if retry count exceeded
        if _get_retry_count(task_data) >= MAX_RETRY_COUNT:
            continue

        # Must be unblocked
        if not is_unblocked(spec_data, task_id, task_data):
            continue

        # Must be in an active phase
        if not _is_in_active_phase(spec_data, task_id, task_data, active_phases):
            continue

        # Prefer leaf tasks (no children)
        children = task_data.get("children", [])
        if children:
            # Skip parent tasks - their children should be selected instead
            continue

        # Validate file path if present
        file_path = _get_task_file_path(task_data)
        if file_path and not _is_within_project_root(file_path, project_root):
            # Skip tasks with invalid paths (security measure)
            continue

        candidates.append((task_id, task_data))

    if not candidates:
        return [], None  # No error, just no candidates

    # Sort candidates by task_id for deterministic ordering
    candidates.sort(key=lambda x: x[0])

    # Greedy selection of independent tasks
    selected: List[Tuple[str, Dict[str, Any]]] = []

    for task_id, task_data in candidates:
        if len(selected) >= max_tasks:
            break

        file_path = _get_task_file_path(task_data)

        # CRITICAL: Tasks without file_path are EXCLUSIVE BARRIERS
        # They cannot run in parallel with anything
        if not file_path:
            if not selected:
                # If nothing selected yet, this barrier task can be the only one
                selected.append((task_id, task_data))
            # Either way, stop selecting more tasks
            break

        # Check independence against all already-selected tasks
        is_independent = True

        for sel_id, sel_data in selected:
            sel_path = _get_task_file_path(sel_data)

            # Check for direct dependency
            if _has_direct_dependency(hierarchy, task_id, task_data, sel_id, sel_data):
                is_independent = False
                break

            # Check for file path conflict
            if _paths_conflict(file_path, sel_path):
                is_independent = False
                break

        if is_independent:
            selected.append((task_id, task_data))

    return selected, None


# Token budget constants
DEFAULT_TOKEN_BUDGET = 50000
"""Default token budget for batch context preparation."""

TOKEN_SAFETY_MARGIN = 0.15
"""15% safety margin for token estimation."""

CHARS_PER_TOKEN = 3.0
"""Conservative character-to-token ratio."""

STALE_TASK_THRESHOLD_HOURS = 1.0
"""Hours before an in_progress task is considered stale."""

# Autonomous mode guardrail constants
MAX_CONSECUTIVE_ERRORS = 3
"""Maximum consecutive errors before autonomous mode pauses."""

CONTEXT_LIMIT_PERCENTAGE = 85.0
"""Context usage percentage that triggers autonomous mode pause."""


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count from text using conservative heuristic.

    Uses char_count / 3.0 with 15% safety margin.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    base_estimate = len(text) / CHARS_PER_TOKEN
    return int(base_estimate * (1 + TOKEN_SAFETY_MARGIN))


def _get_stale_in_progress_tasks(
    spec_data: Dict[str, Any],
    threshold_hours: float = STALE_TASK_THRESHOLD_HOURS,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Find in_progress tasks that have been stale for too long.

    Args:
        spec_data: Loaded spec data
        threshold_hours: Hours before a task is considered stale

    Returns:
        List of (task_id, task_data) tuples for stale tasks
    """
    from datetime import datetime, timezone

    hierarchy = spec_data.get("hierarchy", {})
    stale_tasks: List[Tuple[str, Dict[str, Any]]] = []
    now = datetime.now(timezone.utc)

    for task_id, task_data in hierarchy.items():
        if task_data.get("type") not in ("task", "subtask", "verify"):
            continue
        if task_data.get("status") != "in_progress":
            continue

        # Check started_at timestamp in metadata
        started_at_str = task_data.get("metadata", {}).get("started_at")
        if not started_at_str:
            # No started_at means we can't determine staleness, assume stale
            stale_tasks.append((task_id, task_data))
            continue

        try:
            # Parse ISO format timestamp
            started_at = datetime.fromisoformat(started_at_str.replace("Z", "+00:00"))
            elapsed_hours = (now - started_at).total_seconds() / 3600
            if elapsed_hours >= threshold_hours:
                stale_tasks.append((task_id, task_data))
        except (ValueError, TypeError):
            # Invalid timestamp, assume stale
            stale_tasks.append((task_id, task_data))

    return stale_tasks


def _build_dependency_graph(
    spec_data: Dict[str, Any],
    task_ids: List[str],
) -> Dict[str, Any]:
    """
    Build a dependency graph for the given tasks.

    Includes immediate upstream dependencies for context.

    Args:
        spec_data: Loaded spec data
        task_ids: List of task IDs to include

    Returns:
        Dependency graph with nodes and edges
    """
    hierarchy = spec_data.get("hierarchy", {})
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, str]] = []

    # Add target tasks
    for task_id in task_ids:
        task_data = hierarchy.get(task_id, {})
        nodes[task_id] = {
            "id": task_id,
            "title": task_data.get("title", ""),
            "status": task_data.get("status", ""),
            "file_path": task_data.get("metadata", {}).get("file_path"),
            "is_target": True,
        }

        # Add upstream dependencies
        deps = task_data.get("dependencies", {})
        for dep_id in deps.get("blocked_by", []):
            dep_data = hierarchy.get(dep_id, {})
            if dep_id not in nodes:
                nodes[dep_id] = {
                    "id": dep_id,
                    "title": dep_data.get("title", ""),
                    "status": dep_data.get("status", ""),
                    "file_path": dep_data.get("metadata", {}).get("file_path"),
                    "is_target": False,
                }
            edges.append({"from": dep_id, "to": task_id, "type": "blocks"})

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
    }


def _check_all_blocked(spec_data: Dict[str, Any]) -> bool:
    """
    Check if all remaining tasks are blocked.

    Args:
        spec_data: Loaded spec data

    Returns:
        True if all pending tasks are blocked
    """
    from foundry_mcp.core.task import is_unblocked

    hierarchy = spec_data.get("hierarchy", {})

    for task_id, task_data in hierarchy.items():
        if task_data.get("type") not in ("task", "subtask", "verify"):
            continue
        if task_data.get("status") != "pending":
            continue
        # If any task is unblocked, not all are blocked
        if is_unblocked(spec_data, task_id, task_data):
            return False

    return True


def _check_autonomous_limits(
    autonomous_session: Optional[Any] = None,
    session_stats: Optional[Any] = None,
    session_limits: Optional[Any] = None,
    spec_data: Optional[Dict[str, Any]] = None,
    max_errors: int = MAX_CONSECUTIVE_ERRORS,
    context_limit_pct: float = CONTEXT_LIMIT_PERCENTAGE,
) -> Optional[str]:
    """
    Check if autonomous mode should pause due to resource limits.

    This helper monitors context usage, error rates, and blocking states
    to determine if autonomous execution should pause for user review.

    Args:
        autonomous_session: AutonomousSession instance (from cli.context)
        session_stats: SessionStats instance with error/consultation counts
        session_limits: SessionLimits instance with max thresholds
        spec_data: Loaded spec data for checking blocked tasks
        max_errors: Maximum consecutive errors before pause (default 3)
        context_limit_pct: Context usage % that triggers pause (default 85.0)

    Returns:
        pause_reason string if limits hit, None if OK to continue:
        - "error": Too many consecutive errors
        - "context": Context/token budget nearing limit
        - "blocked": All remaining tasks are blocked
        - "limit": Session consultation/token limit reached
        - None: OK to continue autonomous execution

    Note:
        Updates autonomous_session.pause_reason in-place when limits are hit.
        The caller should check the return value and act accordingly.
    """
    # Early return if no autonomous session
    if autonomous_session is None:
        return None

    # Check if already paused
    if autonomous_session.pause_reason is not None:
        return autonomous_session.pause_reason

    # Check if autonomous mode is not enabled
    if not autonomous_session.enabled:
        return None

    pause_reason: Optional[str] = None

    # 1. Check error rate (consecutive errors)
    if session_stats is not None:
        errors_encountered = getattr(session_stats, "errors_encountered", 0)
        if errors_encountered >= max_errors:
            pause_reason = "error"

    # 2. Check context/token usage
    if pause_reason is None and session_stats is not None and session_limits is not None:
        max_tokens = getattr(session_limits, "max_context_tokens", 0)
        used_tokens = getattr(session_stats, "estimated_tokens_used", 0)

        if max_tokens > 0:
            usage_pct = (used_tokens / max_tokens) * 100
            if usage_pct >= context_limit_pct:
                pause_reason = "context"

        # Also check consultation limit
        max_consultations = getattr(session_limits, "max_consultations", 0)
        consultation_count = getattr(session_stats, "consultation_count", 0)

        if max_consultations > 0 and consultation_count >= max_consultations:
            pause_reason = "limit"

    # 3. Check if all remaining tasks are blocked
    if pause_reason is None and spec_data is not None:
        if _check_all_blocked(spec_data):
            pause_reason = "blocked"

    # Update pause_reason on the session if limits hit
    if pause_reason is not None:
        autonomous_session.pause_reason = pause_reason

    return pause_reason


def prepare_batch_context(
    spec_id: str,
    max_tasks: int = DEFAULT_MAX_TASKS,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    specs_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Prepare context for batch parallel task execution.

    Finds independent tasks and prepares minimal context for each,
    staying within token budget.

    Args:
        spec_id: Specification identifier
        max_tasks: Maximum tasks to include
        token_budget: Maximum tokens for combined context
        specs_dir: Optional specs directory path
        project_root: Optional project root for path validation

    Returns:
        Tuple of:
        - Batch context dict with tasks, warnings, and metadata
        - Error message string if operation failed, None on success
    """
    from foundry_mcp.core.task import (
        check_dependencies,
        get_parent_context,
        get_phase_context,
    )

    # Get independent tasks
    tasks, error = get_independent_tasks(
        spec_id=spec_id,
        max_tasks=max_tasks,
        specs_dir=specs_dir,
        project_root=project_root,
    )

    if error:
        return {}, error

    # Load spec for additional context
    spec_data = load_spec(spec_id, specs_dir)
    if not spec_data:
        return {}, f"Spec '{spec_id}' not found"

    # Check for spec completion
    hierarchy = spec_data.get("hierarchy", {})
    all_tasks = [
        node for node in hierarchy.values()
        if node.get("type") in ("task", "subtask", "verify")
    ]
    completed_count = sum(1 for t in all_tasks if t.get("status") == "completed")
    pending_count = sum(1 for t in all_tasks if t.get("status") == "pending")
    spec_complete = pending_count == 0 and completed_count > 0

    # Check if all remaining are blocked
    all_blocked = _check_all_blocked(spec_data) if not spec_complete else False

    if not tasks:
        return {
            "tasks": [],
            "task_count": 0,
            "spec_complete": spec_complete,
            "all_blocked": all_blocked,
            "warnings": [],
            "stale_tasks": [],
            "dependency_graph": {"nodes": [], "edges": []},
        }, None

    # Prepare context for each task with token budgeting
    task_contexts: List[Dict[str, Any]] = []
    used_tokens = 0
    seen_files: set = set()  # For deduplication
    warnings: List[str] = []

    task_ids = [t[0] for t in tasks]

    for task_id, task_data in tasks:
        task_context: Dict[str, Any] = {
            "task_id": task_id,
            "title": task_data.get("title", ""),
            "type": task_data.get("type", "task"),
            "status": task_data.get("status", "pending"),
            "metadata": task_data.get("metadata", {}),
        }

        # Add dependencies
        deps = check_dependencies(spec_data, task_id)
        task_context["dependencies"] = deps

        # Add phase context (shared across batch, deduplicated)
        phase_context = get_phase_context(spec_data, task_id)
        if phase_context:
            phase_id = phase_context.get("id")
            if phase_id not in seen_files:
                task_context["phase"] = phase_context
                seen_files.add(phase_id)

        # Add parent context
        parent_context = get_parent_context(spec_data, task_id)
        if parent_context:
            task_context["parent"] = {
                "id": parent_context.get("id"),
                "title": parent_context.get("title"),
                "position_label": parent_context.get("position_label"),
            }

        # Estimate tokens for this task context
        import json
        context_json = json.dumps(task_context)
        context_tokens = _estimate_tokens(context_json)

        if used_tokens + context_tokens > token_budget:
            warnings.append(
                f"Token budget exceeded at task {len(task_contexts) + 1}. "
                f"Returning {len(task_contexts)} tasks."
            )
            break

        used_tokens += context_tokens
        task_contexts.append(task_context)

    # Check for stale in_progress tasks
    stale_tasks = _get_stale_in_progress_tasks(spec_data)
    stale_info = [
        {"task_id": t[0], "title": t[1].get("title", "")}
        for t in stale_tasks
    ]
    if stale_info:
        warnings.append(
            f"Found {len(stale_info)} stale in_progress task(s) (>1hr). "
            "Consider resetting them."
        )

    # Build dependency graph
    dep_graph = _build_dependency_graph(spec_data, task_ids[:len(task_contexts)])

    # Add logical coupling warning
    warnings.append(
        "Note: Tasks are file-independent but may have logical coupling "
        "that cannot be detected automatically."
    )

    return {
        "tasks": task_contexts,
        "task_count": len(task_contexts),
        "spec_complete": spec_complete,
        "all_blocked": all_blocked,
        "warnings": warnings,
        "stale_tasks": stale_info,
        "dependency_graph": dep_graph,
        "token_estimate": used_tokens,
    }, None


def start_batch(
    spec_id: str,
    task_ids: List[str],
    specs_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Atomically start multiple tasks as in_progress.

    Validates all tasks can be started before making any changes.
    Uses atomic file write to prevent corruption on crash.

    Args:
        spec_id: Specification identifier
        task_ids: List of task IDs to start
        specs_dir: Optional specs directory path

    Returns:
        Tuple of:
        - Result dict with started task IDs and any warnings
        - Error message string if operation failed, None on success
    """
    from datetime import datetime, timezone
    from foundry_mcp.core.spec import load_spec, save_spec
    from foundry_mcp.core.task import is_unblocked

    if not task_ids:
        return {}, "No task IDs provided"

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if not spec_data:
        return {}, f"Spec '{spec_id}' not found"

    hierarchy = spec_data.get("hierarchy", {})

    # Phase 1: Validate ALL tasks can be started (no changes yet)
    validation_errors: List[str] = []
    tasks_to_start: List[Tuple[str, Dict[str, Any]]] = []

    for task_id in task_ids:
        task_data = hierarchy.get(task_id)
        if not task_data:
            validation_errors.append(f"Task '{task_id}' not found")
            continue

        task_type = task_data.get("type")
        if task_type not in ("task", "subtask", "verify"):
            validation_errors.append(f"'{task_id}' is not a task type (is {task_type})")
            continue

        status = task_data.get("status")
        if status == "in_progress":
            validation_errors.append(f"Task '{task_id}' is already in_progress")
            continue
        if status == "completed":
            validation_errors.append(f"Task '{task_id}' is already completed")
            continue
        if status == "blocked":
            validation_errors.append(f"Task '{task_id}' is blocked")
            continue

        # Check if unblocked
        if not is_unblocked(spec_data, task_id, task_data):
            validation_errors.append(f"Task '{task_id}' has unresolved dependencies")
            continue

        tasks_to_start.append((task_id, task_data))

    # Re-validate independence between selected tasks
    for i, (task_id_a, task_data_a) in enumerate(tasks_to_start):
        for task_id_b, task_data_b in tasks_to_start[i + 1:]:
            # Check for direct dependency
            if _has_direct_dependency(hierarchy, task_id_a, task_data_a, task_id_b, task_data_b):
                validation_errors.append(
                    f"Tasks '{task_id_a}' and '{task_id_b}' have dependencies between them"
                )
                continue

            # Check for file path conflict
            path_a = _get_task_file_path(task_data_a)
            path_b = _get_task_file_path(task_data_b)
            if _paths_conflict(path_a, path_b):
                validation_errors.append(
                    f"Tasks '{task_id_a}' and '{task_id_b}' target conflicting paths"
                )

    # Fail if any validation errors (all-or-nothing)
    if validation_errors:
        return {
            "started": [],
            "errors": validation_errors,
        }, f"Validation failed: {len(validation_errors)} error(s)"

    # Phase 2: Apply changes atomically
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started_ids: List[str] = []

    for task_id, _ in tasks_to_start:
        task = hierarchy[task_id]
        task["status"] = "in_progress"
        # Track when task was started for stale detection
        metadata = task.get("metadata")
        if metadata is None:
            metadata = {}
            task["metadata"] = metadata
        metadata["started_at"] = now
        started_ids.append(task_id)

    # Save atomically (save_spec uses temp file + rename)
    if not save_spec(spec_id, spec_data, specs_dir):
        return {}, "Failed to save spec file atomically"

    return {
        "started": started_ids,
        "started_count": len(started_ids),
        "started_at": now,
    }, None


def complete_batch(
    spec_id: str,
    completions: List[Dict[str, Any]],
    specs_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Complete multiple tasks with individual completion notes.

    Handles partial success where some tasks complete and others fail.
    Failed tasks get 'failed' status with retry_count incremented.

    Args:
        spec_id: Specification identifier
        completions: List of completion dicts with:
            - task_id: Task ID to complete
            - completion_note: Note describing what was accomplished
            - success: True for completed, False for failed
        specs_dir: Optional specs directory path

    Returns:
        Tuple of:
        - Result dict with per-task results and summary
        - Error message string if entire operation failed, None on success
    """
    from datetime import datetime, timezone
    from foundry_mcp.core.spec import load_spec, save_spec
    from foundry_mcp.core.journal import add_journal_entry
    from foundry_mcp.core.progress import sync_computed_fields, update_parent_status

    if not completions:
        return {}, "No completions provided"

    # Validate completions structure
    for i, completion in enumerate(completions):
        if not isinstance(completion, dict):
            return {}, f"Completion {i} must be a dict"
        if "task_id" not in completion:
            return {}, f"Completion {i} missing required 'task_id'"
        if "success" not in completion:
            return {}, f"Completion {i} missing required 'success' flag"

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if not spec_data:
        return {}, f"Spec '{spec_id}' not found"

    hierarchy = spec_data.get("hierarchy", {})
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # Process each completion
    results: Dict[str, Dict[str, Any]] = {}
    completed_count = 0
    failed_count = 0

    for completion in completions:
        task_id = completion["task_id"]
        success = completion.get("success", True)
        completion_note = completion.get("completion_note", "")

        task_data = hierarchy.get(task_id)
        if not task_data:
            results[task_id] = {
                "status": "error",
                "error": f"Task '{task_id}' not found",
            }
            failed_count += 1
            continue

        task_type = task_data.get("type")
        if task_type not in ("task", "subtask", "verify"):
            results[task_id] = {
                "status": "error",
                "error": f"'{task_id}' is not a task type (is {task_type})",
            }
            failed_count += 1
            continue

        current_status = task_data.get("status")
        if current_status == "completed":
            results[task_id] = {
                "status": "skipped",
                "error": "Task already completed",
            }
            continue

        if success:
            # Mark as completed
            task_data["status"] = "completed"
            metadata = task_data.get("metadata")
            if metadata is None:
                metadata = {}
                task_data["metadata"] = metadata
            metadata["completed_at"] = now

            # Auto-calculate actual_hours if started_at exists and not manually set
            if "started_at" in metadata and "actual_hours" not in metadata:
                started_at = datetime.fromisoformat(metadata["started_at"].replace("Z", "+00:00"))
                completed_at = datetime.fromisoformat(now.replace("Z", "+00:00"))
                metadata["actual_hours"] = round((completed_at - started_at).total_seconds() / 3600, 2)

            # Add journal entry for completion
            add_journal_entry(
                spec_data=spec_data,
                title=f"Completed: {task_data.get('title', task_id)}",
                content=completion_note or "Task completed",
                entry_type="status_change",
                task_id=task_id,
            )

            # Update parent status
            update_parent_status(spec_data, task_id)

            results[task_id] = {
                "status": "completed",
                "completed_at": now,
            }
            completed_count += 1
        else:
            # Mark as failed and increment retry count
            task_data["status"] = "failed"
            metadata = task_data.get("metadata")
            if metadata is None:
                metadata = {}
                task_data["metadata"] = metadata

            retry_count = metadata.get("retry_count", 0)
            metadata["retry_count"] = retry_count + 1
            metadata["failed_at"] = now
            metadata["failure_reason"] = completion_note or "Task failed"

            # Add journal entry for failure
            add_journal_entry(
                spec_data=spec_data,
                title=f"Failed: {task_data.get('title', task_id)}",
                content=completion_note or "Task failed",
                entry_type="blocker",
                task_id=task_id,
                metadata={"retry_count": metadata["retry_count"]},
            )

            results[task_id] = {
                "status": "failed",
                "retry_count": metadata["retry_count"],
                "failed_at": now,
            }
            failed_count += 1

    # Recalculate progress
    sync_computed_fields(spec_data)

    # Save atomically
    if not save_spec(spec_id, spec_data, specs_dir):
        return {}, "Failed to save spec file atomically"

    return {
        "results": results,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "total_processed": len(completions),
    }, None


def reset_batch(
    spec_id: str,
    task_ids: Optional[List[str]] = None,
    threshold_hours: float = STALE_TASK_THRESHOLD_HOURS,
    specs_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Reset stale or specified in_progress tasks back to pending.

    If task_ids is provided, resets those specific tasks.
    If task_ids is not provided, finds and resets stale in_progress tasks
    that exceed the threshold_hours.

    Args:
        spec_id: Specification identifier
        task_ids: Optional list of specific task IDs to reset
        threshold_hours: Hours before a task is considered stale (default 1.0)
        specs_dir: Optional specs directory path

    Returns:
        Tuple of:
        - Result dict with reset task IDs and count
        - Error message string if operation failed, None on success
    """
    from foundry_mcp.core.spec import load_spec, save_spec

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if not spec_data:
        return {}, f"Spec '{spec_id}' not found"

    hierarchy = spec_data.get("hierarchy", {})

    # Determine which tasks to reset
    if task_ids:
        # Reset specific tasks
        tasks_to_reset: List[Tuple[str, Dict[str, Any]]] = []
        validation_errors: List[str] = []

        for task_id in task_ids:
            task_data = hierarchy.get(task_id)
            if not task_data:
                validation_errors.append(f"Task '{task_id}' not found")
                continue

            task_type = task_data.get("type")
            if task_type not in ("task", "subtask", "verify"):
                validation_errors.append(f"'{task_id}' is not a task type (is {task_type})")
                continue

            status = task_data.get("status")
            if status != "in_progress":
                validation_errors.append(
                    f"Task '{task_id}' is not in_progress (status: {status})"
                )
                continue

            tasks_to_reset.append((task_id, task_data))

        if validation_errors and not tasks_to_reset:
            return {
                "reset": [],
                "errors": validation_errors,
            }, f"No valid tasks to reset: {len(validation_errors)} error(s)"
    else:
        # Find stale tasks automatically
        tasks_to_reset = _get_stale_in_progress_tasks(spec_data, threshold_hours)
        validation_errors = []

    if not tasks_to_reset:
        return {
            "reset": [],
            "reset_count": 0,
            "message": "No stale in_progress tasks found",
        }, None

    # Reset each task
    reset_ids: List[str] = []

    for task_id, task_data in tasks_to_reset:
        task_data["status"] = "pending"
        metadata = task_data.get("metadata")
        if metadata:
            # Clear started_at timestamp
            metadata.pop("started_at", None)
        reset_ids.append(task_id)

    # Save atomically
    if not save_spec(spec_id, spec_data, specs_dir):
        return {}, "Failed to save spec file atomically"

    result: Dict[str, Any] = {
        "reset": reset_ids,
        "reset_count": len(reset_ids),
    }

    if validation_errors:
        result["errors"] = validation_errors

    return result, None


__all__ = [
    "get_independent_tasks",
    "prepare_batch_context",
    "start_batch",
    "complete_batch",
    "reset_batch",
    "DEFAULT_MAX_TASKS",
    "MAX_RETRY_COUNT",
    "DEFAULT_TOKEN_BUDGET",
    "STALE_TASK_THRESHOLD_HOURS",
    # Autonomous mode guardrails
    "MAX_CONSECUTIVE_ERRORS",
    "CONTEXT_LIMIT_PERCENTAGE",
    "_check_autonomous_limits",
    # Private helpers exposed for testing
    "_get_active_phases",
    "_paths_conflict",
    "_is_within_project_root",
    "_has_direct_dependency",
    "_estimate_tokens",
    "_get_stale_in_progress_tasks",
    "_check_all_blocked",
]
