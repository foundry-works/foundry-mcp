"""
Spec-domain error helpers for MCP tool responses.

Provides error helpers for spec modification operations: circular dependencies,
invalid parents, self-references, missing dependencies, position errors,
regex/pattern errors, and backup/rollback/comparison failures.
"""

from typing import Any, Dict, Optional, Sequence

from foundry_mcp.core.responses.builders import error_response
from foundry_mcp.core.responses.types import ErrorCode, ErrorType, ToolResponse


def circular_dependency_error(
    task_id: str,
    target_id: str,
    *,
    cycle_path: Optional[Sequence[str]] = None,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for circular dependency detection.

    Use when a move or dependency operation would create a cycle.

    Args:
        task_id: The task being moved or modified.
        target_id: The target parent or dependency that would create a cycle.
        cycle_path: Optional sequence showing the dependency cycle path.
        remediation: Guidance on how to resolve.
        request_id: Correlation identifier.

    Example:
        >>> circular_dependency_error("task-3", "task-1", cycle_path=["task-1", "task-2", "task-3"])
    """
    data: Dict[str, Any] = {
        "task_id": task_id,
        "target_id": target_id,
    }
    if cycle_path:
        data["cycle_path"] = list(cycle_path)

    return error_response(
        f"Circular dependency detected: {task_id} cannot depend on {target_id}",
        error_code=ErrorCode.CIRCULAR_DEPENDENCY,
        error_type=ErrorType.CONFLICT,
        data=data,
        remediation=remediation or "Remove an existing dependency to break the cycle before adding this one.",
        request_id=request_id,
    )


def invalid_parent_error(
    task_id: str,
    target_parent: str,
    reason: str,
    *,
    valid_parents: Optional[Sequence[str]] = None,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for invalid parent in move operation.

    Use when a task cannot be moved to the specified parent.

    Args:
        task_id: The task being moved.
        target_parent: The invalid target parent.
        reason: Why the parent is invalid (e.g., "is a task, not a phase").
        valid_parents: Optional list of valid parent IDs.
        remediation: Guidance on how to resolve.
        request_id: Correlation identifier.

    Example:
        >>> invalid_parent_error("task-3-1", "task-2-1", "target is a task, not a phase")
    """
    data: Dict[str, Any] = {
        "task_id": task_id,
        "target_parent": target_parent,
        "reason": reason,
    }
    if valid_parents:
        data["valid_parents"] = list(valid_parents)

    return error_response(
        f"Invalid parent '{target_parent}' for task '{task_id}': {reason}",
        error_code=ErrorCode.INVALID_PARENT,
        error_type=ErrorType.VALIDATION,
        data=data,
        remediation=remediation or "Specify a valid phase or parent task as the target.",
        request_id=request_id,
    )


def self_reference_error(
    task_id: str,
    operation: str,
    *,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for self-referencing operations.

    Use when a task references itself in dependencies or move operations.

    Args:
        task_id: The task that references itself.
        operation: The operation attempted (e.g., "add-dependency", "move").
        remediation: Guidance on how to resolve.
        request_id: Correlation identifier.

    Example:
        >>> self_reference_error("task-1-1", "add-dependency")
    """
    return error_response(
        f"Task '{task_id}' cannot reference itself in {operation}",
        error_code=ErrorCode.SELF_REFERENCE,
        error_type=ErrorType.VALIDATION,
        data={"task_id": task_id, "operation": operation},
        remediation=remediation or "Specify a different task ID as the target.",
        request_id=request_id,
    )


def dependency_not_found_error(
    task_id: str,
    dependency_id: str,
    *,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for missing dependency in remove operation.

    Use when trying to remove a dependency that doesn't exist.

    Args:
        task_id: The task being modified.
        dependency_id: The dependency that wasn't found.
        remediation: Guidance on how to resolve.
        request_id: Correlation identifier.

    Example:
        >>> dependency_not_found_error("task-1-1", "task-2-1")
    """
    return error_response(
        f"Dependency '{dependency_id}' not found on task '{task_id}'",
        error_code=ErrorCode.DEPENDENCY_NOT_FOUND,
        error_type=ErrorType.NOT_FOUND,
        data={"task_id": task_id, "dependency_id": dependency_id},
        remediation=remediation or "Check existing dependencies using task info before removing.",
        request_id=request_id,
    )


def invalid_position_error(
    item_id: str,
    position: int,
    max_position: int,
    *,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for invalid position in move/reorder operation.

    Use when the specified position is out of valid range.

    Args:
        item_id: The item being moved (phase or task ID).
        position: The invalid position specified.
        max_position: The maximum valid position.
        remediation: Guidance on how to resolve.
        request_id: Correlation identifier.

    Example:
        >>> invalid_position_error("phase-3", 10, 5)
    """
    return error_response(
        f"Invalid position {position} for '{item_id}': must be 1-{max_position}",
        error_code=ErrorCode.INVALID_POSITION,
        error_type=ErrorType.VALIDATION,
        data={
            "item_id": item_id,
            "position": position,
            "max_position": max_position,
            "valid_range": f"1-{max_position}",
        },
        remediation=remediation or f"Specify a position between 1 and {max_position}.",
        request_id=request_id,
    )


def invalid_regex_error(
    pattern: str,
    error_detail: str,
    *,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for invalid regex pattern.

    Use when a find/replace pattern is not valid regex.

    Args:
        pattern: The invalid regex pattern.
        error_detail: The regex error message.
        remediation: Guidance on how to fix the pattern.
        request_id: Correlation identifier.

    Example:
        >>> invalid_regex_error("[unclosed", "unterminated character set")
    """
    return error_response(
        f"Invalid regex pattern: {error_detail}",
        error_code=ErrorCode.INVALID_REGEX_PATTERN,
        error_type=ErrorType.VALIDATION,
        data={"pattern": pattern, "error_detail": error_detail},
        remediation=remediation or "Check regex syntax. Use raw strings and escape special characters.",
        request_id=request_id,
    )


def pattern_too_broad_error(
    pattern: str,
    match_count: int,
    max_matches: int,
    *,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for overly broad patterns.

    Use when a find/replace pattern matches too many items.

    Args:
        pattern: The pattern that matched too broadly.
        match_count: Number of matches found.
        max_matches: Maximum allowed matches.
        remediation: Guidance on how to narrow the pattern.
        request_id: Correlation identifier.

    Example:
        >>> pattern_too_broad_error(".*", 500, 100)
    """
    return error_response(
        f"Pattern too broad: {match_count} matches exceeds limit of {max_matches}",
        error_code=ErrorCode.PATTERN_TOO_BROAD,
        error_type=ErrorType.VALIDATION,
        data={
            "pattern": pattern,
            "match_count": match_count,
            "max_matches": max_matches,
        },
        remediation=remediation or "Use a more specific pattern or apply to a narrower scope.",
        request_id=request_id,
    )


def no_matches_error(
    pattern: str,
    scope: str,
    *,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for patterns with no matches.

    Use when a find/replace pattern matches nothing.

    Args:
        pattern: The pattern that found no matches.
        scope: Where the search was performed (e.g., "spec", "phase-1").
        remediation: Guidance on what to check.
        request_id: Correlation identifier.

    Example:
        >>> no_matches_error("deprecated_function", "spec my-spec-001")
    """
    return error_response(
        f"No matches found for pattern '{pattern}' in {scope}",
        error_code=ErrorCode.NO_MATCHES_FOUND,
        error_type=ErrorType.NOT_FOUND,
        data={"pattern": pattern, "scope": scope},
        remediation=remediation or "Verify the pattern and scope. Use dry-run to preview matches.",
        request_id=request_id,
    )


def backup_not_found_error(
    spec_id: str,
    backup_id: Optional[str] = None,
    *,
    available_backups: Optional[Sequence[str]] = None,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for missing backup.

    Use when a rollback or diff references a non-existent backup.

    Args:
        spec_id: The spec whose backup is missing.
        backup_id: The specific backup ID that wasn't found.
        available_backups: Optional list of available backup IDs.
        remediation: Guidance on how to resolve.
        request_id: Correlation identifier.

    Example:
        >>> backup_not_found_error("my-spec-001", "backup-2024-01-15")
    """
    data: Dict[str, Any] = {"spec_id": spec_id}
    if backup_id:
        data["backup_id"] = backup_id
    if available_backups:
        data["available_backups"] = list(available_backups)

    message = f"Backup not found for spec '{spec_id}'"
    if backup_id:
        message = f"Backup '{backup_id}' not found for spec '{spec_id}'"

    return error_response(
        message,
        error_code=ErrorCode.BACKUP_NOT_FOUND,
        error_type=ErrorType.NOT_FOUND,
        data=data,
        remediation=remediation or "List available backups using spec action='history'.",
        request_id=request_id,
    )


def backup_corrupted_error(
    spec_id: str,
    backup_id: str,
    error_detail: str,
    *,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for corrupted backup.

    Use when a backup file exists but cannot be loaded.

    Args:
        spec_id: The spec whose backup is corrupted.
        backup_id: The corrupted backup identifier.
        error_detail: Description of the corruption.
        remediation: Guidance on how to recover.
        request_id: Correlation identifier.

    Example:
        >>> backup_corrupted_error("my-spec", "backup-001", "Invalid JSON structure")
    """
    return error_response(
        f"Backup '{backup_id}' for spec '{spec_id}' is corrupted: {error_detail}",
        error_code=ErrorCode.BACKUP_CORRUPTED,
        error_type=ErrorType.INTERNAL,
        data={
            "spec_id": spec_id,
            "backup_id": backup_id,
            "error_detail": error_detail,
        },
        remediation=remediation or "Try an earlier backup or restore from version control.",
        request_id=request_id,
    )


def rollback_failed_error(
    spec_id: str,
    backup_id: str,
    error_detail: str,
    *,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for failed rollback operation.

    Use when a rollback operation fails after starting.

    Args:
        spec_id: The spec being rolled back.
        backup_id: The backup being restored from.
        error_detail: What went wrong during rollback.
        remediation: Guidance on how to recover.
        request_id: Correlation identifier.

    Example:
        >>> rollback_failed_error("my-spec", "backup-001", "Write permission denied")
    """
    return error_response(
        f"Rollback failed for spec '{spec_id}' from backup '{backup_id}': {error_detail}",
        error_code=ErrorCode.ROLLBACK_FAILED,
        error_type=ErrorType.INTERNAL,
        data={
            "spec_id": spec_id,
            "backup_id": backup_id,
            "error_detail": error_detail,
        },
        remediation=remediation or "Check file permissions. A safety backup was created before rollback attempt.",
        request_id=request_id,
    )


def comparison_failed_error(
    source: str,
    target: str,
    error_detail: str,
    *,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for failed diff/comparison operation.

    Use when a spec comparison operation fails.

    Args:
        source: The source spec or backup being compared.
        target: The target spec or backup being compared.
        error_detail: What went wrong during comparison.
        remediation: Guidance on how to resolve.
        request_id: Correlation identifier.

    Example:
        >>> comparison_failed_error("my-spec-v1", "my-spec-v2", "Schema version mismatch")
    """
    return error_response(
        f"Comparison failed between '{source}' and '{target}': {error_detail}",
        error_code=ErrorCode.COMPARISON_FAILED,
        error_type=ErrorType.INTERNAL,
        data={
            "source": source,
            "target": target,
            "error_detail": error_detail,
        },
        remediation=remediation or "Ensure both specs are valid and use compatible schema versions.",
        request_id=request_id,
    )
