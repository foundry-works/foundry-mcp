"""
Standard response contracts for MCP tool operations.
Provides consistent response structures across all foundry-mcp tools.

Response Schema Contract
========================

All MCP tool responses follow a standard structure:

    {
        "success": bool,       # Required: operation success/failure
        "data": {...},         # Required: primary payload (empty dict on error)
        "error": str | null,   # Required: error message or null on success
        "meta": {              # Required: response metadata
            "version": "response-v2",
            "request_id": "req_abc123"?,
            "warnings": ["..."]?,
            "pagination": { ... }?,
            "rate_limit": { ... }?,
            "telemetry": { ... }?
        }
    }

Metadata Semantics
------------------

Attach operational context through `meta` so every tool shares an identical
envelope. The standard keys are:

* `version` *(required)* – identifies the contract version (`response-v2`).
* `request_id` *(should)* – correlation identifier propagated through logs.
* `warnings` *(should)* – array of non-fatal issues for successful operations.
* `pagination` *(may)* – cursor information (`cursor`, `has_more`, `total_count`).
* `rate_limit` *(may)* – limit, remaining, reset timestamp, retry hints.
* `telemetry` *(may)* – timing/performance metrics, downstream call counts, etc.

Multi-Payload Tools
-------------------

Tools returning multiple payloads should nest each value under a named key:

    data = {
        "spec": {...},          # First payload
        "tasks": [...],         # Second payload
    }

This ensures consumers can access each payload by name rather than relying
on position or implicit structure.

Edge Cases & Partial Payloads
-----------------------------

Empty Results (success=True):
    When a query succeeds but finds no results, return success=True with
    empty/partial data to distinguish from errors:

    {"success": True, "data": {"tasks": [], "count": 0}, "error": None}

Not Found (success=False):
    When the requested resource doesn't exist, return success=False:

    {"success": False, "data": {}, "error": "Spec not found: my-spec"}

Blocked/Conditional States (success=True):
    Dependency checks and similar queries return success=True with state info:

    {
        "success": True,
        "data": {
            "task_id": "task-1-2",
            "can_start": False,
            "blocked_by": [{"id": "task-1-1", "status": "pending"}]
        },
        "error": None,
        "meta": {
            "version": "response-v2",
            "warnings": ["Task currently blocked"]
        }
    }

Key Principle:
    - `success=True` means the operation executed correctly (even if the result is empty).
    - `success=False` means the operation failed to execute; include actionable error details.
    - Keep business data inside `data` and operational context inside `meta`.
"""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from foundry_mcp.core.context import get_correlation_id

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    """Machine-readable error codes for MCP tool responses.

    Use these canonical codes in `error_code` fields to enable consistent
    client-side error handling. Codes follow SCREAMING_SNAKE_CASE convention.

    Categories:
        - Validation (input errors)
        - Resource (not found, conflict)
        - Access (auth, permissions, rate limits)
        - System (internal, unavailable)
    """

    # Validation errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_FORMAT = "INVALID_FORMAT"
    MISSING_REQUIRED = "MISSING_REQUIRED"
    INVALID_PARENT = "INVALID_PARENT"
    INVALID_POSITION = "INVALID_POSITION"
    INVALID_REGEX_PATTERN = "INVALID_REGEX_PATTERN"
    PATTERN_TOO_BROAD = "PATTERN_TOO_BROAD"

    # Resource errors
    NOT_FOUND = "NOT_FOUND"
    SPEC_NOT_FOUND = "SPEC_NOT_FOUND"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    PHASE_NOT_FOUND = "PHASE_NOT_FOUND"
    DEPENDENCY_NOT_FOUND = "DEPENDENCY_NOT_FOUND"
    BACKUP_NOT_FOUND = "BACKUP_NOT_FOUND"
    NO_MATCHES_FOUND = "NO_MATCHES_FOUND"
    DUPLICATE_ENTRY = "DUPLICATE_ENTRY"
    CONFLICT = "CONFLICT"
    CIRCULAR_DEPENDENCY = "CIRCULAR_DEPENDENCY"
    SELF_REFERENCE = "SELF_REFERENCE"

    # Access errors
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    FEATURE_DISABLED = "FEATURE_DISABLED"

    # System errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    UNAVAILABLE = "UNAVAILABLE"
    RESOURCE_BUSY = "RESOURCE_BUSY"
    BACKUP_CORRUPTED = "BACKUP_CORRUPTED"
    ROLLBACK_FAILED = "ROLLBACK_FAILED"
    COMPARISON_FAILED = "COMPARISON_FAILED"

    # AI/LLM Provider errors
    AI_NO_PROVIDER = "AI_NO_PROVIDER"
    AI_PROVIDER_TIMEOUT = "AI_PROVIDER_TIMEOUT"
    AI_PROVIDER_ERROR = "AI_PROVIDER_ERROR"
    AI_CONTEXT_TOO_LARGE = "AI_CONTEXT_TOO_LARGE"
    AI_PROMPT_NOT_FOUND = "AI_PROMPT_NOT_FOUND"
    AI_CACHE_STALE = "AI_CACHE_STALE"


class ErrorType(str, Enum):
    """Error categories for routing and client-side handling.

    Each type corresponds to an HTTP status code analog and indicates
    whether the operation should be retried.

    See docs/codebase_standards/mcp_response_schema.md for the full mapping.
    """

    VALIDATION = "validation"  # 400 - No retry, fix input
    AUTHENTICATION = "authentication"  # 401 - No retry, re-authenticate
    AUTHORIZATION = "authorization"  # 403 - No retry
    NOT_FOUND = "not_found"  # 404 - No retry
    CONFLICT = "conflict"  # 409 - Maybe retry, check state
    RATE_LIMIT = "rate_limit"  # 429 - Yes, after delay
    FEATURE_FLAG = "feature_flag"  # 403 - No retry, check flag status
    INTERNAL = "internal"  # 500 - Yes, with backoff
    UNAVAILABLE = "unavailable"  # 503 - Yes, with backoff
    AI_PROVIDER = "ai_provider"  # AI-specific - Retry varies by error


@dataclass
class ToolResponse:
    """
    Standard response structure for MCP tool operations.

    All tool handlers should return data that can be serialized to this format,
    ensuring consistent API responses across the codebase.

    Attributes:
        success: Whether the operation completed successfully
        data: The primary payload (operation-specific structured data)
        error: Error message if success is False, None otherwise
        meta: Response metadata including version identifier
    """

    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=lambda: {"version": "response-v2"})


def _build_meta(
    *,
    request_id: Optional[str] = None,
    warnings: Optional[Sequence[str]] = None,
    pagination: Optional[Mapping[str, Any]] = None,
    rate_limit: Optional[Mapping[str, Any]] = None,
    telemetry: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
    auto_inject_request_id: bool = True,
) -> Dict[str, Any]:
    """Construct a metadata payload that always includes the response version.

    Args:
        request_id: Explicit correlation ID (takes precedence if provided)
        warnings: Non-fatal issues to surface
        pagination: Cursor metadata for list results
        rate_limit: Rate limit state
        telemetry: Timing/performance metadata
        extra: Arbitrary extra metadata to merge
        auto_inject_request_id: If True (default), auto-inject correlation_id
            from context when request_id is not explicitly provided
    """
    meta: Dict[str, Any] = {"version": "response-v2"}

    # Auto-inject request_id from context if not explicitly provided
    effective_request_id = request_id
    if effective_request_id is None and auto_inject_request_id:
        effective_request_id = get_correlation_id() or None

    if effective_request_id:
        meta["request_id"] = effective_request_id
    if warnings:
        meta["warnings"] = list(warnings)
    if pagination:
        meta["pagination"] = dict(pagination)
    if rate_limit:
        meta["rate_limit"] = dict(rate_limit)
    if telemetry:
        meta["telemetry"] = dict(telemetry)
    if extra:
        meta.update(dict(extra))

    return meta


def success_response(
    data: Optional[Mapping[str, Any]] = None,
    *,
    warnings: Optional[Sequence[str]] = None,
    pagination: Optional[Mapping[str, Any]] = None,
    rate_limit: Optional[Mapping[str, Any]] = None,
    telemetry: Optional[Mapping[str, Any]] = None,
    request_id: Optional[str] = None,
    meta: Optional[Mapping[str, Any]] = None,
    **fields: Any,
) -> ToolResponse:
    """Create a standardized success response.

    Args:
        data: Optional mapping used as the base payload.
        warnings: Non-fatal issues to surface in ``meta.warnings``.
        pagination: Cursor metadata for list results.
        rate_limit: Rate limit state (limit, remaining, reset_at, etc.).
        telemetry: Timing/performance metadata.
        request_id: Correlation identifier propagated through logs/traces.
        meta: Arbitrary extra metadata to merge into ``meta``.
        **fields: Additional payload fields (shorthand for ``data.update``).
    """
    payload: Dict[str, Any] = {}
    if data:
        payload.update(dict(data))
    if fields:
        payload.update(fields)

    meta_payload = _build_meta(
        request_id=request_id,
        warnings=warnings,
        pagination=pagination,
        rate_limit=rate_limit,
        telemetry=telemetry,
        extra=meta,
    )

    return ToolResponse(success=True, data=payload, error=None, meta=meta_payload)


def error_response(
    message: str,
    *,
    data: Optional[Mapping[str, Any]] = None,
    error_code: Optional[Union[ErrorCode, str]] = None,
    error_type: Optional[Union[ErrorType, str]] = None,
    remediation: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
    request_id: Optional[str] = None,
    rate_limit: Optional[Mapping[str, Any]] = None,
    telemetry: Optional[Mapping[str, Any]] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> ToolResponse:
    """Create a standardized error response.

    Args:
        message: Human-readable description of the failure.
        data: Optional mapping with additional machine-readable context.
        error_code: Canonical error code (use ``ErrorCode`` enum or string,
            e.g., ``ErrorCode.VALIDATION_ERROR`` or ``"VALIDATION_ERROR"``).
        error_type: Error category for routing (use ``ErrorType`` enum or string,
            e.g., ``ErrorType.VALIDATION`` or ``"validation"``).
        remediation: User-facing guidance on how to fix the issue.
        details: Nested structure describing validation failures or metadata.
        request_id: Correlation identifier propagated through logs/traces.
        rate_limit: Rate limit state to help clients back off correctly.
        telemetry: Timing/performance metadata captured before failure.
        meta: Arbitrary extra metadata to merge into ``meta``.

    Example:
        >>> error_response(
        ...     "Validation failed: spec_id is required",
        ...     error_code=ErrorCode.MISSING_REQUIRED,
        ...     error_type=ErrorType.VALIDATION,
        ...     remediation="Provide a non-empty spec_id parameter",
        ... )
    """
    payload: Dict[str, Any] = {}
    if data:
        payload.update(dict(data))

    effective_error_code: Union[ErrorCode, str] = (
        error_code if error_code is not None else ErrorCode.INTERNAL_ERROR
    )
    effective_error_type: Union[ErrorType, str] = (
        error_type if error_type is not None else ErrorType.INTERNAL
    )

    if "error_code" not in payload:
        payload["error_code"] = (
            effective_error_code.value
            if isinstance(effective_error_code, Enum)
            else effective_error_code
        )
    if "error_type" not in payload:
        payload["error_type"] = (
            effective_error_type.value
            if isinstance(effective_error_type, Enum)
            else effective_error_type
        )
    if remediation is not None and "remediation" not in payload:
        payload["remediation"] = remediation
    if details and "details" not in payload:
        payload["details"] = dict(details)

    meta_payload = _build_meta(
        request_id=request_id,
        rate_limit=rate_limit,
        telemetry=telemetry,
        extra=meta,
    )

    return ToolResponse(success=False, data=payload, error=message, meta=meta_payload)


# ---------------------------------------------------------------------------
# Specialized Error Helpers
# ---------------------------------------------------------------------------


def validation_error(
    message: str,
    *,
    field: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create a validation error response (HTTP 400 analog).

    Args:
        message: Human-readable description of the validation failure.
        field: The field that failed validation.
        details: Additional context (e.g., constraint violated, value received).
        remediation: Guidance on how to fix the input.
        request_id: Correlation identifier.

    Example:
        >>> validation_error(
        ...     "Invalid email format",
        ...     field="email",
        ...     remediation="Provide email in format: user@domain.com",
        ... )
    """
    error_details = dict(details) if details else {}
    if field and "field" not in error_details:
        error_details["field"] = field

    return error_response(
        message,
        error_code=ErrorCode.VALIDATION_ERROR,
        error_type=ErrorType.VALIDATION,
        details=error_details if error_details else None,
        remediation=remediation,
        request_id=request_id,
    )


def not_found_error(
    resource_type: str,
    resource_id: str,
    *,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create a not found error response (HTTP 404 analog).

    Args:
        resource_type: Type of resource (e.g., "Spec", "Task", "User").
        resource_id: Identifier of the missing resource.
        remediation: Guidance on how to resolve (defaults to verification hint).
        request_id: Correlation identifier.

    Example:
        >>> not_found_error("Spec", "my-spec-001")
    """
    return error_response(
        f"{resource_type} '{resource_id}' not found",
        error_code=ErrorCode.NOT_FOUND,
        error_type=ErrorType.NOT_FOUND,
        data={"resource_type": resource_type, "resource_id": resource_id},
        remediation=remediation or f"Verify the {resource_type.lower()} ID exists.",
        request_id=request_id,
    )


def rate_limit_error(
    limit: int,
    period: str,
    retry_after_seconds: int,
    *,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create a rate limit error response (HTTP 429 analog).

    Args:
        limit: Maximum requests allowed in the period.
        period: Time window (e.g., "minute", "hour").
        retry_after_seconds: Seconds until client can retry.
        remediation: Guidance on how to proceed.
        request_id: Correlation identifier.

    Example:
        >>> rate_limit_error(100, "minute", 45)
    """
    return error_response(
        f"Rate limit exceeded: {limit} requests per {period}",
        error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
        error_type=ErrorType.RATE_LIMIT,
        data={"retry_after_seconds": retry_after_seconds},
        rate_limit={
            "limit": limit,
            "period": period,
            "retry_after": retry_after_seconds,
        },
        remediation=remediation
        or f"Wait {retry_after_seconds} seconds before retrying.",
        request_id=request_id,
    )


def unauthorized_error(
    message: str = "Authentication required",
    *,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an unauthorized error response (HTTP 401 analog).

    Args:
        message: Human-readable description.
        remediation: Guidance on how to authenticate.
        request_id: Correlation identifier.

    Example:
        >>> unauthorized_error("Invalid API key")
    """
    return error_response(
        message,
        error_code=ErrorCode.UNAUTHORIZED,
        error_type=ErrorType.AUTHENTICATION,
        remediation=remediation or "Provide valid authentication credentials.",
        request_id=request_id,
    )


def forbidden_error(
    message: str,
    *,
    required_permission: Optional[str] = None,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create a forbidden error response (HTTP 403 analog).

    Args:
        message: Human-readable description.
        required_permission: The permission needed for this operation.
        remediation: Guidance on how to obtain access.
        request_id: Correlation identifier.

    Example:
        >>> forbidden_error(
        ...     "Cannot delete project",
        ...     required_permission="project:delete",
        ... )
    """
    data: Dict[str, Any] = {}
    if required_permission:
        data["required_permission"] = required_permission

    return error_response(
        message,
        error_code=ErrorCode.FORBIDDEN,
        error_type=ErrorType.AUTHORIZATION,
        data=data if data else None,
        remediation=remediation
        or "Request appropriate permissions from the resource owner.",
        request_id=request_id,
    )


def conflict_error(
    message: str,
    *,
    details: Optional[Mapping[str, Any]] = None,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create a conflict error response (HTTP 409 analog).

    Args:
        message: Human-readable description of the conflict.
        details: Context about the conflicting state.
        remediation: Guidance on how to resolve the conflict.
        request_id: Correlation identifier.

    Example:
        >>> conflict_error(
        ...     "Resource already exists",
        ...     details={"existing_id": "spec-001"},
        ... )
    """
    return error_response(
        message,
        error_code=ErrorCode.CONFLICT,
        error_type=ErrorType.CONFLICT,
        details=details,
        remediation=remediation or "Check current state and retry if appropriate.",
        request_id=request_id,
    )


def internal_error(
    message: str = "An internal error occurred",
    *,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an internal error response (HTTP 500 analog).

    Args:
        message: Human-readable description (keep vague for security).
        request_id: Correlation identifier for log correlation.

    Example:
        >>> internal_error(request_id="req_abc123")
    """
    remediation = "Please try again. If the problem persists, contact support."
    if request_id:
        remediation += f" Reference: {request_id}"

    return error_response(
        message,
        error_code=ErrorCode.INTERNAL_ERROR,
        error_type=ErrorType.INTERNAL,
        remediation=remediation,
        request_id=request_id,
    )


def unavailable_error(
    message: str = "Service temporarily unavailable",
    *,
    retry_after_seconds: Optional[int] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an unavailable error response (HTTP 503 analog).

    Args:
        message: Human-readable description.
        retry_after_seconds: Suggested retry delay.
        request_id: Correlation identifier.

    Example:
        >>> unavailable_error("Database maintenance in progress", retry_after_seconds=300)
    """
    data: Dict[str, Any] = {}
    if retry_after_seconds:
        data["retry_after_seconds"] = retry_after_seconds

    remediation = "Please retry with exponential backoff."
    if retry_after_seconds:
        remediation = f"Retry after {retry_after_seconds} seconds."

    return error_response(
        message,
        error_code=ErrorCode.UNAVAILABLE,
        error_type=ErrorType.UNAVAILABLE,
        data=data if data else None,
        remediation=remediation,
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# Spec Modification Error Helpers
# ---------------------------------------------------------------------------


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
        remediation=remediation
        or "Remove an existing dependency to break the cycle before adding this one.",
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
        remediation=remediation
        or "Check existing dependencies using task info before removing.",
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
        remediation=remediation
        or "Check regex syntax. Use raw strings and escape special characters.",
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
        remediation=remediation
        or "Use a more specific pattern or apply to a narrower scope.",
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
        remediation=remediation
        or "Verify the pattern and scope. Use dry-run to preview matches.",
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
        remediation=remediation
        or "Try an earlier backup or restore from version control.",
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
        remediation=remediation
        or "Check file permissions. A safety backup was created before rollback attempt.",
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
        remediation=remediation
        or "Ensure both specs are valid and use compatible schema versions.",
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# AI/LLM Provider Error Helpers
# ---------------------------------------------------------------------------


def ai_no_provider_error(
    message: str = "No AI provider available",
    *,
    required_providers: Optional[Sequence[str]] = None,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for when no AI provider is available.

    Use when an AI consultation is requested but no providers are configured
    or all configured providers have failed availability checks.

    Args:
        message: Human-readable description.
        required_providers: List of provider IDs that were checked.
        remediation: Guidance on how to configure a provider.
        request_id: Correlation identifier.

    Example:
        >>> ai_no_provider_error(
        ...     "No AI provider available for plan review",
        ...     required_providers=["gemini", "cursor-agent", "codex"],
        ... )
    """
    data: Dict[str, Any] = {}
    if required_providers:
        data["required_providers"] = list(required_providers)

    default_remediation = (
        "Configure an AI provider: set GEMINI_API_KEY, OPENAI_API_KEY, "
        "or ANTHROPIC_API_KEY environment variable, or ensure cursor-agent is available."
    )

    return error_response(
        message,
        error_code=ErrorCode.AI_NO_PROVIDER,
        error_type=ErrorType.AI_PROVIDER,
        data=data if data else None,
        remediation=remediation or default_remediation,
        request_id=request_id,
    )


def ai_provider_timeout_error(
    provider_id: str,
    timeout_seconds: int,
    *,
    message: Optional[str] = None,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for AI provider execution timeout.

    Use when an AI provider call exceeds the configured timeout limit.

    Args:
        provider_id: The provider that timed out (e.g., "gemini", "codex").
        timeout_seconds: The timeout that was exceeded.
        message: Human-readable description (auto-generated if not provided).
        remediation: Guidance on how to handle the timeout.
        request_id: Correlation identifier.

    Example:
        >>> ai_provider_timeout_error("gemini", 300)
    """
    default_message = f"AI provider '{provider_id}' timed out after {timeout_seconds}s"

    return error_response(
        message or default_message,
        error_code=ErrorCode.AI_PROVIDER_TIMEOUT,
        error_type=ErrorType.AI_PROVIDER,
        data={
            "provider_id": provider_id,
            "timeout_seconds": timeout_seconds,
        },
        remediation=remediation
        or (
            "Try again with a smaller context, increase the timeout, "
            "or use a different provider."
        ),
        request_id=request_id,
    )


def ai_provider_error(
    provider_id: str,
    error_detail: str,
    *,
    status_code: Optional[int] = None,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for when an AI provider returns an error.

    Use when an AI provider API call fails with an error response.

    Args:
        provider_id: The provider that returned the error.
        error_detail: The error message from the provider.
        status_code: HTTP status code from the provider (if applicable).
        remediation: Guidance on how to resolve the issue.
        request_id: Correlation identifier.

    Example:
        >>> ai_provider_error("gemini", "Invalid API key", status_code=401)
    """
    data: Dict[str, Any] = {
        "provider_id": provider_id,
        "error_detail": error_detail,
    }
    if status_code is not None:
        data["status_code"] = status_code

    return error_response(
        f"AI provider '{provider_id}' returned error: {error_detail}",
        error_code=ErrorCode.AI_PROVIDER_ERROR,
        error_type=ErrorType.AI_PROVIDER,
        data=data,
        remediation=remediation
        or (
            "Check provider configuration and API key validity. "
            "Consult provider documentation for error details."
        ),
        request_id=request_id,
    )


def ai_context_too_large_error(
    context_tokens: int,
    max_tokens: int,
    *,
    provider_id: Optional[str] = None,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for when context exceeds model limits.

    Use when the prompt/context size exceeds the AI model's token limit.

    Args:
        context_tokens: Number of tokens in the attempted context.
        max_tokens: Maximum tokens allowed by the model.
        provider_id: The provider that rejected the context.
        remediation: Guidance on how to reduce context size.
        request_id: Correlation identifier.

    Example:
        >>> ai_context_too_large_error(150000, 128000, provider_id="gemini")
    """
    data: Dict[str, Any] = {
        "context_tokens": context_tokens,
        "max_tokens": max_tokens,
        "overflow_tokens": context_tokens - max_tokens,
    }
    if provider_id:
        data["provider_id"] = provider_id

    return error_response(
        f"Context size ({context_tokens} tokens) exceeds limit ({max_tokens} tokens)",
        error_code=ErrorCode.AI_CONTEXT_TOO_LARGE,
        error_type=ErrorType.AI_PROVIDER,
        data=data,
        remediation=remediation
        or (
            "Reduce context size by: excluding large files, using incremental mode, "
            "or reviewing only specific tasks/phases instead of the full spec."
        ),
        request_id=request_id,
    )


def ai_prompt_not_found_error(
    prompt_id: str,
    *,
    available_prompts: Optional[Sequence[str]] = None,
    workflow: Optional[str] = None,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for when a prompt template is not found.

    Use when a requested prompt ID doesn't exist in the prompt registry.

    Args:
        prompt_id: The prompt ID that was not found.
        available_prompts: List of valid prompt IDs for the workflow.
        workflow: The workflow context (e.g., "plan_review", "fidelity_review").
        remediation: Guidance on how to find the correct prompt ID.
        request_id: Correlation identifier.

    Example:
        >>> ai_prompt_not_found_error(
        ...     "INVALID_PROMPT",
        ...     available_prompts=["PLAN_REVIEW_FULL_V1", "PLAN_REVIEW_QUICK_V1"],
        ...     workflow="plan_review",
        ... )
    """
    data: Dict[str, Any] = {"prompt_id": prompt_id}
    if available_prompts:
        data["available_prompts"] = list(available_prompts)
    if workflow:
        data["workflow"] = workflow

    available_str = ""
    if available_prompts:
        available_str = f" Available: {', '.join(available_prompts[:5])}"
        if len(available_prompts) > 5:
            available_str += f" (and {len(available_prompts) - 5} more)"

    return error_response(
        f"Prompt '{prompt_id}' not found.{available_str}",
        error_code=ErrorCode.AI_PROMPT_NOT_FOUND,
        error_type=ErrorType.NOT_FOUND,
        data=data,
        remediation=remediation
        or (
            "Use a valid prompt ID from the workflow's prompt builder. "
            "Call list_prompts() to see available templates."
        ),
        request_id=request_id,
    )


def ai_cache_stale_error(
    cache_key: str,
    cache_age_seconds: int,
    max_age_seconds: int,
    *,
    remediation: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """Create an error response for when cached AI result is stale.

    Use when a cached consultation result has expired and needs refresh.

    Args:
        cache_key: Identifier for the cached item.
        cache_age_seconds: Age of the cached result in seconds.
        max_age_seconds: Maximum allowed age for cached results.
        remediation: Guidance on how to refresh the cache.
        request_id: Correlation identifier.

    Example:
        >>> ai_cache_stale_error(
        ...     "plan_review:spec-001:full",
        ...     cache_age_seconds=7200,
        ...     max_age_seconds=3600,
        ... )
    """
    return error_response(
        f"Cached result for '{cache_key}' is stale ({cache_age_seconds}s > {max_age_seconds}s)",
        error_code=ErrorCode.AI_CACHE_STALE,
        error_type=ErrorType.AI_PROVIDER,
        data={
            "cache_key": cache_key,
            "cache_age_seconds": cache_age_seconds,
            "max_age_seconds": max_age_seconds,
        },
        remediation=remediation
        or (
            "Re-run the consultation to refresh cached results, "
            "or use --no-cache to bypass the cache."
        ),
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# Error Message Sanitization
# ---------------------------------------------------------------------------


def sanitize_error_message(
    exc: Exception,
    context: str = "",
    include_type: bool = False,
) -> str:
    """
    Convert exception to user-safe message without internal details.

    Logs full exception server-side for debugging per MCP best practices:
    - "Never expose internal details" (07-error-semantics.md:16)
    - "Log full details server-side" (07-error-semantics.md:23)

    Args:
        exc: The exception to sanitize
        context: Optional context for logging (e.g., "spec validation")
        include_type: Whether to include exception type name in message

    Returns:
        User-safe error message without file paths, stack traces, or internal state
    """
    # Log full details server-side for debugging
    if context:
        logger.debug(f"Error in {context}: {exc}", exc_info=True)
    else:
        logger.debug(f"Error: {exc}", exc_info=True)

    # Map known exception types to safe messages
    type_name = type(exc).__name__

    if isinstance(exc, FileNotFoundError):
        return "Required file or resource not found"
    if isinstance(exc, json.JSONDecodeError):
        return "Invalid JSON format"
    if isinstance(exc, subprocess.TimeoutExpired):
        timeout = getattr(exc, "timeout", "unknown")
        return f"Operation timed out after {timeout} seconds"
    if isinstance(exc, PermissionError):
        return "Permission denied for requested operation"
    if isinstance(exc, ValueError):
        suffix = f" ({type_name})" if include_type else ""
        return f"Invalid value provided{suffix}"
    if isinstance(exc, KeyError):
        return "Required configuration key not found"
    if isinstance(exc, ConnectionError):
        return "Connection failed - service may be unavailable"
    if isinstance(exc, OSError):
        return "System I/O error occurred"

    # Generic fallback - don't expose exception message
    suffix = f" ({type_name})" if include_type else ""
    return f"An internal error occurred{suffix}"
