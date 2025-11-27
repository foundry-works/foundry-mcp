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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Union


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

    # Resource errors
    NOT_FOUND = "NOT_FOUND"
    SPEC_NOT_FOUND = "SPEC_NOT_FOUND"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    DUPLICATE_ENTRY = "DUPLICATE_ENTRY"
    CONFLICT = "CONFLICT"

    # Access errors
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    FEATURE_DISABLED = "FEATURE_DISABLED"

    # System errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    UNAVAILABLE = "UNAVAILABLE"


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
) -> Dict[str, Any]:
    """Construct a metadata payload that always includes the response version."""
    meta: Dict[str, Any] = {"version": "response-v2"}

    if request_id:
        meta["request_id"] = request_id
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

    if error_code is not None and "error_code" not in payload:
        payload["error_code"] = error_code
    if error_type is not None and "error_type" not in payload:
        payload["error_type"] = error_type
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
        rate_limit={"limit": limit, "period": period, "retry_after": retry_after_seconds},
        remediation=remediation or f"Wait {retry_after_seconds} seconds before retrying.",
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
        remediation=remediation or "Request appropriate permissions from the resource owner.",
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
