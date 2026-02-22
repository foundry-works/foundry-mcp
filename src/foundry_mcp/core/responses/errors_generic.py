"""
Generic HTTP-analog error helpers for MCP tool responses.

Provides 8 helpers mapping to standard HTTP status codes:
validation (400), not_found (404), rate_limit (429), unauthorized (401),
forbidden (403), conflict (409), internal (500), unavailable (503).
"""

from typing import Any, Dict, Mapping, Optional

from foundry_mcp.core.responses.builders import error_response
from foundry_mcp.core.responses.types import ErrorCode, ErrorType, ToolResponse


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
