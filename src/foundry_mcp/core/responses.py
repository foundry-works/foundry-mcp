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
from typing import Any, Dict, Mapping, Optional, Sequence


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
    error_code: Optional[str] = None,
    error_type: Optional[str] = None,
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
        error_code: Canonical error code (e.g., ``VALIDATION_ERROR``).
        error_type: Error category (validation, authorization, internal, ...).
        remediation: User-facing guidance on how to fix the issue.
        details: Nested structure describing validation failures or metadata.
        request_id: Correlation identifier propagated through logs/traces.
        rate_limit: Rate limit state to help clients back off correctly.
        telemetry: Timing/performance metadata captured before failure.
        meta: Arbitrary extra metadata to merge into ``meta``.
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
