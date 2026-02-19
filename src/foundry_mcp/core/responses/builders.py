"""
Response builder functions for MCP tool operations.

Provides success_response() and error_response() — the two primary constructors
for creating standardized ToolResponse objects.
"""

from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from foundry_mcp.core.responses.types import (
    ErrorCode,
    ErrorType,
    ToolResponse,
    _build_meta,
)


def success_response(
    data: Optional[Mapping[str, Any]] = None,
    *,
    warnings: Optional[Sequence[str]] = None,
    warning_details: Optional[Sequence[Mapping[str, Any]]] = None,
    pagination: Optional[Mapping[str, Any]] = None,
    rate_limit: Optional[Mapping[str, Any]] = None,
    telemetry: Optional[Mapping[str, Any]] = None,
    content_fidelity: Optional[str] = None,
    dropped_content_ids: Optional[Sequence[str]] = None,
    content_archive_hashes: Optional[Mapping[str, str]] = None,
    request_id: Optional[str] = None,
    meta: Optional[Mapping[str, Any]] = None,
    **fields: Any,
) -> ToolResponse:
    """Create a standardized success response.

    Args:
        data: Optional mapping used as the base payload.
        warnings: Non-fatal issues to surface in ``meta.warnings`` (string array).
        warning_details: Structured warnings with code, severity, message, and context.
            Each warning should have: code (str), severity (info|warning|error),
            message (str), and optional context (dict).
        pagination: Cursor metadata for list results.
        rate_limit: Rate limit state (limit, remaining, reset_at, etc.).
        telemetry: Timing/performance metadata.
        content_fidelity: Response completeness level (full|partial|summary|reference_only).
            When set to anything other than "full", indicates content was truncated
            or summarized due to token limits.
        dropped_content_ids: IDs of content items omitted from response.
            Use with content_fidelity to enable targeted retrieval of dropped items.
        content_archive_hashes: Map of archive IDs to content hashes for retrieval.
            Enables consumers to retrieve dropped content from the archive.
        request_id: Correlation identifier propagated through logs/traces.
        meta: Arbitrary extra metadata to merge into ``meta``.
        **fields: Additional payload fields (shorthand for ``data.update``).

    Example with content fidelity (token management):
        >>> success_response(
        ...     data={"findings": [...]},
        ...     warnings=["CONTENT_TRUNCATED: 3 findings compressed"],
        ...     warning_details=[
        ...         {
        ...             "code": "CONTENT_TRUNCATED",
        ...             "severity": "info",
        ...             "message": "3 findings compressed to fit budget",
        ...             "context": {"dropped_count": 3, "reason": "token_limit"},
        ...         }
        ...     ],
        ...     content_fidelity="partial",
        ...     dropped_content_ids=["finding-003", "finding-004"],
        ... )
    """
    payload: Dict[str, Any] = {}
    if data:
        payload.update(dict(data))
    if fields:
        payload.update(fields)

    meta_payload = _build_meta(
        request_id=request_id,
        warnings=warnings,
        warning_details=warning_details,
        pagination=pagination,
        rate_limit=rate_limit,
        telemetry=telemetry,
        content_fidelity=content_fidelity,
        dropped_content_ids=dropped_content_ids,
        content_archive_hashes=content_archive_hashes,
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

    effective_error_code: Union[ErrorCode, str] = error_code if error_code is not None else ErrorCode.INTERNAL_ERROR
    effective_error_type: Union[ErrorType, str] = error_type if error_type is not None else ErrorType.INTERNAL

    if "error_code" not in payload:
        payload["error_code"] = (
            effective_error_code.value if isinstance(effective_error_code, Enum) else effective_error_code
        )
    if "error_type" not in payload:
        payload["error_type"] = (
            effective_error_type.value if isinstance(effective_error_type, Enum) else effective_error_type
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
