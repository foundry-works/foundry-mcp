"""
Core types for MCP tool response contracts.

Defines the fundamental building blocks: error codes, error types,
the standard ToolResponse dataclass, and the internal _build_meta() helper.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence

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
    AUTHORIZATION = "AUTHORIZATION"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    RATE_LIMITED = "RATE_LIMITED"
    FEATURE_DISABLED = "FEATURE_DISABLED"

    # System errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    UNAVAILABLE = "UNAVAILABLE"
    RESOURCE_BUSY = "RESOURCE_BUSY"
    BACKUP_CORRUPTED = "BACKUP_CORRUPTED"
    ROLLBACK_FAILED = "ROLLBACK_FAILED"
    COMPARISON_FAILED = "COMPARISON_FAILED"
    OPERATION_FAILED = "OPERATION_FAILED"

    # AI/LLM Provider errors
    AI_NO_PROVIDER = "AI_NO_PROVIDER"
    AI_PROVIDER_TIMEOUT = "AI_PROVIDER_TIMEOUT"
    AI_PROVIDER_ERROR = "AI_PROVIDER_ERROR"
    AI_CONTEXT_TOO_LARGE = "AI_CONTEXT_TOO_LARGE"
    AI_PROMPT_NOT_FOUND = "AI_PROMPT_NOT_FOUND"
    AI_CACHE_STALE = "AI_CACHE_STALE"

    # Autonomy errors
    AUTONOMY_WRITE_LOCK_ACTIVE = "AUTONOMY_WRITE_LOCK_ACTIVE"
    SPEC_SESSION_EXISTS = "SPEC_SESSION_EXISTS"
    NO_ACTIVE_SESSION = "NO_ACTIVE_SESSION"
    AMBIGUOUS_ACTIVE_SESSION = "AMBIGUOUS_ACTIVE_SESSION"
    LOCK_TIMEOUT = "LOCK_TIMEOUT"
    STEP_RESULT_REQUIRED = "STEP_RESULT_REQUIRED"
    STEP_MISMATCH = "STEP_MISMATCH"
    INVALID_GATE_EVIDENCE = "INVALID_GATE_EVIDENCE"
    INVALID_CURSOR = "INVALID_CURSOR"
    SESSION_UNRECOVERABLE = "SESSION_UNRECOVERABLE"
    MANUAL_GATE_ACK_REQUIRED = "MANUAL_GATE_ACK_REQUIRED"
    INVALID_GATE_ACK = "INVALID_GATE_ACK"
    INVALID_STATE_TRANSITION = "INVALID_STATE_TRANSITION"
    AUTO_MODE_SPEC_REQUIRED = "AUTO_MODE_SPEC_REQUIRED"
    SPEC_REBASE_REQUIRED = "SPEC_REBASE_REQUIRED"
    REBASE_COMPLETED_TASKS_REMOVED = "REBASE_COMPLETED_TASKS_REMOVED"
    HEARTBEAT_STALE = "HEARTBEAT_STALE"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    REBASE_BACKUP_MISSING = "REBASE_BACKUP_MISSING"


class ErrorType(str, Enum):
    """Error categories for routing and client-side handling.

    Each type corresponds to an HTTP status code analog and indicates
    whether the operation should be retried.

    See dev_docs/codebase_standards/mcp_response_schema.md for the full mapping.
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
    warning_details: Optional[Sequence[Mapping[str, Any]]] = None,
    pagination: Optional[Mapping[str, Any]] = None,
    rate_limit: Optional[Mapping[str, Any]] = None,
    telemetry: Optional[Mapping[str, Any]] = None,
    content_fidelity: Optional[str] = None,
    dropped_content_ids: Optional[Sequence[str]] = None,
    content_archive_hashes: Optional[Mapping[str, str]] = None,
    extra: Optional[Mapping[str, Any]] = None,
    auto_inject_request_id: bool = True,
) -> Dict[str, Any]:
    """Construct a metadata payload that always includes the response version.

    Args:
        request_id: Explicit correlation ID (takes precedence if provided)
        warnings: Non-fatal issues to surface (string array)
        warning_details: Structured warnings with code, severity, message, context
        pagination: Cursor metadata for list results
        rate_limit: Rate limit state
        telemetry: Timing/performance metadata
        content_fidelity: Response completeness level (full|partial|summary|reference_only)
        dropped_content_ids: IDs of content items omitted from response
        content_archive_hashes: Map of archive IDs to content hashes for retrieval
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
    if warning_details:
        meta["warning_details"] = [dict(w) for w in warning_details]
    if pagination:
        meta["pagination"] = dict(pagination)
    if rate_limit:
        meta["rate_limit"] = dict(rate_limit)
    if telemetry:
        meta["telemetry"] = dict(telemetry)
    # Content fidelity metadata (for token management)
    if content_fidelity:
        meta["content_fidelity"] = content_fidelity
        meta["content_fidelity_schema_version"] = "1.0"
    if dropped_content_ids:
        meta["dropped_content_ids"] = list(dropped_content_ids)
    if content_archive_hashes:
        meta["content_archive_hashes"] = dict(content_archive_hashes)
    if extra:
        meta.update(dict(extra))

    return meta
