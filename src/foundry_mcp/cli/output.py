"""JSON output helpers for SDD CLI.

This module provides the sole output mechanism for the CLI.
The CLI is JSON-first and currently emits JSON envelopes only.

Design rationale:
- Primary consumers are AI coding assistants (Claude, Cursor, etc.)
- AI agents parse structured data best - no regex/pattern matching needed
- Consistent output format = reliable integration
- Humans can pipe through `jq` if needed

This module wraps the canonical response helpers from foundry_mcp.core.responses
to ensure CLI output matches the response-v2 schema used by MCP tools.
"""

import json
import sys
from dataclasses import asdict
from typing import Any, Mapping, Sequence, NoReturn

from foundry_mcp.cli.logging import generate_request_id, get_request_id, set_request_id
from foundry_mcp.core.responses import error_response, success_response


def _ensure_request_id() -> str:
    request_id = get_request_id()
    if request_id:
        return request_id
    request_id = generate_request_id()
    set_request_id(request_id)
    return request_id


def emit(data: Any) -> None:
    """Emit JSON to stdout.

    This is the single output function for all CLI commands.
    Data is serialized with indent=2 for readability when debugging.

    Args:
        data: Any JSON-serializable data structure.
    """
    print(json.dumps(data, indent=2, default=str))


def emit_error(
    message: str,
    code: str = "INTERNAL_ERROR",
    *,
    error_type: str = "internal",
    remediation: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> NoReturn:
    """Emit error JSON to stderr and exit with code 1.

    Uses foundry_mcp.core.responses.error_response to ensure response-v2 compliance.
    Error info is structured in the `data` field with `error_code`, `error_type`,
    and `remediation` fields. The `error` field contains the human-readable message.

    Args:
        message: Human-readable error description.
        code: Error code in SCREAMING_SNAKE_CASE (e.g., VALIDATION_ERROR, NOT_FOUND).
        error_type: Error category for routing (validation, not_found, internal, etc.).
        remediation: Actionable guidance for resolving the error.
        details: Optional additional error context.

    Raises:
        SystemExit: Always exits with code 1.
    """
    response = error_response(
        message=message,
        error_code=code,
        error_type=error_type,
        remediation=remediation,
        details=details,
        request_id=_ensure_request_id(),
    )
    print(json.dumps(asdict(response), indent=2, default=str), file=sys.stderr)
    sys.exit(1)


def emit_success(
    data: Any,
    *,
    warnings: Sequence[str] | None = None,
    pagination: Mapping[str, Any] | None = None,
    telemetry: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> None:
    """Emit success response envelope to stdout.

    Uses foundry_mcp.core.responses.success_response to ensure response-v2 compliance.
    All responses include meta.version: "response-v2".

    Args:
        data: The operation-specific payload.
        warnings: Non-fatal issues to surface in meta.warnings.
        pagination: Cursor metadata for list results.
        telemetry: Timing/performance metadata.
        meta: Additional metadata to merge into meta object.
    """
    # Handle both dict and non-dict data
    if isinstance(data, dict):
        response = success_response(
            data=data,
            warnings=warnings,
            pagination=pagination,
            telemetry=telemetry,
            meta=meta,
            request_id=_ensure_request_id(),
        )
    else:
        # Wrap non-dict data in a result key
        response = success_response(
            data={"result": data},
            warnings=warnings,
            pagination=pagination,
            telemetry=telemetry,
            meta=meta,
            request_id=_ensure_request_id(),
        )
    emit(asdict(response))
