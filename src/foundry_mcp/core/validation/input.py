"""Input validation for raw spec data."""

import json
from typing import Any, Dict, Optional

from foundry_mcp.core.security import MAX_INPUT_SIZE
from foundry_mcp.core.validation.models import Diagnostic, ValidationResult


def validate_spec_input(
    raw_input: str | bytes,
    *,
    max_size: Optional[int] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[ValidationResult]]:
    """
    Validate and parse raw spec input with size checks.

    Performs size validation before JSON parsing to prevent resource
    exhaustion attacks from oversized payloads.

    Args:
        raw_input: Raw JSON string or bytes to validate
        max_size: Maximum allowed size in bytes (default: MAX_INPUT_SIZE)

    Returns:
        Tuple of (parsed_data, error_result):
        - On success: (dict, None)
        - On failure: (None, ValidationResult with error)

    Example:
        >>> spec_data, error = validate_spec_input(json_string)
        >>> if error:
        ...     return error_response(error.diagnostics[0].message)
        >>> result = validate_spec(spec_data)
    """
    effective_max_size = max_size if max_size is not None else MAX_INPUT_SIZE

    # Convert to bytes if string for consistent size checking
    if isinstance(raw_input, str):
        input_bytes = raw_input.encode("utf-8")
    else:
        input_bytes = raw_input

    # Check input size
    if len(input_bytes) > effective_max_size:
        error_result = ValidationResult(
            spec_id="unknown",
            is_valid=False,
            error_count=1,
        )
        error_result.diagnostics.append(
            Diagnostic(
                code="INPUT_TOO_LARGE",
                message=f"Input size ({len(input_bytes):,} bytes) exceeds maximum allowed ({effective_max_size:,} bytes)",
                severity="error",
                category="security",
                suggested_fix=f"Reduce input size to under {effective_max_size:,} bytes",
            )
        )
        return None, error_result

    # Try to parse JSON
    try:
        if isinstance(raw_input, bytes):
            spec_data = json.loads(raw_input.decode("utf-8"))
        else:
            spec_data = json.loads(raw_input)
    except json.JSONDecodeError as e:
        error_result = ValidationResult(
            spec_id="unknown",
            is_valid=False,
            error_count=1,
        )
        error_result.diagnostics.append(
            Diagnostic(
                code="INVALID_JSON",
                message=f"Failed to parse JSON: {e}",
                severity="error",
                category="structure",
            )
        )
        return None, error_result

    # Spec data must be a dict
    if not isinstance(spec_data, dict):
        error_result = ValidationResult(
            spec_id="unknown",
            is_valid=False,
            error_count=1,
        )
        error_result.diagnostics.append(
            Diagnostic(
                code="INVALID_SPEC_TYPE",
                message=f"Spec must be a JSON object, got {type(spec_data).__name__}",
                severity="error",
                category="structure",
            )
        )
        return None, error_result

    return spec_data, None
