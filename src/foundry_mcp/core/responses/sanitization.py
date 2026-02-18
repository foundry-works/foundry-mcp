"""
Error message sanitization for MCP tool responses.

Converts exceptions to user-safe messages without exposing internal details,
logging full exception information server-side for debugging.
"""

import json
import logging
import subprocess

logger = logging.getLogger(__name__)


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
