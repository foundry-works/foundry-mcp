"""
Standard response contracts for MCP tool operations.
Provides consistent response structures across all foundry-mcp tools.

Response Schema Contract
========================

All MCP tool responses follow a standard structure:

    {
        "success": bool,       # Required: operation success/failure
        "data": {...},         # Required: primary payload (empty dict on error)
        "error": str | null    # Required: error message or null on success
    }

Meta Payload Convention
-----------------------

For operations returning metadata alongside primary data, use reserved keys
within the data dict:

    data = {
        # Primary operation-specific fields
        "spec_id": "...",
        "tasks": [...],

        # Optional meta fields (reserved keys)
        "_meta": {
            "version": "1.0",           # API/schema version
            "pagination": {             # For paginated results
                "offset": 0,
                "limit": 50,
                "total": 150,
                "has_more": True
            },
            "timing": {                 # Performance info
                "duration_ms": 42
            }
        },
        "_warnings": [                  # Non-fatal issues
            "Spec has validation warnings",
            "Deprecated field 'foo' used"
        ]
    }

Multi-Payload Tools
-------------------

Tools returning multiple distinct payloads should nest them under named keys:

    data = {
        "spec": {...},          # First payload
        "tasks": [...],         # Second payload
        "_meta": {...}          # Metadata about the operation
    }

This ensures consumers can access each payload by key rather than relying
on position or implicit structure.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    """
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


def success_response(**data: Any) -> ToolResponse:
    """
    Create a successful response with the given data.

    Args:
        **data: Keyword arguments to include in the response data

    Returns:
        ToolResponse with success=True and provided data

    Example:
        >>> success_response(spec_id="my-spec", count=5)
        ToolResponse(success=True, data={'spec_id': 'my-spec', 'count': 5}, error=None)
    """
    return ToolResponse(success=True, data=dict(data), error=None)


def error_response(message: str) -> ToolResponse:
    """
    Create an error response with the given message.

    Args:
        message: Error message describing what went wrong

    Returns:
        ToolResponse with success=False and error message

    Example:
        >>> error_response("Spec not found")
        ToolResponse(success=False, data={}, error='Spec not found')
    """
    return ToolResponse(success=False, data={}, error=message)
