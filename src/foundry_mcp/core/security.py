"""
Security utilities for foundry-mcp.

Provides input validation, size limits, and prompt injection defense
for MCP tools. See docs/mcp_best_practices/04-validation-input-hygiene.md
and docs/mcp_best_practices/08-security-trust-boundaries.md for guidance.
"""

from typing import Final

# =============================================================================
# Input Size Limits
# =============================================================================
# These constants define maximum sizes for various input types to prevent
# resource exhaustion and denial-of-service attacks. Adjust based on your
# specific requirements, but be conservative.

MAX_INPUT_SIZE: Final[int] = 100_000
"""Maximum total input payload size in bytes (100KB).

Use this to validate the overall size of request payloads before processing.
Prevents memory exhaustion from oversized requests.
"""

MAX_ARRAY_LENGTH: Final[int] = 1_000
"""Maximum number of items in array/list inputs.

Use this to limit iteration counts and prevent algorithmic complexity attacks.
Arrays larger than this should be paginated or streamed.
"""

MAX_STRING_LENGTH: Final[int] = 10_000
"""Maximum length for individual string fields (10K characters).

Use this for text fields like descriptions, content, etc.
Longer content should use dedicated file/blob handling.
"""

MAX_NESTED_DEPTH: Final[int] = 10
"""Maximum nesting depth for JSON structures.

Prevents stack overflow from deeply nested payloads.
Most legitimate use cases require < 5 levels of nesting.
"""

MAX_FIELD_COUNT: Final[int] = 100
"""Maximum number of fields in an object/dict.

Prevents resource exhaustion from objects with excessive properties.
"""

# Export all constants for easy importing
__all__ = [
    "MAX_INPUT_SIZE",
    "MAX_ARRAY_LENGTH",
    "MAX_STRING_LENGTH",
    "MAX_NESTED_DEPTH",
    "MAX_FIELD_COUNT",
]
