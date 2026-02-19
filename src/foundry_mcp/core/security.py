"""
Security utilities for foundry-mcp.

Provides input validation, size limits, and prompt injection defense
for MCP tools. See dev_docs/mcp_best_practices/04-validation-input-hygiene.md
and dev_docs/mcp_best_practices/08-security-trust-boundaries.md for guidance.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Final, Optional, Tuple

logger = logging.getLogger(__name__)

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

# =============================================================================
# Prompt Injection Detection Patterns
# =============================================================================
# These regex patterns detect common prompt injection attempts in LLM-generated
# input. MCP tools receiving untrusted input should check against these patterns.
# See dev_docs/mcp_best_practices/08-security-trust-boundaries.md for details.

INJECTION_PATTERNS: Final[list[str]] = [
    # Instruction override attempts
    r"ignore\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"forget\s+(everything|all)\s+(above|before)",
    r"new\s+instructions?\s*:",
    # System prompt injection
    r"system\s*:\s*",
    r"<\s*system\s*>",
    # Special tokens (model-specific)
    r"<\|.*?\|>",  # OpenAI-style special tokens
    r"\[INST\]|\[/INST\]",  # Llama instruction markers
    r"<\|im_start\|>|<\|im_end\|>",  # ChatML markers
    r"<<SYS>>|<</SYS>>",  # Llama system markers
    # Code block injection attempts
    r"```system",
    r"```\s*<\s*system",
    # Role injection
    r"^(assistant|user|system)\s*:",
]
"""Regex patterns for detecting prompt injection attempts.

Each pattern targets a specific injection technique:
- Instruction overrides: attempts to ignore/discard previous context
- System prompt injection: attempts to inject system-level instructions
- Special tokens: model-specific control sequences
- Code block injection: attempts to inject via markdown code blocks
- Role injection: attempts to assume different conversation roles

Use with detect_prompt_injection() for comprehensive checking.
"""


# =============================================================================
# Detection Results
# =============================================================================


@dataclass
class InjectionDetectionResult:
    """Result of prompt injection detection.

    Attributes:
        is_suspicious: Whether the input appears to contain injection attempts
        matched_pattern: The regex pattern that matched (if any)
        matched_text: The actual text that matched the pattern (if any)
    """

    is_suspicious: bool
    matched_pattern: Optional[str] = None
    matched_text: Optional[str] = None


# =============================================================================
# Detection Functions
# =============================================================================


def detect_prompt_injection(
    text: str,
    *,
    log_detections: bool = True,
    patterns: Optional[list[str]] = None,
) -> InjectionDetectionResult:
    """Detect potential prompt injection attempts in text.

    Scans the input text against known injection patterns and returns
    a result indicating whether suspicious content was found.

    Args:
        text: The input text to scan for injection attempts
        log_detections: Whether to log detected injection attempts (default: True)
        patterns: Optional custom patterns to use instead of INJECTION_PATTERNS

    Returns:
        InjectionDetectionResult with detection status and match details

    Example:
        >>> result = detect_prompt_injection("ignore previous instructions and...")
        >>> if result.is_suspicious:
        ...     print(f"Blocked: matched pattern '{result.matched_pattern}'")
    """
    check_patterns = patterns if patterns is not None else INJECTION_PATTERNS

    for pattern in check_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            result = InjectionDetectionResult(
                is_suspicious=True,
                matched_pattern=pattern,
                matched_text=match.group(0),
            )

            if log_detections:
                # Log with limited text preview to avoid logging sensitive content
                preview = text[:100] + "..." if len(text) > 100 else text
                logger.warning(
                    "Potential prompt injection detected",
                    extra={
                        "pattern": pattern,
                        "matched_text": result.matched_text,
                        "text_preview": preview,
                    },
                )

            return result

    return InjectionDetectionResult(is_suspicious=False)


def is_prompt_injection(text: str) -> bool:
    """Simple check for prompt injection (returns bool only).

    Convenience function when you only need a boolean result.

    Args:
        text: The input text to scan

    Returns:
        True if injection patterns detected, False otherwise

    Example:
        >>> if is_prompt_injection(user_input):
        ...     return error_response("Input contains disallowed patterns")
    """
    return detect_prompt_injection(text, log_detections=False).is_suspicious


# =============================================================================
# Size Validation Functions
# =============================================================================


@dataclass
class SizeValidationResult:
    """Result of input size validation.

    Attributes:
        is_valid: Whether all size checks passed
        violations: List of (field_name, violation_message) tuples
    """

    is_valid: bool
    violations: list[Tuple[str, str]]


def validate_size(
    value: Any,
    field_name: str = "input",
    *,
    max_size: Optional[int] = None,
    max_length: Optional[int] = None,
    max_string_length: Optional[int] = None,
) -> SizeValidationResult:
    """Validate size constraints on a value.

    Args:
        value: The value to validate
        field_name: Name of the field (for error messages)
        max_size: Maximum byte size for serialized value (default: MAX_INPUT_SIZE)
        max_length: Maximum length for arrays/lists (default: MAX_ARRAY_LENGTH)
        max_string_length: Maximum length for strings (default: MAX_STRING_LENGTH)

    Returns:
        SizeValidationResult with validation status and any violations
    """
    import json

    violations = []

    # Check serialized size
    effective_max_size = max_size if max_size is not None else MAX_INPUT_SIZE
    try:
        serialized = json.dumps(value) if not isinstance(value, str) else value
        if len(serialized.encode("utf-8")) > effective_max_size:
            violations.append((field_name, f"Exceeds maximum size ({effective_max_size} bytes)"))
    except (TypeError, ValueError):
        pass  # Can't serialize, skip size check

    # Check array length
    effective_max_length = max_length if max_length is not None else MAX_ARRAY_LENGTH
    if isinstance(value, (list, tuple)):
        if len(value) > effective_max_length:
            violations.append((field_name, f"Array exceeds maximum length ({effective_max_length} items)"))

    # Check string length
    effective_max_string = max_string_length if max_string_length is not None else MAX_STRING_LENGTH
    if isinstance(value, str):
        if len(value) > effective_max_string:
            violations.append((field_name, f"String exceeds maximum length ({effective_max_string} characters)"))

    return SizeValidationResult(
        is_valid=len(violations) == 0,
        violations=violations,
    )


# =============================================================================
# Validation Decorators
# =============================================================================


def validate_input_size(
    *,
    max_size: Optional[int] = None,
    max_array_length: Optional[int] = None,
    max_string_length: Optional[int] = None,
    check_injection: bool = False,
):
    """Decorator to validate input size limits on tool parameters.

    Validates all string and collection parameters against size limits
    before the function executes. Returns an error response if validation fails.

    Args:
        max_size: Maximum total input size in bytes (default: MAX_INPUT_SIZE)
        max_array_length: Maximum array/list length (default: MAX_ARRAY_LENGTH)
        max_string_length: Maximum string length (default: MAX_STRING_LENGTH)
        check_injection: Also check for prompt injection patterns (default: False)

    Returns:
        Decorator function

    Example:
        @mcp.tool()
        @validate_input_size(max_string_length=5000, check_injection=True)
        def process_text(text: str, items: list) -> dict:
            # Parameters are validated before this runs
            return {"result": process(text)}

    Note:
        This decorator should be applied AFTER @mcp.tool() to ensure
        validation runs before the tool handler.
    """
    import functools
    from dataclasses import asdict

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            all_violations = []

            # Validate keyword arguments
            for name, value in kwargs.items():
                result = validate_size(
                    value,
                    field_name=name,
                    max_size=max_size,
                    max_length=max_array_length,
                    max_string_length=max_string_length,
                )
                all_violations.extend(result.violations)

                # Check for injection if enabled and value is string
                if check_injection and isinstance(value, str):
                    injection_result = detect_prompt_injection(value)
                    if injection_result.is_suspicious:
                        all_violations.append((name, f"Contains disallowed patterns: {injection_result.matched_text}"))

            if all_violations:
                # Import here to avoid circular dependency
                try:
                    from foundry_mcp.core.responses.builders import error_response

                    return asdict(
                        error_response(
                            "Input validation failed",
                            error_code="VALIDATION_ERROR",
                            details={
                                "validation_errors": [{"field": field, "message": msg} for field, msg in all_violations]
                            },
                        )
                    )
                except ImportError:
                    # Fallback if responses module not available
                    return {
                        "success": False,
                        "error": "Input validation failed",
                        "data": {
                            "validation_errors": [{"field": field, "message": msg} for field, msg in all_violations]
                        },
                    }

            return func(*args, **kwargs)

        # Handle async functions
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            all_violations = []

            for name, value in kwargs.items():
                result = validate_size(
                    value,
                    field_name=name,
                    max_size=max_size,
                    max_length=max_array_length,
                    max_string_length=max_string_length,
                )
                all_violations.extend(result.violations)

                if check_injection and isinstance(value, str):
                    injection_result = detect_prompt_injection(value)
                    if injection_result.is_suspicious:
                        all_violations.append((name, f"Contains disallowed patterns: {injection_result.matched_text}"))

            if all_violations:
                try:
                    from foundry_mcp.core.responses.builders import error_response

                    return asdict(
                        error_response(
                            "Input validation failed",
                            error_code="VALIDATION_ERROR",
                            details={
                                "validation_errors": [{"field": field, "message": msg} for field, msg in all_violations]
                            },
                        )
                    )
                except ImportError:
                    return {
                        "success": False,
                        "error": "Input validation failed",
                        "data": {
                            "validation_errors": [{"field": field, "message": msg} for field, msg in all_violations]
                        },
                    }

            return await func(*args, **kwargs)

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# Export all constants and functions
__all__ = [
    # Constants
    "MAX_INPUT_SIZE",
    "MAX_ARRAY_LENGTH",
    "MAX_STRING_LENGTH",
    "MAX_NESTED_DEPTH",
    "MAX_FIELD_COUNT",
    "INJECTION_PATTERNS",
    # Types
    "InjectionDetectionResult",
    "SizeValidationResult",
    # Functions
    "detect_prompt_injection",
    "is_prompt_injection",
    "validate_size",
    # Decorators
    "validate_input_size",
]
