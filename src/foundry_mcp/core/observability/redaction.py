"""Sensitive data redaction utilities.

Provides pattern-based redaction for API keys, passwords, PII, and other
sensitive data. Safe for use before logging or including data in error messages.

See dev_docs/mcp_best_practices/08-security-trust-boundaries.md
"""

import json
import re
from typing import Any, Final, List, Optional, Tuple

SENSITIVE_PATTERNS: Final[List[Tuple[str, str]]] = [
    # API Keys and Tokens
    (r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{20,})['\"]?", "API_KEY"),
    (
        r"(?i)(secret[_-]?key|secretkey)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{20,})['\"]?",
        "SECRET_KEY",
    ),
    (
        r"(?i)(access[_-]?token|accesstoken)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{20,})['\"]?",
        "ACCESS_TOKEN",
    ),
    (r"(?i)bearer\s+([a-zA-Z0-9_\-\.]+)", "BEARER_TOKEN"),
    # Passwords
    (r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?([^\s'\"]{4,})['\"]?", "PASSWORD"),
    # AWS Credentials
    (r"AKIA[0-9A-Z]{16}", "AWS_ACCESS_KEY"),
    (
        r"(?i)(aws[_-]?secret[_-]?access[_-]?key)\s*[:=]\s*['\"]?([a-zA-Z0-9/+=]{40})['\"]?",
        "AWS_SECRET",
    ),
    # Private Keys
    (r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", "PRIVATE_KEY"),
    # Email Addresses (for PII protection)
    (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "EMAIL"),
    # Social Security Numbers (US)
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
    # Credit Card Numbers (basic pattern)
    (r"\b(?:\d{4}[- ]?){3}\d{4}\b", "CREDIT_CARD"),
    # Phone Numbers (various formats)
    (r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "PHONE"),
    # GitHub/GitLab Tokens
    (r"gh[pousr]_[a-zA-Z0-9]{36,}", "GITHUB_TOKEN"),
    (r"glpat-[a-zA-Z0-9\-]{20,}", "GITLAB_TOKEN"),
    # Generic Base64-encoded secrets (long base64 strings in key contexts)
    (
        r"(?i)(token|secret|key|credential)\s*[:=]\s*['\"]?([a-zA-Z0-9+/]{40,}={0,2})['\"]?",
        "BASE64_SECRET",
    ),
]
"""Patterns for detecting sensitive data that should be redacted.

Each tuple contains:
- regex pattern: The pattern to match sensitive data
- label: A human-readable label for the type of sensitive data

Use with redact_sensitive_data() to sanitize logs and error messages.
"""


def redact_sensitive_data(
    data: Any,
    *,
    patterns: Optional[List[Tuple[str, str]]] = None,
    redaction_format: str = "[REDACTED:{label}]",
    max_depth: int = 10,
) -> Any:
    """Recursively redact sensitive data from strings, dicts, and lists.

    Scans input data for sensitive patterns (API keys, passwords, PII, etc.)
    and replaces matches with redaction markers. Safe for use before logging
    or including data in error messages.

    Args:
        data: The data to redact (string, dict, list, or nested structure)
        patterns: Custom patterns to use (default: SENSITIVE_PATTERNS)
        redaction_format: Format string for redaction markers (uses {label})
        max_depth: Maximum recursion depth to prevent stack overflow

    Returns:
        A copy of the data with sensitive values redacted

    Example:
        >>> data = {"api_key": "sk_live_abc123...", "user": "john"}
        >>> safe_data = redact_sensitive_data(data)
        >>> logger.info("Request data", extra={"data": safe_data})
    """
    if max_depth <= 0:
        return "[MAX_DEPTH_EXCEEDED]"

    check_patterns = patterns if patterns is not None else SENSITIVE_PATTERNS

    def redact_string(text: str) -> str:
        """Redact sensitive patterns from a string."""
        result = text
        for pattern, label in check_patterns:
            replacement = redaction_format.format(label=label)
            result = re.sub(pattern, replacement, result)
        return result

    # Handle different data types
    if isinstance(data, str):
        return redact_string(data)

    elif isinstance(data, dict):
        # Check for sensitive key names and redact their values entirely
        sensitive_keys = {
            "password",
            "passwd",
            "pwd",
            "secret",
            "token",
            "api_key",
            "apikey",
            "api-key",
            "access_token",
            "refresh_token",
            "private_key",
            "secret_key",
            "auth",
            "authorization",
            "credential",
            "credentials",
            "ssn",
            "credit_card",
        }
        result = {}
        for key, value in data.items():
            key_lower = str(key).lower().replace("-", "_")
            if key_lower in sensitive_keys:
                # Redact entire value for known sensitive keys
                result[key] = f"[REDACTED:{key_lower.upper()}]"
            else:
                # Recursively process the value
                result[key] = redact_sensitive_data(
                    value,
                    patterns=check_patterns,
                    redaction_format=redaction_format,
                    max_depth=max_depth - 1,
                )
        return result

    elif isinstance(data, (list, tuple)):
        result_list = [
            redact_sensitive_data(
                item,
                patterns=check_patterns,
                redaction_format=redaction_format,
                max_depth=max_depth - 1,
            )
            for item in data
        ]
        return type(data)(result_list) if isinstance(data, tuple) else result_list

    else:
        # For other types (int, float, bool, None), return as-is
        return data


def redact_for_logging(data: Any) -> str:
    """Convenience function to redact and serialize data for logging.

    Combines redaction with JSON serialization for safe logging.

    Args:
        data: Data to redact and serialize

    Returns:
        JSON string with sensitive data redacted

    Example:
        >>> logger.info(f"Processing request: {redact_for_logging(request_data)}")
    """
    redacted = redact_sensitive_data(data)
    try:
        return json.dumps(redacted, default=str)
    except (TypeError, ValueError):
        return str(redacted)
