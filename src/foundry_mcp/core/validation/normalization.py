"""Normalization and value validation utilities for SDD spec files."""

import re
from datetime import datetime, timezone
from difflib import get_close_matches
from typing import Any, Optional

from foundry_mcp.core.validation.constants import VALID_NODE_TYPES, VALID_STATUSES


def _suggest_value(value: str, valid_values: set, n: int = 1) -> Optional[str]:
    """
    Suggest a close match for an invalid value.

    Args:
        value: The invalid value provided
        valid_values: Set of valid values to match against
        n: Number of suggestions to return (default 1)

    Returns:
        Suggestion string like "did you mean 'X'?" or None if no close match
    """
    if not value:
        return None
    matches = get_close_matches(value.lower(), [v.lower() for v in valid_values], n=n, cutoff=0.6)
    if matches:
        # Find the original-case version of the match
        for v in valid_values:
            if v.lower() == matches[0]:
                return f"did you mean '{v}'?"
        return f"did you mean '{matches[0]}'?"
    return None


def _is_valid_spec_id(spec_id: str) -> bool:
    """Check if spec_id follows the recommended format."""
    pattern = r"^[a-z0-9-]+-\d{4}-\d{2}-\d{2}-\d{3}$"
    return bool(re.match(pattern, spec_id))


def _is_valid_iso8601(value: str) -> bool:
    """Check if value is valid ISO 8601 date."""
    try:
        # Try parsing with Z suffix
        if value.endswith("Z"):
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            datetime.fromisoformat(value)
        return True
    except ValueError:
        return False


def _normalize_timestamp(value: Any) -> Optional[str]:
    """Normalize timestamp to ISO 8601 format."""
    if not value:
        return None

    text = str(value).strip()
    candidate = text.replace("Z", "")

    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            dt = datetime.strptime(candidate, fmt)
            return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            continue

    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        return None


def _normalize_status(value: Any) -> str:
    """Normalize status value."""
    if not value:
        return "pending"

    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "inprogress": "in_progress",
        "in__progress": "in_progress",
        "todo": "pending",
        "to_do": "pending",
        "complete": "completed",
        "done": "completed",
    }
    text = mapping.get(text, text)

    if text in VALID_STATUSES:
        return text

    return "pending"


def _normalize_node_type(value: Any) -> str:
    """Normalize node type value."""
    if not value:
        return "task"

    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    mapping = {
        "tasks": "task",
        "sub_task": "subtask",
        "verification": "verify",
        "validate": "verify",
    }
    text = mapping.get(text, text)

    if text in VALID_NODE_TYPES:
        return text

    return "task"
