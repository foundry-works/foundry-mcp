"""
LLM-friendly response patterns for foundry-mcp.

Provides helpers for structuring tool responses to optimize LLM consumption,
including progressive disclosure, batch operation formatting, and context-aware
output sizing.

See docs/mcp_best_practices/15-concurrency-patterns.md for guidance.

Example:
    from foundry_mcp.core.llm_patterns import (
        progressive_disclosure, DetailLevel, batch_response
    )

    # Progressive disclosure based on detail level
    data = {"id": "123", "name": "Item", "details": {...}, "metadata": {...}}
    result = progressive_disclosure(data, level=DetailLevel.SUMMARY)

    # Batch operation response
    response = batch_response(results, errors, total=100)
"""

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

# Schema version for LLM patterns module
SCHEMA_VERSION = "1.0.0"

T = TypeVar("T")


class DetailLevel(str, Enum):
    """Detail levels for progressive disclosure.

    Controls how much information is included in responses:

    SUMMARY: Minimal info for quick overview (IDs, status, counts)
    STANDARD: Default level with common fields (adds descriptions, timestamps)
    FULL: Complete data including all optional/verbose fields

    Example:
        >>> level = DetailLevel.SUMMARY
        >>> if level == DetailLevel.FULL:
        ...     include_metadata = True
    """

    SUMMARY = "summary"
    STANDARD = "standard"
    FULL = "full"


@dataclass
class DisclosureConfig:
    """Configuration for progressive disclosure.

    Attributes:
        summary_fields: Fields to include at SUMMARY level
        standard_fields: Additional fields for STANDARD level
        full_fields: Additional fields for FULL level
        max_list_items: Max items in lists at each level {level: count}
        max_string_length: Max string length at each level {level: length}
        truncation_suffix: Suffix to add when truncating
    """

    summary_fields: List[str] = field(default_factory=lambda: ["id", "name", "status"])
    standard_fields: List[str] = field(
        default_factory=lambda: ["description", "created_at", "updated_at"]
    )
    full_fields: List[str] = field(
        default_factory=lambda: ["metadata", "details", "history"]
    )
    max_list_items: Dict[DetailLevel, int] = field(
        default_factory=lambda: {
            DetailLevel.SUMMARY: 5,
            DetailLevel.STANDARD: 20,
            DetailLevel.FULL: 100,
        }
    )
    max_string_length: Dict[DetailLevel, int] = field(
        default_factory=lambda: {
            DetailLevel.SUMMARY: 100,
            DetailLevel.STANDARD: 500,
            DetailLevel.FULL: 5000,
        }
    )
    truncation_suffix: str = "..."


# Default configuration
DEFAULT_DISCLOSURE_CONFIG = DisclosureConfig()


def progressive_disclosure(
    data: Union[Dict[str, Any], List[Any]],
    level: DetailLevel = DetailLevel.STANDARD,
    *,
    config: Optional[DisclosureConfig] = None,
    include_truncation_info: bool = True,
) -> Dict[str, Any]:
    """Apply progressive disclosure to data based on detail level.

    Filters and truncates data based on the requested detail level,
    making responses more manageable for LLM consumption.

    Args:
        data: Dictionary or list to process
        level: Detail level (SUMMARY, STANDARD, FULL)
        config: Custom disclosure configuration
        include_truncation_info: Add _truncated metadata when content is cut

    Returns:
        Processed data with appropriate fields and truncation

    Example:
        >>> data = {
        ...     "id": "123",
        ...     "name": "Task",
        ...     "status": "active",
        ...     "description": "A long description...",
        ...     "metadata": {"complex": "data"},
        ... }
        >>> result = progressive_disclosure(data, level=DetailLevel.SUMMARY)
        >>> print(result.keys())  # Only id, name, status
    """
    cfg = config or DEFAULT_DISCLOSURE_CONFIG

    if isinstance(data, list):
        return _disclose_list(data, level, cfg, include_truncation_info)

    return _disclose_dict(data, level, cfg, include_truncation_info)


def _disclose_dict(
    data: Dict[str, Any],
    level: DetailLevel,
    config: DisclosureConfig,
    include_truncation_info: bool,
) -> Dict[str, Any]:
    """Apply disclosure to a dictionary."""
    # Determine which fields to include
    allowed_fields = set(config.summary_fields)
    if level in (DetailLevel.STANDARD, DetailLevel.FULL):
        allowed_fields.update(config.standard_fields)
    if level == DetailLevel.FULL:
        allowed_fields.update(config.full_fields)

    result: Dict[str, Any] = {}
    truncated_fields: List[str] = []

    for key, value in data.items():
        # Always include if in allowed fields or if FULL level
        if key in allowed_fields or level == DetailLevel.FULL:
            processed_value, was_truncated = _process_value(value, level, config)
            result[key] = processed_value
            if was_truncated:
                truncated_fields.append(key)
        else:
            truncated_fields.append(key)

    if include_truncation_info and truncated_fields:
        result["_truncated"] = {
            "level": level.value,
            "omitted_fields": [f for f in truncated_fields if f not in result],
            "truncated_fields": [f for f in truncated_fields if f in result],
        }

    return result


def _disclose_list(
    data: List[Any],
    level: DetailLevel,
    config: DisclosureConfig,
    include_truncation_info: bool,
) -> Dict[str, Any]:
    """Apply disclosure to a list."""
    max_items = config.max_list_items.get(level, 20)
    total = len(data)
    truncated = total > max_items

    items = []
    for item in data[:max_items]:
        if isinstance(item, dict):
            items.append(_disclose_dict(item, level, config, include_truncation_info=False))
        else:
            processed, _ = _process_value(item, level, config)
            items.append(processed)

    result: Dict[str, Any] = {
        "items": items,
        "count": len(items),
        "total": total,
    }

    if include_truncation_info and truncated:
        result["_truncated"] = {
            "level": level.value,
            "shown": len(items),
            "total": total,
            "remaining": total - len(items),
        }

    return result


def _process_value(
    value: Any,
    level: DetailLevel,
    config: DisclosureConfig,
) -> tuple[Any, bool]:
    """Process a single value, truncating if necessary.

    Returns:
        Tuple of (processed_value, was_truncated)
    """
    max_length = config.max_string_length.get(level, 500)
    max_items = config.max_list_items.get(level, 20)

    if isinstance(value, str):
        if len(value) > max_length:
            return value[:max_length] + config.truncation_suffix, True
        return value, False

    if isinstance(value, list):
        if len(value) > max_items:
            return value[:max_items], True
        return value, False

    if isinstance(value, dict):
        # Recursively process nested dicts at non-FULL levels
        if level != DetailLevel.FULL and len(str(value)) > max_length:
            # Truncate by keeping only first few keys
            keys = list(value.keys())[:5]
            return {k: value[k] for k in keys}, True
        return value, False

    return value, False


def get_detail_level(
    requested: Optional[str] = None,
    default: DetailLevel = DetailLevel.STANDARD,
) -> DetailLevel:
    """Parse detail level from string parameter.

    Args:
        requested: Requested level as string (or None for default)
        default: Default level if not specified or invalid

    Returns:
        Parsed DetailLevel enum value

    Example:
        >>> level = get_detail_level("summary")
        >>> level == DetailLevel.SUMMARY
        True
    """
    if requested is None:
        return default

    try:
        return DetailLevel(requested.lower())
    except ValueError:
        logger.warning(f"Invalid detail level '{requested}', using default '{default.value}'")
        return default


# Export all public symbols
__all__ = [
    "SCHEMA_VERSION",
    "DetailLevel",
    "DisclosureConfig",
    "DEFAULT_DISCLOSURE_CONFIG",
    "progressive_disclosure",
    "get_detail_level",
]
