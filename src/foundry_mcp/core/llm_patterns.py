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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, Union

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


# -----------------------------------------------------------------------------
# Batch Operation Patterns
# -----------------------------------------------------------------------------


@dataclass
class BatchItemResult:
    """Result for a single item in a batch operation.

    Attributes:
        item_id: Identifier for the item (index or key)
        success: Whether the operation succeeded
        result: Operation result if successful
        error: Error message if failed
        error_code: Machine-readable error code if failed
    """

    item_id: Union[int, str]
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for response."""
        d: Dict[str, Any] = {
            "item_id": self.item_id,
            "success": self.success,
        }
        if self.success:
            d["result"] = self.result
        else:
            d["error"] = self.error
            if self.error_code:
                d["error_code"] = self.error_code
        return d


@dataclass
class BatchResult:
    """Result of a batch operation with separate success/failure tracking.

    Designed for LLM consumption with clear summary and separated results.

    Attributes:
        total: Total items processed
        succeeded: Count of successful operations
        failed: Count of failed operations
        results: List of successful results
        errors: List of failed item details
        warnings: Any warnings generated during processing
    """

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    results: List[BatchItemResult] = field(default_factory=list)
    errors: List[BatchItemResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        """Check if all operations succeeded."""
        return self.failed == 0

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total == 0:
            return 100.0
        return (self.succeeded / self.total) * 100

    def to_response(self, include_details: bool = True) -> Dict[str, Any]:
        """Convert to LLM-friendly response format.

        Args:
            include_details: Include individual results/errors

        Returns:
            Dictionary suitable for tool response
        """
        response: Dict[str, Any] = {
            "summary": f"Processed {self.succeeded}/{self.total} items successfully",
            "counts": {
                "total": self.total,
                "succeeded": self.succeeded,
                "failed": self.failed,
                "success_rate": round(self.success_rate, 1),
            },
        }

        if include_details:
            if self.results:
                response["results"] = [r.to_dict() for r in self.results]
            if self.errors:
                response["errors"] = [e.to_dict() for e in self.errors]

        if self.warnings:
            response["warnings"] = self.warnings

        return response


def batch_response(
    results: List[Any],
    errors: Optional[List[Dict[str, Any]]] = None,
    *,
    total: Optional[int] = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a batch operation response from results and errors.

    Convenience function for creating LLM-friendly batch responses.

    Args:
        results: List of successful results
        errors: List of error dicts with 'item_id', 'error', optional 'error_code'
        total: Total items (defaults to len(results) + len(errors))
        warnings: Optional warnings to include

    Returns:
        LLM-friendly response dictionary

    Example:
        >>> results = [{"id": "1", "data": "..."}, {"id": "2", "data": "..."}]
        >>> errors = [{"item_id": "3", "error": "Not found", "error_code": "NOT_FOUND"}]
        >>> response = batch_response(results, errors)
        >>> print(response["summary"])
        'Processed 2/3 items successfully'
    """
    error_list = errors or []
    actual_total = total or (len(results) + len(error_list))

    batch = BatchResult(
        total=actual_total,
        succeeded=len(results),
        failed=len(error_list),
        warnings=warnings or [],
    )

    # Convert results to BatchItemResult
    for i, result in enumerate(results):
        item_id = result.get("id", i) if isinstance(result, dict) else i
        batch.results.append(BatchItemResult(
            item_id=item_id,
            success=True,
            result=result,
        ))

    # Convert errors to BatchItemResult
    for err in error_list:
        batch.errors.append(BatchItemResult(
            item_id=err.get("item_id", "unknown"),
            success=False,
            error=err.get("error", "Unknown error"),
            error_code=err.get("error_code"),
        ))

    return batch.to_response()


def paginated_batch_response(
    results: List[Any],
    *,
    page_size: int = 50,
    offset: int = 0,
    total: int,
    errors: Optional[List[Dict[str, Any]]] = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a paginated batch response for large result sets.

    Includes pagination metadata for LLMs to understand result scope.

    Args:
        results: Results for current page
        page_size: Number of items per page
        offset: Current page offset
        total: Total items available
        errors: Errors for current page
        warnings: Optional warnings

    Returns:
        Response with pagination metadata

    Example:
        >>> response = paginated_batch_response(
        ...     results=items[:50],
        ...     page_size=50,
        ...     offset=0,
        ...     total=150,
        ... )
        >>> response["pagination"]["has_more"]
        True
    """
    response = batch_response(
        results=results,
        errors=errors,
        total=len(results) + len(errors or []),
        warnings=warnings,
    )

    has_more = offset + len(results) < total
    response["pagination"] = {
        "offset": offset,
        "page_size": page_size,
        "returned": len(results),
        "total": total,
        "has_more": has_more,
        "next_offset": offset + len(results) if has_more else None,
    }

    if has_more:
        remaining = total - (offset + len(results))
        response["warnings"] = response.get("warnings", [])
        response["warnings"].append(
            f"Showing {len(results)} of {total} items. "
            f"{remaining} more available with offset={offset + len(results)}"
        )

    return response


# Export all public symbols
__all__ = [
    # Schema
    "SCHEMA_VERSION",
    # Progressive disclosure
    "DetailLevel",
    "DisclosureConfig",
    "DEFAULT_DISCLOSURE_CONFIG",
    "progressive_disclosure",
    "get_detail_level",
    # Batch operations
    "BatchItemResult",
    "BatchResult",
    "batch_response",
    "paginated_batch_response",
]
