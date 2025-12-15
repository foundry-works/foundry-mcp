"""
Pagination utilities for MCP tool operations.

Provides cursor-based pagination with opaque cursors, encoding/decoding,
and response formatting helpers for list-style operations.

Pagination Defaults
===================

Use these constants for consistent pagination across tools:

    DEFAULT_PAGE_SIZE (100)  - Default number of items per page
    MAX_PAGE_SIZE (1000)     - Maximum allowed page size

Example usage:

    from foundry_mcp.core.pagination import (
        DEFAULT_PAGE_SIZE,
        MAX_PAGE_SIZE,
        encode_cursor,
        decode_cursor,
        paginated_response,
    )

    @mcp.tool()
    def list_items(cursor: str = None, limit: int = DEFAULT_PAGE_SIZE) -> dict:
        limit = min(max(1, limit), MAX_PAGE_SIZE)

        # Decode cursor if provided
        start_after = None
        if cursor:
            cursor_data = decode_cursor(cursor)
            start_after = cursor_data.get("last_id")

        # Fetch items (one extra to detect has_more)
        items = db.list_items(start_after=start_after, limit=limit + 1)
        has_more = len(items) > limit
        if has_more:
            items = items[:limit]

        # Build response with pagination
        next_cursor = None
        if has_more and items:
            next_cursor = encode_cursor({"last_id": items[-1]["id"]})

        return paginated_response(
            data={"items": items},
            cursor=next_cursor,
            has_more=has_more,
            page_size=limit,
        )
"""

import base64
import json
from dataclasses import asdict
from typing import Any, Dict, Optional

from foundry_mcp.core.responses import success_response


# ---------------------------------------------------------------------------
# Pagination Constants
# ---------------------------------------------------------------------------

#: Default number of items per page
DEFAULT_PAGE_SIZE: int = 100

#: Maximum allowed page size
MAX_PAGE_SIZE: int = 1000

#: Cursor format version (for future compatibility)
CURSOR_VERSION: int = 1


# ---------------------------------------------------------------------------
# Cursor Encoding/Decoding
# ---------------------------------------------------------------------------


class CursorError(Exception):
    """Error during cursor encoding or decoding.

    Attributes:
        cursor: The invalid cursor string (if decoding).
        reason: Description of what went wrong.
    """

    def __init__(
        self,
        message: str,
        cursor: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        super().__init__(message)
        self.cursor = cursor
        self.reason = reason


def encode_cursor(data: Dict[str, Any]) -> str:
    """Encode cursor data as opaque Base64 token.

    The cursor is a URL-safe Base64-encoded JSON object containing
    position information for resuming pagination.

    Args:
        data: Dictionary containing cursor data (typically last_id,
              timestamp, or other position markers).

    Returns:
        Opaque cursor string (URL-safe Base64 encoded).

    Example:
        >>> cursor = encode_cursor({"last_id": "item_123"})
        >>> # Returns: "eyJsYXN0X2lkIjogIml0ZW1fMTIzIiwgInZlcnNpb24iOiAxfQ=="
    """
    # Add version for future format migrations
    cursor_data = {**data, "version": CURSOR_VERSION}
    json_str = json.dumps(cursor_data, separators=(",", ":"))
    return base64.urlsafe_b64encode(json_str.encode()).decode()


def decode_cursor(cursor: str) -> Dict[str, Any]:
    """Decode cursor token to dictionary.

    Args:
        cursor: Opaque cursor string from previous response.

    Returns:
        Dictionary with cursor data including position markers.

    Raises:
        CursorError: If cursor is invalid or cannot be decoded.

    Example:
        >>> data = decode_cursor("eyJsYXN0X2lkIjogIml0ZW1fMTIzIiwgInZlcnNpb24iOiAxfQ==")
        >>> print(data["last_id"])
        "item_123"
    """
    if not cursor:
        raise CursorError("Cursor cannot be empty", cursor=cursor, reason="empty")

    try:
        decoded_bytes = base64.urlsafe_b64decode(cursor.encode())
        data = json.loads(decoded_bytes.decode())

        if not isinstance(data, dict):
            raise CursorError(
                "Invalid cursor format",
                cursor=cursor,
                reason="not_a_dict",
            )

        return data

    except (ValueError, json.JSONDecodeError) as e:
        raise CursorError(
            f"Failed to decode cursor: {str(e)}",
            cursor=cursor,
            reason="decode_failed",
        )


def validate_cursor(cursor: str) -> bool:
    """Check if cursor is valid without raising exceptions.

    Args:
        cursor: Cursor string to validate.

    Returns:
        True if cursor is valid, False otherwise.
    """
    try:
        decode_cursor(cursor)
        return True
    except CursorError:
        return False


# ---------------------------------------------------------------------------
# Pagination Response Helper
# ---------------------------------------------------------------------------


def paginated_response(
    data: Dict[str, Any],
    cursor: Optional[str] = None,
    has_more: bool = False,
    page_size: int = DEFAULT_PAGE_SIZE,
    total_count: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Create a success response with pagination metadata.

    Wraps the data in a standard MCP response envelope with
    meta.pagination containing cursor and pagination info.

    Args:
        data: Response data (typically contains items list).
        cursor: Next page cursor (None if no more pages).
        has_more: Whether more items exist after this page.
        page_size: Number of items in this page.
        total_count: Total count of items (optional, only if efficient).
        **kwargs: Additional arguments passed to success_response.

    Returns:
        Dict formatted as MCP response with pagination metadata.

    Example:
        >>> response = paginated_response(
        ...     data={"items": [...]},
        ...     cursor="abc123",
        ...     has_more=True,
        ...     page_size=100,
        ... )
        >>> # Response includes meta.pagination with cursor, has_more, etc.
    """
    pagination = {
        "cursor": cursor,
        "has_more": has_more,
        "page_size": page_size,
    }

    if total_count is not None:
        pagination["total_count"] = total_count

    return asdict(success_response(data=data, pagination=pagination, **kwargs))


def normalize_page_size(
    requested: Optional[int],
    default: int = DEFAULT_PAGE_SIZE,
    maximum: int = MAX_PAGE_SIZE,
) -> int:
    """Normalize requested page size to valid range.

    Args:
        requested: Requested page size (may be None or out of range).
        default: Default page size if None provided.
        maximum: Maximum allowed page size.

    Returns:
        Valid page size between 1 and maximum.

    Example:
        >>> normalize_page_size(None)
        100
        >>> normalize_page_size(5000)
        1000
        >>> normalize_page_size(-1)
        1
    """
    if requested is None:
        return default
    return min(max(1, requested), maximum)
