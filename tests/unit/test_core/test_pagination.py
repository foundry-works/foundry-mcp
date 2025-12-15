"""Tests for pagination utilities."""

import pytest

from foundry_mcp.core.pagination import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    CURSOR_VERSION,
    CursorError,
    encode_cursor,
    decode_cursor,
    validate_cursor,
    normalize_page_size,
    paginated_response,
)


class TestCursorEncoding:
    """Tests for cursor encoding/decoding functions."""

    def test_encode_cursor_basic(self):
        """Should encode simple cursor data."""
        cursor = encode_cursor({"last_id": "item_123"})
        assert cursor is not None
        assert isinstance(cursor, str)
        # Should be base64 URL-safe encoded
        assert "+" not in cursor
        assert "/" not in cursor

    def test_encode_cursor_with_complex_data(self):
        """Should encode complex cursor data."""
        cursor = encode_cursor(
            {
                "last_id": "item_456",
                "timestamp": "2025-01-15T10:30:00Z",
                "offset": 100,
            }
        )
        assert cursor is not None
        assert isinstance(cursor, str)

    def test_decode_cursor_basic(self):
        """Should decode cursor back to original data."""
        original = {"last_id": "item_789"}
        cursor = encode_cursor(original)
        decoded = decode_cursor(cursor)

        assert decoded["last_id"] == "item_789"
        assert decoded["version"] == CURSOR_VERSION

    def test_decode_cursor_preserves_all_fields(self):
        """Should preserve all fields in cursor data."""
        original = {
            "last_id": "item_abc",
            "timestamp": "2025-01-15T10:30:00Z",
            "custom_field": "custom_value",
        }
        cursor = encode_cursor(original)
        decoded = decode_cursor(cursor)

        assert decoded["last_id"] == original["last_id"]
        assert decoded["timestamp"] == original["timestamp"]
        assert decoded["custom_field"] == original["custom_field"]

    def test_decode_cursor_includes_version(self):
        """Should include version in decoded cursor."""
        cursor = encode_cursor({"last_id": "test"})
        decoded = decode_cursor(cursor)
        assert "version" in decoded
        assert decoded["version"] == CURSOR_VERSION

    def test_encode_decode_roundtrip(self):
        """Should preserve data through encode/decode cycle."""
        original = {"last_id": "roundtrip_test", "page": 5}
        cursor = encode_cursor(original)
        decoded = decode_cursor(cursor)

        assert decoded["last_id"] == original["last_id"]
        assert decoded["page"] == original["page"]


class TestCursorDecodeErrors:
    """Tests for cursor decode error handling."""

    def test_decode_empty_cursor(self):
        """Should raise CursorError for empty cursor."""
        with pytest.raises(CursorError) as exc_info:
            decode_cursor("")
        assert exc_info.value.reason == "empty"

    def test_decode_none_cursor(self):
        """Should raise CursorError for None cursor."""
        with pytest.raises(CursorError) as exc_info:
            decode_cursor(None)
        assert exc_info.value.reason == "empty"

    def test_decode_invalid_base64(self):
        """Should raise CursorError for invalid base64."""
        with pytest.raises(CursorError) as exc_info:
            decode_cursor("not_valid_base64!!!")
        assert exc_info.value.reason == "decode_failed"

    def test_decode_invalid_json(self):
        """Should raise CursorError for invalid JSON after decoding."""
        import base64

        invalid_json = base64.urlsafe_b64encode(b"not json").decode()
        with pytest.raises(CursorError) as exc_info:
            decode_cursor(invalid_json)
        assert exc_info.value.reason == "decode_failed"

    def test_decode_non_dict_json(self):
        """Should raise CursorError for non-dict JSON."""
        import base64

        list_json = base64.urlsafe_b64encode(b'["item"]').decode()
        with pytest.raises(CursorError) as exc_info:
            decode_cursor(list_json)
        assert exc_info.value.reason == "not_a_dict"

    def test_cursor_error_includes_cursor(self):
        """CursorError should include the invalid cursor."""
        invalid = "invalid_cursor_value"
        try:
            decode_cursor(invalid)
        except CursorError as e:
            assert e.cursor == invalid


class TestValidateCursor:
    """Tests for validate_cursor function."""

    def test_validate_valid_cursor(self):
        """Should return True for valid cursor."""
        cursor = encode_cursor({"last_id": "test"})
        assert validate_cursor(cursor) is True

    def test_validate_invalid_cursor(self):
        """Should return False for invalid cursor."""
        assert validate_cursor("invalid") is False

    def test_validate_empty_cursor(self):
        """Should return False for empty cursor."""
        assert validate_cursor("") is False


class TestNormalizePageSize:
    """Tests for normalize_page_size function."""

    def test_normalize_none_returns_default(self):
        """Should return default when None provided."""
        result = normalize_page_size(None)
        assert result == DEFAULT_PAGE_SIZE

    def test_normalize_valid_value(self):
        """Should return valid value unchanged."""
        assert normalize_page_size(50) == 50
        assert normalize_page_size(100) == 100
        assert normalize_page_size(500) == 500

    def test_normalize_exceeds_max(self):
        """Should cap at maximum page size."""
        assert normalize_page_size(5000) == MAX_PAGE_SIZE
        assert normalize_page_size(MAX_PAGE_SIZE + 1) == MAX_PAGE_SIZE

    def test_normalize_below_minimum(self):
        """Should floor at 1."""
        assert normalize_page_size(0) == 1
        assert normalize_page_size(-1) == 1
        assert normalize_page_size(-100) == 1

    def test_normalize_custom_default(self):
        """Should use custom default if provided."""
        assert normalize_page_size(None, default=50) == 50

    def test_normalize_custom_maximum(self):
        """Should use custom maximum if provided."""
        assert normalize_page_size(500, maximum=200) == 200


class TestPaginationConstants:
    """Tests for pagination constants."""

    def test_default_page_size(self):
        """DEFAULT_PAGE_SIZE should be 100."""
        assert DEFAULT_PAGE_SIZE == 100

    def test_max_page_size(self):
        """MAX_PAGE_SIZE should be 1000."""
        assert MAX_PAGE_SIZE == 1000

    def test_cursor_version(self):
        """CURSOR_VERSION should be 1."""
        assert CURSOR_VERSION == 1


class TestPaginatedResponse:
    """Tests for paginated_response helper."""

    def test_basic_response_structure(self):
        """Should return standard response structure."""
        result = paginated_response(data={"items": [1, 2, 3]})

        assert result["success"] is True
        assert result["error"] is None
        assert "data" in result
        assert "meta" in result

    def test_data_is_preserved(self):
        """Should preserve data in response."""
        data = {"items": [{"id": "1"}, {"id": "2"}], "count": 2}
        result = paginated_response(data=data)

        assert result["data"] == data

    def test_pagination_metadata_included(self):
        """Should include pagination in meta."""
        result = paginated_response(
            data={"items": []},
            cursor="abc123",
            has_more=True,
            page_size=50,
        )

        assert "pagination" in result["meta"]
        pagination = result["meta"]["pagination"]
        assert pagination["cursor"] == "abc123"
        assert pagination["has_more"] is True
        assert pagination["page_size"] == 50

    def test_default_pagination_values(self):
        """Should use default pagination values."""
        result = paginated_response(data={"items": []})

        pagination = result["meta"]["pagination"]
        assert pagination["cursor"] is None
        assert pagination["has_more"] is False
        assert pagination["page_size"] == DEFAULT_PAGE_SIZE

    def test_total_count_optional(self):
        """Should include total_count when provided."""
        result = paginated_response(
            data={"items": []},
            total_count=500,
        )

        pagination = result["meta"]["pagination"]
        assert pagination["total_count"] == 500

    def test_total_count_not_included_by_default(self):
        """Should not include total_count if not provided."""
        result = paginated_response(data={"items": []})

        pagination = result["meta"]["pagination"]
        assert "total_count" not in pagination

    def test_response_version_included(self):
        """Should include response version in meta."""
        result = paginated_response(data={"items": []})

        assert result["meta"]["version"] == "response-v2"

    def test_additional_kwargs_passed_through(self):
        """Should pass through additional kwargs to success_response."""
        result = paginated_response(
            data={"items": []},
            warnings=["Partial results"],
        )

        assert "warnings" in result["meta"]
        assert result["meta"]["warnings"] == ["Partial results"]

    def test_empty_data(self):
        """Should handle empty data dict."""
        result = paginated_response(data={})

        assert result["success"] is True
        assert result["data"] == {}

    def test_complex_data_structure(self):
        """Should handle complex nested data."""
        complex_data = {
            "items": [
                {"id": "1", "nested": {"key": "value"}},
                {"id": "2", "nested": {"key": "other"}},
            ],
            "metadata": {"source": "test"},
        }
        result = paginated_response(data=complex_data)

        assert result["data"] == complex_data

    def test_cursor_with_has_more_false(self):
        """Should allow cursor with has_more=False (edge case)."""
        result = paginated_response(
            data={"items": []},
            cursor="last_page",
            has_more=False,
        )

        pagination = result["meta"]["pagination"]
        assert pagination["cursor"] == "last_page"
        assert pagination["has_more"] is False


class TestPaginatedListToolIntegration:
    """Integration tests for paginated list tools pattern.

    These tests validate the common pagination pattern used across
    list tools (spec-list, task-query, etc.).
    """

    def test_pagination_pattern_first_page(self):
        """Test first page pagination pattern."""
        # Simulate fetching first page of 3 items with limit=2
        items = ["item-1", "item-2", "item-3"]
        page_size = 2

        # Fetch one extra to detect has_more
        page_items = items[: page_size + 1]
        has_more = len(page_items) > page_size
        if has_more:
            page_items = page_items[:page_size]

        assert page_items == ["item-1", "item-2"]
        assert has_more is True

        # Build cursor for next page
        next_cursor = encode_cursor({"last_id": page_items[-1]}) if has_more else None
        assert next_cursor is not None

    def test_pagination_pattern_with_cursor(self):
        """Test pagination pattern with cursor continuation."""
        items = ["item-1", "item-2", "item-3", "item-4", "item-5"]
        page_size = 2

        # First page
        cursor = encode_cursor({"last_id": "item-2"})

        # Decode cursor
        cursor_data = decode_cursor(cursor)
        start_after_id = cursor_data.get("last_id")

        # Find index of cursor position
        start_index = 0
        for i, item in enumerate(items):
            if item == start_after_id:
                start_index = i + 1
                break
        remaining_items = items[start_index:]

        # Fetch page
        page_items = remaining_items[: page_size + 1]
        has_more = len(page_items) > page_size
        if has_more:
            page_items = page_items[:page_size]

        assert page_items == ["item-3", "item-4"]
        assert has_more is True

    def test_pagination_pattern_last_page(self):
        """Test pagination pattern on last page."""
        page_size = 2

        # Start after item-2
        remaining_items = ["item-3"]

        # Fetch page
        page_items = remaining_items[: page_size + 1]
        has_more = len(page_items) > page_size
        if has_more:
            page_items = page_items[:page_size]

        assert page_items == ["item-3"]
        assert has_more is False

        # No next cursor for last page
        next_cursor = encode_cursor({"last_id": page_items[-1]}) if has_more else None
        assert next_cursor is None

    def test_pagination_pattern_empty_results(self):
        """Test pagination pattern with no results."""
        items = []
        page_size = 10

        page_items = items[: page_size + 1]
        has_more = len(page_items) > page_size
        if has_more:
            page_items = page_items[:page_size]

        assert page_items == []
        assert has_more is False

    def test_pagination_pattern_exact_page_boundary(self):
        """Test pagination when items exactly match page size."""
        items = ["item-1", "item-2"]
        page_size = 2

        # Fetch page (no extra item means no more pages)
        page_items = items[: page_size + 1]
        has_more = len(page_items) > page_size
        if has_more:
            page_items = page_items[:page_size]

        assert page_items == ["item-1", "item-2"]
        assert has_more is False

    def test_cursor_roundtrip_preserves_position(self):
        """Cursor encode/decode roundtrip preserves position info."""
        original_position = {"last_id": "task-5-3", "timestamp": "2025-01-15"}

        cursor = encode_cursor(original_position)
        decoded = decode_cursor(cursor)

        assert decoded["last_id"] == original_position["last_id"]
        assert decoded["timestamp"] == original_position["timestamp"]

    def test_pagination_response_format(self):
        """Test that pagination response matches expected format."""
        result = paginated_response(
            data={"specs": [{"spec_id": "spec-1"}, {"spec_id": "spec-2"}], "count": 2},
            cursor="next_cursor_token",
            has_more=True,
            page_size=100,
        )

        # Verify structure
        assert result["success"] is True
        assert result["data"]["count"] == 2
        assert len(result["data"]["specs"]) == 2

        # Verify pagination metadata
        pagination = result["meta"]["pagination"]
        assert pagination["cursor"] == "next_cursor_token"
        assert pagination["has_more"] is True
        assert pagination["page_size"] == 100

    def test_invalid_cursor_handling(self):
        """Test that invalid cursors are properly detected."""
        # Should raise CursorError for invalid cursor
        with pytest.raises(CursorError):
            decode_cursor("invalid_cursor_value")

        # validate_cursor should return False
        assert validate_cursor("invalid") is False
        assert validate_cursor("") is False
