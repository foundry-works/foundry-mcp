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
        cursor = encode_cursor({
            "last_id": "item_456",
            "timestamp": "2025-01-15T10:30:00Z",
            "offset": 100,
        })
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
