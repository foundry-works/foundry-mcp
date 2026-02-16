"""Tests for M3: reason_detail parameter length validation."""

import pytest

from foundry_mcp.tools.unified.task_handlers._helpers import (
    _validate_reason_detail,
    _REASON_DETAIL_MAX_LENGTH,
)


class TestValidateReasonDetail:
    """Unit tests for _validate_reason_detail helper."""

    def test_within_limit_returns_none(self):
        """reason_detail within limit is accepted."""
        result = _validate_reason_detail("x" * 1999, "session-end", "req-1")
        assert result is None

    def test_at_limit_returns_none(self):
        """reason_detail at exact limit is accepted."""
        result = _validate_reason_detail("x" * _REASON_DETAIL_MAX_LENGTH, "session-end", "req-1")
        assert result is None

    def test_exceeding_limit_returns_error(self):
        """reason_detail exceeding limit is rejected."""
        result = _validate_reason_detail("x" * 2001, "session-end", "req-1")
        assert result is not None
        assert result["success"] is False
        assert result["data"]["error_code"] == "MISSING_REQUIRED"
        assert "reason_detail" in result["error"]
        assert "2001" in result["error"]

    def test_none_returns_none(self):
        """None reason_detail is accepted."""
        result = _validate_reason_detail(None, "session-end", "req-1")
        assert result is None

    def test_empty_string_returns_none(self):
        """Empty string reason_detail is accepted."""
        result = _validate_reason_detail("", "session-end", "req-1")
        assert result is None
