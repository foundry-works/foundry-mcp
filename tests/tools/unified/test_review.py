"""Tests for unified review tool dispatch exception handling.

Tests that _dispatch_review_action catches exceptions and returns error responses
instead of crashing the MCP server.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_config():
    """Create a mock ServerConfig."""
    config = MagicMock()
    return config


class TestReviewDispatchExceptionHandling:
    """Tests for _dispatch_review_action exception handling."""

    def test_dispatch_catches_exceptions(self, mock_config):
        """_dispatch_review_action should catch exceptions and return error response."""
        from foundry_mcp.tools.unified.review import _dispatch_review_action

        with patch(
            "foundry_mcp.tools.unified.review._REVIEW_ROUTER"
        ) as mock_router:
            mock_router.allowed_actions.return_value = ["run"]
            mock_router.dispatch.side_effect = RuntimeError("Review analysis failed")

            result = _dispatch_review_action(
                action="run",
                payload={"spec_id": "test-spec"},
                config=mock_config,
            )

        # Should return error response, not raise exception
        assert result["success"] is False
        assert "Review analysis failed" in result["error"]
        assert result["data"]["error_type"] == "internal"
        assert result["data"]["details"]["action"] == "run"
        assert result["data"]["details"]["error_type"] == "RuntimeError"

    def test_dispatch_handles_empty_exception_message(self, mock_config):
        """_dispatch_review_action should handle exceptions with empty messages."""
        from foundry_mcp.tools.unified.review import _dispatch_review_action

        with patch(
            "foundry_mcp.tools.unified.review._REVIEW_ROUTER"
        ) as mock_router:
            mock_router.allowed_actions.return_value = ["run"]
            mock_router.dispatch.side_effect = RuntimeError()

            result = _dispatch_review_action(
                action="run",
                payload={"spec_id": "test-spec"},
                config=mock_config,
            )

        # Should use class name when message is empty
        assert result["success"] is False
        assert "RuntimeError" in result["error"]

    def test_dispatch_logs_exception(self, mock_config, caplog):
        """_dispatch_review_action should log exceptions."""
        import logging

        from foundry_mcp.tools.unified.review import _dispatch_review_action

        with caplog.at_level(logging.ERROR):
            with patch(
                "foundry_mcp.tools.unified.review._REVIEW_ROUTER"
            ) as mock_router:
                mock_router.allowed_actions.return_value = ["run"]
                mock_router.dispatch.side_effect = ValueError("test error")

                _dispatch_review_action(
                    action="run",
                    payload={"spec_id": "test-spec"},
                    config=mock_config,
                )

        assert "test error" in caplog.text
