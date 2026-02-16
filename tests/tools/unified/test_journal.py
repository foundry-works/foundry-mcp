"""Tests for unified journal tool dispatch exception handling.

Tests that _dispatch_journal_action catches exceptions and returns error responses
instead of crashing the MCP server.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_config():
    """Create a mock ServerConfig."""
    config = MagicMock()
    return config


class TestJournalDispatchExceptionHandling:
    """Tests for _dispatch_journal_action exception handling."""

    def test_dispatch_catches_exceptions(self, mock_config):
        """_dispatch_journal_action should catch exceptions and return error response."""
        from foundry_mcp.tools.unified.journal import _dispatch_journal_action

        with patch(
            "foundry_mcp.tools.unified.journal._JOURNAL_ROUTER"
        ) as mock_router:
            mock_router.allowed_actions.return_value = ["list"]
            mock_router.dispatch.side_effect = RuntimeError("Journal storage failed")

            result = _dispatch_journal_action(
                action="list",
                payload={"spec_id": "test-spec"},
                config=mock_config,
            )

        # Should return error response, not raise exception
        assert result["success"] is False
        assert "Journal storage failed" in result["error"]
        assert result["data"]["error_type"] == "internal"
        assert result["data"]["details"]["action"] == "list"
        assert result["data"]["details"]["error_type"] == "RuntimeError"

    def test_dispatch_handles_empty_exception_message(self, mock_config):
        """_dispatch_journal_action should handle exceptions with empty messages."""
        from foundry_mcp.tools.unified.journal import _dispatch_journal_action

        with patch(
            "foundry_mcp.tools.unified.journal._JOURNAL_ROUTER"
        ) as mock_router:
            mock_router.allowed_actions.return_value = ["list"]
            mock_router.dispatch.side_effect = RuntimeError()

            result = _dispatch_journal_action(
                action="list",
                payload={"spec_id": "test-spec"},
                config=mock_config,
            )

        # Should use class name when message is empty
        assert result["success"] is False
        assert "RuntimeError" in result["error"]

    def test_dispatch_logs_exception(self, mock_config, caplog):
        """_dispatch_journal_action should log exceptions."""
        import logging

        from foundry_mcp.tools.unified.journal import _dispatch_journal_action

        with caplog.at_level(logging.ERROR):
            with patch(
                "foundry_mcp.tools.unified.journal._JOURNAL_ROUTER"
            ) as mock_router:
                mock_router.allowed_actions.return_value = ["list"]
                mock_router.dispatch.side_effect = ValueError("test error")

                _dispatch_journal_action(
                    action="list",
                    payload={"spec_id": "test-spec"},
                    config=mock_config,
                )

        assert "test error" in caplog.text
