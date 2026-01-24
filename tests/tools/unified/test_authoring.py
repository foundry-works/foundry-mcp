"""Tests for unified authoring tool dispatch exception handling.

Tests that _dispatch_authoring_action catches exceptions and returns error responses
instead of crashing the MCP server.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_config():
    """Create a mock ServerConfig."""
    config = MagicMock()
    return config


class TestAuthoringDispatchExceptionHandling:
    """Tests for _dispatch_authoring_action exception handling."""

    def test_dispatch_catches_exceptions(self, mock_config):
        """_dispatch_authoring_action should catch exceptions and return error response."""
        from foundry_mcp.tools.unified.authoring import _dispatch_authoring_action

        with patch(
            "foundry_mcp.tools.unified.authoring._AUTHORING_ROUTER"
        ) as mock_router:
            mock_router.dispatch.side_effect = RuntimeError("Database connection failed")

            result = _dispatch_authoring_action(
                action="spec-create",
                payload={"name": "test"},
                config=mock_config,
            )

        # Should return error response, not raise exception
        assert result["success"] is False
        assert "Database connection failed" in result["error"]
        assert result["data"]["error_type"] == "internal"
        assert result["data"]["details"]["action"] == "spec-create"
        assert result["data"]["details"]["error_type"] == "RuntimeError"

    def test_dispatch_handles_empty_exception_message(self, mock_config):
        """_dispatch_authoring_action should handle exceptions with empty messages."""
        from foundry_mcp.tools.unified.authoring import _dispatch_authoring_action

        with patch(
            "foundry_mcp.tools.unified.authoring._AUTHORING_ROUTER"
        ) as mock_router:
            mock_router.dispatch.side_effect = RuntimeError()

            result = _dispatch_authoring_action(
                action="spec-create",
                payload={"name": "test"},
                config=mock_config,
            )

        # Should use class name when message is empty
        assert result["success"] is False
        assert "RuntimeError" in result["error"]

    def test_dispatch_logs_exception(self, mock_config, caplog):
        """_dispatch_authoring_action should log exceptions."""
        import logging

        from foundry_mcp.tools.unified.authoring import _dispatch_authoring_action

        with caplog.at_level(logging.ERROR):
            with patch(
                "foundry_mcp.tools.unified.authoring._AUTHORING_ROUTER"
            ) as mock_router:
                mock_router.dispatch.side_effect = ValueError("test error")

                _dispatch_authoring_action(
                    action="spec-create",
                    payload={"name": "test"},
                    config=mock_config,
                )

        assert "test error" in caplog.text
