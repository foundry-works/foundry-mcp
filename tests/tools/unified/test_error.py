"""Tests for unified error tool dispatch exception handling.

Tests that _dispatch_error_action catches exceptions and returns error responses
instead of crashing the MCP server.
"""

from unittest.mock import patch


class TestErrorDispatchExceptionHandling:
    """Tests for _dispatch_error_action exception handling."""

    def test_dispatch_catches_exceptions(self, mock_config):
        """_dispatch_error_action should catch exceptions and return error response."""
        from foundry_mcp.tools.unified.error import _dispatch_error_action

        with patch("foundry_mcp.tools.unified.error._ERROR_ROUTER") as mock_router:
            mock_router.allowed_actions.return_value = ["list"]
            mock_router.dispatch.side_effect = RuntimeError("Database connection failed")

            result = _dispatch_error_action(
                action="list",
                payload={},
                config=mock_config,
            )

        # Should return error response, not raise exception
        assert result["success"] is False
        assert "Database connection failed" in result["error"]
        assert result["data"]["error_type"] == "internal"
        assert result["data"]["details"]["action"] == "list"
        assert result["data"]["details"]["error_type"] == "RuntimeError"

    def test_dispatch_handles_empty_exception_message(self, mock_config):
        """_dispatch_error_action should handle exceptions with empty messages."""
        from foundry_mcp.tools.unified.error import _dispatch_error_action

        with patch("foundry_mcp.tools.unified.error._ERROR_ROUTER") as mock_router:
            mock_router.allowed_actions.return_value = ["list"]
            mock_router.dispatch.side_effect = RuntimeError()

            result = _dispatch_error_action(
                action="list",
                payload={},
                config=mock_config,
            )

        # Should use class name when message is empty
        assert result["success"] is False
        assert "RuntimeError" in result["error"]

    def test_dispatch_logs_exception(self, mock_config, caplog):
        """_dispatch_error_action should log exceptions."""
        import logging

        from foundry_mcp.tools.unified.error import _dispatch_error_action

        with caplog.at_level(logging.ERROR):
            with patch("foundry_mcp.tools.unified.error._ERROR_ROUTER") as mock_router:
                mock_router.allowed_actions.return_value = ["list"]
                mock_router.dispatch.side_effect = ValueError("test error")

                _dispatch_error_action(
                    action="list",
                    payload={},
                    config=mock_config,
                )

        assert "test error" in caplog.text
