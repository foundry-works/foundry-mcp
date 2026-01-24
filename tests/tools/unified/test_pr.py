"""Tests for unified PR tool dispatch exception handling.

Tests that _dispatch_pr_action catches exceptions and returns error responses
instead of crashing the MCP server.
"""

from unittest.mock import patch


class TestPRDispatchExceptionHandling:
    """Tests for _dispatch_pr_action exception handling."""

    def test_dispatch_catches_exceptions(self):
        """_dispatch_pr_action should catch exceptions and return error response."""
        from foundry_mcp.tools.unified.pr import _dispatch_pr_action

        with patch(
            "foundry_mcp.tools.unified.pr._PR_ROUTER"
        ) as mock_router:
            mock_router.dispatch.side_effect = RuntimeError("Git operation failed")

            result = _dispatch_pr_action(
                action="create",
                payload={"spec_id": "test-spec"},
            )

        # Should return error response, not raise exception
        assert result["success"] is False
        assert "Git operation failed" in result["error"]
        assert result["data"]["error_type"] == "internal"
        assert result["data"]["details"]["action"] == "create"
        assert result["data"]["details"]["error_type"] == "RuntimeError"

    def test_dispatch_handles_empty_exception_message(self):
        """_dispatch_pr_action should handle exceptions with empty messages."""
        from foundry_mcp.tools.unified.pr import _dispatch_pr_action

        with patch(
            "foundry_mcp.tools.unified.pr._PR_ROUTER"
        ) as mock_router:
            mock_router.dispatch.side_effect = RuntimeError()

            result = _dispatch_pr_action(
                action="create",
                payload={"spec_id": "test-spec"},
            )

        # Should use class name when message is empty
        assert result["success"] is False
        assert "RuntimeError" in result["error"]

    def test_dispatch_logs_exception(self, caplog):
        """_dispatch_pr_action should log exceptions."""
        import logging

        from foundry_mcp.tools.unified.pr import _dispatch_pr_action

        with caplog.at_level(logging.ERROR):
            with patch(
                "foundry_mcp.tools.unified.pr._PR_ROUTER"
            ) as mock_router:
                mock_router.dispatch.side_effect = ValueError("test error")

                _dispatch_pr_action(
                    action="create",
                    payload={"spec_id": "test-spec"},
                )

        assert "test error" in caplog.text
