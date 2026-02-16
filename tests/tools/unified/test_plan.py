"""Tests for unified plan tool dispatch exception handling.

Tests that _dispatch_plan_action catches exceptions and returns error responses
instead of crashing the MCP server.
"""

from unittest.mock import patch


class TestPlanDispatchExceptionHandling:
    """Tests for _dispatch_plan_action exception handling."""

    def test_dispatch_catches_exceptions(self):
        """_dispatch_plan_action should catch exceptions and return error response."""
        from foundry_mcp.tools.unified.plan import _dispatch_plan_action

        with patch(
            "foundry_mcp.tools.unified.common.get_server_role",
            return_value="maintainer",
        ):
            with patch(
                "foundry_mcp.tools.unified.plan._PLAN_ROUTER"
            ) as mock_router:
                mock_router.allowed_actions.return_value = ["create"]
                mock_router.dispatch.side_effect = RuntimeError("Plan execution failed")

                result = _dispatch_plan_action(
                    action="create",
                    payload={"name": "test-plan"},
                )

        # Should return error response, not raise exception
        assert result["success"] is False
        assert "Plan execution failed" in result["error"]
        assert result["data"]["error_type"] == "internal"
        assert result["data"]["details"]["action"] == "create"
        assert result["data"]["details"]["error_type"] == "RuntimeError"

    def test_dispatch_handles_empty_exception_message(self):
        """_dispatch_plan_action should handle exceptions with empty messages."""
        from foundry_mcp.tools.unified.plan import _dispatch_plan_action

        with patch(
            "foundry_mcp.tools.unified.common.get_server_role",
            return_value="maintainer",
        ):
            with patch(
                "foundry_mcp.tools.unified.plan._PLAN_ROUTER"
            ) as mock_router:
                mock_router.allowed_actions.return_value = ["create"]
                mock_router.dispatch.side_effect = RuntimeError()

                result = _dispatch_plan_action(
                    action="create",
                    payload={"name": "test-plan"},
                )

        # Should use class name when message is empty
        assert result["success"] is False
        assert "RuntimeError" in result["error"]

    def test_dispatch_logs_exception(self, caplog):
        """_dispatch_plan_action should log exceptions."""
        import logging

        from foundry_mcp.tools.unified.plan import _dispatch_plan_action

        with caplog.at_level(logging.ERROR):
            with patch(
                "foundry_mcp.tools.unified.common.get_server_role",
                return_value="maintainer",
            ):
                with patch(
                    "foundry_mcp.tools.unified.plan._PLAN_ROUTER"
                ) as mock_router:
                    mock_router.allowed_actions.return_value = ["create"]
                    mock_router.dispatch.side_effect = ValueError("test error")

                    _dispatch_plan_action(
                        action="create",
                        payload={"name": "test-plan"},
                    )

        assert "test error" in caplog.text
