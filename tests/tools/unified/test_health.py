"""Tests for unified health tool dispatch exception handling.

Tests that _dispatch_health_action catches exceptions and returns error responses
instead of crashing the MCP server.
"""

from unittest.mock import MagicMock, patch


class TestHealthDispatchExceptionHandling:
    """Tests for _dispatch_health_action exception handling."""

    def test_dispatch_catches_exceptions(self):
        """_dispatch_health_action should catch exceptions and return error response."""
        from foundry_mcp.tools.unified.health import _dispatch_health_action

        with patch(
            "foundry_mcp.tools.unified.common.get_server_role",
            return_value="maintainer",
        ):
            with patch("foundry_mcp.tools.unified.health._HEALTH_ROUTER") as mock_router:
                mock_router.allowed_actions.return_value = ["check"]
                mock_router.dispatch.side_effect = RuntimeError("Health check failed")

                result = _dispatch_health_action(action="check", include_details=True)

        # Should return error response, not raise exception
        assert result["success"] is False
        assert "Health check failed" in result["error"]
        assert result["data"]["error_type"] == "internal"
        assert result["data"]["details"]["action"] == "check"
        assert result["data"]["details"]["error_type"] == "RuntimeError"

    def test_dispatch_handles_empty_exception_message(self):
        """_dispatch_health_action should handle exceptions with empty messages."""
        from foundry_mcp.tools.unified.health import _dispatch_health_action

        with patch(
            "foundry_mcp.tools.unified.common.get_server_role",
            return_value="maintainer",
        ):
            with patch("foundry_mcp.tools.unified.health._HEALTH_ROUTER") as mock_router:
                mock_router.allowed_actions.return_value = ["check"]
                mock_router.dispatch.side_effect = RuntimeError()

                result = _dispatch_health_action(action="check")

        # Should use class name when message is empty
        assert result["success"] is False
        assert "RuntimeError" in result["error"]

    def test_dispatch_logs_exception(self, caplog):
        """_dispatch_health_action should log exceptions."""
        import logging

        from foundry_mcp.tools.unified.health import _dispatch_health_action

        with caplog.at_level(logging.ERROR):
            with patch(
                "foundry_mcp.tools.unified.common.get_server_role",
                return_value="maintainer",
            ):
                with patch("foundry_mcp.tools.unified.health._HEALTH_ROUTER") as mock_router:
                    mock_router.allowed_actions.return_value = ["check"]
                    mock_router.dispatch.side_effect = ValueError("test error")

                    _dispatch_health_action(action="check")

        assert "test error" in caplog.text

    def test_dispatch_forwards_config_to_router(self):
        """_dispatch_health_action should pass config through dispatch kwargs."""
        from foundry_mcp.tools.unified.health import _dispatch_health_action

        config = MagicMock()
        with patch(
            "foundry_mcp.tools.unified.common.get_server_role",
            return_value="maintainer",
        ):
            with patch("foundry_mcp.tools.unified.health._HEALTH_ROUTER") as mock_router:
                mock_router.allowed_actions.return_value = ["check"]
                mock_router.dispatch.return_value = {"success": True}

                _dispatch_health_action(
                    action="check",
                    include_details=False,
                    config=config,
                )

        mock_router.dispatch.assert_called_once_with(
            action="check",
            include_details=False,
            config=config,
        )

    def test_dispatch_with_config_still_enforces_authorization(self):
        """Observer role should be denied even when config is provided."""
        from foundry_mcp.tools.unified.health import _dispatch_health_action

        config = MagicMock()
        with patch(
            "foundry_mcp.tools.unified.common.get_server_role",
            return_value="observer",
        ):
            result = _dispatch_health_action(
                action="check",
                include_details=True,
                config=config,
            )

        assert result["success"] is False
        assert result["data"]["error_code"] == "AUTHORIZATION"
