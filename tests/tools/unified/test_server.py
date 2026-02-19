"""Tests for unified server tool dispatch exception handling.

Tests that _dispatch_server_action catches exceptions and returns error responses
instead of crashing the MCP server.
"""

import json
from pathlib import Path
from unittest.mock import patch

from foundry_mcp.config.server import ServerConfig


class TestServerDispatchExceptionHandling:
    """Tests for _dispatch_server_action exception handling."""

    def test_dispatch_catches_exceptions(self, mock_config):
        """_dispatch_server_action should catch exceptions and return error response."""
        from foundry_mcp.tools.unified.server import _dispatch_server_action

        with patch("foundry_mcp.tools.unified.server._SERVER_ROUTER") as mock_router:
            mock_router.allowed_actions.return_value = ["info"]
            mock_router.dispatch.side_effect = RuntimeError("Server introspection failed")

            result = _dispatch_server_action(
                action="info",
                payload={},
                config=mock_config,
            )

        # Should return error response, not raise exception
        assert result["success"] is False
        assert "Server introspection failed" in result["error"]
        assert result["data"]["error_type"] == "internal"
        assert result["data"]["details"]["action"] == "info"
        assert result["data"]["details"]["error_type"] == "RuntimeError"

    def test_dispatch_handles_empty_exception_message(self, mock_config):
        """_dispatch_server_action should handle exceptions with empty messages."""
        from foundry_mcp.tools.unified.server import _dispatch_server_action

        with patch("foundry_mcp.tools.unified.server._SERVER_ROUTER") as mock_router:
            mock_router.allowed_actions.return_value = ["info"]
            mock_router.dispatch.side_effect = RuntimeError()

            result = _dispatch_server_action(
                action="info",
                payload={},
                config=mock_config,
            )

        # Should use class name when message is empty
        assert result["success"] is False
        assert "RuntimeError" in result["error"]

    def test_dispatch_logs_exception(self, mock_config, caplog):
        """_dispatch_server_action should log exceptions."""
        import logging

        from foundry_mcp.tools.unified.server import _dispatch_server_action

        with caplog.at_level(logging.ERROR):
            with patch("foundry_mcp.tools.unified.server._SERVER_ROUTER") as mock_router:
                mock_router.allowed_actions.return_value = ["info"]
                mock_router.dispatch.side_effect = ValueError("test error")

                _dispatch_server_action(
                    action="info",
                    payload={},
                    config=mock_config,
                )

        assert "test error" in caplog.text


class TestCapabilitiesAction:
    """Tests for server(action='capabilities')."""

    def test_capabilities_report_autonomy_always_enabled(self):
        """Autonomy capabilities should always be reported as enabled."""
        from foundry_mcp.tools.unified.server import _handle_capabilities

        config = ServerConfig()

        result = _handle_capabilities(config=config, payload={})

        assert result["success"] is True
        assert result["data"]["capabilities"]["autonomy_sessions"] is True
        assert result["data"]["capabilities"]["autonomy_fidelity_gates"] is True
        assert result["data"]["runtime"]["autonomy"]["enabled_now"]["autonomy_sessions"] is True
        assert result["data"]["runtime"]["autonomy"]["supported_by_binary"]["autonomy_fidelity_gates"] is True

    def test_capabilities_include_role_preflight_convention(self):
        """Capabilities include role preflight guidance for consumers."""
        from foundry_mcp.tools.unified.server import _handle_capabilities

        config = ServerConfig()
        result = _handle_capabilities(config=config, payload={})

        assert result["success"] is True
        role_preflight = result["data"]["runtime"]["conventions"]["role_preflight"]
        assert role_preflight["recommended_call"]["tool"] == "task"
        assert role_preflight["recommended_call"]["params"]["action"] == "session"
        assert role_preflight["recommended_call"]["params"]["command"] == "list"
        assert "AUTHORIZATION" in role_preflight["fail_fast_on"]

    def test_capabilities_include_posture_profile_and_session_defaults(self):
        """Capabilities expose runtime posture/session defaults for operators."""
        from foundry_mcp.tools.unified.server import _handle_capabilities

        config = ServerConfig()
        config.autonomy_posture.profile = "unattended"
        config.autonomy_security.role = "autonomy_runner"
        config.autonomy_security.allow_lock_bypass = False
        config.autonomy_security.allow_gate_waiver = False
        config.autonomy_security.enforce_required_phase_gates = True
        config.autonomy_session_defaults.gate_policy = "strict"
        config.autonomy_session_defaults.stop_on_phase_completion = True
        config.autonomy_session_defaults.auto_retry_fidelity_gate = True

        result = _handle_capabilities(config=config, payload={})

        assert result["success"] is True
        autonomy_runtime = result["data"]["runtime"]["autonomy"]
        assert autonomy_runtime["posture_profile"] == "unattended"
        assert autonomy_runtime["security"]["role"] == "autonomy_runner"
        assert autonomy_runtime["session_defaults"]["gate_policy"] == "strict"
        assert autonomy_runtime["session_defaults"]["stop_on_phase_completion"] is True

    def test_manifest_server_examples_are_supported_actions(self):
        """Manifest examples for server tool should be routable actions."""
        from foundry_mcp.tools.unified.server import _SERVER_ROUTER

        manifest_path = Path(__file__).resolve().parents[3] / "mcp" / "capabilities_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        server_tool = next(tool for tool in manifest["tools"]["unified"] if tool["name"] == "server")
        allowed_actions = set(_SERVER_ROUTER.allowed_actions())

        for example in server_tool.get("examples", []):
            action = example["input"]["action"]
            assert action in allowed_actions, example["description"]
