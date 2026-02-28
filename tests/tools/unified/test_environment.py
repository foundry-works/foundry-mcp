"""Tests for unified environment tool dispatch and get-config action.

Tests that _dispatch_environment_action catches exceptions and returns error responses
instead of crashing the MCP server. Also tests get-config autonomy section support.
"""

from unittest.mock import patch

import pytest

from foundry_mcp.config.autonomy import (
    AutonomyPostureConfig,
    AutonomySecurityConfig,
    AutonomySessionDefaultsConfig,
)


class TestEnvironmentDispatchExceptionHandling:
    """Tests for _dispatch_environment_action exception handling."""

    def test_dispatch_catches_exceptions(self, mock_config):
        """_dispatch_environment_action should catch exceptions and return error response."""
        from foundry_mcp.tools.unified.environment import _dispatch_environment_action

        with patch("foundry_mcp.tools.unified.environment._ENVIRONMENT_ROUTER") as mock_router:
            mock_router.allowed_actions.return_value = ["info"]
            mock_router.dispatch.side_effect = RuntimeError("File system error")

            result = _dispatch_environment_action(
                action="info",
                payload={},
                config=mock_config,
            )

        # Should return error response, not raise exception
        assert result["success"] is False
        assert "File system error" in result["error"]
        assert result["data"]["error_type"] == "internal"
        assert result["data"]["details"]["action"] == "info"
        assert result["data"]["details"]["error_type"] == "RuntimeError"

    def test_dispatch_handles_empty_exception_message(self, mock_config):
        """_dispatch_environment_action should handle exceptions with empty messages."""
        from foundry_mcp.tools.unified.environment import _dispatch_environment_action

        with patch("foundry_mcp.tools.unified.environment._ENVIRONMENT_ROUTER") as mock_router:
            mock_router.allowed_actions.return_value = ["info"]
            mock_router.dispatch.side_effect = RuntimeError()

            result = _dispatch_environment_action(
                action="info",
                payload={},
                config=mock_config,
            )

        # Should use class name when message is empty
        assert result["success"] is False
        assert "RuntimeError" in result["error"]

    def test_dispatch_logs_exception(self, mock_config, caplog):
        """_dispatch_environment_action should log exceptions."""
        import logging

        from foundry_mcp.tools.unified.environment import _dispatch_environment_action

        with caplog.at_level(logging.ERROR):
            with patch("foundry_mcp.tools.unified.environment._ENVIRONMENT_ROUTER") as mock_router:
                mock_router.allowed_actions.return_value = ["info"]
                mock_router.dispatch.side_effect = ValueError("test error")

                _dispatch_environment_action(
                    action="info",
                    payload={},
                    config=mock_config,
                )

        assert "test error" in caplog.text


def _make_config_with_autonomy(
    profile=None,
    role="maintainer",
    allow_lock_bypass=False,
    allow_gate_waiver=False,
    enforce_required_phase_gates=True,
    gate_policy="strict",
    stop_on_phase_completion=False,
    auto_retry_fidelity_gate=True,
    max_tasks_per_session=100,
    max_consecutive_errors=3,
    max_fidelity_review_cycles_per_phase=3,
):
    """Build a mock ServerConfig with real autonomy dataclasses."""
    from unittest.mock import MagicMock

    config = MagicMock()
    config.autonomy_posture = AutonomyPostureConfig(profile=profile)
    config.autonomy_security = AutonomySecurityConfig(
        role=role,
        allow_lock_bypass=allow_lock_bypass,
        allow_gate_waiver=allow_gate_waiver,
        enforce_required_phase_gates=enforce_required_phase_gates,
    )
    config.autonomy_session_defaults = AutonomySessionDefaultsConfig(
        gate_policy=gate_policy,
        stop_on_phase_completion=stop_on_phase_completion,
        auto_retry_fidelity_gate=auto_retry_fidelity_gate,
        max_tasks_per_session=max_tasks_per_session,
        max_consecutive_errors=max_consecutive_errors,
        max_fidelity_review_cycles_per_phase=max_fidelity_review_cycles_per_phase,
    )
    return config


class TestGetConfigAutonomy:
    """Tests for get-config action with autonomy section support."""

    def _call(self, config, **kwargs):
        from foundry_mcp.tools.unified.environment import _handle_get_config

        return _handle_get_config(config=config, **kwargs)

    def test_returns_defaults_when_no_autonomy_config(self, tmp_path):
        """get-config returns dataclass defaults when no autonomy config is set."""
        toml_file = tmp_path / ".foundry-mcp.toml"
        toml_file.write_text("[workspace]\nspecs_dir = './specs'\n")
        config = _make_config_with_autonomy()

        result = self._call(config, sections=["autonomy"], path=str(toml_file))

        assert result["success"] is True
        autonomy = result["data"]["sections"]["autonomy"]
        assert autonomy["posture"]["profile"] is None
        assert autonomy["security"]["role"] == "maintainer"
        assert autonomy["security"]["allow_lock_bypass"] is False
        assert autonomy["security"]["allow_gate_waiver"] is False
        assert autonomy["security"]["enforce_required_phase_gates"] is True
        assert autonomy["session_defaults"]["gate_policy"] == "strict"
        assert autonomy["session_defaults"]["max_tasks_per_session"] == 100
        assert autonomy["session_defaults"]["max_consecutive_errors"] == 3

    def test_returns_posture_profile_when_set(self, tmp_path):
        """get-config returns posture profile when configured."""
        toml_file = tmp_path / ".foundry-mcp.toml"
        toml_file.write_text("[workspace]\nspecs_dir = './specs'\n")
        config = _make_config_with_autonomy(profile="unattended")

        result = self._call(config, sections=["autonomy"], path=str(toml_file))

        assert result["success"] is True
        assert result["data"]["sections"]["autonomy"]["posture"]["profile"] == "unattended"

    def test_returns_merged_security_and_session_defaults(self, tmp_path):
        """get-config returns full security + session defaults from config."""
        toml_file = tmp_path / ".foundry-mcp.toml"
        toml_file.write_text("[workspace]\nspecs_dir = './specs'\n")
        config = _make_config_with_autonomy(
            profile="supervised",
            role="maintainer",
            allow_lock_bypass=True,
            allow_gate_waiver=True,
            gate_policy="strict",
            stop_on_phase_completion=True,
            max_consecutive_errors=5,
        )

        result = self._call(config, sections=["autonomy"], path=str(toml_file))

        assert result["success"] is True
        autonomy = result["data"]["sections"]["autonomy"]
        assert autonomy["security"]["allow_lock_bypass"] is True
        assert autonomy["security"]["allow_gate_waiver"] is True
        assert autonomy["session_defaults"]["stop_on_phase_completion"] is True
        assert autonomy["session_defaults"]["max_consecutive_errors"] == 5

    def test_rejects_implement_as_unsupported(self, tmp_path):
        """get-config rejects 'implement' as an unsupported section."""
        toml_file = tmp_path / ".foundry-mcp.toml"
        toml_file.write_text("[workspace]\nspecs_dir = './specs'\n")
        config = _make_config_with_autonomy()

        result = self._call(config, sections=["implement"], path=str(toml_file))

        assert result["success"] is False
        assert "Unsupported sections: implement" in result["error"]

    def test_key_parameter_works_for_autonomy_sub_keys(self, tmp_path):
        """get-config key parameter extracts autonomy sub-keys."""
        toml_file = tmp_path / ".foundry-mcp.toml"
        toml_file.write_text("[workspace]\nspecs_dir = './specs'\n")
        config = _make_config_with_autonomy(profile="debug")

        result = self._call(
            config, sections=["autonomy"], key="posture", path=str(toml_file)
        )

        assert result["success"] is True
        autonomy = result["data"]["sections"]["autonomy"]
        assert autonomy == {"posture": {"profile": "debug"}}
        # Should not include security or session_defaults
        assert "security" not in autonomy
        assert "session_defaults" not in autonomy

    def test_key_parameter_not_found_returns_error(self, tmp_path):
        """get-config returns error for unknown key within autonomy."""
        toml_file = tmp_path / ".foundry-mcp.toml"
        toml_file.write_text("[workspace]\nspecs_dir = './specs'\n")
        config = _make_config_with_autonomy()

        result = self._call(
            config, sections=["autonomy"], key="nonexistent", path=str(toml_file)
        )

        assert result["success"] is False
        assert "nonexistent" in result["error"]

    def test_returns_both_autonomy_and_git(self, tmp_path):
        """get-config returns both sections when requested."""
        toml_file = tmp_path / ".foundry-mcp.toml"
        toml_file.write_text(
            "[workspace]\nspecs_dir = './specs'\n\n"
            "[git]\nauto_commit = true\n"
        )
        config = _make_config_with_autonomy(profile="unattended")

        result = self._call(
            config, sections=["autonomy", "git"], path=str(toml_file)
        )

        assert result["success"] is True
        sections = result["data"]["sections"]
        assert "autonomy" in sections
        assert "git" in sections
        assert sections["autonomy"]["posture"]["profile"] == "unattended"
        assert sections["git"]["auto_commit"] is True

    def test_default_sections_returns_autonomy_and_git(self, tmp_path):
        """get-config with no sections param returns both autonomy and git."""
        toml_file = tmp_path / ".foundry-mcp.toml"
        toml_file.write_text("[workspace]\nspecs_dir = './specs'\n")
        config = _make_config_with_autonomy()

        result = self._call(config, path=str(toml_file))

        assert result["success"] is True
        sections = result["data"]["sections"]
        assert "autonomy" in sections
        assert "git" in sections
        assert "implement" not in sections

    # -- Phase 3 backward compatibility tests --

    def test_old_implement_section_in_toml_is_silently_ignored(self, tmp_path):
        """TOML with old [implement] section is silently ignored — no crash or error."""
        toml_file = tmp_path / ".foundry-mcp.toml"
        toml_file.write_text(
            "[workspace]\nspecs_dir = './specs'\n\n"
            "[implement]\ndelegate = true\nparallel = true\n\n"
            "[git]\nauto_commit = true\n"
        )
        config = _make_config_with_autonomy(profile="unattended")

        # Default sections (autonomy + git) — should NOT include implement
        result = self._call(config, path=str(toml_file))

        assert result["success"] is True
        sections = result["data"]["sections"]
        assert "autonomy" in sections
        assert "git" in sections
        assert "implement" not in sections
        # Autonomy comes from config object, not TOML
        assert sections["autonomy"]["posture"]["profile"] == "unattended"
        # Git still read from TOML correctly
        assert sections["git"]["auto_commit"] is True

    def test_no_autonomy_config_returns_safe_defaults(self, tmp_path):
        """No autonomy config at all returns safe interactive defaults."""
        toml_file = tmp_path / ".foundry-mcp.toml"
        toml_file.write_text("[workspace]\nspecs_dir = './specs'\n")
        # Config with all defaults (no profile, standard security)
        config = _make_config_with_autonomy()

        result = self._call(config, path=str(toml_file))

        assert result["success"] is True
        autonomy = result["data"]["sections"]["autonomy"]
        # Safe defaults for interactive behavior
        assert autonomy["posture"]["profile"] is None
        assert autonomy["security"]["role"] == "maintainer"
        assert autonomy["security"]["allow_lock_bypass"] is False
        assert autonomy["security"]["allow_gate_waiver"] is False
        assert autonomy["security"]["enforce_required_phase_gates"] is True
        assert autonomy["session_defaults"]["gate_policy"] == "strict"
        assert autonomy["session_defaults"]["stop_on_phase_completion"] is False
        assert autonomy["session_defaults"]["auto_retry_fidelity_gate"] is True
        assert autonomy["session_defaults"]["max_tasks_per_session"] == 100
        assert autonomy["session_defaults"]["max_consecutive_errors"] == 3
        assert autonomy["session_defaults"]["max_fidelity_review_cycles_per_phase"] == 3

    def test_posture_overrides_applied_correctly(self, tmp_path):
        """Explicit autonomy overrides take priority over defaults."""
        toml_file = tmp_path / ".foundry-mcp.toml"
        toml_file.write_text("[workspace]\nspecs_dir = './specs'\n")
        config = _make_config_with_autonomy(
            profile="unattended",
            role="maintainer",
            allow_lock_bypass=False,
            allow_gate_waiver=False,
            enforce_required_phase_gates=True,
            gate_policy="strict",
            stop_on_phase_completion=True,
            auto_retry_fidelity_gate=True,
            max_tasks_per_session=50,
            max_consecutive_errors=5,
            max_fidelity_review_cycles_per_phase=5,
        )

        result = self._call(config, sections=["autonomy"], path=str(toml_file))

        assert result["success"] is True
        autonomy = result["data"]["sections"]["autonomy"]
        assert autonomy["posture"]["profile"] == "unattended"
        assert autonomy["session_defaults"]["max_tasks_per_session"] == 50
        assert autonomy["session_defaults"]["max_consecutive_errors"] == 5
        assert autonomy["session_defaults"]["stop_on_phase_completion"] is True
        assert autonomy["session_defaults"]["max_fidelity_review_cycles_per_phase"] == 5
