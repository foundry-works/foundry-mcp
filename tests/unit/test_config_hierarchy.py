"""Tests for layered config file loading hierarchy."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from foundry_mcp.config import ServerConfig


class TestConfigHierarchy:
    """Test layered configuration loading (XDG -> home -> project -> env)."""

    @pytest.fixture
    def xdg_config_content(self):
        """XDG config directory content (system-wide user defaults)."""
        return """
[logging]
level = "NOTSET"
structured = true

[tools]
disabled_tools = ["environment"]
"""

    @pytest.fixture
    def home_config_content(self):
        """Home directory config content (user defaults)."""
        return """
[logging]
level = "DEBUG"
structured = false

[tools]
disabled_tools = ["health"]
"""

    @pytest.fixture
    def project_config_content(self):
        """Project directory config content (project overrides)."""
        return """
[logging]
level = "WARNING"

[workspace]
specs_dir = "./my-specs"
"""

    @pytest.fixture
    def legacy_config_content(self):
        """Legacy .foundry-mcp.toml content."""
        return """
[logging]
level = "ERROR"

[workspace]
specs_dir = "./legacy-specs"
"""

    def test_home_config_loaded_as_base_layer(self, tmp_path, home_config_content):
        """Home config is loaded as the base configuration layer."""
        home_config = tmp_path / ".foundry-mcp.toml"
        home_config.write_text(home_config_content)

        with patch.object(Path, "home", return_value=tmp_path):
            with patch.dict(os.environ, {}, clear=True):
                # Run from a directory without any project config
                original_cwd = os.getcwd()
                os.chdir(tmp_path)
                try:
                    config = ServerConfig.from_env()
                    assert config.log_level == "DEBUG"
                    assert config.structured_logging is False
                    assert config.disabled_tools == ["health"]
                finally:
                    os.chdir(original_cwd)

    def test_project_config_overrides_home_config(
        self, tmp_path, home_config_content, project_config_content
    ):
        """Project config values override home config values."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        home_config = home_dir / ".foundry-mcp.toml"
        home_config.write_text(home_config_content)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_config = project_dir / "foundry-mcp.toml"
        project_config.write_text(project_config_content)

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(os.environ, {}, clear=True):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    # Project overrides home
                    assert config.log_level == "WARNING"
                    # Home values preserved for unset project values
                    assert config.structured_logging is False
                    assert config.disabled_tools == ["health"]
                    # Project-specific value
                    assert config.specs_dir == Path("./my-specs")
                finally:
                    os.chdir(original_cwd)

    def test_env_vars_override_both_toml_layers(
        self, tmp_path, home_config_content, project_config_content
    ):
        """Environment variables override both home and project config."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        home_config = home_dir / ".foundry-mcp.toml"
        home_config.write_text(home_config_content)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_config = project_dir / "foundry-mcp.toml"
        project_config.write_text(project_config_content)

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(
                os.environ,
                {
                    "FOUNDRY_MCP_LOG_LEVEL": "INFO",
                    "FOUNDRY_MCP_SPECS_DIR": "/env/specs",
                },
                clear=True,
            ):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    # Env vars override everything
                    assert config.log_level == "INFO"
                    assert config.specs_dir == Path("/env/specs")
                finally:
                    os.chdir(original_cwd)

    def test_explicit_config_file_skips_layered_loading(
        self, tmp_path, home_config_content, project_config_content
    ):
        """Explicit config_file parameter skips layered loading."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        home_config = home_dir / ".foundry-mcp.toml"
        home_config.write_text(home_config_content)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_config = project_dir / "foundry-mcp.toml"
        project_config.write_text(project_config_content)

        explicit_config = tmp_path / "explicit.toml"
        explicit_config.write_text("""
[logging]
level = "CRITICAL"
""")

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(os.environ, {}, clear=True):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env(config_file=str(explicit_config))
                    # Only explicit config loaded
                    assert config.log_level == "CRITICAL"
                    # Defaults used, not home or project values
                    assert config.structured_logging is True  # default
                    assert config.disabled_tools == []  # default
                finally:
                    os.chdir(original_cwd)

    def test_env_var_config_file_skips_layered_loading(
        self, tmp_path, home_config_content, project_config_content
    ):
        """FOUNDRY_MCP_CONFIG_FILE env var skips layered loading."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        home_config = home_dir / ".foundry-mcp.toml"
        home_config.write_text(home_config_content)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_config = project_dir / "foundry-mcp.toml"
        project_config.write_text(project_config_content)

        explicit_config = tmp_path / "explicit.toml"
        explicit_config.write_text("""
[logging]
level = "CRITICAL"
""")

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(
                os.environ,
                {"FOUNDRY_MCP_CONFIG_FILE": str(explicit_config)},
                clear=True,
            ):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    # Only explicit config loaded
                    assert config.log_level == "CRITICAL"
                    assert config.structured_logging is True  # default
                finally:
                    os.chdir(original_cwd)

    def test_missing_home_config_no_error(self, tmp_path, project_config_content):
        """Missing home config doesn't cause errors."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        # No home config file

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_config = project_dir / "foundry-mcp.toml"
        project_config.write_text(project_config_content)

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(os.environ, {}, clear=True):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    # Project config loaded, defaults for rest
                    assert config.log_level == "WARNING"
                    assert config.structured_logging is True  # default
                finally:
                    os.chdir(original_cwd)

    def test_missing_project_config_uses_home_only(self, tmp_path, home_config_content):
        """Missing project config uses home config values only."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        home_config = home_dir / ".foundry-mcp.toml"
        home_config.write_text(home_config_content)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        # No project config file

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(os.environ, {}, clear=True):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    # Home config values used
                    assert config.log_level == "DEBUG"
                    assert config.structured_logging is False
                finally:
                    os.chdir(original_cwd)

    def test_neither_config_exists_uses_defaults(self, tmp_path):
        """When neither config exists, defaults are used."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(os.environ, {}, clear=True):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    # All defaults
                    assert config.log_level == "INFO"  # default
                    assert config.structured_logging is True  # default
                    assert config.disabled_tools == []  # default
                finally:
                    os.chdir(original_cwd)

    def test_legacy_config_loaded_as_fallback(
        self, tmp_path, home_config_content, legacy_config_content
    ):
        """Legacy .foundry-mcp.toml in project dir is loaded as fallback."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        home_config = home_dir / ".foundry-mcp.toml"
        home_config.write_text(home_config_content)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        # Only legacy config, no new config
        legacy_config = project_dir / ".foundry-mcp.toml"
        legacy_config.write_text(legacy_config_content)

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(os.environ, {}, clear=True):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    # Legacy config overrides home
                    assert config.log_level == "ERROR"
                    # Home values preserved for unset legacy values
                    assert config.structured_logging is False
                    # Legacy-specific value
                    assert config.specs_dir == Path("./legacy-specs")
                finally:
                    os.chdir(original_cwd)

    def test_new_config_preferred_over_legacy(
        self, tmp_path, home_config_content, project_config_content, legacy_config_content
    ):
        """New foundry-mcp.toml is preferred over legacy .foundry-mcp.toml."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        home_config = home_dir / ".foundry-mcp.toml"
        home_config.write_text(home_config_content)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        # Both new and legacy configs exist
        project_config = project_dir / "foundry-mcp.toml"
        project_config.write_text(project_config_content)
        legacy_config = project_dir / ".foundry-mcp.toml"
        legacy_config.write_text(legacy_config_content)

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(os.environ, {}, clear=True):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    # New config used, not legacy
                    assert config.log_level == "WARNING"  # from project
                    assert config.specs_dir == Path("./my-specs")  # from project
                    # Not legacy values
                    assert config.log_level != "ERROR"
                finally:
                    os.chdir(original_cwd)

    def test_partial_override_preserves_unset_values(self, tmp_path):
        """Partial project config preserves unset values from home."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        home_config = home_dir / ".foundry-mcp.toml"
        home_config.write_text("""
[logging]
level = "DEBUG"
structured = false

[tools]
disabled_tools = ["health", "error"]

[research]
default_timeout = 500.0
""")

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_config = project_dir / "foundry-mcp.toml"
        # Only override logging level
        project_config.write_text("""
[logging]
level = "INFO"
""")

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(os.environ, {}, clear=True):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    # Project override
                    assert config.log_level == "INFO"
                    # Home values preserved
                    assert config.structured_logging is False
                    assert config.research.default_timeout == 500.0
                    assert set(config.disabled_tools) == {"health", "error"}
                finally:
                    os.chdir(original_cwd)

    def test_xdg_config_loaded_as_base_layer(self, tmp_path, xdg_config_content):
        """XDG config (~/.config/foundry-mcp/config.toml) is loaded as the base layer."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        xdg_config_dir = home_dir / ".config" / "foundry-mcp"
        xdg_config_dir.mkdir(parents=True)
        xdg_config = xdg_config_dir / "config.toml"
        xdg_config.write_text(xdg_config_content)

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(os.environ, {}, clear=True):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    assert config.log_level == "NOTSET"
                    assert config.structured_logging is True
                    assert config.disabled_tools == ["environment"]
                finally:
                    os.chdir(original_cwd)

    def test_home_config_overrides_xdg_config(
        self, tmp_path, xdg_config_content, home_config_content
    ):
        """Home config (~/.foundry-mcp.toml) overrides XDG config."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()

        # XDG config (lowest priority of user configs)
        xdg_config_dir = home_dir / ".config" / "foundry-mcp"
        xdg_config_dir.mkdir(parents=True)
        xdg_config = xdg_config_dir / "config.toml"
        xdg_config.write_text(xdg_config_content)

        # Home config (overrides XDG)
        home_config = home_dir / ".foundry-mcp.toml"
        home_config.write_text(home_config_content)

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(os.environ, {}, clear=True):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    # Home overrides XDG
                    assert config.log_level == "DEBUG"
                    assert config.structured_logging is False
                    assert config.disabled_tools == ["health"]
                finally:
                    os.chdir(original_cwd)

    def test_xdg_config_home_env_var_respected(self, tmp_path, xdg_config_content):
        """XDG_CONFIG_HOME environment variable is respected."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()

        # Custom XDG config location
        custom_xdg = tmp_path / "custom-config"
        custom_xdg.mkdir()
        xdg_config_dir = custom_xdg / "foundry-mcp"
        xdg_config_dir.mkdir()
        xdg_config = xdg_config_dir / "config.toml"
        xdg_config.write_text(xdg_config_content)

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(custom_xdg)}, clear=True):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    # XDG config loaded from custom location
                    assert config.log_level == "NOTSET"
                    assert config.disabled_tools == ["environment"]
                finally:
                    os.chdir(original_cwd)

    def test_full_hierarchy_xdg_home_project_env(
        self, tmp_path, xdg_config_content, home_config_content, project_config_content
    ):
        """Full config hierarchy: XDG < home < project < env."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()

        # XDG config (base layer)
        xdg_config_dir = home_dir / ".config" / "foundry-mcp"
        xdg_config_dir.mkdir(parents=True)
        xdg_config = xdg_config_dir / "config.toml"
        xdg_config.write_text(xdg_config_content)

        # Home config (overrides XDG)
        home_config = home_dir / ".foundry-mcp.toml"
        home_config.write_text(home_config_content)

        # Project config (overrides home)
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_config = project_dir / "foundry-mcp.toml"
        project_config.write_text(project_config_content)

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(
                os.environ,
                {"FOUNDRY_MCP_LOG_LEVEL": "CRITICAL"},
                clear=True,
            ):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    # Env var overrides everything
                    assert config.log_level == "CRITICAL"
                    # Project overrides home
                    assert config.specs_dir == Path("./my-specs")
                    # Home overrides XDG (structured=false from home)
                    assert config.structured_logging is False
                    # Home value (disabled_tools from home, not XDG)
                    assert config.disabled_tools == ["health"]
                finally:
                    os.chdir(original_cwd)

    def test_autonomy_security_rate_limit_loaded_from_toml(self, tmp_path):
        """Autonomy security rate-limit fields are loaded from TOML."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_config = project_dir / "foundry-mcp.toml"
        project_config.write_text("""
[autonomy_security]
rate_limit_max_consecutive_denials = 7
rate_limit_denial_window_seconds = 45
rate_limit_retry_after_seconds = 12
""")

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(os.environ, {}, clear=True):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    assert config.autonomy_security.rate_limit_max_consecutive_denials == 7
                    assert config.autonomy_security.rate_limit_denial_window_seconds == 45
                    assert config.autonomy_security.rate_limit_retry_after_seconds == 12
                finally:
                    os.chdir(original_cwd)

    def test_autonomy_security_rate_limit_env_overrides(self, tmp_path):
        """Environment variables override autonomy security rate-limit values."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_config = project_dir / "foundry-mcp.toml"
        project_config.write_text("""
[autonomy_security]
rate_limit_max_consecutive_denials = 7
rate_limit_denial_window_seconds = 45
rate_limit_retry_after_seconds = 12
""")

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(
                os.environ,
                {
                    "FOUNDRY_MCP_AUTONOMY_SECURITY_RATE_LIMIT_MAX_CONSECUTIVE_DENIALS": "3",
                    "FOUNDRY_MCP_AUTONOMY_SECURITY_RATE_LIMIT_DENIAL_WINDOW_SECONDS": "22",
                    "FOUNDRY_MCP_AUTONOMY_SECURITY_RATE_LIMIT_RETRY_AFTER_SECONDS": "9",
                },
                clear=True,
            ):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    assert config.autonomy_security.rate_limit_max_consecutive_denials == 3
                    assert config.autonomy_security.rate_limit_denial_window_seconds == 22
                    assert config.autonomy_security.rate_limit_retry_after_seconds == 9
                finally:
                    os.chdir(original_cwd)

    def test_feature_flags_loaded_from_toml(self, tmp_path):
        """[feature_flags] table is parsed from TOML config."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_config = project_dir / "foundry-mcp.toml"
        project_config.write_text("""
[feature_flags]
autonomy_sessions = true
autonomy_fidelity_gates = false
""")

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(os.environ, {}, clear=True):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    assert config.feature_flags["autonomy_sessions"] is True
                    assert config.feature_flags["autonomy_fidelity_gates"] is False
                finally:
                    os.chdir(original_cwd)

    def test_feature_flags_env_overrides_toml(self, tmp_path):
        """FOUNDRY_MCP_FEATURE_FLAGS overrides TOML feature flags."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_config = project_dir / "foundry-mcp.toml"
        project_config.write_text("""
[feature_flags]
autonomy_sessions = false
autonomy_fidelity_gates = false
""")

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(
                os.environ,
                {
                    "FOUNDRY_MCP_FEATURE_FLAGS": "autonomy_sessions=true,autonomy_fidelity_gates=true"
                },
                clear=True,
            ):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    assert config.feature_flags["autonomy_sessions"] is True
                    assert config.feature_flags["autonomy_fidelity_gates"] is True
                finally:
                    os.chdir(original_cwd)

    def test_feature_flag_per_flag_env_overrides_bulk_env(self, tmp_path):
        """FOUNDRY_MCP_FEATURE_FLAG_<NAME> takes precedence over bulk env."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(
                os.environ,
                {
                    "FOUNDRY_MCP_FEATURE_FLAGS": "autonomy_sessions=true,autonomy_fidelity_gates=true",
                    "FOUNDRY_MCP_FEATURE_FLAG_AUTONOMY_FIDELITY_GATES": "false",
                },
                clear=True,
            ):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    assert config.feature_flags["autonomy_sessions"] is True
                    assert config.feature_flags["autonomy_fidelity_gates"] is False
                finally:
                    os.chdir(original_cwd)

    def test_feature_flag_dependency_warning(self, tmp_path):
        """Startup warnings include dependency issues for inconsistent flags."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(
                os.environ,
                {"FOUNDRY_MCP_FEATURE_FLAGS": "autonomy_fidelity_gates=true"},
                clear=True,
            ):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    assert any(
                        "autonomy_fidelity_gates" in warning
                        and "autonomy_sessions" in warning
                        for warning in config.startup_warnings
                    )
                finally:
                    os.chdir(original_cwd)

    def test_autonomy_posture_profile_expands_defaults(self, tmp_path):
        """[autonomy_posture] profile applies role/security/session defaults."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_config = project_dir / "foundry-mcp.toml"
        project_config.write_text("""
[autonomy_posture]
profile = "unattended"
""")

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(os.environ, {}, clear=True):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    assert config.autonomy_posture.profile == "unattended"
                    assert config.autonomy_security.role == "autonomy_runner"
                    assert config.autonomy_security.allow_lock_bypass is False
                    assert config.autonomy_security.allow_gate_waiver is False
                    assert config.autonomy_security.enforce_required_phase_gates is True
                    assert config.autonomy_session_defaults.gate_policy == "strict"
                    assert config.autonomy_session_defaults.stop_on_phase_completion is True
                finally:
                    os.chdir(original_cwd)

    def test_autonomy_posture_supervised_profile_expands_defaults(self, tmp_path):
        """[autonomy_posture] supervised profile keeps guardrails with escape hatches enabled."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_config = project_dir / "foundry-mcp.toml"
        project_config.write_text("""
[autonomy_posture]
profile = "supervised"
""")

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(os.environ, {}, clear=True):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    assert config.autonomy_posture.profile == "supervised"
                    assert config.autonomy_security.role == "maintainer"
                    assert config.autonomy_security.allow_lock_bypass is True
                    assert config.autonomy_security.allow_gate_waiver is True
                    assert config.autonomy_security.enforce_required_phase_gates is True
                    assert config.autonomy_session_defaults.gate_policy == "strict"
                    assert config.autonomy_session_defaults.stop_on_phase_completion is True
                finally:
                    os.chdir(original_cwd)

    def test_autonomy_posture_allows_direct_override_with_unsafe_warning(self, tmp_path):
        """Direct security/session defaults can override profile and trigger safety warnings."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_config = project_dir / "foundry-mcp.toml"
        project_config.write_text("""
[autonomy_posture]
profile = "unattended"

[autonomy_security]
role = "maintainer"
allow_lock_bypass = true
allow_gate_waiver = true
enforce_required_phase_gates = false

[autonomy_session_defaults]
gate_policy = "manual"
stop_on_phase_completion = false
""")

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(os.environ, {}, clear=True):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    assert config.autonomy_posture.profile == "unattended"
                    assert config.autonomy_security.role == "maintainer"
                    assert config.autonomy_security.allow_lock_bypass is True
                    assert config.autonomy_security.allow_gate_waiver is True
                    assert config.autonomy_security.enforce_required_phase_gates is False
                    assert config.autonomy_session_defaults.gate_policy == "manual"
                    assert config.autonomy_session_defaults.stop_on_phase_completion is False
                    assert any(
                        "UNSAFE unattended posture configuration" in warning
                        for warning in config.startup_warnings
                    )
                finally:
                    os.chdir(original_cwd)

    def test_autonomy_posture_env_applies_debug_defaults(self, tmp_path):
        """FOUNDRY_MCP_AUTONOMY_POSTURE applies debug profile defaults."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with patch.object(Path, "home", return_value=home_dir):
            with patch.dict(
                os.environ,
                {"FOUNDRY_MCP_AUTONOMY_POSTURE": "debug"},
                clear=True,
            ):
                original_cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    config = ServerConfig.from_env()
                    assert config.autonomy_posture.profile == "debug"
                    assert config.autonomy_security.role == "maintainer"
                    assert config.autonomy_security.allow_lock_bypass is True
                    assert config.autonomy_security.allow_gate_waiver is True
                    assert config.autonomy_session_defaults.gate_policy == "manual"
                    assert config.autonomy_session_defaults.stop_on_phase_completion is False
                finally:
                    os.chdir(original_cwd)
