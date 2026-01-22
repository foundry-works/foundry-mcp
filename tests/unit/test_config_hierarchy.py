"""Tests for layered config file loading hierarchy."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from foundry_mcp.config import ServerConfig


class TestConfigHierarchy:
    """Test layered configuration loading (home -> project -> env)."""

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
