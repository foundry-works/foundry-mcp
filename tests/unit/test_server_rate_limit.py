"""Tests for server authorization rate-limit initialization."""

from pathlib import Path
from unittest.mock import patch

from foundry_mcp.config.server import ServerConfig
from foundry_mcp.server import create_server


def _make_specs_dir(tmp_path: Path) -> Path:
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    (specs_dir / "active").mkdir()
    (specs_dir / "pending").mkdir()
    (specs_dir / "completed").mkdir()
    (specs_dir / "archived").mkdir()
    return specs_dir


def test_create_server_initializes_authorization_rate_limit_from_config(tmp_path):
    specs_dir = _make_specs_dir(tmp_path)
    config = ServerConfig(
        server_name="foundry-mcp-test",
        server_version="0.1.0",
        specs_dir=specs_dir,
        log_level="WARNING",
    )
    config.autonomy_security.rate_limit_max_consecutive_denials = 4
    config.autonomy_security.rate_limit_denial_window_seconds = 33
    config.autonomy_security.rate_limit_retry_after_seconds = 11

    with patch("foundry_mcp.server.get_rate_limit_tracker") as mock_get_tracker:
        create_server(config)

    assert mock_get_tracker.call_count == 1
    rate_config = mock_get_tracker.call_args.args[0]
    assert rate_config.max_consecutive_denials == 4
    assert rate_config.denial_window_seconds == 33
    assert rate_config.retry_after_seconds == 11
