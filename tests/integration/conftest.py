"""Shared fixtures for integration tests."""

import pytest

from foundry_mcp.config.server import ServerConfig
from foundry_mcp.server import create_server


@pytest.fixture
def test_specs_dir(tmp_path):
    """Create a minimal test specs directory with standard status subdirectories."""
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    for d in ("active", "pending", "completed", "archived"):
        (specs_dir / d).mkdir()
    return specs_dir


@pytest.fixture
def test_config(test_specs_dir):
    """Create a standard test server configuration."""
    return ServerConfig(
        server_name="foundry-mcp-test",
        server_version="0.1.0",
        specs_dir=test_specs_dir,
        log_level="WARNING",
    )


@pytest.fixture
def mcp_server(test_config):
    """Create a test MCP server instance."""
    return create_server(test_config)
