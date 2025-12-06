"""Unit tests for dashboard server module."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Skip all tests if aiohttp is not available
pytest.importorskip("aiohttp")


class TestServerHelpers:
    """Test server helper functions."""

    def test_get_static_dir(self):
        """Test static directory path resolution."""
        from foundry_mcp.dashboard.server import get_static_dir

        static_dir = get_static_dir()
        assert isinstance(static_dir, Path)
        assert static_dir.name == "static"

    def test_get_dashboard_url_default(self):
        """Test default dashboard URL."""
        from foundry_mcp.dashboard.server import get_dashboard_url

        url = get_dashboard_url()
        assert url == "http://127.0.0.1:8080"

    def test_get_dashboard_url_with_config(self):
        """Test dashboard URL with custom config."""
        from foundry_mcp.dashboard.server import get_dashboard_url
        from foundry_mcp.config import DashboardConfig

        config = DashboardConfig(host="0.0.0.0", port=9090)
        url = get_dashboard_url(config)
        assert url == "http://0.0.0.0:9090"

    def test_is_dashboard_running_initially_false(self):
        """Test that dashboard is not running initially."""
        from foundry_mcp.dashboard.server import is_dashboard_running, _server_running

        # Reset state
        _server_running.clear()

        assert is_dashboard_running() is False


class TestStartDashboard:
    """Test dashboard start functionality."""

    def test_start_dashboard_disabled(self):
        """Test starting dashboard when disabled."""
        from foundry_mcp.dashboard.server import start_dashboard
        from foundry_mcp.config import DashboardConfig

        config = DashboardConfig(enabled=False)
        result = start_dashboard(config)
        assert result is False

    def test_start_dashboard_already_running(self):
        """Test starting dashboard when already running."""
        from foundry_mcp.dashboard.server import start_dashboard, _server_running
        from foundry_mcp.config import DashboardConfig
        from unittest.mock import MagicMock
        import foundry_mcp.dashboard.server as server_module

        # Simulate already running
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        original_thread = server_module._server_thread
        server_module._server_thread = mock_thread

        try:
            config = DashboardConfig(enabled=True)
            result = start_dashboard(config)
            assert result is True
        finally:
            server_module._server_thread = original_thread


class TestStopDashboard:
    """Test dashboard stop functionality."""

    def test_stop_dashboard_not_running(self):
        """Test stopping dashboard when not running."""
        from foundry_mcp.dashboard.server import stop_dashboard
        import foundry_mcp.dashboard.server as server_module

        # Ensure no server thread
        server_module._server_thread = None
        server_module._current_config = None

        # Should not raise
        stop_dashboard()

    def test_stop_dashboard_running(self):
        """Test stopping running dashboard."""
        from foundry_mcp.dashboard.server import stop_dashboard, _server_running
        import foundry_mcp.dashboard.server as server_module

        # Create mock thread
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False

        server_module._server_thread = mock_thread
        server_module._current_config = MagicMock()
        _server_running.set()

        stop_dashboard()

        assert server_module._server_thread is None
        assert server_module._current_config is None
        assert not _server_running.is_set()


class TestAppCreation:
    """Test app creation."""

    @pytest.mark.asyncio
    async def test_create_app(self):
        """Test creating aiohttp application."""
        from foundry_mcp.dashboard.server import _create_app
        from foundry_mcp.config import DashboardConfig

        config = DashboardConfig(enabled=True)
        app = await _create_app(config)

        assert app is not None
        assert "config" in app

        # Check that routes are registered
        routes = list(app.router.routes())
        assert len(routes) > 0


class TestModuleExports:
    """Test module exports."""

    def test_module_exports(self):
        """Test that module exports expected functions."""
        from foundry_mcp.dashboard import (
            start_dashboard,
            stop_dashboard,
            get_dashboard_url,
        )

        assert callable(start_dashboard)
        assert callable(stop_dashboard)
        assert callable(get_dashboard_url)
