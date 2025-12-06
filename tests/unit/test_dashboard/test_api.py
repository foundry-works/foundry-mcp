"""Unit tests for dashboard API endpoints."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone

# Skip all tests if aiohttp is not available
pytest.importorskip("aiohttp")


class TestApiHelpers:
    """Test API helper functions."""

    def test_json_response_success(self):
        """Test successful JSON response."""
        from foundry_mcp.dashboard.api import _json_response

        response = _json_response({"key": "value"})
        assert response.status == 200
        assert response.content_type == "application/json"

    def test_json_response_custom_status(self):
        """Test JSON response with custom status."""
        from foundry_mcp.dashboard.api import _json_response

        response = _json_response({"error": "not found"}, status=404)
        assert response.status == 404

    def test_error_response(self):
        """Test error response helper."""
        from foundry_mcp.dashboard.api import _error_response

        response = _error_response("Something went wrong", status=500)
        assert response.status == 500


class TestErrorsEndpoint:
    """Test error endpoints."""

    @pytest.mark.asyncio
    async def test_handle_errors_list_disabled(self):
        """Test errors list when error collection is disabled."""
        from foundry_mcp.dashboard.api import handle_errors_list

        mock_request = MagicMock()
        mock_request.query = {}

        with patch("foundry_mcp.dashboard.api._get_error_store", return_value=None):
            response = await handle_errors_list(mock_request)

        assert response.status == 200
        # Response contains disabled message

    @pytest.mark.asyncio
    async def test_handle_errors_list_with_store(self):
        """Test errors list with active store."""
        from foundry_mcp.dashboard.api import handle_errors_list

        mock_request = MagicMock()
        mock_request.query = {"limit": "10"}

        # Create mock record
        mock_record = MagicMock()
        mock_record.to_dict.return_value = {
            "id": "err-123",
            "timestamp": "2024-01-01T00:00:00Z",
            "tool_name": "test-tool",
            "error_code": "TEST_ERROR",
        }

        mock_store = MagicMock()
        mock_store.query.return_value = [mock_record]
        mock_store.count.return_value = 1

        with patch("foundry_mcp.dashboard.api._get_error_store", return_value=mock_store):
            response = await handle_errors_list(mock_request)

        assert response.status == 200
        mock_store.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_error_get_not_found(self):
        """Test getting non-existent error."""
        from foundry_mcp.dashboard.api import handle_error_get

        mock_request = MagicMock()
        mock_request.match_info = {"error_id": "nonexistent"}

        mock_store = MagicMock()
        mock_store.get.return_value = None

        with patch("foundry_mcp.dashboard.api._get_error_store", return_value=mock_store):
            response = await handle_error_get(mock_request)

        assert response.status == 404

    @pytest.mark.asyncio
    async def test_handle_errors_stats(self):
        """Test error stats endpoint."""
        from foundry_mcp.dashboard.api import handle_errors_stats

        mock_request = MagicMock()

        mock_store = MagicMock()
        mock_store.get_stats.return_value = {
            "total_errors": 100,
            "unique_patterns": 5,
            "by_tool": {"test-tool": 50},
        }

        with patch("foundry_mcp.dashboard.api._get_error_store", return_value=mock_store):
            response = await handle_errors_stats(mock_request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_handle_errors_patterns(self):
        """Test error patterns endpoint."""
        from foundry_mcp.dashboard.api import handle_errors_patterns

        mock_request = MagicMock()
        mock_request.query = {"min_count": "3"}

        mock_store = MagicMock()
        mock_store.get_patterns.return_value = [
            {"fingerprint": "fp-123", "count": 10},
            {"fingerprint": "fp-456", "count": 5},
        ]

        with patch("foundry_mcp.dashboard.api._get_error_store", return_value=mock_store):
            response = await handle_errors_patterns(mock_request)

        assert response.status == 200


class TestMetricsEndpoint:
    """Test metrics endpoints."""

    @pytest.mark.asyncio
    async def test_handle_metrics_list_disabled(self):
        """Test metrics list when persistence is disabled."""
        from foundry_mcp.dashboard.api import handle_metrics_list

        mock_request = MagicMock()

        with patch("foundry_mcp.dashboard.api._get_metrics_store", return_value=None):
            response = await handle_metrics_list(mock_request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_handle_metrics_list_with_store(self):
        """Test metrics list with active store."""
        from foundry_mcp.dashboard.api import handle_metrics_list

        mock_request = MagicMock()

        mock_store = MagicMock()
        mock_store.list_metrics.return_value = [
            {"name": "tool_duration_ms", "count": 100},
            {"name": "tool_invocations", "count": 50},
        ]

        with patch("foundry_mcp.dashboard.api._get_metrics_store", return_value=mock_store):
            response = await handle_metrics_list(mock_request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_handle_metrics_summary(self):
        """Test metrics summary endpoint."""
        from foundry_mcp.dashboard.api import handle_metrics_summary

        mock_request = MagicMock()
        mock_request.match_info = {"metric_name": "tool_duration_ms"}
        mock_request.query = {}

        mock_store = MagicMock()
        mock_store.get_summary.return_value = {
            "min": 10,
            "max": 500,
            "avg": 125,
            "count": 100,
        }

        with patch("foundry_mcp.dashboard.api._get_metrics_store", return_value=mock_store):
            response = await handle_metrics_summary(mock_request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_handle_metrics_timeseries(self):
        """Test metrics timeseries endpoint."""
        from foundry_mcp.dashboard.api import handle_metrics_timeseries

        mock_request = MagicMock()
        mock_request.match_info = {"metric_name": "tool_duration_ms"}
        mock_request.query = {}

        mock_record = MagicMock()
        mock_record.timestamp = "2024-01-01T00:00:00Z"
        mock_record.value = 100.0
        mock_record.labels = {}

        mock_store = MagicMock()
        mock_store.query.return_value = [mock_record]

        with patch("foundry_mcp.dashboard.api._get_metrics_store", return_value=mock_store):
            response = await handle_metrics_timeseries(mock_request)

        assert response.status == 200


class TestHealthEndpoint:
    """Test health endpoint."""

    @pytest.mark.asyncio
    async def test_handle_health_success(self):
        """Test health endpoint with healthy status."""
        from foundry_mcp.dashboard.api import handle_health

        mock_request = MagicMock()

        mock_result = MagicMock()
        mock_result.status.value = "healthy"
        mock_result.is_healthy = True
        mock_result.is_degraded = False
        mock_result.dependencies = {}

        mock_manager = MagicMock()
        mock_manager.check_health.return_value = mock_result

        with patch("foundry_mcp.core.health.get_health_manager", return_value=mock_manager):
            response = await handle_health(mock_request)

        assert response.status == 200


class TestProvidersEndpoint:
    """Test providers endpoint."""

    @pytest.mark.asyncio
    async def test_handle_providers_list(self):
        """Test providers list endpoint."""
        from foundry_mcp.dashboard.api import handle_providers_list

        mock_request = MagicMock()

        with patch("foundry_mcp.core.providers.available_providers", return_value=["gemini"]):
            with patch(
                "foundry_mcp.core.providers.get_provider_metadata",
                return_value={
                    "description": "Google Gemini",
                    "models": ["gemini-pro"],
                },
            ):
                response = await handle_providers_list(mock_request)

        assert response.status == 200


class TestConfigEndpoint:
    """Test config endpoint."""

    @pytest.mark.asyncio
    async def test_handle_config(self):
        """Test config endpoint."""
        from foundry_mcp.dashboard.api import handle_config

        mock_config = MagicMock()
        mock_config.refresh_interval_ms = 5000
        mock_config.host = "127.0.0.1"
        mock_config.port = 8080

        mock_request = MagicMock()
        mock_request.app = {"config": mock_config}

        response = await handle_config(mock_request)
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_handle_config_no_config(self):
        """Test config endpoint with no config."""
        from foundry_mcp.dashboard.api import handle_config

        mock_app = MagicMock()
        mock_app.get.return_value = None

        mock_request = MagicMock()
        mock_request.app = mock_app

        response = await handle_config(mock_request)
        assert response.status == 200


class TestRouteSetup:
    """Test API route setup."""

    def test_setup_api_routes(self):
        """Test that API routes are registered correctly."""
        from foundry_mcp.dashboard.api import setup_api_routes
        from aiohttp import web

        app = web.Application()
        setup_api_routes(app)

        # Check that routes were registered
        routes = {route.resource.canonical for route in app.router.routes()}

        assert "/api/errors" in routes
        assert "/api/errors/stats" in routes
        assert "/api/errors/patterns" in routes
        assert "/api/metrics" in routes
        assert "/api/health" in routes
        assert "/api/providers" in routes
        assert "/api/config" in routes
