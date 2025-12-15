"""Integration tests for unified provider tool (`provider(action=...)`)."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import error_response, success_response
from foundry_mcp.server import create_server


@pytest.fixture
def test_specs_dir(tmp_path):
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    (specs_dir / "active").mkdir()
    (specs_dir / "pending").mkdir()
    (specs_dir / "completed").mkdir()
    (specs_dir / "archived").mkdir()
    return specs_dir


@pytest.fixture
def test_config(test_specs_dir):
    return ServerConfig(
        server_name="foundry-mcp-test",
        server_version="0.1.0",
        specs_dir=test_specs_dir,
        log_level="WARNING",
    )


@pytest.fixture
def mcp_server(test_config):
    return create_server(test_config)


class TestProviderToolResponseEnvelopes:
    def test_success_response_has_required_fields(self):
        result = asdict(success_response(data={"providers": [], "available_count": 0}))
        assert result["success"] is True
        assert result["meta"]["version"] == "response-v2"

    def test_error_response_has_required_fields(self):
        result = asdict(
            error_response(
                "Provider not found",
                error_code="NOT_FOUND",
                error_type="not_found",
            )
        )
        assert result["success"] is False
        assert result["data"]["error_code"] == "NOT_FOUND"
        assert result["data"]["error_type"] == "not_found"


class TestProviderToolRegistration:
    def test_provider_tool_registered(self, mcp_server):
        tools = mcp_server._tool_manager._tools
        assert "provider" in tools
        assert callable(tools["provider"].fn)


class TestProviderListTool:
    def test_provider_list_returns_envelope(self, mcp_server):
        tools = mcp_server._tool_manager._tools
        result = tools["provider"].fn(action="list")
        assert "success" in result
        assert "meta" in result


class TestProviderStatusTool:
    def test_provider_status_requires_provider_id(self, mcp_server):
        tools = mcp_server._tool_manager._tools
        result = tools["provider"].fn(action="status")
        assert result["success"] is False
        assert result["data"]["error_type"] in {"validation", "not_found"}
