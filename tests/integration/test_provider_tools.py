"""Integration tests for unified provider tool (`provider(action=...)`)."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from foundry_mcp.core.responses import error_response, success_response
from tests.conftest import extract_response_dict


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
        result = extract_response_dict(tools["provider"].fn(action="list"))
        assert "success" in result
        assert "meta" in result


class TestProviderStatusTool:
    def test_provider_status_requires_provider_id(self, mcp_server):
        tools = mcp_server._tool_manager._tools
        result = extract_response_dict(tools["provider"].fn(action="status"))
        assert result["success"] is False
        assert result["data"]["error_type"] in {"validation", "not_found"}
