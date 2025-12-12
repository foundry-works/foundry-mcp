"""
Integration tests for provider MCP tools.

Tests:
- Response envelope compliance (success/error structure)
- Tool registration and schema validation
- Provider listing behavior
- Provider status queries
- Provider execution with mock providers
- Error handling and edge cases
"""

import json
import pytest
from dataclasses import asdict
from unittest.mock import patch, MagicMock

from foundry_mcp.server import create_server
from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response
from foundry_mcp.core.feature_flags import get_flag_service


@pytest.fixture
def test_specs_dir(tmp_path):
    """Create a test specs directory structure."""
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    (specs_dir / "active").mkdir()
    (specs_dir / "pending").mkdir()
    (specs_dir / "completed").mkdir()
    (specs_dir / "archived").mkdir()
    return specs_dir


@pytest.fixture
def test_config(test_specs_dir):
    """Create a test server configuration."""
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


def _unified_manifest_enabled() -> bool:
    return get_flag_service().is_enabled("unified_manifest")


def _provider_tool(tools, action: str):
    if _unified_manifest_enabled():
        return tools["provider"], action

    mapping = {
        "list": "provider-list",
        "status": "provider-status",
        "execute": "provider-execute",
    }
    return tools[mapping[action]], None


def _call_provider(tools, action: str, **kwargs):
    tool, unified_action = _provider_tool(tools, action)
    if unified_action is None:
        return tool.fn(**kwargs)
    return tool.fn(action=unified_action, **kwargs)


class TestProviderToolResponseEnvelopes:
    """Integration tests for provider tool response envelope compliance."""

    def test_success_response_has_required_fields(self):
        """Test that success responses include required envelope fields."""
        result = asdict(success_response(data={"providers": [], "available_count": 0}))

        # Required fields per MCP best practices
        assert "success" in result
        assert result["success"] is True
        assert "data" in result
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"

    def test_error_response_has_required_fields(self):
        """Test that error responses include required envelope fields."""
        result = asdict(
            error_response(
                "Provider not found",
                error_code="NOT_FOUND",
                error_type="not_found",
            )
        )

        # Required fields per MCP best practices
        assert "success" in result
        assert result["success"] is False
        assert "error" in result
        assert "data" in result
        assert result["data"]["error_code"] == "NOT_FOUND"
        assert result["data"]["error_type"] == "not_found"
        assert "meta" in result

    def test_error_response_with_remediation(self):
        """Test that error responses can include remediation guidance."""
        result = asdict(
            error_response(
                "Provider unavailable",
                error_code="UNAVAILABLE",
                remediation="Use provider(action=list) to see available providers.",
            )
        )

        assert result["success"] is False
        assert "remediation" in result["data"]
        assert "provider(action=list)" in result["data"]["remediation"]


class TestProviderToolRegistration:
    """Integration tests for provider tool registration."""

    def test_provider_list_tool_registered(self, mcp_server):
        """Test that provider list entrypoint is registered."""
        tools = mcp_server._tool_manager._tools
        if _unified_manifest_enabled():
            assert "provider" in tools
            assert callable(tools["provider"].fn)
        else:
            assert "provider-list" in tools
            assert callable(tools["provider-list"].fn)

    def test_provider_status_tool_registered(self, mcp_server):
        """Test that provider status entrypoint is registered."""
        tools = mcp_server._tool_manager._tools
        if _unified_manifest_enabled():
            assert "provider" in tools
            assert callable(tools["provider"].fn)
        else:
            assert "provider-status" in tools
            assert callable(tools["provider-status"].fn)

    def test_provider_execute_tool_registered(self, mcp_server):
        """Test that provider execute entrypoint is registered."""
        tools = mcp_server._tool_manager._tools
        if _unified_manifest_enabled():
            assert "provider" in tools
            assert callable(tools["provider"].fn)
        else:
            assert "provider-execute" in tools
            assert callable(tools["provider-execute"].fn)


class TestProviderListTool:
    """Integration tests for provider-list tool."""

    def test_provider_list_returns_success_envelope(self, mcp_server):
        """Test provider-list returns proper success envelope."""
        tools = mcp_server._tool_manager._tools
        result = _call_provider(tools, "list")

        assert result["success"] is True
        assert "data" in result
        assert "providers" in result["data"]
        assert "available_count" in result["data"]
        assert "total_count" in result["data"]

    def test_provider_list_includes_expected_providers(self, mcp_server):
        """Test provider-list includes expected provider types."""
        tools = mcp_server._tool_manager._tools

        # With include_unavailable=True to see all registered providers
        result = _call_provider(tools, "list", include_unavailable=True)

        # Check structure even if providers are unavailable
        assert result["success"] is True
        assert isinstance(result["data"]["providers"], list)
        assert result["data"]["total_count"] >= 0

    def test_provider_list_filters_unavailable_by_default(self, mcp_server):
        """Test provider-list filters unavailable providers by default."""
        tools = mcp_server._tool_manager._tools

        # Default behavior (include_unavailable=False)
        result_filtered = _call_provider(tools, "list", include_unavailable=False)

        # All returned providers should be available
        for provider in result_filtered["data"]["providers"]:
            assert provider.get("available", False) is True

    def test_provider_list_with_include_unavailable(self, mcp_server):
        """Test provider-list with include_unavailable=True shows all providers."""
        tools = mcp_server._tool_manager._tools

        result_all = _call_provider(tools, "list", include_unavailable=True)
        result_filtered = _call_provider(tools, "list", include_unavailable=False)

        # Total with all should be >= filtered
        assert (
            result_all["data"]["total_count"]
            >= result_filtered["data"]["available_count"]
        )


class TestProviderStatusTool:
    """Integration tests for provider-status tool."""

    def test_provider_status_requires_provider_id(self, mcp_server):
        """Test provider status returns error when provider_id is missing."""
        tools = mcp_server._tool_manager._tools
        result = _call_provider(tools, "status", provider_id="")

        assert result["success"] is False
        assert result["data"]["error_code"] == "MISSING_REQUIRED"

    def test_provider_status_returns_not_found_for_unknown(self, mcp_server):
        """Test provider-status returns not found for unknown provider."""
        tools = mcp_server._tool_manager._tools
        result = _call_provider(tools, "status", provider_id="nonexistent-provider-xyz")

        assert result["success"] is False
        assert result["data"]["error_code"] == "NOT_FOUND"
        assert "remediation" in result["data"]

    def test_provider_status_returns_success_for_known_provider(self, mcp_server):
        """Test provider-status returns success envelope for registered provider."""
        tools = mcp_server._tool_manager._tools

        # Test with gemini provider (registered by default)
        result = _call_provider(tools, "status", provider_id="gemini")

        assert result["success"] is True
        assert "data" in result
        assert result["data"]["provider_id"] == "gemini"
        assert "available" in result["data"]
        # Metadata may be None if provider unavailable, but key should exist
        assert "metadata" in result["data"]
        assert "capabilities" in result["data"]
        assert "health" in result["data"]

    def test_provider_status_includes_metadata_when_available(self, mcp_server):
        """Test provider-status includes metadata for configured providers."""
        tools = mcp_server._tool_manager._tools

        # Test with a provider that has metadata registered
        result = _call_provider(tools, "status", provider_id="gemini")

        if result["success"] and result["data"]["metadata"]:
            metadata = result["data"]["metadata"]
            assert "name" in metadata
            assert "version" in metadata
            assert "default_model" in metadata
            assert "supported_models" in metadata


class TestProviderExecuteTool:
    """Integration tests for provider-execute tool."""

    def test_provider_execute_requires_provider_id(self, mcp_server):
        """Test provider-execute returns error when provider_id is missing."""
        tools = mcp_server._tool_manager._tools
        result = _call_provider(tools, "execute", provider_id="", prompt="Hello")

        assert result["success"] is False
        assert result["data"]["error_code"] == "MISSING_REQUIRED"

    def test_provider_execute_requires_prompt(self, mcp_server):
        """Test provider-execute returns error when prompt is missing."""
        tools = mcp_server._tool_manager._tools
        result = _call_provider(tools, "execute", provider_id="gemini", prompt="")

        assert result["success"] is False
        assert result["data"]["error_code"] == "MISSING_REQUIRED"

    def test_provider_execute_validates_whitespace_prompt(self, mcp_server):
        """Test provider-execute rejects whitespace-only prompt."""
        tools = mcp_server._tool_manager._tools
        result = _call_provider(
            tools, "execute", provider_id="gemini", prompt="   \n\t  "
        )

        assert result["success"] is False
        assert result["data"]["error_code"] == "MISSING_REQUIRED"

    def test_provider_execute_returns_unavailable_for_missing_provider(
        self, mcp_server
    ):
        """Test provider execute returns unavailable error for missing provider."""
        tools = mcp_server._tool_manager._tools

        # Most providers won't be available in test environment
        result = _call_provider(
            tools, "execute", provider_id="gemini", prompt="Hello world"
        )

        # Should either succeed (if gemini is available) or return unavailable
        if not result["success"]:
            assert result["data"]["error_code"] in ["UNAVAILABLE", "NOT_FOUND"]


class TestProviderExecuteErrorHandling:
    """Integration tests for provider-execute error handling.

    Note: These tests verify the tool's response structure without mocking internal
    provider functions, since those are imported inside register_provider_tools
    at runtime.
    """

    def test_provider_execute_unavailable_provider_returns_error(self, mcp_server):
        """Test provider execute returns proper error for unavailable provider."""
        tools = mcp_server._tool_manager._tools

        # Use a provider that doesn't exist in registry
        result = _call_provider(
            tools, "execute", provider_id="nonexistent-provider-xyz", prompt="Hello"
        )

        assert result["success"] is False
        assert result["data"]["error_code"] in ["UNAVAILABLE", "NOT_FOUND"]
        assert "remediation" in result["data"]

    def test_provider_execute_returns_expected_fields_on_error(self, mcp_server):
        """Test provider execute error response has expected structure."""
        tools = mcp_server._tool_manager._tools

        result = _call_provider(
            tools, "execute", provider_id="fake-provider", prompt="Hello"
        )

        # Check error response structure
        assert "success" in result
        assert "error" in result
        assert "data" in result
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"

    def test_provider_execute_with_real_unavailable_provider(self, mcp_server):
        """Test provider execute with a registered but unavailable provider."""
        tools = mcp_server._tool_manager._tools

        # Gemini is registered but likely unavailable in test environment
        result = _call_provider(
            tools, "execute", provider_id="gemini", prompt="Hello world"
        )

        # Should return either success or unavailable error
        if not result["success"]:
            # If unavailable, verify error structure
            assert result["data"]["error_code"] in [
                "UNAVAILABLE",
                "NOT_FOUND",
                "TIMEOUT",
                "EXECUTION_ERROR",
            ]
        else:
            # If available, verify success structure
            assert "content" in result["data"]
            assert "model" in result["data"]


class TestProviderToolsEndToEnd:
    """End-to-end integration tests for provider tools workflow."""

    def test_list_then_status_workflow(self, mcp_server):
        """Test workflow: list providers, then check status of one."""
        tools = mcp_server._tool_manager._tools

        # Step 1: List all providers
        list_result = _call_provider(tools, "list", include_unavailable=True)
        assert list_result["success"] is True

        # Step 2: If any providers exist, check status of first one
        providers = list_result["data"]["providers"]
        if providers:
            first_provider_id = providers[0].get("id")
            if first_provider_id:
                status_result = _call_provider(
                    tools, "status", provider_id=first_provider_id
                )
                assert status_result["success"] is True
                assert status_result["data"]["provider_id"] == first_provider_id

    def test_provider_metadata_structure(self, mcp_server):
        """Test that provider metadata has expected structure."""
        tools = mcp_server._tool_manager._tools

        # Check gemini provider structure (always registered)
        result = _call_provider(tools, "status", provider_id="gemini")

        assert result["success"] is True
        data = result["data"]

        # Required fields
        assert "provider_id" in data
        assert "available" in data
        assert "metadata" in data
        assert "capabilities" in data
        assert "health" in data

        # Types
        assert isinstance(data["provider_id"], str)
        assert isinstance(data["available"], bool)

        # If metadata present, check structure
        if data["metadata"]:
            metadata = data["metadata"]
            assert "name" in metadata
            assert "version" in metadata
            assert "default_model" in metadata
            assert "supported_models" in metadata
            assert isinstance(metadata["supported_models"], list)


class TestProviderToolsErrorCases:
    """Integration tests for provider tools error handling."""

    def test_provider_list_handles_missing_providers_gracefully(self, mcp_server):
        """Test provider list returns valid response even when no providers available."""
        tools = mcp_server._tool_manager._tools

        # This should always succeed, returning available providers or empty list
        result = _call_provider(tools, "list", include_unavailable=False)

        assert result["success"] is True
        assert "providers" in result["data"]
        assert isinstance(result["data"]["providers"], list)

    def test_provider_status_handles_empty_provider_id(self, mcp_server):
        """Test provider status returns error for empty provider_id."""
        tools = mcp_server._tool_manager._tools

        result = _call_provider(tools, "status", provider_id="")

        assert result["success"] is False
        assert result["data"]["error_code"] == "MISSING_REQUIRED"

    def test_provider_execute_handles_empty_provider_and_prompt(self, mcp_server):
        """Test provider execute returns error for missing required params."""
        tools = mcp_server._tool_manager._tools

        # Empty provider_id
        result1 = _call_provider(tools, "execute", provider_id="", prompt="Hello")
        assert result1["success"] is False
        assert result1["data"]["error_code"] == "MISSING_REQUIRED"

        # Empty prompt
        result2 = _call_provider(tools, "execute", provider_id="gemini", prompt="")
        assert result2["success"] is False
        assert result2["data"]["error_code"] == "MISSING_REQUIRED"

    def test_provider_status_unknown_provider_returns_not_found(self, mcp_server):
        """Test provider status returns NOT_FOUND for unknown provider."""
        tools = mcp_server._tool_manager._tools

        result = _call_provider(
            tools, "status", provider_id="completely-unknown-provider-xyz"
        )

        assert result["success"] is False
        assert result["data"]["error_code"] == "NOT_FOUND"
        assert "remediation" in result["data"]
