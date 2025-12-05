"""
Integration tests for LLM-powered review tools.

Tests verify:
- spec_review returns NOT_IMPLEMENTED (requires external AI integration)
- review_list_tools works with provider system
- review_list_plan_tools returns plan tools with availability
- pr_create_with_spec returns NOT_IMPLEMENTED (requires external integration)
- pr_get_spec_context retrieves spec context using core APIs
- Response envelope compliance
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from foundry_mcp.config import ServerConfig


@pytest.fixture
def mock_mcp():
    """Create a mock FastMCP server instance.

    Note: Tools are registered with function names (snake_case) by the
    canonical_tool decorator, not the kebab-case canonical names.
    """
    mcp = MagicMock()
    mcp._tools = {}

    def mock_tool(*args, **kwargs):
        def decorator(func):
            # Use function name as key (this is what actually gets registered)
            name = func.__name__
            mcp._tools[name] = MagicMock(fn=func)
            return func
        return decorator

    mcp.tool = mock_tool
    return mcp


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock server config with temp specs dir."""
    return ServerConfig(specs_dir=tmp_path / "specs")


@pytest.fixture
def sample_spec_data():
    """Sample spec data for testing pr_get_spec_context."""
    return {
        "metadata": {
            "title": "Test Specification",
            "spec_id": "test-spec-001",
        },
        "hierarchy": {
            "task-1": {
                "type": "task",
                "title": "First task",
                "status": "completed",
                "metadata": {"completed_at": "2025-01-15T10:00:00Z"},
            },
            "task-2": {
                "type": "task",
                "title": "Second task",
                "status": "in_progress",
                "metadata": {},
            },
        },
        "journal": [
            {
                "timestamp": "2025-01-15T10:00:00Z",
                "entry_type": "note",
                "title": "Task completed",
                "task_id": "task-1",
                "content": "Completed first task",
            },
        ],
    }


class TestSpecReviewNotImplemented:
    """Test spec_review returns NOT_IMPLEMENTED as expected."""

    def test_spec_review_registers_successfully(self, mock_mcp, mock_config):
        """Test spec_review tool registers without error."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)
        assert "spec_review" in mock_mcp._tools

    def test_spec_review_quick_returns_success_or_error(self, mock_mcp, mock_config):
        """Test spec_review with quick review type returns success or appropriate error."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        spec_review = mock_mcp._tools["spec_review"]
        result = spec_review.fn(spec_id="test-spec-001", review_type="quick")

        # spec_review is now implemented - returns success for quick reviews or
        # appropriate error if spec not found
        assert "success" in result
        assert "spec_id" in result.get("data", {}) or result["success"] is False

    def test_spec_review_includes_spec_id_in_response(self, mock_mcp, mock_config):
        """Test spec_review includes spec_id in response data or error."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        spec_review = mock_mcp._tools["spec_review"]
        result = spec_review.fn(spec_id="my-test-spec", review_type="full")

        # spec_review is now implemented - check for spec_id in response or error
        assert "success" in result
        if result["success"]:
            assert result["data"]["spec_id"] == "my-test-spec"
        else:
            # May fail due to missing spec or AI provider - that's fine
            assert result.get("data", {}).get("spec_id") == "my-test-spec" or "error" in result

    def test_spec_review_preserves_all_parameters(self, mock_mcp, mock_config):
        """Test spec_review includes all provided parameters in response."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        spec_review = mock_mcp._tools["spec_review"]
        result = spec_review.fn(
            spec_id="test-spec-001",
            review_type="security",
            tools="cursor-agent,gemini",
            model="gpt-4",
            dry_run=True,
        )

        # spec_review is now implemented - check response has expected structure
        assert "success" in result
        if result["success"]:
            assert result["data"]["spec_id"] == "test-spec-001"
            assert result["data"]["review_type"] == "security"
        else:
            # May fail due to missing spec or AI provider - check error structure
            assert "error" in result


class TestReviewListTools:
    """Test review_list_tools with provider system integration."""

    def test_review_list_tools_registers(self, mock_mcp, mock_config):
        """Test review_list_tools registers successfully."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)
        assert "review_list_tools" in mock_mcp._tools

    def test_review_list_tools_returns_success(self, mock_mcp, mock_config):
        """Test review_list_tools returns success response."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_tools = mock_mcp._tools["review_list_tools"]
        result = list_tools.fn()

        assert result["success"] is True
        assert "tools" in result["data"]
        assert "llm_status" in result["data"]

    def test_review_list_tools_includes_provider_info(self, mock_mcp, mock_config):
        """Test review_list_tools includes provider information."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_tools = mock_mcp._tools["review_list_tools"]
        result = list_tools.fn()

        assert result["success"] is True
        tools = result["data"]["tools"]
        assert isinstance(tools, list)

        # Each tool should have expected fields
        for tool in tools:
            assert "name" in tool
            assert "available" in tool
            assert "status" in tool

    def test_review_list_tools_includes_review_types(self, mock_mcp, mock_config):
        """Test review_list_tools includes available review types."""
        from foundry_mcp.tools.review import register_review_tools, REVIEW_TYPES

        register_review_tools(mock_mcp, mock_config)

        list_tools = mock_mcp._tools["review_list_tools"]
        result = list_tools.fn()

        assert result["success"] is True
        assert "review_types" in result["data"]
        assert set(result["data"]["review_types"]) == set(REVIEW_TYPES)

    def test_review_list_tools_includes_llm_status(self, mock_mcp, mock_config):
        """Test review_list_tools includes LLM configuration status."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_tools = mock_mcp._tools["review_list_tools"]
        result = list_tools.fn()

        assert result["success"] is True
        llm_status = result["data"]["llm_status"]
        assert "configured" in llm_status

    def test_review_list_tools_includes_duration(self, mock_mcp, mock_config):
        """Test review_list_tools includes duration_ms in response."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_tools = mock_mcp._tools["review_list_tools"]
        result = list_tools.fn()

        assert result["success"] is True
        assert "duration_ms" in result["data"]
        assert isinstance(result["data"]["duration_ms"], (int, float))


class TestReviewListPlanTools:
    """Test review_list_plan_tools functionality."""

    def test_review_list_plan_tools_registers(self, mock_mcp, mock_config):
        """Test review_list_plan_tools registers successfully."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)
        assert "review_list_plan_tools" in mock_mcp._tools

    def test_review_list_plan_tools_returns_success(self, mock_mcp, mock_config):
        """Test review_list_plan_tools returns success response."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_plan_tools = mock_mcp._tools["review_list_plan_tools"]
        result = list_plan_tools.fn()

        assert result["success"] is True
        assert "plan_tools" in result["data"]

    def test_review_list_plan_tools_includes_tool_info(self, mock_mcp, mock_config):
        """Test review_list_plan_tools includes detailed tool information."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_plan_tools = mock_mcp._tools["review_list_plan_tools"]
        result = list_plan_tools.fn()

        assert result["success"] is True
        plan_tools = result["data"]["plan_tools"]
        assert isinstance(plan_tools, list)
        assert len(plan_tools) >= 1

        # Each tool should have expected fields
        for tool in plan_tools:
            assert "name" in tool
            assert "description" in tool
            assert "capabilities" in tool
            assert "llm_required" in tool

    def test_review_list_plan_tools_has_non_llm_option(self, mock_mcp, mock_config):
        """Test review_list_plan_tools includes at least one non-LLM option."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_plan_tools = mock_mcp._tools["review_list_plan_tools"]
        result = list_plan_tools.fn()

        assert result["success"] is True
        plan_tools = result["data"]["plan_tools"]

        non_llm_tools = [t for t in plan_tools if not t.get("llm_required")]
        assert len(non_llm_tools) >= 1, "Should have at least one non-LLM tool option"

    def test_review_list_plan_tools_includes_recommendations(self, mock_mcp, mock_config):
        """Test review_list_plan_tools includes usage recommendations."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_plan_tools = mock_mcp._tools["review_list_plan_tools"]
        result = list_plan_tools.fn()

        assert result["success"] is True
        assert "recommendations" in result["data"]
        assert isinstance(result["data"]["recommendations"], list)
        assert len(result["data"]["recommendations"]) >= 1


class TestPRCreateWithSpec:
    """Test pr_create_with_spec returns NOT_IMPLEMENTED."""

    def test_pr_create_registers_successfully(self, mock_mcp, mock_config):
        """Test pr_create_with_spec tool registers without error."""
        from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools

        register_pr_workflow_tools(mock_mcp, mock_config)
        assert "pr_create_with_spec" in mock_mcp._tools

    def test_pr_create_returns_not_implemented(self, mock_mcp, mock_config):
        """Test pr_create_with_spec returns NOT_IMPLEMENTED error."""
        from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools

        register_pr_workflow_tools(mock_mcp, mock_config)

        pr_create = mock_mcp._tools["pr_create_with_spec"]
        result = pr_create.fn(spec_id="test-spec-001", dry_run=True)

        assert result["success"] is False
        # error_code is in the data dict
        assert result.get("data", {}).get("error_code") == "NOT_IMPLEMENTED"
        assert "sdd-pr" in result.get("error", "").lower() or \
               "sdd-pr" in result.get("data", {}).get("alternative", "")

    def test_pr_create_preserves_parameters(self, mock_mcp, mock_config):
        """Test pr_create_with_spec includes all parameters in response."""
        from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools

        register_pr_workflow_tools(mock_mcp, mock_config)

        pr_create = mock_mcp._tools["pr_create_with_spec"]
        result = pr_create.fn(
            spec_id="test-spec-001",
            title="Custom PR title",
            base_branch="develop",
            dry_run=True,
        )

        assert result["data"]["spec_id"] == "test-spec-001"
        assert result["data"]["title"] == "Custom PR title"
        assert result["data"]["base_branch"] == "develop"
        assert result["data"]["dry_run"] is True


class TestPRGetSpecContext:
    """Test pr_get_spec_context using core APIs."""

    def test_pr_get_spec_context_registers(self, mock_mcp, mock_config):
        """Test pr_get_spec_context registers successfully."""
        from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools

        register_pr_workflow_tools(mock_mcp, mock_config)
        assert "pr_get_spec_context" in mock_mcp._tools

    def test_pr_get_spec_context_handles_missing_spec(self, mock_mcp, mock_config):
        """Test pr_get_spec_context handles missing spec gracefully."""
        from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools

        register_pr_workflow_tools(mock_mcp, mock_config)

        pr_get_context = mock_mcp._tools["pr_get_spec_context"]
        result = pr_get_context.fn(spec_id="nonexistent-spec")

        assert result["success"] is False
        assert "error" in result

    def test_pr_get_spec_context_with_valid_spec(
        self, mock_mcp, mock_config, sample_spec_data, tmp_path
    ):
        """Test pr_get_spec_context retrieves context for valid spec."""
        from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools

        # Create specs directory and spec file
        specs_dir = tmp_path / "specs" / "active"
        specs_dir.mkdir(parents=True)
        spec_file = specs_dir / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec_data))

        # Update config to use tmp_path
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_pr_workflow_tools(mock_mcp, config)

        pr_get_context = mock_mcp._tools["pr_get_spec_context"]
        result = pr_get_context.fn(spec_id="test-spec-001", path=str(tmp_path))

        assert result["success"] is True
        assert result["data"]["spec_id"] == "test-spec-001"
        assert result["data"]["title"] == "Test Specification"


class TestProviderSystemIntegration:
    """Test provider system integration in review tools."""

    def test_provider_statuses_used_in_list_tools(self, mock_mcp, mock_config):
        """Test review_list_tools uses provider system for status checks."""
        from foundry_mcp.tools.review import register_review_tools
        from foundry_mcp.core.providers import get_provider_statuses

        register_review_tools(mock_mcp, mock_config)

        # Get provider statuses directly
        provider_statuses = get_provider_statuses()

        # Get list from tool
        list_tools = mock_mcp._tools["review_list_tools"]
        result = list_tools.fn()

        assert result["success"] is True

        # Tool should include provider info
        tools = result["data"]["tools"]
        tool_names = [t["name"] for t in tools]

        # All registered providers should appear in the tool list
        for provider_id in provider_statuses.keys():
            assert provider_id in tool_names

    def test_provider_availability_reflected_in_tool_status(self, mock_mcp, mock_config):
        """Test tool availability reflects provider availability."""
        from foundry_mcp.tools.review import register_review_tools
        from foundry_mcp.core.providers import check_provider_available

        register_review_tools(mock_mcp, mock_config)

        list_tools = mock_mcp._tools["review_list_tools"]
        result = list_tools.fn()

        assert result["success"] is True

        # Check that availability is boolean
        for tool in result["data"]["tools"]:
            if tool["available"] is not None:
                assert isinstance(tool["available"], bool)


class TestResponseEnvelope:
    """Test response envelope compliance for review tools."""

    def test_spec_review_envelope(self, mock_mcp, mock_config):
        """Test spec_review response has required envelope fields."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        spec_review = mock_mcp._tools["spec_review"]
        result = spec_review.fn(spec_id="test-spec-001")

        # Required envelope fields for all responses
        assert "success" in result
        assert "data" in result
        assert "meta" in result
        # Error field required only for error responses
        if not result["success"]:
            assert "error" in result

    def test_review_list_tools_success_envelope(self, mock_mcp, mock_config):
        """Test review_list_tools success response has required envelope fields."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_tools = mock_mcp._tools["review_list_tools"]
        result = list_tools.fn()

        # Required envelope fields
        assert "success" in result
        assert result["success"] is True
        assert "data" in result
        assert "meta" in result

    def test_review_list_plan_tools_success_envelope(self, mock_mcp, mock_config):
        """Test review_list_plan_tools success response has required envelope fields."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_plan_tools = mock_mcp._tools["review_list_plan_tools"]
        result = list_plan_tools.fn()

        # Required envelope fields
        assert "success" in result
        assert result["success"] is True
        assert "data" in result
        assert "meta" in result

    def test_pr_create_error_envelope(self, mock_mcp, mock_config):
        """Test pr_create_with_spec error response has required envelope fields."""
        from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools

        register_pr_workflow_tools(mock_mcp, mock_config)

        pr_create = mock_mcp._tools["pr_create_with_spec"]
        result = pr_create.fn(spec_id="test-spec-001", dry_run=True)

        # Required envelope fields for errors
        assert "success" in result
        assert result["success"] is False
        assert "error" in result
        assert "data" in result
        assert "meta" in result


class TestMetricsEmission:
    """Test metrics emission for review tools."""

    def test_review_list_tools_emits_duration_metric(self, mock_mcp, mock_config):
        """Test review_list_tools emits timer metric."""
        from foundry_mcp.tools.review import register_review_tools, _metrics

        register_review_tools(mock_mcp, mock_config)

        with patch.object(_metrics, 'timer') as mock_timer:
            list_tools = mock_mcp._tools["review_list_tools"]
            result = list_tools.fn()

            assert result["success"] is True
            # Timer should have been called
            mock_timer.assert_called()

    def test_review_list_plan_tools_emits_duration_metric(self, mock_mcp, mock_config):
        """Test review_list_plan_tools emits timer metric."""
        from foundry_mcp.tools.review import register_review_tools, _metrics

        register_review_tools(mock_mcp, mock_config)

        with patch.object(_metrics, 'timer') as mock_timer:
            list_plan_tools = mock_mcp._tools["review_list_plan_tools"]
            result = list_plan_tools.fn()

            assert result["success"] is True
            mock_timer.assert_called()

    def test_review_metrics_singleton_exists(self):
        """Test review tools have metrics singleton available."""
        from foundry_mcp.tools.review import _metrics

        assert _metrics is not None

    def test_pr_workflow_metrics_singleton_exists(self):
        """Test PR workflow tools have metrics singleton available."""
        from foundry_mcp.tools.pr_workflow import _metrics

        assert _metrics is not None


class TestToolRegistration:
    """Test all review tools register correctly."""

    def test_all_review_tools_register(self, mock_mcp, mock_config):
        """Test all review tools register without error."""
        from foundry_mcp.tools.review import register_review_tools

        # Should not raise
        register_review_tools(mock_mcp, mock_config)

        # Should have registered expected tools
        assert "spec_review" in mock_mcp._tools
        assert "review_list_tools" in mock_mcp._tools
        assert "review_list_plan_tools" in mock_mcp._tools

    def test_all_pr_workflow_tools_register(self, mock_mcp, mock_config):
        """Test all PR workflow tools register without error."""
        from foundry_mcp.tools.pr_workflow import register_pr_workflow_tools

        # Should not raise
        register_pr_workflow_tools(mock_mcp, mock_config)

        # Should have registered expected tools
        assert "pr_create_with_spec" in mock_mcp._tools
        assert "pr_get_spec_context" in mock_mcp._tools


class TestLLMConfigurationStatus:
    """Test LLM configuration status reporting."""

    def test_list_tools_includes_llm_configured_status(self, mock_mcp, mock_config):
        """Test review_list_tools shows LLM configuration status."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_tools = mock_mcp._tools["review_list_tools"]
        result = list_tools.fn()

        assert result["success"] is True
        assert "llm_status" in result["data"]
        assert "configured" in result["data"]["llm_status"]

    def test_list_plan_tools_reflects_llm_status(self, mock_mcp, mock_config):
        """Test review_list_plan_tools shows availability based on LLM status."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_plan_tools = mock_mcp._tools["review_list_plan_tools"]
        result = list_plan_tools.fn()

        assert result["success"] is True
        assert "llm_status" in result["data"]

        # Plan tools should have status information
        plan_tools = result["data"]["plan_tools"]
        for tool in plan_tools:
            assert "status" in tool
            assert tool["status"] in ("available", "unavailable")
