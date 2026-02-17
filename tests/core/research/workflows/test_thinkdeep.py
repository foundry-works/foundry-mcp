"""Unit tests for ThinkDeepWorkflow exception handling.

Tests that ThinkDeepWorkflow.execute() catches exceptions and returns error WorkflowResult
instead of crashing the MCP server.
"""

from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.research.workflows.base import WorkflowResult


@pytest.fixture
def mock_config(mock_config):
    """Extend base mock_config with thinkdeep-specific attributes."""
    mock_config.thinkdeep_max_depth = 5
    return mock_config


@pytest.fixture
def mock_memory(mock_memory):
    """Extend base mock_memory with investigation-specific methods."""
    mock_memory.load_investigation = MagicMock(return_value=None)
    return mock_memory


class TestThinkDeepWorkflowExceptionHandling:
    """Tests for ThinkDeepWorkflow.execute() exception handling."""

    def test_execute_catches_exceptions_on_memory_access(self, mock_config, mock_memory):
        """ThinkDeepWorkflow.execute() should catch exceptions and return error WorkflowResult."""
        from foundry_mcp.core.research.workflows.thinkdeep import ThinkDeepWorkflow

        # Mock memory to throw exception when load_investigation is called
        mock_memory.load_investigation.side_effect = RuntimeError("Storage unavailable")

        workflow = ThinkDeepWorkflow(mock_config, mock_memory)
        result = workflow.execute(investigation_id="test-inv-123")

        # Should return error result, not raise exception
        assert isinstance(result, WorkflowResult)
        assert result.success is False
        assert result.error is not None
        assert "Storage unavailable" in result.error
        assert result.metadata["workflow"] == "thinkdeep"
        assert result.metadata["error_type"] == "RuntimeError"

    def test_execute_catches_exceptions_on_new_investigation(self, mock_config, mock_memory):
        """ThinkDeepWorkflow.execute() should catch exceptions when starting new investigation."""
        from foundry_mcp.core.research.workflows.thinkdeep import ThinkDeepWorkflow

        workflow = ThinkDeepWorkflow(mock_config, mock_memory)

        # Mock _generate_initial_query to raise an exception
        with patch.object(
            workflow, "_generate_initial_query", side_effect=RuntimeError("Query generation failed")
        ):
            result = workflow.execute(topic="Test topic")

        # Should return error result, not raise exception
        assert result.success is False
        assert result.error is not None
        assert "Query generation failed" in result.error
        assert result.metadata["error_type"] == "RuntimeError"

    def test_execute_handles_empty_exception_message(self, mock_config, mock_memory):
        """ThinkDeepWorkflow.execute() should handle exceptions with empty messages."""
        from foundry_mcp.core.research.workflows.thinkdeep import ThinkDeepWorkflow

        # Mock memory to throw exception with no message
        mock_memory.load_investigation.side_effect = RuntimeError()

        workflow = ThinkDeepWorkflow(mock_config, mock_memory)
        result = workflow.execute(investigation_id="test-inv-123")

        # Should use class name when message is empty
        assert result.success is False
        assert result.error is not None
        assert "RuntimeError" in result.error
        assert result.metadata["error_type"] == "RuntimeError"
