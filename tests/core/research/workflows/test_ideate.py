"""Unit tests for IdeateWorkflow exception handling.

Tests that IdeateWorkflow.execute() catches exceptions and returns error WorkflowResult
instead of crashing the MCP server.
"""

from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.research.workflows.base import WorkflowResult


@pytest.fixture
def mock_config():
    """Create a mock ResearchConfig."""
    config = MagicMock()
    config.default_provider = "test-provider"
    config.ideate_perspectives = ["user", "business", "technical"]
    return config


@pytest.fixture
def mock_memory():
    """Create a mock ResearchMemory."""
    memory = MagicMock()
    memory.load_ideation = MagicMock(return_value=None)
    return memory


class TestIdeateWorkflowExceptionHandling:
    """Tests for IdeateWorkflow.execute() exception handling."""

    def test_execute_catches_exceptions_on_memory_access(self, mock_config, mock_memory):
        """IdeateWorkflow.execute() should catch exceptions and return error WorkflowResult."""
        from foundry_mcp.core.research.workflows.ideate import IdeateWorkflow

        # Mock memory to throw exception when load_ideation is called
        mock_memory.load_ideation.side_effect = RuntimeError("Storage unavailable")

        workflow = IdeateWorkflow(mock_config, mock_memory)
        result = workflow.execute(ideation_id="test-ideation-123")

        # Should return error result, not raise exception
        assert isinstance(result, WorkflowResult)
        assert result.success is False
        assert result.error is not None
        assert "Storage unavailable" in result.error
        assert result.metadata["workflow"] == "ideate"
        assert result.metadata["error_type"] == "RuntimeError"

    def test_execute_catches_generate_exceptions(self, mock_config, mock_memory):
        """IdeateWorkflow.execute() should catch _generate_ideas exceptions."""
        from foundry_mcp.core.research.workflows.ideate import IdeateWorkflow

        workflow = IdeateWorkflow(mock_config, mock_memory)

        # Mock _generate_ideas to raise an exception
        with patch.object(
            workflow, "_generate_ideas", side_effect=RuntimeError("Idea generation failed")
        ):
            result = workflow.execute(topic="Test topic", action="generate")

        # Should return error result, not raise exception
        assert result.success is False
        assert result.error is not None
        assert "Idea generation failed" in result.error
        assert result.metadata["error_type"] == "RuntimeError"

    def test_execute_handles_empty_exception_message(self, mock_config, mock_memory):
        """IdeateWorkflow.execute() should handle exceptions with empty messages."""
        from foundry_mcp.core.research.workflows.ideate import IdeateWorkflow

        # Mock memory to throw exception with no message
        mock_memory.load_ideation.side_effect = RuntimeError()

        workflow = IdeateWorkflow(mock_config, mock_memory)
        result = workflow.execute(ideation_id="test-ideation-123")

        # Should use class name when message is empty
        assert result.success is False
        assert result.error is not None
        assert "RuntimeError" in result.error
        assert result.metadata["error_type"] == "RuntimeError"
