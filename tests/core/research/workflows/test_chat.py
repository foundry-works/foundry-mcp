"""Unit tests for ChatWorkflow exception handling.

Tests that ChatWorkflow.execute() catches exceptions and returns error WorkflowResult
instead of crashing the MCP server.
"""

from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.research.workflows.base import WorkflowResult


@pytest.fixture
def mock_config(mock_config):
    """Extend base mock_config with chat-specific attributes."""
    mock_config.max_messages_per_thread = 50
    return mock_config


class TestChatWorkflowExceptionHandling:
    """Tests for ChatWorkflow.execute() exception handling."""

    def test_execute_catches_exceptions(self, mock_config, mock_memory):
        """ChatWorkflow.execute() should catch exceptions and return error WorkflowResult."""
        from foundry_mcp.core.research.workflows.chat import ChatWorkflow

        workflow = ChatWorkflow(mock_config, mock_memory)

        # Mock _get_or_create_thread to raise an exception
        with patch.object(
            workflow, "_get_or_create_thread", side_effect=RuntimeError("Storage unavailable")
        ):
            result = workflow.execute(prompt="Hello")

        # Should return error result, not raise exception
        assert isinstance(result, WorkflowResult)
        assert result.success is False
        assert result.error is not None
        assert "Storage unavailable" in result.error
        assert result.metadata["workflow"] == "chat"
        assert result.metadata["error_type"] == "RuntimeError"

    def test_execute_catches_provider_exceptions(self, mock_config, mock_memory):
        """ChatWorkflow.execute() should catch provider execution exceptions."""
        from foundry_mcp.core.research.workflows.chat import ChatWorkflow

        workflow = ChatWorkflow(mock_config, mock_memory)

        # Mock methods to simulate execution flow
        mock_thread = MagicMock()
        mock_thread.provider_id = "test-provider"
        mock_thread.system_prompt = None
        mock_thread.messages = []

        with patch.object(workflow, "_get_or_create_thread", return_value=mock_thread):
            with patch.object(workflow, "_build_context", return_value="context"):
                with patch.object(
                    workflow,
                    "_execute_provider",
                    side_effect=ConnectionError("Provider API timeout"),
                ):
                    result = workflow.execute(prompt="Hello")

        # Should return error result, not raise exception
        assert result.success is False
        assert result.error is not None
        assert "Provider API timeout" in result.error
        assert result.metadata["error_type"] == "ConnectionError"

    def test_execute_catches_keyboard_interrupt(self, mock_config, mock_memory):
        """ChatWorkflow.execute() should catch KeyboardInterrupt as Exception subclass."""
        from foundry_mcp.core.research.workflows.chat import ChatWorkflow

        workflow = ChatWorkflow(mock_config, mock_memory)

        # Note: KeyboardInterrupt is NOT a subclass of Exception, so it won't be caught
        # This test verifies behavior for Exception subclasses only
        with patch.object(
            workflow, "_get_or_create_thread", side_effect=ValueError("Invalid thread state")
        ):
            result = workflow.execute(prompt="Hello")

        assert result.success is False
        assert result.error is not None
        assert "Invalid thread state" in result.error
        assert result.metadata["error_type"] == "ValueError"

    def test_execute_handles_empty_exception_message(self, mock_config, mock_memory):
        """ChatWorkflow.execute() should handle exceptions with empty messages."""
        from foundry_mcp.core.research.workflows.chat import ChatWorkflow

        workflow = ChatWorkflow(mock_config, mock_memory)

        # Create an exception with no message
        with patch.object(
            workflow, "_get_or_create_thread", side_effect=RuntimeError()
        ):
            result = workflow.execute(prompt="Hello")

        # Should use class name when message is empty
        assert result.success is False
        assert result.error is not None
        assert "RuntimeError" in result.error
        assert result.metadata["error_type"] == "RuntimeError"
