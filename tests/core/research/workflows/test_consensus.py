"""Unit tests for ConsensusWorkflow exception handling.

Tests that ConsensusWorkflow.execute() catches exceptions and returns error WorkflowResult
instead of crashing the MCP server.
"""

from unittest.mock import patch

import pytest

from foundry_mcp.core.research.workflows.base import WorkflowResult


@pytest.fixture
def mock_config(mock_config):
    """Extend base mock_config with consensus-specific attributes."""
    mock_config.consensus_providers = ["openai", "anthropic"]
    mock_config.consensus_strategy = "synthesize"
    return mock_config


class TestConsensusWorkflowExceptionHandling:
    """Tests for ConsensusWorkflow.execute() exception handling."""

    def test_execute_catches_exceptions(self, mock_config, mock_memory):
        """ConsensusWorkflow.execute() should catch exceptions and return error WorkflowResult."""
        from foundry_mcp.core.research.workflows.consensus import ConsensusWorkflow

        workflow = ConsensusWorkflow(mock_config, mock_memory)

        # Mock available_providers to raise an exception
        with patch(
            "foundry_mcp.core.research.workflows.consensus.available_providers",
            side_effect=RuntimeError("Provider pool unavailable"),
        ):
            result = workflow.execute(prompt="Test prompt")

        # Should return error result, not raise exception
        assert isinstance(result, WorkflowResult)
        assert result.success is False
        assert result.error is not None
        assert "Provider pool unavailable" in result.error
        assert result.metadata["workflow"] == "consensus"
        assert result.metadata["error_type"] == "RuntimeError"

    def test_execute_catches_provider_spec_exceptions(self, mock_config, mock_memory):
        """ConsensusWorkflow.execute() should catch provider spec parsing exceptions."""
        from foundry_mcp.core.research.workflows.consensus import ConsensusWorkflow

        workflow = ConsensusWorkflow(mock_config, mock_memory)

        # Mock ProviderSpec.parse_flexible to raise an exception
        with patch(
            "foundry_mcp.core.research.workflows.consensus.ProviderSpec.parse_flexible",
            side_effect=RuntimeError("Invalid provider spec"),
        ):
            with patch(
                "foundry_mcp.core.research.workflows.consensus.available_providers",
                return_value=["openai", "anthropic"],
            ):
                result = workflow.execute(prompt="Test prompt")

        # Should return error result, not raise exception
        assert result.success is False
        assert result.error is not None
        assert "Invalid provider spec" in result.error
        assert result.metadata["error_type"] == "RuntimeError"

    def test_execute_handles_empty_exception_message(self, mock_config, mock_memory):
        """ConsensusWorkflow.execute() should handle exceptions with empty messages."""
        from foundry_mcp.core.research.workflows.consensus import ConsensusWorkflow

        workflow = ConsensusWorkflow(mock_config, mock_memory)

        # Mock available_providers to raise exception with no message
        with patch(
            "foundry_mcp.core.research.workflows.consensus.available_providers",
            side_effect=RuntimeError(),
        ):
            result = workflow.execute(prompt="Test prompt")

        # Should use class name when message is empty
        assert result.success is False
        assert result.error is not None
        assert "RuntimeError" in result.error
        assert result.metadata["error_type"] == "RuntimeError"
