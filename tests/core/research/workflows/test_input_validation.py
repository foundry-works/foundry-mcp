"""Tests for input bounds validation across research workflows.

Phase 2c: validates MAX_PROMPT_LENGTH, MAX_ITERATIONS, MAX_SUB_QUERIES,
MAX_SOURCES_PER_QUERY, and MAX_CONCURRENT_PROVIDERS limits.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.research.workflows.base import (
    MAX_PROMPT_LENGTH,
    WorkflowResult,
)
from foundry_mcp.core.research.workflows.deep_research._constants import (
    MAX_CONCURRENT_PROVIDERS,
    MAX_ITERATIONS,
    MAX_SOURCES_PER_QUERY,
    MAX_SUB_QUERIES,
)

# ──────────────────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_config():
    """Create a mock ResearchConfig."""
    config = MagicMock()
    config.default_provider = "test-provider"
    config.default_timeout = 30
    config.ttl_hours = 24
    config.deep_research_mode = "general"
    config.deep_research_timeout = 600
    config.resolve_phase_provider = MagicMock(return_value=("test-provider", None))
    config.get_phase_fallback_providers = MagicMock(return_value=[])
    return config


@pytest.fixture
def mock_memory():
    """Create a mock ResearchMemory."""
    memory = MagicMock()
    memory.save_deep_research = MagicMock()
    return memory


# ──────────────────────────────────────────────────────────────────────
#  Prompt Length Validation (base workflow)
# ──────────────────────────────────────────────────────────────────────


class TestPromptLengthValidation:
    """Tests for MAX_PROMPT_LENGTH enforcement in _execute_provider_async."""

    def test_prompt_at_limit_accepted(self, mock_config, mock_memory):
        """Prompt exactly at MAX_PROMPT_LENGTH should not be rejected."""
        from foundry_mcp.core.research.workflows.deep_research import (
            DeepResearchWorkflow,
        )

        workflow = DeepResearchWorkflow(mock_config, mock_memory)
        prompt = "x" * MAX_PROMPT_LENGTH

        # Mock provider resolution to return a provider that succeeds
        mock_provider = MagicMock()
        mock_result = MagicMock()
        mock_result.status.value = "success"
        mock_result.content = "response"
        mock_result.provider_id = "test"
        mock_result.model_used = "test-model"
        mock_result.tokens = None
        # Use the SUCCESS enum value
        from foundry_mcp.core.providers import ProviderStatus

        mock_result.status = ProviderStatus.SUCCESS
        mock_provider.generate.return_value = mock_result

        with patch.object(workflow, "_resolve_provider", return_value=mock_provider):
            result = asyncio.run(
                workflow._execute_provider_async(
                    prompt=prompt,
                    phase="test",
                )
            )

        # Should succeed (not rejected by validation)
        assert result.success is True

    def test_prompt_over_limit_rejected(self, mock_config, mock_memory):
        """Prompt exceeding MAX_PROMPT_LENGTH should return error."""
        from foundry_mcp.core.research.workflows.deep_research import (
            DeepResearchWorkflow,
        )

        workflow = DeepResearchWorkflow(mock_config, mock_memory)
        prompt = "x" * (MAX_PROMPT_LENGTH + 1)

        result = asyncio.run(
            workflow._execute_provider_async(
                prompt=prompt,
                phase="test",
            )
        )

        assert result.success is False
        assert result.error is not None
        assert "exceeds maximum" in result.error
        assert str(MAX_PROMPT_LENGTH) in result.error
        assert result.metadata.get("validation_error") == "prompt_too_long"

    def test_prompt_well_under_limit_accepted(self, mock_config, mock_memory):
        """Normal-length prompt should not trigger validation."""
        from foundry_mcp.core.research.workflows.deep_research import (
            DeepResearchWorkflow,
        )

        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        mock_provider = MagicMock()
        mock_result = MagicMock()
        from foundry_mcp.core.providers import ProviderStatus

        mock_result.status = ProviderStatus.SUCCESS
        mock_result.content = "response"
        mock_result.provider_id = "test"
        mock_result.model_used = "test-model"
        mock_result.tokens = None
        mock_provider.generate.return_value = mock_result

        with patch.object(workflow, "_resolve_provider", return_value=mock_provider):
            result = asyncio.run(
                workflow._execute_provider_async(
                    prompt="What is quantum computing?",
                    phase="test",
                )
            )

        assert result.success is True

    def test_max_prompt_length_constant_is_generous(self):
        """MAX_PROMPT_LENGTH should allow ~50k tokens worth of text."""
        # At ~4 chars/token, 200k chars ≈ 50k tokens
        assert MAX_PROMPT_LENGTH >= 100_000, (
            f"MAX_PROMPT_LENGTH={MAX_PROMPT_LENGTH} is too restrictive; should allow at least 100k characters"
        )


# ──────────────────────────────────────────────────────────────────────
#  Deep Research Input Bounds Validation
# ──────────────────────────────────────────────────────────────────────


class TestDeepResearchInputBounds:
    """Tests for deep research parameter validation in _start_research."""

    def test_max_iterations_at_limit_accepted(self, mock_config, mock_memory):
        """max_iterations at MAX_ITERATIONS should be accepted."""
        from foundry_mcp.core.research.workflows.deep_research import (
            DeepResearchWorkflow,
        )

        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        # We test that validation doesn't reject at-limit values.
        # Mock the subsequent workflow execution to avoid real provider calls.
        with patch.object(workflow, "_start_background_task") as mock_bg:
            mock_bg.return_value = WorkflowResult(success=True, content="started")
            result = workflow.execute(
                query="test query",
                action="start",
                max_iterations=MAX_ITERATIONS,
                background=True,
            )

        assert result.success is True

    def test_max_iterations_over_limit_rejected(self, mock_config, mock_memory):
        """max_iterations exceeding MAX_ITERATIONS should return error."""
        from foundry_mcp.core.research.workflows.deep_research import (
            DeepResearchWorkflow,
        )

        workflow = DeepResearchWorkflow(mock_config, mock_memory)
        result = workflow.execute(
            query="test query",
            action="start",
            max_iterations=MAX_ITERATIONS + 1,
        )

        assert result.success is False
        assert result.error is not None
        assert "max_iterations" in result.error
        assert "validation" in result.error.lower()

    def test_max_sub_queries_over_limit_rejected(self, mock_config, mock_memory):
        """max_sub_queries exceeding MAX_SUB_QUERIES should return error."""
        from foundry_mcp.core.research.workflows.deep_research import (
            DeepResearchWorkflow,
        )

        workflow = DeepResearchWorkflow(mock_config, mock_memory)
        result = workflow.execute(
            query="test query",
            action="start",
            max_sub_queries=MAX_SUB_QUERIES + 1,
        )

        assert result.success is False
        assert result.error is not None
        assert "max_sub_queries" in result.error

    def test_max_sources_per_query_over_limit_rejected(self, mock_config, mock_memory):
        """max_sources_per_query exceeding limit should return error."""
        from foundry_mcp.core.research.workflows.deep_research import (
            DeepResearchWorkflow,
        )

        workflow = DeepResearchWorkflow(mock_config, mock_memory)
        result = workflow.execute(
            query="test query",
            action="start",
            max_sources_per_query=MAX_SOURCES_PER_QUERY + 1,
        )

        assert result.success is False
        assert result.error is not None
        assert "max_sources_per_query" in result.error

    def test_max_concurrent_over_limit_rejected(self, mock_config, mock_memory):
        """max_concurrent exceeding MAX_CONCURRENT_PROVIDERS should return error."""
        from foundry_mcp.core.research.workflows.deep_research import (
            DeepResearchWorkflow,
        )

        workflow = DeepResearchWorkflow(mock_config, mock_memory)
        result = workflow.execute(
            query="test query",
            action="start",
            max_concurrent=MAX_CONCURRENT_PROVIDERS + 1,
        )

        assert result.success is False
        assert result.error is not None
        assert "max_concurrent" in result.error

    def test_multiple_violations_reported(self, mock_config, mock_memory):
        """Multiple bound violations should all be reported."""
        from foundry_mcp.core.research.workflows.deep_research import (
            DeepResearchWorkflow,
        )

        workflow = DeepResearchWorkflow(mock_config, mock_memory)
        result = workflow.execute(
            query="test query",
            action="start",
            max_iterations=MAX_ITERATIONS + 1,
            max_sub_queries=MAX_SUB_QUERIES + 1,
        )

        assert result.success is False
        assert result.error is not None
        assert "max_iterations" in result.error
        assert "max_sub_queries" in result.error
        # Metadata should contain individual violations
        assert len(result.metadata.get("validation_errors", [])) == 2

    def test_query_too_long_rejected(self, mock_config, mock_memory):
        """Query exceeding MAX_PROMPT_LENGTH should return error."""
        from foundry_mcp.core.research.workflows.deep_research import (
            DeepResearchWorkflow,
        )

        workflow = DeepResearchWorkflow(mock_config, mock_memory)
        result = workflow.execute(
            query="x" * (MAX_PROMPT_LENGTH + 1),
            action="start",
        )

        assert result.success is False
        assert result.error is not None
        assert "query length" in result.error

    def test_all_params_at_default_accepted(self, mock_config, mock_memory):
        """Default parameter values should all pass validation."""
        from foundry_mcp.core.research.workflows.deep_research import (
            DeepResearchWorkflow,
        )

        workflow = DeepResearchWorkflow(mock_config, mock_memory)

        with patch.object(workflow, "_start_background_task") as mock_bg:
            mock_bg.return_value = WorkflowResult(success=True, content="started")
            result = workflow.execute(
                query="test query",
                action="start",
                background=True,
            )

        assert result.success is True

    def test_validation_errors_in_metadata(self, mock_config, mock_memory):
        """Validation errors should be available in metadata."""
        from foundry_mcp.core.research.workflows.deep_research import (
            DeepResearchWorkflow,
        )

        workflow = DeepResearchWorkflow(mock_config, mock_memory)
        result = workflow.execute(
            query="test query",
            action="start",
            max_iterations=MAX_ITERATIONS + 5,
        )

        assert result.success is False
        errors = result.metadata.get("validation_errors", [])
        assert len(errors) == 1
        assert "max_iterations" in errors[0]


# ──────────────────────────────────────────────────────────────────────
#  Constants sanity checks
# ──────────────────────────────────────────────────────────────────────


class TestValidationConstants:
    """Sanity checks for validation constant values."""

    def test_constants_are_positive(self):
        """All limits must be positive."""
        assert MAX_PROMPT_LENGTH > 0
        assert MAX_ITERATIONS > 0
        assert MAX_SUB_QUERIES > 0
        assert MAX_SOURCES_PER_QUERY > 0
        assert MAX_CONCURRENT_PROVIDERS > 0

    def test_constants_are_generous(self):
        """Limits should be generous enough for legitimate use."""
        assert MAX_ITERATIONS >= 10
        assert MAX_SUB_QUERIES >= 20
        assert MAX_SOURCES_PER_QUERY >= 50
        assert MAX_CONCURRENT_PROVIDERS >= 10
