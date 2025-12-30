"""Unit tests for research workflow classes.

Tests WorkflowResult dataclass, ResearchWorkflowBase, and all workflow
implementations with mocked providers.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.config import ResearchConfig
from foundry_mcp.core.research.memory import ResearchMemory
from foundry_mcp.core.research.models import (
    ConfidenceLevel,
    ConsensusStrategy,
    IdeationPhase,
    ThreadStatus,
)
from foundry_mcp.core.research.workflows.base import (
    ResearchWorkflowBase,
    WorkflowResult,
)
from foundry_mcp.core.research.workflows.chat import ChatWorkflow
from foundry_mcp.core.research.workflows.consensus import ConsensusWorkflow
from foundry_mcp.core.research.workflows.ideate import IdeateWorkflow
from foundry_mcp.core.research.workflows.thinkdeep import ThinkDeepWorkflow


# =============================================================================
# WorkflowResult Dataclass Tests
# =============================================================================


class TestWorkflowResult:
    """Tests for WorkflowResult dataclass."""

    def test_creation_success(self):
        """Should create a successful result."""
        result = WorkflowResult(
            success=True,
            content="Generated response",
            provider_id="gemini",
            model_used="gemini-2.0-flash",
            tokens_used=150,
            duration_ms=1234.5,
        )
        assert result.success is True
        assert result.content == "Generated response"
        assert result.provider_id == "gemini"
        assert result.model_used == "gemini-2.0-flash"
        assert result.tokens_used == 150
        assert result.duration_ms == 1234.5
        assert result.error is None
        assert result.metadata == {}

    def test_creation_failure(self):
        """Should create a failure result."""
        result = WorkflowResult(
            success=False,
            content="",
            error="Provider timeout after 30s",
        )
        assert result.success is False
        assert result.content == ""
        assert result.error == "Provider timeout after 30s"

    def test_metadata_default(self):
        """Should default metadata to empty dict via __post_init__."""
        result = WorkflowResult(success=True, content="test")
        assert result.metadata == {}
        assert isinstance(result.metadata, dict)

    def test_metadata_custom(self):
        """Should preserve custom metadata."""
        result = WorkflowResult(
            success=True,
            content="test",
            metadata={"thread_id": "t-123", "message_count": 5},
        )
        assert result.metadata["thread_id"] == "t-123"
        assert result.metadata["message_count"] == 5

    def test_minimal_creation(self):
        """Should create with only required fields."""
        result = WorkflowResult(success=True, content="minimal")
        assert result.success is True
        assert result.content == "minimal"
        assert result.provider_id is None
        assert result.model_used is None
        assert result.tokens_used is None
        assert result.duration_ms is None
        assert result.error is None

    def test_asdict_conversion(self):
        """Should convert to dict correctly."""
        result = WorkflowResult(
            success=True,
            content="test",
            provider_id="openai",
            metadata={"key": "value"},
        )
        data = asdict(result)
        assert data["success"] is True
        assert data["content"] == "test"
        assert data["provider_id"] == "openai"
        assert data["metadata"] == {"key": "value"}


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def research_config(tmp_path: Path) -> ResearchConfig:
    """Create a ResearchConfig for testing."""
    return ResearchConfig(
        enabled=True,
        storage_path=str(tmp_path / "research"),
        ttl_hours=24,
        default_provider="gemini",
        consensus_providers=["gemini", "claude"],
        thinkdeep_max_depth=3,
        ideate_perspectives=["technical", "creative"],
    )


@pytest.fixture
def mock_memory(tmp_path: Path) -> ResearchMemory:
    """Create a ResearchMemory instance for testing."""
    return ResearchMemory(base_path=tmp_path / "memory", ttl_hours=24)


@pytest.fixture
def mock_provider_result():
    """Create a mock provider result factory."""
    from foundry_mcp.core.providers import ProviderResult, ProviderStatus, TokenUsage

    def _create(
        content: str = "Mock response",
        success: bool = True,
        provider_id: str = "gemini",
        model: str = "gemini-2.0-flash",
    ):
        return ProviderResult(
            content=content,
            status=ProviderStatus.SUCCESS if success else ProviderStatus.ERROR,
            provider_id=provider_id,
            model_used=model,
            tokens=TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150),
            duration_ms=500.0,
        )

    return _create


@pytest.fixture
def mock_ideate_provider_context():
    """Create a mock provider context that returns ideation-formatted response."""
    from foundry_mcp.core.providers import ProviderResult, ProviderStatus, TokenUsage

    context = MagicMock()
    # Return bullet-formatted ideas that _parse_ideas can parse
    context.generate.return_value = ProviderResult(
        content="- First creative idea for the topic\n- Second innovative idea\n- Third practical suggestion",
        status=ProviderStatus.SUCCESS,
        provider_id="gemini",
        model_used="gemini-2.0-flash",
        tokens=TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150),
        duration_ms=500.0,
    )
    return context


@pytest.fixture
def mock_provider_context(mock_provider_result):
    """Create a mock provider context."""
    context = MagicMock()
    context.generate.return_value = mock_provider_result()
    return context


# =============================================================================
# ChatWorkflow Tests
# =============================================================================


class TestChatWorkflow:
    """Tests for ChatWorkflow class with mocked providers."""

    def test_init(self, research_config: ResearchConfig, mock_memory: ResearchMemory):
        """Should initialize with config and memory."""
        workflow = ChatWorkflow(research_config, mock_memory)
        assert workflow.config == research_config
        assert workflow.memory == mock_memory

    def test_execute_new_thread(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_provider_context,
    ):
        """Should create new thread and return response."""
        workflow = ChatWorkflow(research_config, mock_memory)

        with patch.object(
            workflow, "_resolve_provider", return_value=mock_provider_context
        ):
            result = workflow.execute(prompt="Hello, how are you?")

        assert result.success is True
        assert result.content == "Mock response"
        assert "thread_id" in result.metadata

    def test_execute_continue_thread(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_provider_context,
    ):
        """Should continue existing thread."""
        workflow = ChatWorkflow(research_config, mock_memory)

        # First message - create thread
        with patch.object(
            workflow, "_resolve_provider", return_value=mock_provider_context
        ):
            result1 = workflow.execute(prompt="First message")

        thread_id = result1.metadata["thread_id"]

        # Second message - continue thread
        with patch.object(
            workflow, "_resolve_provider", return_value=mock_provider_context
        ):
            result2 = workflow.execute(prompt="Second message", thread_id=thread_id)

        assert result2.success is True
        assert result2.metadata["thread_id"] == thread_id
        assert result2.metadata["message_count"] >= 2

    def test_execute_provider_unavailable(
        self, research_config: ResearchConfig, mock_memory: ResearchMemory
    ):
        """Should return error when provider unavailable."""
        workflow = ChatWorkflow(research_config, mock_memory)

        with patch.object(workflow, "_resolve_provider", return_value=None):
            result = workflow.execute(prompt="Hello")

        assert result.success is False
        assert "not available" in result.error.lower()

    def test_list_threads(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_provider_context,
    ):
        """Should list created threads."""
        workflow = ChatWorkflow(research_config, mock_memory)

        # Create some threads
        with patch.object(
            workflow, "_resolve_provider", return_value=mock_provider_context
        ):
            workflow.execute(prompt="Thread 1", title="First Thread")
            workflow.execute(prompt="Thread 2", title="Second Thread")

        threads = workflow.list_threads()
        assert len(threads) >= 2

    def test_get_thread(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_provider_context,
    ):
        """Should get thread by ID."""
        workflow = ChatWorkflow(research_config, mock_memory)

        with patch.object(
            workflow, "_resolve_provider", return_value=mock_provider_context
        ):
            result = workflow.execute(prompt="Test", title="My Thread")

        thread_id = result.metadata["thread_id"]
        thread = workflow.get_thread(thread_id)

        assert thread is not None
        assert thread["id"] == thread_id

    def test_delete_thread(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_provider_context,
    ):
        """Should delete thread."""
        workflow = ChatWorkflow(research_config, mock_memory)

        with patch.object(
            workflow, "_resolve_provider", return_value=mock_provider_context
        ):
            result = workflow.execute(prompt="Test")

        thread_id = result.metadata["thread_id"]
        assert workflow.delete_thread(thread_id) is True
        assert workflow.get_thread(thread_id) is None

    def test_response_structure(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_provider_context,
    ):
        """Should return properly structured response."""
        workflow = ChatWorkflow(research_config, mock_memory)

        with patch.object(
            workflow, "_resolve_provider", return_value=mock_provider_context
        ):
            result = workflow.execute(prompt="Test")

        # Verify structure
        assert isinstance(result, WorkflowResult)
        assert isinstance(result.success, bool)
        assert isinstance(result.content, str)
        assert isinstance(result.metadata, dict)
        assert "thread_id" in result.metadata
        assert "message_count" in result.metadata


# =============================================================================
# ConsensusWorkflow Tests
# =============================================================================


class TestConsensusWorkflow:
    """Tests for ConsensusWorkflow class with mocked providers."""

    def test_init(self, research_config: ResearchConfig, mock_memory: ResearchMemory):
        """Should initialize with config and memory."""
        workflow = ConsensusWorkflow(research_config, mock_memory)
        assert workflow.config == research_config

    def test_execute_single_provider(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_provider_context,
    ):
        """Should execute with single provider."""
        workflow = ConsensusWorkflow(research_config, mock_memory)

        with patch.object(
            workflow, "_resolve_provider", return_value=mock_provider_context
        ):
            result = workflow.execute(
                prompt="What is 2+2?",
                providers=["gemini"],
                strategy=ConsensusStrategy.FIRST_VALID,
            )

        assert result.success is True
        assert result.content is not None

    def test_execute_all_responses_strategy(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_provider_context,
    ):
        """Should return all responses with ALL_RESPONSES strategy."""
        workflow = ConsensusWorkflow(research_config, mock_memory)

        with patch.object(
            workflow, "_resolve_provider", return_value=mock_provider_context
        ):
            result = workflow.execute(
                prompt="Explain X",
                providers=["gemini", "claude"],
                strategy=ConsensusStrategy.ALL_RESPONSES,
            )

        assert result.success is True
        assert "providers_consulted" in result.metadata

    def test_execute_provider_failure(
        self, research_config: ResearchConfig, mock_memory: ResearchMemory
    ):
        """Should handle provider failures gracefully."""
        workflow = ConsensusWorkflow(research_config, mock_memory)

        with patch.object(workflow, "_resolve_provider", return_value=None):
            result = workflow.execute(
                prompt="Test",
                providers=["nonexistent"],
            )

        assert result.success is False

    def test_response_structure(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_provider_context,
    ):
        """Should return properly structured response."""
        workflow = ConsensusWorkflow(research_config, mock_memory)

        with patch.object(
            workflow, "_resolve_provider", return_value=mock_provider_context
        ):
            result = workflow.execute(prompt="Test", providers=["gemini"])

        assert isinstance(result, WorkflowResult)
        assert isinstance(result.success, bool)
        assert isinstance(result.content, str)
        assert isinstance(result.metadata, dict)


# =============================================================================
# ThinkDeepWorkflow Tests
# =============================================================================


class TestThinkDeepWorkflow:
    """Tests for ThinkDeepWorkflow class with mocked providers."""

    def test_init(self, research_config: ResearchConfig, mock_memory: ResearchMemory):
        """Should initialize with config and memory."""
        workflow = ThinkDeepWorkflow(research_config, mock_memory)
        assert workflow.config == research_config

    def test_execute_new_investigation(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_provider_context,
    ):
        """Should start new investigation with topic."""
        workflow = ThinkDeepWorkflow(research_config, mock_memory)

        with patch.object(
            workflow, "_resolve_provider", return_value=mock_provider_context
        ):
            result = workflow.execute(topic="Why do databases use B-trees?")

        assert result.success is True
        assert "investigation_id" in result.metadata
        assert "current_depth" in result.metadata

    def test_execute_continue_investigation(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_provider_context,
    ):
        """Should continue existing investigation."""
        workflow = ThinkDeepWorkflow(research_config, mock_memory)

        # Start investigation
        with patch.object(
            workflow, "_resolve_provider", return_value=mock_provider_context
        ):
            result1 = workflow.execute(topic="Test investigation")

        investigation_id = result1.metadata["investigation_id"]

        # Continue investigation
        with patch.object(
            workflow, "_resolve_provider", return_value=mock_provider_context
        ):
            result2 = workflow.execute(
                investigation_id=investigation_id,
                query="What else should we consider?",
            )

        assert result2.success is True
        assert result2.metadata["investigation_id"] == investigation_id

    def test_execute_max_depth(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_provider_context,
    ):
        """Should respect max_depth configuration."""
        workflow = ThinkDeepWorkflow(research_config, mock_memory)

        with patch.object(
            workflow, "_resolve_provider", return_value=mock_provider_context
        ):
            result = workflow.execute(topic="Test", max_depth=2)

        assert result.metadata.get("max_depth", research_config.thinkdeep_max_depth) <= 3

    def test_response_structure(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_provider_context,
    ):
        """Should return properly structured response."""
        workflow = ThinkDeepWorkflow(research_config, mock_memory)

        with patch.object(
            workflow, "_resolve_provider", return_value=mock_provider_context
        ):
            result = workflow.execute(topic="Test topic")

        assert isinstance(result, WorkflowResult)
        assert isinstance(result.success, bool)
        assert isinstance(result.content, str)
        assert isinstance(result.metadata, dict)
        # ThinkDeep-specific fields
        assert "investigation_id" in result.metadata
        assert "current_depth" in result.metadata


# =============================================================================
# IdeateWorkflow Tests
# =============================================================================


class TestIdeateWorkflow:
    """Tests for IdeateWorkflow class with mocked providers."""

    def test_init(self, research_config: ResearchConfig, mock_memory: ResearchMemory):
        """Should initialize with config and memory."""
        workflow = IdeateWorkflow(research_config, mock_memory)
        assert workflow.config == research_config

    def test_execute_generate_ideas(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_ideate_provider_context,
    ):
        """Should generate ideas for a topic."""
        workflow = IdeateWorkflow(research_config, mock_memory)

        with patch.object(
            workflow, "_resolve_provider", return_value=mock_ideate_provider_context
        ):
            result = workflow.execute(
                topic="New features for the app",
                action="generate",
            )

        assert result.success is True
        assert "ideation_id" in result.metadata
        assert "phase" in result.metadata

    def test_execute_cluster_ideas(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_ideate_provider_context,
    ):
        """Should cluster existing ideas."""
        workflow = IdeateWorkflow(research_config, mock_memory)

        # First generate ideas
        with patch.object(
            workflow, "_resolve_provider", return_value=mock_ideate_provider_context
        ):
            result1 = workflow.execute(
                topic="Test ideas",
                action="generate",
            )

        assert result1.success is True
        ideation_id = result1.metadata["ideation_id"]

        # Mock cluster response
        from foundry_mcp.core.providers import ProviderResult, ProviderStatus, TokenUsage

        cluster_context = MagicMock()
        cluster_context.generate.return_value = ProviderResult(
            content="CLUSTER: Technical Ideas\nDESCRIPTION: Technical improvements\nIDEAS: 1, 2\n\nCLUSTER: User Ideas\nDESCRIPTION: User-facing features\nIDEAS: 3",
            status=ProviderStatus.SUCCESS,
            provider_id="gemini",
            model_used="gemini-2.0-flash",
            tokens=TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150),
            duration_ms=500.0,
        )

        # Then cluster them
        with patch.object(
            workflow, "_resolve_provider", return_value=cluster_context
        ):
            result2 = workflow.execute(
                ideation_id=ideation_id,
                action="cluster",
            )

        assert result2.success is True

    def test_execute_with_perspectives(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_ideate_provider_context,
    ):
        """Should generate ideas from multiple perspectives."""
        workflow = IdeateWorkflow(research_config, mock_memory)

        with patch.object(
            workflow, "_resolve_provider", return_value=mock_ideate_provider_context
        ):
            result = workflow.execute(
                topic="Product improvements",
                action="generate",
                perspectives=["user", "developer", "business"],
            )

        assert result.success is True

    def test_response_structure(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_ideate_provider_context,
    ):
        """Should return properly structured response."""
        workflow = IdeateWorkflow(research_config, mock_memory)

        with patch.object(
            workflow, "_resolve_provider", return_value=mock_ideate_provider_context
        ):
            result = workflow.execute(topic="Test", action="generate")

        assert isinstance(result, WorkflowResult)
        assert isinstance(result.success, bool)
        assert isinstance(result.content, str)
        assert isinstance(result.metadata, dict)
        # Ideate-specific fields
        assert "ideation_id" in result.metadata
        assert "phase" in result.metadata


# =============================================================================
# ResearchWorkflowBase Tests
# =============================================================================


class TestResearchWorkflowBase:
    """Tests for ResearchWorkflowBase abstract class."""

    def test_resolve_provider_caches(
        self, research_config: ResearchConfig, mock_memory: ResearchMemory
    ):
        """Should cache resolved providers."""
        # Use ChatWorkflow as concrete implementation
        workflow = ChatWorkflow(research_config, mock_memory)

        with patch(
            "foundry_mcp.core.research.workflows.base.available_providers",
            return_value=["gemini"],
        ):
            with patch(
                "foundry_mcp.core.research.workflows.base.resolve_provider"
            ) as mock_resolve:
                mock_context = MagicMock()
                mock_resolve.return_value = mock_context

                # First call
                result1 = workflow._resolve_provider("gemini")
                # Second call should use cache
                result2 = workflow._resolve_provider("gemini")

                assert result1 is result2
                mock_resolve.assert_called_once()

    def test_resolve_provider_unavailable(
        self, research_config: ResearchConfig, mock_memory: ResearchMemory
    ):
        """Should return None for unavailable provider."""
        workflow = ChatWorkflow(research_config, mock_memory)

        with patch(
            "foundry_mcp.core.research.workflows.base.available_providers",
            return_value=[],
        ):
            result = workflow._resolve_provider("nonexistent")

        assert result is None

    def test_get_available_providers(
        self, research_config: ResearchConfig, mock_memory: ResearchMemory
    ):
        """Should return list of available providers."""
        workflow = ChatWorkflow(research_config, mock_memory)

        with patch(
            "foundry_mcp.core.research.workflows.base.available_providers",
            return_value=["gemini", "claude", "openai"],
        ):
            providers = workflow.get_available_providers()

        assert providers == ["gemini", "claude", "openai"]
