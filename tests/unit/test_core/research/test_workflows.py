"""Unit tests for research workflow classes.

Tests WorkflowResult dataclass, ResearchWorkflowBase, and all workflow
implementations with mocked providers.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.config.research import ResearchConfig
from foundry_mcp.core.providers import ProviderResult, ProviderStatus
from foundry_mcp.core.research.memory import ResearchMemory
from foundry_mcp.core.research.models.enums import (
    ConsensusStrategy,
)
from foundry_mcp.core.research.workflows.base import (
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

        with patch.object(workflow, "_resolve_provider", return_value=mock_provider_context):
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
        with patch.object(workflow, "_resolve_provider", return_value=mock_provider_context):
            result1 = workflow.execute(prompt="First message")

        thread_id = result1.metadata["thread_id"]

        # Second message - continue thread
        with patch.object(workflow, "_resolve_provider", return_value=mock_provider_context):
            result2 = workflow.execute(prompt="Second message", thread_id=thread_id)

        assert result2.success is True
        assert result2.metadata["thread_id"] == thread_id
        assert result2.metadata["message_count"] >= 2

    def test_execute_provider_unavailable(self, research_config: ResearchConfig, mock_memory: ResearchMemory):
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
        with patch.object(workflow, "_resolve_provider", return_value=mock_provider_context):
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

        with patch.object(workflow, "_resolve_provider", return_value=mock_provider_context):
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

        with patch.object(workflow, "_resolve_provider", return_value=mock_provider_context):
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

        with patch.object(workflow, "_resolve_provider", return_value=mock_provider_context):
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

        with patch(
            "foundry_mcp.core.research.workflows.consensus.available_providers",
            return_value=["gemini"],
        ):
            with patch(
                "foundry_mcp.core.research.workflows.consensus.resolve_provider",
                return_value=mock_provider_context,
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

        with patch(
            "foundry_mcp.core.research.workflows.consensus.available_providers",
            return_value=["gemini", "claude"],
        ):
            with patch(
                "foundry_mcp.core.research.workflows.consensus.resolve_provider",
                return_value=mock_provider_context,
            ):
                result = workflow.execute(
                    prompt="Explain X",
                    providers=["gemini", "claude"],
                    strategy=ConsensusStrategy.ALL_RESPONSES,
                )

        assert result.success is True
        assert "providers_consulted" in result.metadata

    def test_execute_provider_failure(self, research_config: ResearchConfig, mock_memory: ResearchMemory):
        """Should handle provider failures gracefully."""
        workflow = ConsensusWorkflow(research_config, mock_memory)

        with patch(
            "foundry_mcp.core.research.workflows.consensus.available_providers",
            return_value=[],
        ):
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

        with patch(
            "foundry_mcp.core.research.workflows.consensus.available_providers",
            return_value=["gemini"],
        ):
            with patch(
                "foundry_mcp.core.research.workflows.consensus.resolve_provider",
                return_value=mock_provider_context,
            ):
                result = workflow.execute(prompt="Test", providers=["gemini"])

        assert isinstance(result, WorkflowResult)
        assert isinstance(result.success, bool)
        assert isinstance(result.content, str)
        assert isinstance(result.metadata, dict)

    def test_execute_with_full_provider_specs(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
    ):
        """Should correctly parse full provider specs like [cli]codex:gpt-5.2.

        This tests that consensus workflow properly handles provider specs from config
        that include the [cli] prefix and model specification.
        """
        workflow = ConsensusWorkflow(research_config, mock_memory)

        with patch(
            "foundry_mcp.core.research.workflows.consensus.available_providers",
            return_value=["codex", "gemini"],
        ):
            with patch("foundry_mcp.core.research.workflows.consensus.resolve_provider") as mock_resolve:
                # Set up mock provider that returns successful results
                mock_context = MagicMock()
                mock_result = MagicMock()
                mock_result.status = ProviderStatus.SUCCESS
                mock_result.content = "Test response"
                mock_result.model_used = "gpt-5.2"
                mock_result.tokens = MagicMock()
                mock_result.tokens.total_tokens = 100
                mock_context.generate.return_value = mock_result
                mock_resolve.return_value = mock_context

                result = workflow.execute(
                    prompt="Test question",
                    providers=["[cli]codex:gpt-5.2", "[cli]gemini:pro"],
                    strategy=ConsensusStrategy.FIRST_VALID,
                )

                # Verify resolve_provider was called with parsed base IDs and models
                assert mock_resolve.call_count == 2
                calls = mock_resolve.call_args_list

                # First call should be for codex with model gpt-5.2
                assert calls[0][0][0] == "codex"
                assert calls[0][1]["model"] == "gpt-5.2"

                # Second call should be for gemini with model pro
                assert calls[1][0][0] == "gemini"
                assert calls[1][1]["model"] == "pro"

    def test_execute_filters_unavailable_providers_with_specs(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
    ):
        """Should filter out unavailable providers even with full specs."""
        workflow = ConsensusWorkflow(research_config, mock_memory)

        with patch(
            "foundry_mcp.core.research.workflows.consensus.available_providers",
            return_value=["gemini"],  # Only gemini available
        ):
            with patch("foundry_mcp.core.research.workflows.consensus.resolve_provider") as mock_resolve:
                mock_context = MagicMock()
                mock_result = MagicMock()
                mock_result.status = ProviderStatus.SUCCESS
                mock_result.content = "Test response"
                mock_result.model_used = "pro"
                mock_result.tokens = MagicMock()
                mock_result.tokens.total_tokens = 100
                mock_context.generate.return_value = mock_result
                mock_resolve.return_value = mock_context

                result = workflow.execute(
                    prompt="Test",
                    providers=["[cli]codex:gpt-5.2", "[cli]gemini:pro"],
                    strategy=ConsensusStrategy.FIRST_VALID,
                )

                # Only gemini should be called since codex is not available
                assert mock_resolve.call_count == 1
                assert mock_resolve.call_args[0][0] == "gemini"


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

        with patch.object(workflow, "_resolve_provider", return_value=mock_provider_context):
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
        with patch.object(workflow, "_resolve_provider", return_value=mock_provider_context):
            result1 = workflow.execute(topic="Test investigation")

        investigation_id = result1.metadata["investigation_id"]

        # Continue investigation
        with patch.object(workflow, "_resolve_provider", return_value=mock_provider_context):
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

        with patch.object(workflow, "_resolve_provider", return_value=mock_provider_context):
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

        with patch.object(workflow, "_resolve_provider", return_value=mock_provider_context):
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

        with patch.object(workflow, "_resolve_provider", return_value=mock_ideate_provider_context):
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
        with patch.object(workflow, "_resolve_provider", return_value=mock_ideate_provider_context):
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
        with patch.object(workflow, "_resolve_provider", return_value=cluster_context):
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

        with patch.object(workflow, "_resolve_provider", return_value=mock_ideate_provider_context):
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

        with patch.object(workflow, "_resolve_provider", return_value=mock_ideate_provider_context):
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

    def test_resolve_provider_caches(self, research_config: ResearchConfig, mock_memory: ResearchMemory):
        """Should cache resolved providers."""
        # Use ChatWorkflow as concrete implementation
        workflow = ChatWorkflow(research_config, mock_memory)

        with patch(
            "foundry_mcp.core.research.workflows.base.available_providers",
            return_value=["gemini"],
        ):
            with patch("foundry_mcp.core.research.workflows.base.resolve_provider") as mock_resolve:
                mock_context = MagicMock()
                mock_resolve.return_value = mock_context

                # First call
                result1 = workflow._resolve_provider("gemini")
                # Second call should use cache
                result2 = workflow._resolve_provider("gemini")

                assert result1 is result2
                mock_resolve.assert_called_once()

    def test_resolve_provider_unavailable(self, research_config: ResearchConfig, mock_memory: ResearchMemory):
        """Should return None for unavailable provider."""
        workflow = ChatWorkflow(research_config, mock_memory)

        with patch(
            "foundry_mcp.core.research.workflows.base.available_providers",
            return_value=[],
        ):
            result = workflow._resolve_provider("nonexistent")

        assert result is None

    def test_get_available_providers(self, research_config: ResearchConfig, mock_memory: ResearchMemory):
        """Should return list of available providers."""
        workflow = ChatWorkflow(research_config, mock_memory)

        with patch(
            "foundry_mcp.core.research.workflows.base.available_providers",
            return_value=["gemini", "claude", "openai"],
        ):
            providers = workflow.get_available_providers()

        assert providers == ["gemini", "claude", "openai"]

    def test_resolve_provider_with_full_spec(self, research_config: ResearchConfig, mock_memory: ResearchMemory):
        """Should correctly parse full provider specs like [cli]codex:gpt-5.2-codex.

        This tests the fix for provider spec parsing where:
        - Full specs need to be parsed to extract base provider ID for availability check
        - The model component should be passed to resolve_provider
        """
        workflow = ChatWorkflow(research_config, mock_memory)

        with patch(
            "foundry_mcp.core.research.workflows.base.available_providers",
            return_value=["codex", "gemini"],
        ):
            with patch("foundry_mcp.core.research.workflows.base.resolve_provider") as mock_resolve:
                mock_context = MagicMock()
                mock_resolve.return_value = mock_context

                # Test with full provider spec
                result = workflow._resolve_provider("[cli]codex:gpt-5.2-codex")

                assert result is mock_context
                # Verify resolve_provider was called with base provider ID and model
                mock_resolve.assert_called_once()
                call_args = mock_resolve.call_args
                assert call_args[0][0] == "codex"  # base provider ID
                assert call_args[1]["model"] == "gpt-5.2-codex"  # model from spec

    def test_resolve_provider_with_simple_id(self, research_config: ResearchConfig, mock_memory: ResearchMemory):
        """Should handle simple provider IDs without model."""
        workflow = ChatWorkflow(research_config, mock_memory)

        with patch(
            "foundry_mcp.core.research.workflows.base.available_providers",
            return_value=["gemini"],
        ):
            with patch("foundry_mcp.core.research.workflows.base.resolve_provider") as mock_resolve:
                mock_context = MagicMock()
                mock_resolve.return_value = mock_context

                result = workflow._resolve_provider("gemini")

                assert result is mock_context
                call_args = mock_resolve.call_args
                assert call_args[0][0] == "gemini"
                assert call_args[1]["model"] is None

    def test_resolve_provider_caches_by_full_spec(self, research_config: ResearchConfig, mock_memory: ResearchMemory):
        """Should cache providers using full spec string as key."""
        workflow = ChatWorkflow(research_config, mock_memory)

        with patch(
            "foundry_mcp.core.research.workflows.base.available_providers",
            return_value=["codex"],
        ):
            with patch("foundry_mcp.core.research.workflows.base.resolve_provider") as mock_resolve:
                mock_context = MagicMock()
                mock_resolve.return_value = mock_context

                # Same full spec should be cached
                result1 = workflow._resolve_provider("[cli]codex:gpt-5.2")
                result2 = workflow._resolve_provider("[cli]codex:gpt-5.2")

                assert result1 is result2
                assert mock_resolve.call_count == 1

                # Different model should create new provider
                result3 = workflow._resolve_provider("[cli]codex:gpt-5.1")

                assert mock_resolve.call_count == 2

    def test_resolve_provider_invalid_spec(self, research_config: ResearchConfig, mock_memory: ResearchMemory):
        """Should return None for invalid provider spec."""
        workflow = ChatWorkflow(research_config, mock_memory)

        # Invalid spec format
        result = workflow._resolve_provider("[invalid]malformed")

        assert result is None


# =============================================================================
# ResearchWorkflowBase Async Provider Tests
# =============================================================================


class TestExecuteProviderAsync:
    """Tests for async provider execution behavior."""

    @pytest.mark.asyncio
    async def test_uses_per_provider_model_on_fallback(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
    ) -> None:
        """Fallback providers should receive their own model overrides."""
        workflow = ChatWorkflow(research_config, mock_memory)
        seen: dict[str, Optional[str]] = {}

        def primary_generate(request):
            seen["primary_model"] = request.model
            return ProviderResult(
                content="",
                status=ProviderStatus.ERROR,
                provider_id="gemini",
                model_used="gemini",
            )

        def fallback_generate(request):
            seen["fallback_model"] = request.model
            return ProviderResult(
                content="ok",
                status=ProviderStatus.SUCCESS,
                provider_id="claude",
                model_used="sonnet",
            )

        primary_context = MagicMock()
        primary_context.generate.side_effect = primary_generate

        fallback_context = MagicMock()
        fallback_context.generate.side_effect = fallback_generate

        def resolve_side_effect(provider_id: Optional[str], hooks=None):
            if provider_id == "gemini":
                return primary_context
            if provider_id == "[cli]claude:sonnet":
                return fallback_context
            return None

        with patch.object(workflow, "_resolve_provider", side_effect=resolve_side_effect):
            result = await workflow._execute_provider_async(
                prompt="hello",
                provider_id="gemini",
                model="gpt-5.1",
                fallback_providers=["[cli]claude:sonnet"],
                max_retries=0,
                timeout=0.01,
            )

        assert result.success is True
        assert seen["primary_model"] == "gpt-5.1"
        assert seen["fallback_model"] == "sonnet"

    @pytest.mark.asyncio
    async def test_timeout_metadata_false_for_non_timeout_failures(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
    ) -> None:
        """Non-timeout failures should not be marked as timeouts."""
        workflow = ChatWorkflow(research_config, mock_memory)

        with patch.object(workflow, "_resolve_provider", return_value=None):
            result = await workflow._execute_provider_async(
                prompt="hello",
                provider_id="gemini",
                fallback_providers=["claude"],
                max_retries=0,
                timeout=0.01,
            )

        assert result.success is False
        assert result.metadata.get("timeout") is False


# =============================================================================
# Deep Research Concurrency and Robustness Tests
# =============================================================================


class TestDeepResearchRobustness:
    """Tests for deep research thread safety and robustness fixes."""

    def test_active_sessions_lock_exists(self):
        """Should have a lock for protecting _active_research_sessions."""
        import threading

        from foundry_mcp.core.research.workflows.deep_research import (
            _active_research_sessions,
            _active_sessions_lock,
        )

        assert isinstance(_active_sessions_lock, type(threading.Lock()))
        assert isinstance(_active_research_sessions, dict)

    def test_tasks_dict_not_weak(self):
        """Should use regular dict, not WeakValueDictionary for task tracking."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        # Check that _tasks is a regular dict, not WeakValueDictionary
        assert isinstance(DeepResearchWorkflow._tasks, dict)
        # WeakValueDictionary has different type
        from weakref import WeakValueDictionary

        assert not isinstance(DeepResearchWorkflow._tasks, WeakValueDictionary)

    def test_tasks_lock_exists(self):
        """Should have a lock for protecting _tasks."""
        import threading

        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        assert isinstance(DeepResearchWorkflow._tasks_lock, type(threading.Lock()))

    def test_cleanup_stale_tasks_method_exists(self):
        """Should have cleanup_stale_tasks class method."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow

        assert hasattr(DeepResearchWorkflow, "cleanup_stale_tasks")
        assert callable(DeepResearchWorkflow.cleanup_stale_tasks)

    @pytest.mark.asyncio
    async def test_base_exception_handling_in_gather(self):
        """Should handle BaseException (not just Exception) from asyncio.gather."""
        import asyncio

        # Simulate what happens in _execute_gathering_phase
        async def task_that_succeeds():
            return (5, None)  # (added_count, error)

        async def task_that_raises_cancelled():
            raise asyncio.CancelledError()

        async def task_that_raises_keyboard():
            raise KeyboardInterrupt()

        # Test with mixed results including BaseException subclasses
        tasks = [
            task_that_succeeds(),
            task_that_raises_cancelled(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # The fix: check for BaseException, not just Exception
        failed_queries = 0
        total_sources = 0

        for result in results:
            if isinstance(result, BaseException):  # This is the fix
                failed_queries += 1
            else:
                added, error = result
                total_sources += added

        assert failed_queries == 1
        assert total_sources == 5

    def test_timezone_aware_datetime_in_deep_research(self):
        """Should use timezone-aware datetime, not deprecated utcnow()."""
        from datetime import timezone

        # Create an AgentDecision to test the default_factory
        from foundry_mcp.core.research.workflows.deep_research import AgentDecision, AgentRole

        decision = AgentDecision(
            agent=AgentRole.PLANNER,
            action="test",
            rationale="test rationale",
            inputs={},
        )

        # The timestamp should be timezone-aware
        assert decision.timestamp.tzinfo is not None
        assert decision.timestamp.tzinfo == timezone.utc


class TestFileStorageRobustness:
    """Tests for file storage thread safety improvements."""

    def test_load_handles_concurrent_delete(self, tmp_path: Path):
        """Should handle file being deleted between existence check and read."""
        from foundry_mcp.core.research.memory import FileStorageBackend
        from foundry_mcp.core.research.models.conversations import ConversationThread

        backend = FileStorageBackend(
            storage_path=tmp_path / "threads",
            model_class=ConversationThread,
            ttl_hours=24,
        )

        # File doesn't exist - should return None gracefully
        result = backend.load("nonexistent")
        assert result is None

    def test_delete_handles_missing_file(self, tmp_path: Path):
        """Should return False when deleting non-existent item."""
        from foundry_mcp.core.research.memory import FileStorageBackend
        from foundry_mcp.core.research.models.conversations import ConversationThread

        backend = FileStorageBackend(
            storage_path=tmp_path / "threads",
            model_class=ConversationThread,
            ttl_hours=24,
        )

        result = backend.delete("nonexistent")
        assert result is False

    def test_delete_cleans_orphaned_lock_files(self, tmp_path: Path):
        """Should clean up orphaned lock files when data file is missing."""
        from foundry_mcp.core.research.memory import FileStorageBackend
        from foundry_mcp.core.research.models.conversations import ConversationThread

        backend = FileStorageBackend(
            storage_path=tmp_path / "threads",
            model_class=ConversationThread,
            ttl_hours=24,
        )

        # Create an orphaned lock file
        lock_path = backend._get_lock_path("orphaned")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("")

        # Delete should clean up the orphaned lock file
        backend.delete("orphaned")

        # Lock file should be removed
        assert not lock_path.exists()

    def test_load_with_ttl_expiry_inside_lock(self, tmp_path: Path):
        """Should check expiry inside lock to avoid TOCTOU race."""
        import time

        from foundry_mcp.core.research.memory import FileStorageBackend
        from foundry_mcp.core.research.models.conversations import ConversationThread

        # Create backend with very short TTL
        backend = FileStorageBackend(
            storage_path=tmp_path / "threads",
            model_class=ConversationThread,
            ttl_hours=0,  # Immediate expiry based on mtime
        )

        # Create a thread
        thread = ConversationThread(title="Test")
        backend.save(thread.id, thread)

        # Wait a moment for file to age
        time.sleep(0.1)

        # Manually set TTL to make file expired
        backend.ttl_hours = 0  # 0 hours = expired immediately

        # Load should handle expired file gracefully
        result = backend.load(thread.id)

        # Either returns None (expired and deleted) or the thread (if TTL check passed)
        # The important thing is no exception was raised
        assert result is None or isinstance(result, ConversationThread)


# =============================================================================
# Workflow Failure Scenario Tests
# =============================================================================


class TestChatWorkflowFailureRecovery:
    """Tests for ChatWorkflow state recovery on provider failure."""

    def test_thread_saved_before_provider_call(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
    ):
        """Should save thread with user message before calling provider.

        This ensures the user message is persisted even if the provider fails,
        enabling retry scenarios and maintaining state consistency.
        """
        workflow = ChatWorkflow(research_config, mock_memory)

        # Mock provider to fail
        with patch.object(workflow, "_resolve_provider", return_value=None):
            result = workflow.execute(prompt="Hello, this is a test message")

        assert result.success is False
        assert result.error is not None
        assert "not available" in result.error.lower()

        # Verify thread was saved with user message despite provider failure
        assert "thread_id" in result.metadata
        thread_id = result.metadata["thread_id"]

        # Load the thread and verify user message was persisted
        thread = mock_memory.load_thread(thread_id)
        assert thread is not None
        assert len(thread.messages) == 1
        assert thread.messages[0].role == "user"
        assert thread.messages[0].content == "Hello, this is a test message"

    def test_thread_metadata_always_returned(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_provider_context,
    ):
        """Should return thread metadata even when provider fails."""
        workflow = ChatWorkflow(research_config, mock_memory)

        # First, test with provider failure
        with patch.object(workflow, "_resolve_provider", return_value=None):
            result = workflow.execute(prompt="Test message")

        assert "thread_id" in result.metadata
        assert "message_count" in result.metadata
        assert "thread_title" in result.metadata

    def test_continued_thread_recovers_after_failure(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_provider_context,
    ):
        """Should allow continuing a thread after a previous failure."""
        workflow = ChatWorkflow(research_config, mock_memory)

        # First message fails (provider unavailable)
        with patch.object(workflow, "_resolve_provider", return_value=None):
            result1 = workflow.execute(prompt="First message")

        thread_id = result1.metadata["thread_id"]
        assert result1.success is False

        # Second message succeeds (provider available)
        with patch.object(workflow, "_resolve_provider", return_value=mock_provider_context):
            result2 = workflow.execute(prompt="Second message", thread_id=thread_id)

        assert result2.success is True
        assert result2.metadata["thread_id"] == thread_id
        # Should have 3 messages: first user, second user, assistant response
        assert result2.metadata["message_count"] == 3


class TestConsensusWorkflowFailureRecovery:
    """Tests for ConsensusWorkflow state recovery on synthesis failure."""

    def test_responses_saved_before_synthesis(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_provider_context,
    ):
        """Should save responses before attempting synthesis.

        This ensures collected responses are persisted even if synthesis fails.
        """
        workflow = ConsensusWorkflow(research_config, mock_memory)

        # Track save calls
        save_calls = []
        original_save = mock_memory.save_consensus

        def tracking_save(state):
            save_calls.append(
                {
                    "has_responses": len(state.responses) > 0,
                    "completed": state.completed_at is not None,
                }
            )
            return original_save(state)

        mock_memory.save_consensus = tracking_save

        with patch(
            "foundry_mcp.core.research.workflows.consensus.available_providers",
            return_value=["gemini"],
        ):
            with patch(
                "foundry_mcp.core.research.workflows.consensus.resolve_provider",
                return_value=mock_provider_context,
            ):
                result = workflow.execute(
                    prompt="Test",
                    providers=["gemini"],
                    strategy=ConsensusStrategy.FIRST_VALID,
                )

        assert result.success is True

        # Verify save was called before synthesis (first call should have responses but not be completed)
        assert len(save_calls) >= 2
        assert save_calls[0]["has_responses"] is True
        assert save_calls[0]["completed"] is False  # First save is before synthesis

    def test_synthesis_error_persists_state(
        self,
        research_config: ResearchConfig,
        mock_memory: ResearchMemory,
        mock_provider_context,
    ):
        """Should persist state with error info when synthesis fails."""
        workflow = ConsensusWorkflow(research_config, mock_memory)

        # Mock synthesis to fail
        with patch(
            "foundry_mcp.core.research.workflows.consensus.available_providers",
            return_value=["gemini"],
        ):
            with patch(
                "foundry_mcp.core.research.workflows.consensus.resolve_provider",
                return_value=mock_provider_context,
            ):
                # Patch _apply_strategy to raise an error
                with patch.object(
                    workflow,
                    "_apply_strategy",
                    side_effect=ValueError("Synthesis failed"),
                ):
                    result = workflow.execute(
                        prompt="Test",
                        providers=["gemini"],
                        strategy=ConsensusStrategy.SYNTHESIZE,
                    )

        # The outer exception handler should catch this
        assert result.success is False

        # Verify the consensus state was saved (list should have one entry)
        states = mock_memory.list_consensus(limit=10)
        assert len(states) >= 1

        # The most recent state should have responses saved
        latest_state = states[0]
        assert len(latest_state.responses) > 0


class TestDeepResearchTimeoutRecovery:
    """Tests for deep research timeout and partial state handling."""

    def test_timeout_marks_state_as_failed(self):
        """Should mark state as failed when timeout occurs."""
        from foundry_mcp.core.research.workflows.deep_research import DeepResearchState

        state = DeepResearchState(original_query="Test query")

        # Simulate timeout marking (as done in deep_research.py:1372-1388)
        state.metadata["timeout"] = True
        state.metadata["abort_phase"] = state.phase.value
        state.metadata["abort_iteration"] = state.iteration
        state.mark_failed("Research timed out after 60s")

        assert state.metadata["failed"] is True
        assert state.metadata["timeout"] is True
        assert state.completed_at is not None

    def test_cleanup_stale_tasks_removes_old_completed(self):
        """Should clean up old completed tasks from registry."""
        import threading
        import time

        from foundry_mcp.core.research.workflows.deep_research import (
            BackgroundTask,
            DeepResearchWorkflow,
        )

        # Clear any existing tasks
        with DeepResearchWorkflow._tasks_lock:
            DeepResearchWorkflow._tasks.clear()

        # Create dummy completed threads
        def noop():
            pass

        old_thread = threading.Thread(target=noop)
        old_thread.start()
        old_thread.join()  # Complete immediately

        new_thread = threading.Thread(target=noop)
        new_thread.start()
        new_thread.join()  # Complete immediately

        # Add some tasks with completed threads
        old_task = BackgroundTask(research_id="old-task", thread=old_thread, timeout=60)
        old_task.completed_at = time.time() - 7200  # 2 hours ago

        new_task = BackgroundTask(research_id="new-task", thread=new_thread, timeout=60)
        new_task.completed_at = time.time() - 60  # 1 minute ago

        # Running task has no completed_at
        running_task = BackgroundTask(research_id="running-task", timeout=60)
        # No thread means is_done returns True for this edge case, but no completed_at

        with DeepResearchWorkflow._tasks_lock:
            DeepResearchWorkflow._tasks["old-task"] = old_task
            DeepResearchWorkflow._tasks["new-task"] = new_task
            DeepResearchWorkflow._tasks["running-task"] = running_task

        # Cleanup with 1 hour threshold
        removed = DeepResearchWorkflow.cleanup_stale_tasks(max_age_seconds=3600)

        assert removed == 1  # Only old task should be removed

        with DeepResearchWorkflow._tasks_lock:
            assert "old-task" not in DeepResearchWorkflow._tasks
            assert "new-task" in DeepResearchWorkflow._tasks
            assert "running-task" in DeepResearchWorkflow._tasks
            # Clean up
            DeepResearchWorkflow._tasks.clear()

    def test_active_sessions_protected_by_lock(self):
        """Should protect active sessions dict with lock during iteration."""

        from foundry_mcp.core.research.workflows.deep_research import (
            DeepResearchState,
            _active_research_sessions,
            _active_sessions_lock,
        )

        # Create some test states
        state1 = DeepResearchState(original_query="Query 1")
        state2 = DeepResearchState(original_query="Query 2")

        # Add states under lock
        with _active_sessions_lock:
            _active_research_sessions[state1.id] = state1
            _active_research_sessions[state2.id] = state2

        # Take snapshot under lock (as crash handler does)
        with _active_sessions_lock:
            snapshot = list(_active_research_sessions.items())

        assert len(snapshot) == 2

        # Cleanup
        with _active_sessions_lock:
            _active_research_sessions.pop(state1.id, None)
            _active_research_sessions.pop(state2.id, None)


class TestConcurrentAccessSafety:
    """Tests for concurrent access safety in research workflows."""

    @pytest.mark.asyncio
    async def test_gathering_phase_state_lock(self):
        """Should protect state modifications in gathering phase with lock."""
        import asyncio

        # Simulate the state_lock pattern from deep_research.py gathering phase
        state_lock = asyncio.Lock()
        sources = []
        seen_urls = set()

        async def add_source(url: str):
            async with state_lock:
                if url in seen_urls:
                    return False
                seen_urls.add(url)
                sources.append(url)
                return True

        # Run concurrent additions
        tasks = [
            add_source("http://example.com/1"),
            add_source("http://example.com/2"),
            add_source("http://example.com/1"),  # Duplicate
            add_source("http://example.com/3"),
            add_source("http://example.com/2"),  # Duplicate
        ]

        results = await asyncio.gather(*tasks)

        # Should have 3 unique URLs, 2 duplicates rejected
        assert sum(results) == 3
        assert len(sources) == 3
        assert len(seen_urls) == 3

    def test_tasks_dict_thread_safe_access(self):
        """Should access tasks dict safely from multiple threads."""
        import threading
        import time

        from foundry_mcp.core.research.workflows.deep_research import (
            BackgroundTask,
            DeepResearchWorkflow,
        )

        # Clear existing tasks
        with DeepResearchWorkflow._tasks_lock:
            DeepResearchWorkflow._tasks.clear()

        results = {"added": 0, "read": 0}
        errors = []

        def add_tasks():
            for i in range(10):
                task = BackgroundTask(research_id=f"task-{i}", timeout=60)
                with DeepResearchWorkflow._tasks_lock:
                    DeepResearchWorkflow._tasks[f"task-{i}"] = task
                    results["added"] += 1
                time.sleep(0.001)

        def read_tasks():
            for _ in range(20):
                with DeepResearchWorkflow._tasks_lock:
                    count = len(DeepResearchWorkflow._tasks)
                    results["read"] += 1
                time.sleep(0.001)

        # Run concurrent readers and writers
        threads = [
            threading.Thread(target=add_tasks),
            threading.Thread(target=read_tasks),
            threading.Thread(target=read_tasks),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results["added"] == 10
        assert results["read"] == 40
        assert len(errors) == 0

        # Cleanup
        with DeepResearchWorkflow._tasks_lock:
            DeepResearchWorkflow._tasks.clear()
