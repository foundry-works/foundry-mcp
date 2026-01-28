"""
Integration tests for Deep Research workflow with Tavily configuration.

Tests the integration between ResearchConfig Tavily settings and DeepResearchWorkflow,
including:
- Research-mode smart defaults
- Config override behavior
- Tavily search parameter propagation
- Extract follow-up integration
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.config import ResearchConfig
from foundry_mcp.core.research.models import (
    DeepResearchPhase,
    DeepResearchState,
    ResearchMode,
    ResearchSource,
    SourceType,
)
from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def research_dir(tmp_path: Path) -> Path:
    """Create temporary research directory."""
    research_path = tmp_path / "research"
    research_path.mkdir(parents=True)
    return research_path


@pytest.fixture
def mock_memory(research_dir: Path):
    """Mock research memory that persists to temp dir."""
    from foundry_mcp.core.research.memory import ResearchMemory

    memory = ResearchMemory(base_path=research_dir)
    return memory


@pytest.fixture
def base_config() -> ResearchConfig:
    """Base research config with Tavily API key set."""
    return ResearchConfig(
        enabled=True,
        tavily_api_key="tvly-test-key-12345",
        deep_research_providers=["tavily"],
        deep_research_max_iterations=1,
        deep_research_max_sub_queries=2,
        deep_research_max_sources=3,
    )


@pytest.fixture
def mock_tavily_search_response():
    """Mock Tavily search response fixture."""
    return [
        ResearchSource(
            title="Test Result 1",
            url="https://example.com/article1",
            source_type=SourceType.WEB,
            snippet="This is the first search result snippet.",
            content="Full content of the first article.",
        ),
        ResearchSource(
            title="Test Result 2",
            url="https://example.com/article2",
            source_type=SourceType.WEB,
            snippet="This is the second search result snippet.",
            content="Full content of the second article.",
        ),
    ]


# =============================================================================
# Research Mode Smart Defaults Tests
# =============================================================================


class TestResearchModeSmartDefaults:
    """Tests for research-mode smart default behavior."""

    def test_general_mode_uses_basic_search_depth(self, base_config):
        """General research mode should use basic search depth by default."""
        config = ResearchConfig(
            **{**base_config.__dict__, "deep_research_mode": "general"}
        )
        assert config.deep_research_mode == "general"
        assert config.tavily_search_depth == "basic"

    def test_academic_mode_prefers_advanced_depth(self):
        """Academic research mode should benefit from advanced search depth."""
        config = ResearchConfig(
            deep_research_mode="academic",
            tavily_search_depth="advanced",
        )
        assert config.deep_research_mode == "academic"
        assert config.tavily_search_depth == "advanced"

    def test_technical_mode_can_use_advanced_depth(self):
        """Technical research mode can use advanced search for deeper results."""
        config = ResearchConfig(
            deep_research_mode="technical",
            tavily_search_depth="advanced",
        )
        assert config.deep_research_mode == "technical"
        assert config.tavily_search_depth == "advanced"

    def test_news_topic_requires_days_limit(self):
        """News topic should work with days limit for recent results."""
        config = ResearchConfig(
            tavily_topic="news",
            tavily_news_days=7,
        )
        assert config.tavily_topic == "news"
        assert config.tavily_news_days == 7


# =============================================================================
# Config Override Behavior Tests
# =============================================================================


class TestConfigOverrideBehavior:
    """Tests for config override behavior in deep research."""

    def test_config_search_depth_propagates_to_workflow(
        self, base_config, mock_memory, mock_tavily_search_response
    ):
        """Search depth from config should propagate to Tavily search calls."""
        config = ResearchConfig(
            **{**base_config.__dict__, "tavily_search_depth": "advanced"}
        )

        workflow = DeepResearchWorkflow(config=config, memory=mock_memory)

        # Verify config is stored in workflow
        assert workflow.config.tavily_search_depth == "advanced"

    def test_config_topic_propagates_to_workflow(
        self, base_config, mock_memory
    ):
        """Topic from config should propagate to Tavily search calls."""
        config = ResearchConfig(
            **{**base_config.__dict__, "tavily_topic": "news", "tavily_news_days": 30}
        )

        workflow = DeepResearchWorkflow(config=config, memory=mock_memory)

        assert workflow.config.tavily_topic == "news"
        assert workflow.config.tavily_news_days == 30

    def test_config_country_propagates_to_workflow(
        self, base_config, mock_memory
    ):
        """Country from config should propagate to Tavily search calls."""
        config = ResearchConfig(
            **{**base_config.__dict__, "tavily_country": "US"}
        )

        workflow = DeepResearchWorkflow(config=config, memory=mock_memory)

        assert workflow.config.tavily_country == "US"

    def test_extract_in_deep_research_flag(self, base_config, mock_memory):
        """Extract in deep research flag should be accessible in workflow."""
        config = ResearchConfig(
            **{
                **base_config.__dict__,
                "tavily_extract_in_deep_research": True,
                "tavily_extract_max_urls": 10,
            }
        )

        workflow = DeepResearchWorkflow(config=config, memory=mock_memory)

        assert workflow.config.tavily_extract_in_deep_research is True
        assert workflow.config.tavily_extract_max_urls == 10


# =============================================================================
# Tavily Search Parameter Propagation Tests
# =============================================================================


class TestTavilySearchParameterPropagation:
    """Tests for Tavily search parameter propagation through workflow."""

    @pytest.mark.asyncio
    async def test_get_tavily_search_kwargs_includes_configured_params(
        self, base_config, mock_memory
    ):
        """_get_tavily_search_kwargs should include all configured parameters."""
        config = ResearchConfig(
            **{
                **base_config.__dict__,
                "tavily_search_depth": "advanced",
                "tavily_topic": "news",
                "tavily_news_days": 7,
                "tavily_include_images": True,
                "tavily_country": "US",
                "tavily_chunks_per_source": 5,
                "tavily_auto_parameters": True,
            }
        )

        workflow = DeepResearchWorkflow(config=config, memory=mock_memory)

        # Create a mock state
        state = DeepResearchState(
            original_query="test query",
            research_mode=ResearchMode.GENERAL,
            follow_links=True,
        )

        kwargs = workflow._get_tavily_search_kwargs(state)

        assert kwargs["search_depth"] == "advanced"
        assert kwargs["topic"] == "news"
        assert kwargs["days"] == 7
        assert kwargs["include_images"] is True
        assert kwargs["country"] == "US"
        assert kwargs["chunks_per_source"] == 5
        assert kwargs["auto_parameters"] is True

    @pytest.mark.asyncio
    async def test_get_tavily_search_kwargs_omits_none_values(
        self, base_config, mock_memory
    ):
        """_get_tavily_search_kwargs should omit None values."""
        config = ResearchConfig(
            **{
                **base_config.__dict__,
                "tavily_search_depth": "basic",
                "tavily_topic": "general",
                # tavily_news_days is None by default
                # tavily_country is None by default
            }
        )

        workflow = DeepResearchWorkflow(config=config, memory=mock_memory)

        state = DeepResearchState(
            original_query="test query",
            research_mode=ResearchMode.GENERAL,
        )

        kwargs = workflow._get_tavily_search_kwargs(state)

        assert "days" not in kwargs
        assert "country" not in kwargs

    @pytest.mark.asyncio
    async def test_get_tavily_search_kwargs_respects_basic_override(
        self, mock_memory
    ):
        """Explicit config should override mode defaults even when matching base defaults."""
        config = ResearchConfig(
            enabled=True,
            tavily_api_key="tvly-test-key-12345",
            deep_research_providers=["tavily"],
            deep_research_max_iterations=1,
            deep_research_max_sub_queries=2,
            deep_research_max_sources=3,
            tavily_search_depth="basic",
            tavily_chunks_per_source=3,
        )
        config.tavily_search_depth_configured = True
        config.tavily_chunks_per_source_configured = True

        workflow = DeepResearchWorkflow(config=config, memory=mock_memory)

        state = DeepResearchState(
            original_query="test query",
            research_mode=ResearchMode.ACADEMIC,
            follow_links=False,
        )

        kwargs = workflow._get_tavily_search_kwargs(state)

        assert kwargs["search_depth"] == "basic"
        assert kwargs["chunks_per_source"] == 3


# =============================================================================
# Extract Follow-up Integration Tests
# =============================================================================


class TestExtractFollowupIntegration:
    """Tests for Tavily extract follow-up integration in deep research."""

    @pytest.mark.asyncio
    async def test_extract_followup_disabled_by_default(
        self, base_config, mock_memory
    ):
        """Extract follow-up should be disabled by default."""
        workflow = DeepResearchWorkflow(config=base_config, memory=mock_memory)

        assert workflow.config.tavily_extract_in_deep_research is False

    @pytest.mark.asyncio
    async def test_extract_followup_enabled_when_configured(
        self, base_config, mock_memory
    ):
        """Extract follow-up should be enabled when config flag is True."""
        config = ResearchConfig(
            **{
                **base_config.__dict__,
                "tavily_extract_in_deep_research": True,
            }
        )

        workflow = DeepResearchWorkflow(config=config, memory=mock_memory)

        assert workflow.config.tavily_extract_in_deep_research is True

    @pytest.mark.asyncio
    async def test_extract_max_urls_configurable(
        self, base_config, mock_memory
    ):
        """Extract max URLs should be configurable."""
        config = ResearchConfig(
            **{
                **base_config.__dict__,
                "tavily_extract_in_deep_research": True,
                "tavily_extract_max_urls": 10,
            }
        )

        workflow = DeepResearchWorkflow(config=config, memory=mock_memory)

        assert workflow.config.tavily_extract_max_urls == 10

    @pytest.mark.asyncio
    async def test_extract_followup_method_exists(
        self, base_config, mock_memory
    ):
        """_execute_extract_followup_async method should exist on workflow."""
        config = ResearchConfig(
            **{
                **base_config.__dict__,
                "tavily_extract_in_deep_research": True,
            }
        )

        workflow = DeepResearchWorkflow(config=config, memory=mock_memory)

        assert hasattr(workflow, "_execute_extract_followup_async")
        assert callable(workflow._execute_extract_followup_async)


# =============================================================================
# Workflow State Integration Tests
# =============================================================================


class TestWorkflowStateIntegration:
    """Tests for workflow state integration with Tavily config."""

    def test_state_preserves_research_mode(self, base_config, mock_memory):
        """State should preserve research mode for source quality scoring."""
        workflow = DeepResearchWorkflow(config=base_config, memory=mock_memory)

        state = DeepResearchState(
            original_query="test query",
            research_mode=ResearchMode.ACADEMIC,
        )

        assert state.research_mode == ResearchMode.ACADEMIC

    def test_state_tracks_follow_links(self, base_config, mock_memory):
        """State should track follow_links setting."""
        config = ResearchConfig(
            **{**base_config.__dict__, "deep_research_follow_links": True}
        )

        state = DeepResearchState(
            original_query="test query",
            follow_links=config.deep_research_follow_links,
        )

        assert state.follow_links is True

    def test_state_from_config_settings(self, base_config, mock_memory):
        """State should be initializable from config settings."""
        config = ResearchConfig(
            **{
                **base_config.__dict__,
                "deep_research_max_iterations": 5,
                "deep_research_max_sub_queries": 10,
                "deep_research_max_sources": 15,
            }
        )

        state = DeepResearchState(
            original_query="test query",
            max_iterations=config.deep_research_max_iterations,
            max_sub_queries=config.deep_research_max_sub_queries,
            max_sources_per_query=config.deep_research_max_sources,
        )

        assert state.max_iterations == 5
        assert state.max_sub_queries == 10
        assert state.max_sources_per_query == 15
