"""Tests for Phase 1: provider resolution from state.metadata['active_providers'].

Verifies that gathering and supervision phases read providers from
state.metadata['active_providers'] (set by the brief phase) instead of
always falling back to the global config default.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    ResearchDirective,
)
from foundry_mcp.core.research.models.sources import (
    ResearchSource,
    SourceType,
    SubQuery,
)
from foundry_mcp.core.research.workflows.deep_research.phases.gathering import (
    GatheringPhaseMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases.supervision import (
    SupervisionPhaseMixin,
)

from tests.core.research.workflows.deep_research.conftest import (
    make_gathering_state,
    make_supervision_state,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_provider(name: str) -> MagicMock:
    """Create a mock search provider that returns one source."""
    provider = MagicMock()
    provider.get_provider_name.return_value = name
    provider.search = AsyncMock(
        return_value=[
            ResearchSource(
                title=f"Result from {name}",
                url=f"https://{name}.example.com/result",
                source_type=SourceType.WEB,
            )
        ]
    )
    return provider


class StubGathering(GatheringPhaseMixin):
    """Concrete class for testing GatheringPhaseMixin in isolation."""

    def __init__(self) -> None:
        self.config = MagicMock()
        self.config.deep_research_providers = ["tavily", "google", "semantic_scholar"]
        self.memory = MagicMock()
        self._search_providers: dict = {}

    def _write_audit_event(self, state: Any, event_type: Any, *, data: Any = None) -> None:
        pass

    def _check_cancellation(self, state: Any) -> None:
        pass


class StubSupervision(SupervisionPhaseMixin):
    """Concrete class for testing SupervisionPhaseMixin in isolation."""

    def __init__(self) -> None:
        self.config = MagicMock()
        self.config.deep_research_providers = ["tavily", "google", "semantic_scholar"]
        self.config.deep_research_max_concurrent_research_units = 5
        self.config.deep_research_topic_max_tool_calls = 1
        self.config.deep_research_reflection_timeout = 60.0
        self.config.deep_research_coverage_confidence_threshold = 0.75
        self.config.deep_research_coverage_confidence_weights = None
        self.config.deep_research_supervision_wall_clock_timeout = 1800.0
        self.config.deep_research_supervision_min_sources_per_query = 2
        self.memory = MagicMock()
        self._search_providers: dict = {}
        self._provider_lookup_fn: Callable[[str], Any] | None = None

    def _write_audit_event(self, state: Any, event_type: Any, *, data: Any = None) -> None:
        pass

    def _check_cancellation(self, state: Any) -> None:
        pass

    def _get_search_provider(self, provider_name: str) -> Any:
        if self._provider_lookup_fn is not None:
            return self._provider_lookup_fn(provider_name)
        return None


# ===========================================================================
# Gathering phase tests
# ===========================================================================


class TestGatheringProviderResolution:
    """Verify gathering phase reads active_providers from state metadata."""

    @pytest.mark.asyncio
    async def test_uses_active_providers_from_metadata(self):
        """When state.metadata has active_providers, only those providers are used."""
        stub = StubGathering()
        state = make_gathering_state(
            num_sub_queries=0,
            sources_per_query=0,
            metadata={"active_providers": ["tavily"]},
        )
        state.phase = DeepResearchPhase.GATHERING
        sq = SubQuery(
            id="sq-test",
            query="Test query",
            rationale="Test",
            priority=1,
            status="pending",
        )
        state.sub_queries.append(sq)

        instantiated_providers: list[str] = []

        def provider_lookup(name: str):
            instantiated_providers.append(name)
            return _make_mock_provider(name)

        with (
            patch.object(stub, "_get_search_provider", side_effect=provider_lookup),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.gathering.get_resilience_manager"
            ) as mock_rm,
        ):
            mock_rm.return_value.is_provider_available.return_value = True
            result = await stub._execute_gathering_async(
                state=state,
                provider_id=None,
                timeout=30.0,
                max_concurrent=2,
            )

        # Only "tavily" should have been looked up — not google or semantic_scholar
        assert instantiated_providers == ["tavily"]
        assert result.metadata["providers_used"] == ["tavily"]

    @pytest.mark.asyncio
    async def test_falls_back_to_config_when_no_active_providers(self):
        """When active_providers is absent from metadata, use config fallback."""
        stub = StubGathering()
        stub.config.deep_research_providers = ["tavily", "google"]

        state = make_gathering_state(
            num_sub_queries=0,
            sources_per_query=0,
        )
        state.metadata.pop("active_providers", None)
        state.phase = DeepResearchPhase.GATHERING
        sq = SubQuery(
            id="sq-test",
            query="Test query",
            rationale="Test",
            priority=1,
            status="pending",
        )
        state.sub_queries.append(sq)

        instantiated_providers: list[str] = []

        def provider_lookup(name: str):
            instantiated_providers.append(name)
            return _make_mock_provider(name)

        with (
            patch.object(stub, "_get_search_provider", side_effect=provider_lookup),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.gathering.get_resilience_manager"
            ) as mock_rm,
        ):
            mock_rm.return_value.is_provider_available.return_value = True
            result = await stub._execute_gathering_async(
                state=state,
                provider_id=None,
                timeout=30.0,
                max_concurrent=2,
            )

        # Should fall back to config: ["tavily", "google"]
        assert instantiated_providers == ["tavily", "google"]
        assert result.metadata["providers_used"] == ["tavily", "google"]

    @pytest.mark.asyncio
    async def test_falls_back_to_default_when_no_config(self):
        """When active_providers absent AND config attr missing, use hardcoded default."""
        stub = StubGathering()
        # Remove the config attribute entirely
        del stub.config.deep_research_providers

        state = make_gathering_state(
            num_sub_queries=0,
            sources_per_query=0,
        )
        state.metadata.pop("active_providers", None)
        state.phase = DeepResearchPhase.GATHERING
        sq = SubQuery(
            id="sq-test",
            query="Test query",
            rationale="Test",
            priority=1,
            status="pending",
        )
        state.sub_queries.append(sq)

        instantiated_providers: list[str] = []

        def provider_lookup(name: str):
            instantiated_providers.append(name)
            return _make_mock_provider(name)

        with (
            patch.object(stub, "_get_search_provider", side_effect=provider_lookup),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.gathering.get_resilience_manager"
            ) as mock_rm,
        ):
            mock_rm.return_value.is_provider_available.return_value = True
            result = await stub._execute_gathering_async(
                state=state,
                provider_id=None,
                timeout=30.0,
                max_concurrent=2,
            )

        # Hardcoded default is ["tavily"] (not the old 3-provider list)
        assert instantiated_providers == ["tavily"]
        assert result.metadata["providers_used"] == ["tavily"]

    @pytest.mark.asyncio
    async def test_empty_active_providers_falls_back_to_config(self):
        """An empty active_providers list is falsy, so config fallback kicks in."""
        stub = StubGathering()
        stub.config.deep_research_providers = ["tavily"]

        state = make_gathering_state(
            num_sub_queries=0,
            sources_per_query=0,
            metadata={"active_providers": []},
        )
        state.phase = DeepResearchPhase.GATHERING
        sq = SubQuery(
            id="sq-test",
            query="Test query",
            rationale="Test",
            priority=1,
            status="pending",
        )
        state.sub_queries.append(sq)

        instantiated_providers: list[str] = []

        def provider_lookup(name: str):
            instantiated_providers.append(name)
            return _make_mock_provider(name)

        with (
            patch.object(stub, "_get_search_provider", side_effect=provider_lookup),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.gathering.get_resilience_manager"
            ) as mock_rm,
        ):
            mock_rm.return_value.is_provider_available.return_value = True
            await stub._execute_gathering_async(
                state=state,
                provider_id=None,
                timeout=30.0,
                max_concurrent=2,
            )

        # Empty list is falsy -> falls back to config
        assert instantiated_providers == ["tavily"]


# ===========================================================================
# Supervision phase tests
# ===========================================================================


class TestSupervisionProviderResolution:
    """Verify supervision directive execution reads active_providers from metadata."""

    @pytest.mark.asyncio
    async def test_uses_active_providers_from_metadata(self):
        """When state.metadata has active_providers, only those providers are used."""
        stub = StubSupervision()
        state = make_supervision_state(
            num_sub_queries=2,
            sources_per_query=2,
            metadata={"active_providers": ["tavily"]},
        )

        directive = ResearchDirective(
            research_topic="Follow-up query about the topic",
            priority=1,
        )

        instantiated_providers: list[str] = []

        def provider_lookup(name: str):
            instantiated_providers.append(name)
            return _make_mock_provider(name)

        stub._provider_lookup_fn = provider_lookup
        await stub._execute_directives_async(
            state=state,
            directives=[directive],
            timeout=30.0,
        )

        # Only "tavily" should have been instantiated
        assert instantiated_providers == ["tavily"]

    @pytest.mark.asyncio
    async def test_falls_back_to_config_when_no_active_providers(self):
        """When active_providers is absent from metadata, use config fallback."""
        stub = StubSupervision()
        stub.config.deep_research_providers = ["tavily", "google"]

        state = make_supervision_state(
            num_sub_queries=2,
            sources_per_query=2,
        )
        state.metadata.pop("active_providers", None)

        directive = ResearchDirective(
            research_topic="Follow-up query about the topic",
            priority=1,
        )

        instantiated_providers: list[str] = []

        def provider_lookup(name: str):
            instantiated_providers.append(name)
            return _make_mock_provider(name)

        stub._provider_lookup_fn = provider_lookup
        await stub._execute_directives_async(
            state=state,
            directives=[directive],
            timeout=30.0,
        )

        # Falls back to config: ["tavily", "google"]
        assert instantiated_providers == ["tavily", "google"]

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_providers_available(self):
        """When no providers can be resolved, return empty results list."""
        stub = StubSupervision()
        state = make_supervision_state(
            metadata={"active_providers": ["nonexistent"]},
        )

        directive = ResearchDirective(
            research_topic="Follow-up query about the topic",
            priority=1,
        )

        stub._provider_lookup_fn = lambda name: None
        results = await stub._execute_directives_async(
            state=state,
            directives=[directive],
            timeout=30.0,
        )

        assert results == []
