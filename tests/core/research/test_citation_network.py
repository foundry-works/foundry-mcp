"""Tests for PLAN-4 Item 2: Citation Network / Connected Papers Graph.

Tests cover:
1. CitationNode, CitationEdge, CitationNetwork model serialization
2. Network builder with mocked provider responses
3. Foundational paper identification
4. Research thread detection with known graph
5. Role classification
6. Graceful handling when < 3 academic sources
7. Action handler wiring and response format
"""

import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.core.research.models.deep_research import (
    CitationEdge,
    CitationNetwork,
    CitationNode,
    DeepResearchState,
    ResearchExtensions,
    ResearchThread,
)
from foundry_mcp.core.research.models.sources import ResearchSource, SourceType
from foundry_mcp.core.research.workflows.deep_research.phases.citation_network import (
    MIN_ACADEMIC_SOURCES,
    CitationNetworkBuilder,
    _extract_paper_id,
    _node_from_source,
)


# =============================================================================
# Fixtures
# =============================================================================


def _make_source(
    source_id: str,
    title: str,
    *,
    openalex_id: Optional[str] = None,
    paper_id: Optional[str] = None,
    doi: Optional[str] = None,
    year: Optional[int] = None,
    citation_count: Optional[int] = None,
    authors: str = "Author A",
    source_type: SourceType = SourceType.ACADEMIC,
) -> ResearchSource:
    """Helper to create a ResearchSource with metadata."""
    meta: dict[str, Any] = {"authors": authors}
    if openalex_id:
        meta["openalex_id"] = openalex_id
    if paper_id:
        meta["paper_id"] = paper_id
    if doi:
        meta["doi"] = doi
    if year:
        meta["year"] = year
    if citation_count is not None:
        meta["citation_count"] = citation_count
    return ResearchSource(
        id=source_id,
        title=title,
        source_type=source_type,
        metadata=meta,
    )


def _make_ref_source(
    openalex_id: str,
    title: str,
    year: int = 2020,
) -> ResearchSource:
    """Helper to create a provider-returned ResearchSource for refs/cites."""
    return ResearchSource(
        title=title,
        source_type=SourceType.ACADEMIC,
        metadata={
            "openalex_id": openalex_id,
            "authors": "Ref Author",
            "year": year,
            "citation_count": 10,
        },
    )


@pytest.fixture
def academic_sources() -> list[ResearchSource]:
    """Five academic sources with OpenAlex IDs."""
    return [
        _make_source("src-1", "Paper Alpha", openalex_id="W100", year=2022, citation_count=50),
        _make_source("src-2", "Paper Beta", openalex_id="W200", year=2021, citation_count=120),
        _make_source("src-3", "Paper Gamma", openalex_id="W300", year=2023, citation_count=30),
        _make_source("src-4", "Paper Delta", openalex_id="W400", year=2020, citation_count=200),
        _make_source("src-5", "Paper Epsilon", openalex_id="W500", year=2024, citation_count=15),
    ]


@pytest.fixture
def shared_reference() -> ResearchSource:
    """A reference paper cited by multiple discovered papers."""
    return _make_ref_source("W999", "Foundational Paper", year=2015)


# =============================================================================
# Model Tests
# =============================================================================


class TestCitationModels:
    """Test CitationNode, CitationEdge, CitationNetwork serialization."""

    def test_citation_node_defaults(self):
        node = CitationNode(paper_id="W100", title="Test Paper")
        assert node.role == "peripheral"
        assert node.is_discovered is False
        assert node.year is None

    def test_citation_node_full(self):
        node = CitationNode(
            paper_id="W100",
            title="Test Paper",
            authors="Smith, Jones",
            year=2023,
            citation_count=42,
            is_discovered=True,
            source_id="src-1",
            role="discovered",
        )
        data = node.model_dump()
        assert data["paper_id"] == "W100"
        assert data["role"] == "discovered"
        assert data["citation_count"] == 42

    def test_citation_edge(self):
        edge = CitationEdge(citing_paper_id="W100", cited_paper_id="W200")
        data = edge.model_dump()
        assert data["citing_paper_id"] == "W100"
        assert data["cited_paper_id"] == "W200"

    def test_research_thread(self):
        thread = ResearchThread(
            name="thread-1",
            paper_ids=["W100", "W200", "W300"],
            description="A cluster of related papers",
        )
        assert len(thread.paper_ids) == 3

    def test_citation_network_empty(self):
        network = CitationNetwork()
        assert network.nodes == []
        assert network.edges == []
        assert network.foundational_papers == []

    def test_citation_network_roundtrip(self):
        network = CitationNetwork(
            nodes=[CitationNode(paper_id="W1", title="Paper 1")],
            edges=[CitationEdge(citing_paper_id="W1", cited_paper_id="W2")],
            foundational_papers=["W999"],
            research_threads=[ResearchThread(name="t1", paper_ids=["W1", "W2", "W3"])],
            stats={"total_nodes": 1, "total_edges": 1},
        )
        data = network.model_dump(mode="json")
        restored = CitationNetwork.model_validate(data)
        assert len(restored.nodes) == 1
        assert len(restored.edges) == 1
        assert restored.foundational_papers == ["W999"]

    def test_research_extensions_citation_network(self):
        ext = ResearchExtensions(
            citation_network=CitationNetwork(
                nodes=[CitationNode(paper_id="W1", title="P1")],
                stats={"total_nodes": 1},
            )
        )
        assert ext.citation_network is not None
        assert ext.citation_network.stats["total_nodes"] == 1

    def test_deep_research_state_accessor(self):
        state = DeepResearchState(original_query="test")
        assert state.citation_network is None

        state.extensions.citation_network = CitationNetwork(
            stats={"total_nodes": 5}
        )
        assert state.citation_network is not None
        assert state.citation_network.stats["total_nodes"] == 5


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelpers:
    def test_extract_paper_id_openalex(self):
        src = _make_source("s1", "T", openalex_id="W100")
        assert _extract_paper_id(src) == "W100"

    def test_extract_paper_id_semantic_scholar(self):
        src = _make_source("s1", "T", paper_id="abc123")
        assert _extract_paper_id(src) == "abc123"

    def test_extract_paper_id_doi(self):
        src = _make_source("s1", "T", doi="10.1234/test")
        assert _extract_paper_id(src) == "https://doi.org/10.1234/test"

    def test_extract_paper_id_doi_full_url(self):
        src = _make_source("s1", "T", doi="https://doi.org/10.1234/test")
        assert _extract_paper_id(src) == "https://doi.org/10.1234/test"

    def test_extract_paper_id_none(self):
        src = _make_source("s1", "T", source_type=SourceType.WEB)
        assert _extract_paper_id(src) is None

    def test_extract_paper_id_preference_order(self):
        """openalex_id takes precedence over paper_id and doi."""
        src = _make_source("s1", "T", openalex_id="W100", paper_id="abc", doi="10.1/x")
        assert _extract_paper_id(src) == "W100"

    def test_node_from_source(self):
        src = _make_source("s1", "Paper A", openalex_id="W100", year=2023, authors="Smith")
        node = _node_from_source(src, "W100")
        assert node.is_discovered is True
        assert node.source_id == "s1"
        assert node.role == "discovered"
        assert node.year == 2023


# =============================================================================
# Network Builder Tests
# =============================================================================


class TestCitationNetworkBuilder:
    """Test CitationNetworkBuilder with mocked providers."""

    def test_skip_when_few_sources(self):
        """Should skip when fewer than MIN_ACADEMIC_SOURCES."""
        builder = CitationNetworkBuilder()
        sources = [
            _make_source("s1", "P1", openalex_id="W1"),
            _make_source("s2", "P2", openalex_id="W2"),
        ]
        network = asyncio.run(builder.build_network(sources))
        assert network.stats.get("status") == "skipped"
        assert "fewer than" in network.stats.get("reason", "")

    def test_skip_when_no_academic_sources(self):
        """Should skip when no academic sources have paper IDs."""
        builder = CitationNetworkBuilder()
        sources = [
            _make_source("s1", "P1", source_type=SourceType.WEB),
            _make_source("s2", "P2", source_type=SourceType.WEB),
            _make_source("s3", "P3", source_type=SourceType.WEB),
        ]
        network = asyncio.run(builder.build_network(sources))
        assert network.stats.get("status") == "skipped"

    def test_skip_academic_without_ids(self):
        """Academic sources without paper IDs should be filtered out."""
        builder = CitationNetworkBuilder()
        sources = [
            _make_source("s1", "P1"),  # academic but no ID
            _make_source("s2", "P2"),
            _make_source("s3", "P3"),
        ]
        network = asyncio.run(builder.build_network(sources))
        assert network.stats.get("status") == "skipped"

    def test_build_network_with_mocked_providers(self, academic_sources, shared_reference):
        """Build network with mocked OpenAlex returning shared references."""
        mock_openalex = AsyncMock()

        # All 5 sources reference the same foundational paper (W999)
        async def mock_get_references(work_id, max_results=20):
            return [shared_reference]

        async def mock_get_citations(work_id, max_results=20):
            return []  # no forward citations for simplicity

        mock_openalex.get_references = mock_get_references
        mock_openalex.get_citations = mock_get_citations

        builder = CitationNetworkBuilder(
            openalex_provider=mock_openalex,
            max_references_per_paper=20,
            max_citations_per_paper=20,
        )

        network = asyncio.run(builder.build_network(academic_sources))

        assert network.stats["total_nodes"] == 6  # 5 discovered + 1 shared ref
        assert network.stats["discovered_sources"] == 5
        assert len(network.edges) == 5  # each source -> W999

        # W999 should be foundational
        assert "W999" in network.foundational_papers

    def test_build_network_with_citations(self, academic_sources):
        """Build network with forward citations."""
        mock_openalex = AsyncMock()
        extension_paper = _make_ref_source("W600", "Extension Paper", year=2025)

        async def mock_get_references(work_id, max_results=20):
            return []

        async def mock_get_citations(work_id, max_results=20):
            # W600 cites all discovered papers
            return [extension_paper]

        mock_openalex.get_references = mock_get_references
        mock_openalex.get_citations = mock_get_citations

        builder = CitationNetworkBuilder(openalex_provider=mock_openalex)
        network = asyncio.run(builder.build_network(academic_sources))

        # W600 should exist and cite all discovered papers
        assert network.stats["total_nodes"] == 6
        w600_node = next((n for n in network.nodes if n.paper_id == "W600"), None)
        assert w600_node is not None
        assert w600_node.role == "extension"

    def test_provider_failure_graceful(self, academic_sources):
        """Provider failures should not crash the builder."""
        mock_openalex = AsyncMock()

        async def mock_get_references(work_id, max_results=20):
            raise RuntimeError("API error")

        async def mock_get_citations(work_id, max_results=20):
            raise RuntimeError("API error")

        mock_openalex.get_references = mock_get_references
        mock_openalex.get_citations = mock_get_citations

        builder = CitationNetworkBuilder(openalex_provider=mock_openalex)
        network = asyncio.run(builder.build_network(academic_sources))

        # Should still return a network with just the discovered nodes
        assert network.stats["total_nodes"] == 5
        assert network.stats["total_edges"] == 0

    def test_edge_deduplication(self, academic_sources):
        """Same edge from different sources should not be duplicated."""
        mock_openalex = AsyncMock()
        shared_ref = _make_ref_source("W999", "Shared", year=2015)

        call_count = 0

        async def mock_get_references(work_id, max_results=20):
            nonlocal call_count
            call_count += 1
            return [shared_ref]

        async def mock_get_citations(work_id, max_results=20):
            return []

        mock_openalex.get_references = mock_get_references
        mock_openalex.get_citations = mock_get_citations

        builder = CitationNetworkBuilder(openalex_provider=mock_openalex)
        network = asyncio.run(builder.build_network(academic_sources))

        # Each edge W100->W999, W200->W999, etc. should be unique
        edge_keys = {(e.citing_paper_id, e.cited_paper_id) for e in network.edges}
        assert len(edge_keys) == len(network.edges)

    def test_semantic_scholar_fallback(self, academic_sources):
        """Should fall back to Semantic Scholar when OpenAlex fails."""
        mock_openalex = AsyncMock()
        mock_s2 = AsyncMock()

        async def mock_oa_get_references(work_id, max_results=20):
            raise RuntimeError("OA down")

        async def mock_oa_get_citations(work_id, max_results=20):
            raise RuntimeError("OA down")

        async def mock_s2_get_citations(paper_id, max_results=20):
            return [_make_ref_source("W600", "S2 Citation", year=2025)]

        mock_openalex.get_references = mock_oa_get_references
        mock_openalex.get_citations = mock_oa_get_citations
        mock_s2.get_citations = mock_s2_get_citations

        builder = CitationNetworkBuilder(
            openalex_provider=mock_openalex,
            semantic_scholar_provider=mock_s2,
        )
        network = asyncio.run(builder.build_network(academic_sources))

        # Should have citation edges from S2 fallback
        assert network.stats["total_edges"] > 0


# =============================================================================
# Foundational Paper Identification
# =============================================================================


class TestFoundationalPapers:
    def test_foundational_by_threshold(self):
        """Papers cited by 3+ discovered papers should be foundational."""
        discovered = {"W1", "W2", "W3", "W4", "W5"}
        nodes = {pid: CitationNode(paper_id=pid, title=f"P{pid}", is_discovered=True) for pid in discovered}
        nodes["W999"] = CitationNode(paper_id="W999", title="Foundational")

        edges = [
            CitationEdge(citing_paper_id="W1", cited_paper_id="W999"),
            CitationEdge(citing_paper_id="W2", cited_paper_id="W999"),
            CitationEdge(citing_paper_id="W3", cited_paper_id="W999"),
        ]

        foundational = CitationNetworkBuilder._identify_foundational_papers(nodes, edges, discovered)
        assert "W999" in foundational

    def test_not_foundational_below_threshold(self):
        """Papers cited by < 3 discovered papers should not be foundational."""
        discovered = {"W1", "W2", "W3", "W4", "W5"}
        nodes = {pid: CitationNode(paper_id=pid, title=f"P{pid}", is_discovered=True) for pid in discovered}
        nodes["W999"] = CitationNode(paper_id="W999", title="Peripheral")

        edges = [
            CitationEdge(citing_paper_id="W1", cited_paper_id="W999"),
            CitationEdge(citing_paper_id="W2", cited_paper_id="W999"),
        ]

        foundational = CitationNetworkBuilder._identify_foundational_papers(nodes, edges, discovered)
        assert "W999" not in foundational

    def test_discovered_not_foundational(self):
        """Discovered papers should not be marked as foundational even if cited."""
        discovered = {"W1", "W2", "W3", "W4"}
        nodes = {pid: CitationNode(paper_id=pid, title=f"P{pid}", is_discovered=True) for pid in discovered}

        edges = [
            CitationEdge(citing_paper_id="W2", cited_paper_id="W1"),
            CitationEdge(citing_paper_id="W3", cited_paper_id="W1"),
            CitationEdge(citing_paper_id="W4", cited_paper_id="W1"),
        ]

        foundational = CitationNetworkBuilder._identify_foundational_papers(nodes, edges, discovered)
        assert "W1" not in foundational


# =============================================================================
# Research Thread Detection
# =============================================================================


class TestResearchThreads:
    def test_detect_threads(self):
        """Connected components with 3+ nodes should become threads."""
        nodes = {
            "W1": CitationNode(paper_id="W1", title="P1", is_discovered=True),
            "W2": CitationNode(paper_id="W2", title="P2", is_discovered=True),
            "W3": CitationNode(paper_id="W3", title="P3"),
            # Isolated pair (should not be a thread)
            "W4": CitationNode(paper_id="W4", title="P4"),
            "W5": CitationNode(paper_id="W5", title="P5"),
        }
        edges = [
            CitationEdge(citing_paper_id="W1", cited_paper_id="W3"),
            CitationEdge(citing_paper_id="W2", cited_paper_id="W3"),
            # Isolated pair
            CitationEdge(citing_paper_id="W4", cited_paper_id="W5"),
        ]

        threads = CitationNetworkBuilder._identify_research_threads(nodes, edges)
        assert len(threads) == 1
        assert set(threads[0].paper_ids) == {"W1", "W2", "W3"}

    def test_no_threads_below_minimum(self):
        """Pairs of nodes should not form threads."""
        nodes = {
            "W1": CitationNode(paper_id="W1", title="P1"),
            "W2": CitationNode(paper_id="W2", title="P2"),
        }
        edges = [CitationEdge(citing_paper_id="W1", cited_paper_id="W2")]

        threads = CitationNetworkBuilder._identify_research_threads(nodes, edges)
        assert len(threads) == 0

    def test_empty_graph(self):
        threads = CitationNetworkBuilder._identify_research_threads({}, [])
        assert threads == []

    def test_multiple_threads(self):
        """Multiple connected components should yield multiple threads."""
        nodes = {}
        edges = []
        # Thread 1: W1-W2-W3
        for i in range(1, 4):
            nodes[f"W{i}"] = CitationNode(paper_id=f"W{i}", title=f"P{i}")
        edges.append(CitationEdge(citing_paper_id="W1", cited_paper_id="W2"))
        edges.append(CitationEdge(citing_paper_id="W2", cited_paper_id="W3"))

        # Thread 2: W10-W11-W12
        for i in range(10, 13):
            nodes[f"W{i}"] = CitationNode(paper_id=f"W{i}", title=f"P{i}")
        edges.append(CitationEdge(citing_paper_id="W10", cited_paper_id="W11"))
        edges.append(CitationEdge(citing_paper_id="W11", cited_paper_id="W12"))

        threads = CitationNetworkBuilder._identify_research_threads(nodes, edges)
        assert len(threads) == 2


# =============================================================================
# Role Classification
# =============================================================================


class TestRoleClassification:
    def test_classify_roles(self):
        discovered = {"W1", "W2", "W3"}
        foundational = ["W999"]
        nodes = {
            "W1": CitationNode(paper_id="W1", title="P1", is_discovered=True),
            "W2": CitationNode(paper_id="W2", title="P2", is_discovered=True),
            "W3": CitationNode(paper_id="W3", title="P3", is_discovered=True),
            "W999": CitationNode(paper_id="W999", title="Foundation"),
            "W600": CitationNode(paper_id="W600", title="Extension"),
            "W700": CitationNode(paper_id="W700", title="Peripheral"),
        }
        edges = [
            # W999 is cited by all discovered
            CitationEdge(citing_paper_id="W1", cited_paper_id="W999"),
            CitationEdge(citing_paper_id="W2", cited_paper_id="W999"),
            CitationEdge(citing_paper_id="W3", cited_paper_id="W999"),
            # W600 cites all discovered (extension)
            CitationEdge(citing_paper_id="W600", cited_paper_id="W1"),
            CitationEdge(citing_paper_id="W600", cited_paper_id="W2"),
            CitationEdge(citing_paper_id="W600", cited_paper_id="W3"),
            # W700 doesn't cite any discovered papers (truly peripheral)
            CitationEdge(citing_paper_id="W700", cited_paper_id="W999"),
        ]

        CitationNetworkBuilder._classify_roles(nodes, discovered, foundational, edges)

        assert nodes["W1"].role == "discovered"
        assert nodes["W999"].role == "foundational"
        assert nodes["W600"].role == "extension"
        assert nodes["W700"].role == "peripheral"


# =============================================================================
# Action Handler Tests
# =============================================================================


class TestActionHandler:
    """Test _handle_deep_research_network handler."""

    def test_missing_research_id(self):
        from foundry_mcp.tools.unified.research_handlers.handlers_deep_research import (
            _handle_deep_research_network,
        )

        result = _handle_deep_research_network(research_id=None)
        assert result["success"] is False

    def test_research_not_found(self):
        from foundry_mcp.tools.unified.research_handlers.handlers_deep_research import (
            _handle_deep_research_network,
        )

        with patch(
            "foundry_mcp.tools.unified.research_handlers.handlers_deep_research._get_memory"
        ) as mock_mem, patch(
            "foundry_mcp.tools.unified.research_handlers.handlers_deep_research._get_config"
        ) as mock_cfg:
            mock_mem.return_value.load_deep_research.return_value = None
            mock_cfg.return_value = MagicMock()

            result = _handle_deep_research_network(research_id="nonexistent")
            assert result["success"] is False

    def test_action_wired_in_router(self):
        """Verify deep-research-network is registered in the action router."""
        from foundry_mcp.tools.unified.research_handlers import _ACTION_DEFINITIONS

        action_names = [a.name for a in _ACTION_DEFINITIONS]
        assert "deep-research-network" in action_names

    def test_action_summary_exists(self):
        from foundry_mcp.tools.unified.research_handlers._helpers import _ACTION_SUMMARY

        assert "deep-research-network" in _ACTION_SUMMARY
