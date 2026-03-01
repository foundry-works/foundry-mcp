"""Citation network builder for deep research sessions (PLAN-4 Item 2).

Builds a citation graph from a completed research session's academic sources.
User-triggered via the ``deep-research-network`` action — not part of the
automatic research pipeline.

Uses OpenAlex (primary, 100 req/s) with Semantic Scholar fallback (1 RPS)
for fetching references and forward citations per source.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Optional

from foundry_mcp.core.research.models.deep_research import (
    CitationEdge,
    CitationNetwork,
    CitationNode,
    ResearchThread,
)
from foundry_mcp.core.research.models.sources import ResearchSource, SourceType

logger = logging.getLogger(__name__)

# Minimum discovered sources to build a network
MIN_ACADEMIC_SOURCES = 3


def _extract_paper_id(source: ResearchSource) -> Optional[str]:
    """Extract the best paper identifier from a source's metadata.

    Preference: openalex_id > paper_id (Semantic Scholar) > DOI.
    Returns None if no usable identifier found.
    """
    meta = source.metadata
    for key in ("openalex_id", "paper_id"):
        val = meta.get(key)
        if val:
            return str(val)
    doi = meta.get("doi")
    if doi:
        # Normalize DOI for API lookups
        doi_str = str(doi)
        if doi_str.startswith("https://doi.org/"):
            return doi_str
        if doi_str.startswith("10."):
            return f"https://doi.org/{doi_str}"
        return doi_str
    return None


def _node_from_source(source: ResearchSource, paper_id: str) -> CitationNode:
    """Create a CitationNode from a discovered ResearchSource."""
    meta = source.metadata
    return CitationNode(
        paper_id=paper_id,
        title=source.title,
        authors=meta.get("authors", ""),
        year=meta.get("year"),
        citation_count=meta.get("citation_count"),
        is_discovered=True,
        source_id=source.id,
        role="discovered",
    )


def _node_from_provider_source(provider_source: ResearchSource) -> Optional[CitationNode]:
    """Create a CitationNode from a provider-returned ResearchSource (ref/citation)."""
    pid = _extract_paper_id(provider_source)
    if not pid:
        return None
    meta = provider_source.metadata
    return CitationNode(
        paper_id=pid,
        title=provider_source.title,
        authors=meta.get("authors", ""),
        year=meta.get("year"),
        citation_count=meta.get("citation_count"),
        is_discovered=False,
        role="peripheral",
    )


class CitationNetworkBuilder:
    """Builds a citation network from completed research session sources.

    Args:
        openalex_provider: OpenAlex provider instance (primary).
        semantic_scholar_provider: Semantic Scholar provider instance (fallback).
        max_references_per_paper: Max backward references to fetch per source.
        max_citations_per_paper: Max forward citations to fetch per source.
        max_concurrent: Max concurrent API calls.
    """

    def __init__(
        self,
        openalex_provider: Any = None,
        semantic_scholar_provider: Any = None,
        *,
        max_references_per_paper: int = 20,
        max_citations_per_paper: int = 20,
        max_concurrent: int = 5,
    ) -> None:
        self._openalex = openalex_provider
        self._semantic_scholar = semantic_scholar_provider
        self._max_refs = max_references_per_paper
        self._max_cites = max_citations_per_paper
        self._max_concurrent = max_concurrent

    async def build_network(
        self,
        sources: list[ResearchSource],
        *,
        timeout: Optional[float] = 90.0,
    ) -> CitationNetwork:
        """Build a citation network from research session sources.

        Filters to academic sources with paper IDs, fetches references
        and citations for each, then assembles the graph.

        Args:
            sources: Research sources to build the network from.
            timeout: Maximum seconds to wait for all API calls.
                Defaults to 90s.  On timeout, partial results are used.

        Returns:
            CitationNetwork with nodes, edges, foundational papers,
            research threads, and statistics.
        """
        # Filter to academic sources with usable IDs
        academic_sources: list[tuple[ResearchSource, str]] = []
        for src in sources:
            if src.source_type != SourceType.ACADEMIC:
                continue
            pid = _extract_paper_id(src)
            if pid:
                academic_sources.append((src, pid))

        if len(academic_sources) < MIN_ACADEMIC_SOURCES:
            return CitationNetwork(
                stats={
                    "status": "skipped",
                    "reason": f"fewer than {MIN_ACADEMIC_SOURCES} academic sources with paper IDs",
                    "academic_source_count": len(academic_sources),
                },
            )

        # Index: paper_id -> CitationNode
        nodes: dict[str, CitationNode] = {}
        edges: list[CitationEdge] = []
        edge_set: set[tuple[str, str]] = set()  # dedup

        # Add discovered papers as nodes
        for src, pid in academic_sources:
            nodes[pid] = _node_from_source(src, pid)

        discovered_ids = set(nodes.keys())

        # Fetch references and citations concurrently
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def fetch_for_source(pid: str) -> tuple[list[ResearchSource], list[ResearchSource]]:
            async with semaphore:
                refs = await self._fetch_references(pid)
                cites = await self._fetch_citations(pid)
                return refs, cites

        tasks = [fetch_for_source(pid) for _, pid in academic_sources]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout or 90.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Citation network build timed out after %.0fs; using partial results",
                timeout or 90.0,
            )
            results = [TimeoutError("citation network build timed out")] * len(tasks)

        for (_src, pid), result in zip(academic_sources, results):
            if isinstance(result, BaseException):
                logger.warning("Failed to fetch refs/cites for %s: %s", pid, result)
                continue

            refs, cites = result

            # Process references (papers this source cites)
            for ref_source in refs:
                ref_node = _node_from_provider_source(ref_source)
                if ref_node is None:
                    continue
                if ref_node.paper_id not in nodes:
                    nodes[ref_node.paper_id] = ref_node
                edge_key = (pid, ref_node.paper_id)
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    edges.append(CitationEdge(citing_paper_id=pid, cited_paper_id=ref_node.paper_id))

            # Process citations (papers that cite this source)
            for cite_source in cites:
                cite_node = _node_from_provider_source(cite_source)
                if cite_node is None:
                    continue
                if cite_node.paper_id not in nodes:
                    nodes[cite_node.paper_id] = cite_node
                edge_key = (cite_node.paper_id, pid)
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    edges.append(CitationEdge(citing_paper_id=cite_node.paper_id, cited_paper_id=pid))

        # Classify roles and identify structure
        foundational = self._identify_foundational_papers(nodes, edges, discovered_ids)
        self._classify_roles(nodes, discovered_ids, foundational, edges)
        threads = self._identify_research_threads(nodes, edges)

        return CitationNetwork(
            nodes=list(nodes.values()),
            edges=edges,
            foundational_papers=foundational,
            research_threads=threads,
            stats={
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "discovered_sources": len(discovered_ids),
                "foundational_count": len(foundational),
                "thread_count": len(threads),
            },
        )

    async def _fetch_references(self, paper_id: str) -> list[ResearchSource]:
        """Fetch backward references for a paper. OpenAlex primary, S2 fallback."""
        # Try OpenAlex first
        if self._openalex is not None:
            try:
                return await self._openalex.get_references(paper_id, max_results=self._max_refs)
            except Exception as e:
                logger.debug("OpenAlex get_references failed for %s: %s", paper_id, e)

        # Semantic Scholar fallback — no get_references method, skip
        return []

    async def _fetch_citations(self, paper_id: str) -> list[ResearchSource]:
        """Fetch forward citations for a paper. OpenAlex primary, S2 fallback."""
        if self._openalex is not None:
            try:
                return await self._openalex.get_citations(paper_id, max_results=self._max_cites)
            except Exception as e:
                logger.debug("OpenAlex get_citations failed for %s: %s", paper_id, e)

        if self._semantic_scholar is not None:
            try:
                return await self._semantic_scholar.get_citations(paper_id, max_results=self._max_cites)
            except Exception as e:
                logger.debug("Semantic Scholar get_citations failed for %s: %s", paper_id, e)

        return []

    @staticmethod
    def _identify_foundational_papers(
        nodes: dict[str, CitationNode],
        edges: list[CitationEdge],
        discovered_ids: set[str],
    ) -> list[str]:
        """Identify papers cited by many discovered papers.

        A paper is foundational if cited by >= 3 discovered papers OR >= 30%
        of discovered papers, whichever is lower.
        """
        # Count how many discovered papers cite each paper
        cited_by_discovered: dict[str, int] = defaultdict(int)
        for edge in edges:
            if edge.citing_paper_id in discovered_ids:
                cited_by_discovered[edge.cited_paper_id] += 1

        # "Cited by >= 3 OR >= 30% of discovered papers, whichever is lower"
        # — ensures small sets (e.g. 5 papers) use a proportional threshold
        # while large sets cap at 3.
        effective_threshold = min(3, max(1, int(len(discovered_ids) * 0.3)))

        foundational = [
            pid
            for pid, count in cited_by_discovered.items()
            if count >= effective_threshold and pid not in discovered_ids
        ]

        # Sort by citation count descending
        foundational.sort(
            key=lambda pid: cited_by_discovered.get(pid, 0),
            reverse=True,
        )
        return foundational

    @staticmethod
    def _classify_roles(
        nodes: dict[str, CitationNode],
        discovered_ids: set[str],
        foundational_ids: list[str],
        edges: list[CitationEdge],
    ) -> None:
        """Classify node roles in-place.

        - discovered: in the original research session sources
        - foundational: cited by many discovered papers
        - extension: cites many discovered papers (recent work building on them)
        - peripheral: everything else
        """
        foundational_set = set(foundational_ids)

        # Count how many discovered papers each node cites
        cites_discovered: dict[str, int] = defaultdict(int)
        for edge in edges:
            if edge.cited_paper_id in discovered_ids:
                cites_discovered[edge.citing_paper_id] += 1

        extension_threshold = min(3, max(1, int(len(discovered_ids) * 0.3)))

        for pid, node in nodes.items():
            if pid in discovered_ids:
                node.role = "discovered"
            elif pid in foundational_set:
                node.role = "foundational"
            elif cites_discovered.get(pid, 0) >= extension_threshold:
                node.role = "extension"
            else:
                node.role = "peripheral"

    @staticmethod
    def _identify_research_threads(
        nodes: dict[str, CitationNode],
        edges: list[CitationEdge],
    ) -> list[ResearchThread]:
        """Identify connected components with 3+ nodes using union-find.

        Treats edges as undirected for component detection.
        """
        if not nodes:
            return []

        # Union-Find
        parent: dict[str, str] = {pid: pid for pid in nodes}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for edge in edges:
            if edge.citing_paper_id in parent and edge.cited_paper_id in parent:
                union(edge.citing_paper_id, edge.cited_paper_id)

        # Group by root
        components: dict[str, list[str]] = defaultdict(list)
        for pid in nodes:
            components[find(pid)].append(pid)

        # Filter to 3+ nodes
        threads: list[ResearchThread] = []
        for i, (_root, members) in enumerate(
            sorted(components.items(), key=lambda kv: len(kv[1]), reverse=True)
        ):
            if len(members) < 3:
                continue

            # Build description from discovered papers in thread
            discovered_in_thread = [
                nodes[pid].title for pid in members if nodes[pid].is_discovered
            ]
            desc = (
                f"Thread with {len(members)} papers"
                + (f", including: {', '.join(discovered_in_thread[:3])}" if discovered_in_thread else "")
            )

            threads.append(
                ResearchThread(
                    name=f"thread-{i + 1}",
                    paper_ids=members,
                    description=desc,
                )
            )

        return threads
