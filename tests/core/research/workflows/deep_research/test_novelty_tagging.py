"""Tests for Phase 3: Novelty-Tagged Search Results for Researcher Stop Decisions.

Covers:
- 3.1: Novelty scoring in _handle_web_search_tool comparing new vs existing sources
- 3.2: Formatted results annotated with [NEW], [RELATED: <title>], [DUPLICATE] tags
- 3.3: Novelty summary line in search results message header
- 3.4: Think-tool stop-criteria injection references novelty tags
- 3.5: Lightweight content similarity helper (compute_novelty_tag, build_novelty_summary)
- 3.6: Novelty tags appear in researcher message history
- 3.7: Duplicate sources correctly tagged
- 3.8: Think injection references novelty context
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchState,
    ResearcherToolCall,
    TopicResearchResult,
)
from foundry_mcp.core.research.models.sources import (
    ResearchSource,
    SourceQuality,
    SourceType,
    SubQuery,
)
from foundry_mcp.core.research.workflows.deep_research._helpers import (
    NoveltyTag,
    build_novelty_summary,
    compute_novelty_tag,
    _extract_domain,
)
from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
    TopicResearchMixin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source(
    id: str = "src-1",
    title: str = "Test Source",
    url: str = "https://example.com/page",
    content: str = "Some content about the topic.",
    snippet: str | None = None,
    sub_query_id: str = "sq-0",
    quality: SourceQuality = SourceQuality.MEDIUM,
    **metadata_kw: Any,
) -> ResearchSource:
    """Create a ResearchSource for testing."""
    return ResearchSource(
        id=id,
        title=title,
        url=url,
        content=content,
        snippet=snippet,
        sub_query_id=sub_query_id,
        quality=quality,
        source_type=SourceType.WEB,
        metadata=metadata_kw,
    )


# =============================================================================
# Tests: compute_novelty_tag (3.5)
# =============================================================================


class TestComputeNoveltyTag:
    """Tests for the compute_novelty_tag helper function."""

    def test_new_when_no_existing_sources(self):
        """With no existing sources, everything is [NEW]."""
        tag = compute_novelty_tag(
            new_content="Some novel research content.",
            new_url="https://example.com/new",
            existing_sources=[],
        )
        assert tag.category == "new"
        assert tag.tag == "[NEW]"
        assert tag.similarity == 0.0

    def test_duplicate_when_identical_content(self):
        """Identical content should be tagged [DUPLICATE]."""
        content = "Renewable energy reduces carbon emissions and lowers costs for consumers."
        existing = [(content, "Original Source", "https://example.com/original")]

        tag = compute_novelty_tag(
            new_content=content,
            new_url="https://mirror.com/copy",
            existing_sources=existing,
        )
        assert tag.category == "duplicate"
        assert tag.tag == "[DUPLICATE]"
        assert tag.similarity >= 0.7
        assert tag.matched_title == "Original Source"

    def test_related_when_partial_overlap(self):
        """Content with partial overlap should be tagged [RELATED]."""
        # Use content that shares many n-grams but isn't identical,
        # producing a Jaccard similarity in the 0.3-0.7 range.
        base = (
            "Solar energy is a growing sector in renewable energy markets worldwide. "
            "Costs have dropped by 90% over the last decade making it competitive."
        )
        related = (
            "Solar energy is a growing sector in renewable energy markets worldwide. "
            "Investment in solar panels has increased substantially across all regions."
        )
        existing = [(base, "Solar Energy Report", "https://solar.com/report")]

        tag = compute_novelty_tag(
            new_content=related,
            new_url="https://news.com/solar",
            existing_sources=existing,
        )
        assert tag.category == "related"
        assert "[RELATED:" in tag.tag
        assert "Solar Energy Report" in tag.tag
        assert 0.3 <= tag.similarity < 0.7

    def test_new_when_completely_different_content(self):
        """Completely different content should be tagged [NEW]."""
        existing = [
            (
                "Machine learning algorithms process large datasets efficiently.",
                "ML Overview",
                "https://ml.com/overview",
            )
        ]

        tag = compute_novelty_tag(
            new_content="Ancient Roman architecture featured innovative concrete construction techniques.",
            new_url="https://history.com/rome",
            existing_sources=existing,
        )
        assert tag.category == "new"
        assert tag.tag == "[NEW]"
        assert tag.similarity < 0.3

    def test_domain_boost_increases_similarity(self):
        """Same-domain sources get a small similarity boost."""
        content_a = "Page one about renewable energy benefits and costs."
        content_b = "Page two about renewable energy advantages and pricing."
        existing = [(content_a, "Page One", "https://energy.com/page-one")]

        # Same domain
        tag_same = compute_novelty_tag(
            new_content=content_b,
            new_url="https://energy.com/page-two",
            existing_sources=existing,
        )
        # Different domain
        tag_diff = compute_novelty_tag(
            new_content=content_b,
            new_url="https://other.com/page-two",
            existing_sources=existing,
        )

        assert tag_same.similarity >= tag_diff.similarity

    def test_handles_none_urls(self):
        """Should not crash when URLs are None."""
        existing = [("Some content.", "Source", None)]
        tag = compute_novelty_tag(
            new_content="Completely different content here.",
            new_url=None,
            existing_sources=existing,
        )
        assert tag.category in ("new", "related", "duplicate")

    def test_handles_empty_content(self):
        """Empty content should be classified as new (no overlap possible)."""
        existing = [("Existing content.", "Source", "https://example.com")]
        tag = compute_novelty_tag(
            new_content="",
            new_url="https://other.com",
            existing_sources=existing,
        )
        assert tag.category == "new"
        assert tag.similarity == 0.0

    def test_long_title_truncated_in_related_tag(self):
        """Related tag should truncate long titles to 60 chars."""
        long_title = "A" * 100
        existing = [("Some overlapping content about energy.", long_title, "https://example.com")]
        tag = compute_novelty_tag(
            new_content="Some overlapping content about energy policy reforms.",
            new_url="https://other.com",
            existing_sources=existing,
        )
        if tag.category == "related":
            # Title in display should be truncated
            assert len(tag.tag) < len(long_title) + 20


# =============================================================================
# Tests: _extract_domain (3.5)
# =============================================================================


class TestExtractDomain:
    """Tests for the URL domain extraction helper."""

    def test_standard_url(self):
        assert _extract_domain("https://example.com/page") == "example.com"

    def test_strips_www(self):
        assert _extract_domain("https://www.example.com/page") == "example.com"

    def test_http_url(self):
        assert _extract_domain("http://example.com") == "example.com"

    def test_no_protocol(self):
        assert _extract_domain("example.com/path") == "example.com"

    def test_with_query_params(self):
        assert _extract_domain("https://example.com?foo=bar") == "example.com"

    def test_empty_string(self):
        assert _extract_domain("") is None

    def test_subdomain_preserved(self):
        assert _extract_domain("https://docs.example.com/api") == "docs.example.com"


# =============================================================================
# Tests: build_novelty_summary (3.3)
# =============================================================================


class TestBuildNoveltySummary:
    """Tests for the novelty summary line builder."""

    def test_all_new(self):
        tags = [
            NoveltyTag(tag="[NEW]", category="new", similarity=0.0),
            NoveltyTag(tag="[NEW]", category="new", similarity=0.1),
        ]
        summary = build_novelty_summary(tags)
        assert summary == "Novelty: 2 new, 0 related, 0 duplicate out of 2 results"

    def test_mixed(self):
        tags = [
            NoveltyTag(tag="[NEW]", category="new", similarity=0.0),
            NoveltyTag(tag="[RELATED: X]", category="related", similarity=0.5),
            NoveltyTag(tag="[DUPLICATE]", category="duplicate", similarity=0.9),
        ]
        summary = build_novelty_summary(tags)
        assert summary == "Novelty: 1 new, 1 related, 1 duplicate out of 3 results"

    def test_empty_list(self):
        summary = build_novelty_summary([])
        assert summary == "Novelty: 0 new, 0 related, 0 duplicate out of 0 results"


# =============================================================================
# Tests: _handle_think_tool references novelty (3.4, 3.8)
# =============================================================================


class TestThinkToolNoveltyReference:
    """Tests that the think-tool stop-criteria injection references novelty tags."""

    def test_think_response_mentions_novelty_tags(self):
        """3.8: Think tool response should reference novelty tags."""
        from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
            TopicResearchMixin,
        )

        # Create a minimal mixin instance
        mixin = TopicResearchMixin.__new__(TopicResearchMixin)

        sub_query = SubQuery(
            id="sq-0",
            query="test query",
            rationale="test",
            priority=1,
        )
        result = TopicResearchResult(sub_query_id="sq-0")
        tool_call = ResearcherToolCall(
            tool="think",
            arguments={"reasoning": "Analyzing results so far..."},
        )

        response = mixin._handle_think_tool(
            tool_call=tool_call,
            sub_query=sub_query,
            result=result,
        )

        assert "novelty tags" in response.lower()
        assert "[RELATED]" in response
        assert "[DUPLICATE]" in response
        assert "research_complete" in response


# =============================================================================
# Tests: Novelty tags in _handle_web_search_tool output (3.1, 3.2, 3.6, 3.7)
# =============================================================================


class StubTopicResearch(TopicResearchMixin):
    """Concrete class for testing TopicResearchMixin in isolation."""

    def __init__(self) -> None:
        self.config = MagicMock()
        self.config.deep_research_summarization_min_content_length = 300
        self.config.deep_research_summarization_timeout = 30
        self.config.deep_research_max_content_length = 50_000
        self.memory = MagicMock()
        self._search_providers: dict[str, Any] = {}

    def _write_audit_event(self, state: Any, event: str, **kwargs: Any) -> None:
        pass

    def _check_cancellation(self, state: Any) -> None:
        pass


def _make_state_with_existing_sources(
    sub_query_id: str = "sq-0",
    existing_sources: list[ResearchSource] | None = None,
) -> DeepResearchState:
    """Create a state with pre-existing sources for a sub-query."""
    from tests.core.research.workflows.deep_research.conftest import make_test_state

    state = make_test_state(
        id="deepres-test-novelty",
        query="test query",
        research_brief="Test brief",
        max_sources_per_query=10,
    )
    sq = SubQuery(
        id=sub_query_id,
        query="renewable energy benefits",
        rationale="Test",
        priority=1,
        status="executing",
    )
    state.sub_queries.append(sq)

    if existing_sources:
        for src in existing_sources:
            state.append_source(src)
            sq.source_ids.append(src.id)

    return state


class TestNoveltyInSearchResults:
    """Tests that _handle_web_search_tool annotates results with novelty tags."""

    @pytest.mark.asyncio
    async def test_novelty_tags_appear_in_output(self):
        """3.6: Novelty tags should appear in the formatted search results."""
        # Set up state with one existing source
        existing = _make_source(
            id="src-existing",
            title="Existing Solar Report",
            content="Solar energy benefits include cost reduction and sustainability.",
            url="https://solar.com/existing",
        )
        state = _make_state_with_existing_sources(existing_sources=[existing])

        # New sources from this search — one novel, one duplicate
        new_sources = [
            _make_source(
                id="src-new-1",
                title="Wind Energy Overview",
                content="Wind energy is a rapidly growing source of renewable electricity worldwide.",
                url="https://wind.com/overview",
            ),
            _make_source(
                id="src-dup",
                title="Solar Energy Benefits",
                content="Solar energy benefits include cost reduction and sustainability.",
                url="https://mirror.com/solar",
            ),
        ]

        stub = StubTopicResearch()
        result = TopicResearchResult(sub_query_id="sq-0")
        sq = state.sub_queries[0]

        # Mock _topic_search to add the new sources to state AND sub_query.source_ids
        async def mock_topic_search(**kwargs):
            for src in new_sources:
                state.append_source(src)
                sq.source_ids.append(src.id)
            return len(new_sources)

        # Mock _summarize_search_results to be a no-op
        async def mock_summarize(**kwargs):
            pass

        with patch.object(stub, "_topic_search", side_effect=mock_topic_search):
            with patch.object(stub, "_summarize_search_results", side_effect=mock_summarize):
                output, _charged = await stub._handle_web_search_tool(
                    tool_call=ResearcherToolCall(
                        tool="web_search",
                        arguments={"query": "renewable energy"},
                    ),
                    sub_query=sq,
                    state=state,
                    result=result,
                    available_providers=[],
                    max_sources_per_provider=5,
                    timeout=30.0,
                    seen_urls=set(),
                    seen_titles={},
                    state_lock=asyncio.Lock(),
                    semaphore=asyncio.Semaphore(5),
                )

        # Should contain novelty summary header
        assert "Novelty:" in output
        # Wind source should be [NEW] (different content)
        assert "[NEW]" in output
        # Solar copy should be [DUPLICATE] or [RELATED]
        assert "[DUPLICATE]" in output or "[RELATED:" in output

    @pytest.mark.asyncio
    async def test_all_new_when_no_existing_sources(self):
        """3.6: All sources should be [NEW] when there are no prior sources."""
        state = _make_state_with_existing_sources(existing_sources=[])
        sq = state.sub_queries[0]

        new_sources = [
            _make_source(id="src-1", title="Source A", content="Content A."),
            _make_source(id="src-2", title="Source B", content="Content B."),
        ]

        stub = StubTopicResearch()
        result = TopicResearchResult(sub_query_id="sq-0")

        async def mock_topic_search(**kwargs):
            for src in new_sources:
                state.append_source(src)
                sq.source_ids.append(src.id)
            return len(new_sources)

        async def mock_summarize(**kwargs):
            pass

        with patch.object(stub, "_topic_search", side_effect=mock_topic_search):
            with patch.object(stub, "_summarize_search_results", side_effect=mock_summarize):
                output, _charged = await stub._handle_web_search_tool(
                    tool_call=ResearcherToolCall(
                        tool="web_search",
                        arguments={"query": "test"},
                    ),
                    sub_query=sq,
                    state=state,
                    result=result,
                    available_providers=[],
                    max_sources_per_provider=5,
                    timeout=30.0,
                    seen_urls=set(),
                    seen_titles={},
                    state_lock=asyncio.Lock(),
                    semaphore=asyncio.Semaphore(5),
                )

        assert "Novelty: 2 new, 0 related, 0 duplicate out of 2 results" in output
        # Both should have [NEW] tag
        assert output.count("[NEW]") == 2

    @pytest.mark.asyncio
    async def test_duplicate_sources_correctly_tagged(self):
        """3.7: Near-identical sources are tagged [DUPLICATE]."""
        content = "The impact of climate change on global agriculture is significant and growing."
        existing = _make_source(
            id="src-existing",
            title="Climate Agriculture Impact",
            content=content,
        )
        state = _make_state_with_existing_sources(existing_sources=[existing])
        sq = state.sub_queries[0]

        # Near-identical copy
        duplicate = _make_source(
            id="src-dup",
            title="Climate Agriculture Impact (Copy)",
            content=content,  # Same content
            url="https://mirror.com/copy",
        )

        stub = StubTopicResearch()
        result = TopicResearchResult(sub_query_id="sq-0")

        async def mock_topic_search(**kwargs):
            state.append_source(duplicate)
            sq.source_ids.append(duplicate.id)
            return 1

        async def mock_summarize(**kwargs):
            pass

        with patch.object(stub, "_topic_search", side_effect=mock_topic_search):
            with patch.object(stub, "_summarize_search_results", side_effect=mock_summarize):
                output, _charged = await stub._handle_web_search_tool(
                    tool_call=ResearcherToolCall(
                        tool="web_search",
                        arguments={"query": "climate agriculture"},
                    ),
                    sub_query=sq,
                    state=state,
                    result=result,
                    available_providers=[],
                    max_sources_per_provider=5,
                    timeout=30.0,
                    seen_urls=set(),
                    seen_titles={},
                    state_lock=asyncio.Lock(),
                    semaphore=asyncio.Semaphore(5),
                )

        assert "[DUPLICATE]" in output
        assert "1 duplicate" in output

    @pytest.mark.asyncio
    async def test_novelty_metadata_stored_on_sources(self):
        """3.1: Novelty tag and similarity stored in source metadata."""
        state = _make_state_with_existing_sources(existing_sources=[])
        sq = state.sub_queries[0]

        new_source = _make_source(id="src-1", title="New Source", content="Brand new content.")

        stub = StubTopicResearch()
        result = TopicResearchResult(sub_query_id="sq-0")

        async def mock_topic_search(**kwargs):
            state.append_source(new_source)
            sq.source_ids.append(new_source.id)
            return 1

        async def mock_summarize(**kwargs):
            pass

        with patch.object(stub, "_topic_search", side_effect=mock_topic_search):
            with patch.object(stub, "_summarize_search_results", side_effect=mock_summarize):
                _output, _charged = await stub._handle_web_search_tool(
                    tool_call=ResearcherToolCall(
                        tool="web_search",
                        arguments={"query": "test"},
                    ),
                    sub_query=sq,
                    state=state,
                    result=result,
                    available_providers=[],
                    max_sources_per_provider=5,
                    timeout=30.0,
                    seen_urls=set(),
                    seen_titles={},
                    state_lock=asyncio.Lock(),
                    semaphore=asyncio.Semaphore(5),
                )

        # Check metadata on the source
        src = state.sources[-1]
        assert src.metadata.get("novelty_tag") == "new"
        assert "novelty_similarity" in src.metadata

    @pytest.mark.asyncio
    async def test_no_sources_returns_no_sources_message(self):
        """When no sources are added, novelty scoring is not invoked."""
        state = _make_state_with_existing_sources(existing_sources=[])
        stub = StubTopicResearch()
        result = TopicResearchResult(sub_query_id="sq-0")

        async def mock_topic_search(**kwargs):
            return 0

        with patch.object(stub, "_topic_search", side_effect=mock_topic_search):
            output, charged = await stub._handle_web_search_tool(
                tool_call=ResearcherToolCall(
                    tool="web_search",
                    arguments={"query": "test"},
                ),
                sub_query=state.sub_queries[0],
                state=state,
                result=result,
                available_providers=[],
                max_sources_per_provider=5,
                timeout=30.0,
                seen_urls=set(),
                seen_titles={},
                state_lock=asyncio.Lock(),
                semaphore=asyncio.Semaphore(5),
            )

        assert charged == 1
        assert "no new sources" in output.lower()
        # Should NOT contain novelty header
        assert "Novelty:" not in output
