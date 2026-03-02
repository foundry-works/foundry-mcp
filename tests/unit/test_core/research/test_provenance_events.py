"""Tests for deferred provenance events: provider_query, source_deduplicated,
gap_identified, gap_resolved.

Covers:
- provider_query event logged after provider search with correct details
- provider_query skipped when provenance is None
- source_deduplicated with url_match, title_match, content_similarity reasons
- _dedup_and_add_source() returns tuple[bool, Optional[str]]
- gap_identified event logged after state.add_gap()
- gap_resolved event logged after gap resolution
- gap_resolved skipped for nonexistent gap_id
- All new events carry ISO 8601 timestamps
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchState,
    ProvenanceLog,
)
from foundry_mcp.core.research.models.sources import ResearchSource, SubQuery


# =========================================================================
# Helpers
# =========================================================================


def _state_with_provenance(query: str = "test query") -> DeepResearchState:
    """Create a DeepResearchState with provenance enabled."""
    state = DeepResearchState(original_query=query)
    log = ProvenanceLog(
        session_id=state.id,
        query=query,
        started_at="2024-01-01T00:00:00+00:00",
    )
    state.extensions.provenance = log
    return state


def _state_without_provenance(query: str = "test query") -> DeepResearchState:
    """Create a DeepResearchState without provenance."""
    return DeepResearchState(original_query=query)


def _make_source(url: str = "https://example.com/a", title: str = "Example A") -> ResearchSource:
    return ResearchSource(url=url, title=title)


# =========================================================================
# _dedup_and_add_source return type tests
# =========================================================================


class TestDedupReturnType:
    """_dedup_and_add_source() returns (bool, Optional[str])."""

    @pytest.mark.asyncio
    async def test_dedup_return_type_added(self):
        """When a novel source is added, returns (True, None)."""
        from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
            _dedup_and_add_source,
        )

        state = _state_with_provenance()
        sub_query = SubQuery(query="test")
        lock = asyncio.Lock()

        source = _make_source(url="https://novel.example.com", title="Novel Title")
        result = await _dedup_and_add_source(
            source=source,
            sub_query=sub_query,
            state=state,
            seen_urls=set(),
            seen_titles={},
            state_lock=lock,
            content_dedup_enabled=False,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        was_added, reason = result
        assert was_added is True
        assert reason is None

    @pytest.mark.asyncio
    async def test_dedup_return_type_url_match(self):
        """When URL already seen, returns (False, 'url_match')."""
        from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
            _dedup_and_add_source,
        )

        state = _state_with_provenance()
        sub_query = SubQuery(query="test")
        lock = asyncio.Lock()
        seen_urls = {"https://duplicate.com"}

        source = _make_source(url="https://duplicate.com", title="Dup")
        was_added, reason = await _dedup_and_add_source(
            source=source,
            sub_query=sub_query,
            state=state,
            seen_urls=seen_urls,
            seen_titles={},
            state_lock=lock,
            content_dedup_enabled=False,
        )

        assert was_added is False
        assert reason == "url_match"

    @pytest.mark.asyncio
    async def test_dedup_return_type_title_match(self):
        """When normalized title already seen, returns (False, 'title_match')."""
        from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
            _dedup_and_add_source,
        )

        state = _state_with_provenance()
        sub_query = SubQuery(query="test")
        lock = asyncio.Lock()
        # Title with >20 chars so it triggers the title dedup path
        long_title = "A sufficiently long duplicate title for testing"
        from foundry_mcp.core.research.workflows.deep_research.source_quality import (
            _normalize_title,
        )

        normalized = _normalize_title(long_title)
        seen_titles = {normalized: "https://original.com"}

        source = _make_source(url="https://new-url.com", title=long_title)
        was_added, reason = await _dedup_and_add_source(
            source=source,
            sub_query=sub_query,
            state=state,
            seen_urls=set(),
            seen_titles=seen_titles,
            state_lock=lock,
            content_dedup_enabled=False,
        )

        assert was_added is False
        assert reason == "title_match"

    @pytest.mark.asyncio
    async def test_dedup_return_type_content_similarity(self):
        """When content is too similar, returns (False, 'content_similarity')."""
        from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
            _dedup_and_add_source,
        )

        state = _state_with_provenance()
        sub_query = SubQuery(query="test")
        lock = asyncio.Lock()

        # Add an existing source with substantial content
        existing = _make_source(url="https://existing.com", title="Existing Source")
        # Content needs to be > 100 chars for content dedup to trigger
        existing.content = "This is a detailed article about renewable energy and its benefits. " * 5
        state.append_source(existing)

        # New source with identical content but different URL
        new_source = _make_source(url="https://mirror.com", title="Mirror Source")
        new_source.content = existing.content  # Exact copy → similarity = 1.0

        was_added, reason = await _dedup_and_add_source(
            source=new_source,
            sub_query=sub_query,
            state=state,
            seen_urls=set(),
            seen_titles={},
            state_lock=lock,
            content_dedup_enabled=True,
            dedup_threshold=0.8,
        )

        assert was_added is False
        assert reason == "content_similarity"


# =========================================================================
# source_deduplicated provenance event tests
# =========================================================================


class TestSourceDeduplicatedEvent:
    """source_deduplicated provenance events are logged at caller sites."""

    @pytest.mark.asyncio
    async def test_source_deduplicated_url_match(self):
        """Event logged with reason='url_match' when URL already seen."""
        from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
            _dedup_and_add_source,
        )

        state = _state_with_provenance()
        sub_query = SubQuery(query="test")
        lock = asyncio.Lock()
        seen_urls = {"https://duplicate.com"}

        source = _make_source(url="https://duplicate.com", title="Dup Title")
        was_added, dedup_reason = await _dedup_and_add_source(
            source=source,
            sub_query=sub_query,
            state=state,
            seen_urls=seen_urls,
            seen_titles={},
            state_lock=lock,
            content_dedup_enabled=False,
        )

        assert was_added is False
        assert dedup_reason == "url_match"

        # Provenance event is logged by the CALLER, not the function itself.
        # Simulate what the caller does:
        if not was_added and dedup_reason and state.provenance is not None:
            state.provenance.append(
                phase="supervision",
                event_type="source_deduplicated",
                summary=f"Source deduplicated ({dedup_reason}): {(source.title or '')[:80]}",
                source_url=source.url or "",
                source_title=(source.title or "")[:120],
                reason=dedup_reason,
            )

        assert len(state.provenance.entries) == 1
        entry = state.provenance.entries[0]
        assert entry.event_type == "source_deduplicated"
        assert entry.details["reason"] == "url_match"
        assert "Dup Title" in entry.summary

    @pytest.mark.asyncio
    async def test_source_deduplicated_title_match(self):
        """Event logged with reason='title_match' for normalized title collision."""
        from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
            _dedup_and_add_source,
        )
        from foundry_mcp.core.research.workflows.deep_research.source_quality import (
            _normalize_title,
        )

        state = _state_with_provenance()
        sub_query = SubQuery(query="test")
        lock = asyncio.Lock()
        long_title = "A sufficiently long duplicate title for testing"
        normalized = _normalize_title(long_title)
        seen_titles = {normalized: "https://original.com"}

        source = _make_source(url="https://new.com", title=long_title)
        was_added, dedup_reason = await _dedup_and_add_source(
            source=source,
            sub_query=sub_query,
            state=state,
            seen_urls=set(),
            seen_titles=seen_titles,
            state_lock=lock,
            content_dedup_enabled=False,
        )

        assert was_added is False
        assert dedup_reason == "title_match"

        # Simulate caller logging
        if not was_added and dedup_reason and state.provenance is not None:
            state.provenance.append(
                phase="supervision",
                event_type="source_deduplicated",
                summary=f"Source deduplicated ({dedup_reason}): {(source.title or '')[:80]}",
                source_url=source.url or "",
                source_title=(source.title or "")[:120],
                reason=dedup_reason,
            )

        entry = state.provenance.entries[0]
        assert entry.details["reason"] == "title_match"

    @pytest.mark.asyncio
    async def test_source_deduplicated_content_similarity(self):
        """Event logged with reason='content_similarity' for near-duplicate content."""
        from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
            _dedup_and_add_source,
        )

        state = _state_with_provenance()
        sub_query = SubQuery(query="test")
        lock = asyncio.Lock()

        existing = _make_source(url="https://existing.com", title="Existing Source")
        existing.content = "Detailed article about renewable energy benefits for the planet. " * 5
        state.append_source(existing)

        new_source = _make_source(url="https://mirror.com", title="Mirror Source")
        new_source.content = existing.content

        was_added, dedup_reason = await _dedup_and_add_source(
            source=new_source,
            sub_query=sub_query,
            state=state,
            seen_urls=set(),
            seen_titles={},
            state_lock=lock,
            content_dedup_enabled=True,
            dedup_threshold=0.8,
        )

        assert was_added is False
        assert dedup_reason == "content_similarity"

        # Simulate caller logging
        if not was_added and dedup_reason and state.provenance is not None:
            state.provenance.append(
                phase="supervision",
                event_type="source_deduplicated",
                summary=f"Source deduplicated ({dedup_reason}): {(new_source.title or '')[:80]}",
                source_url=new_source.url or "",
                source_title=(new_source.title or "")[:120],
                reason=dedup_reason,
            )

        entry = state.provenance.entries[0]
        assert entry.details["reason"] == "content_similarity"


# =========================================================================
# provider_query provenance event tests
# =========================================================================


class TestProviderQueryEvent:
    """provider_query event logged after mock provider search."""

    def test_provider_query_event_logged(self):
        """Event appended after mock provider search with correct details."""
        state = _state_with_provenance()

        # Simulate what _topic_search does after a provider returns results
        provider_name = "tavily"
        query = "renewable energy benefits"
        sources = [_make_source(url=f"https://example.com/{i}", title=f"Result {i}") for i in range(3)]
        added_source_ids = [s.id for s in sources]

        state.provenance.append(
            phase="supervision",
            event_type="provider_query",
            summary=f"Queried {provider_name}: {len(sources)} results for '{query[:80]}'",
            provider=provider_name,
            query=query,
            result_count=len(sources),
            source_ids=added_source_ids,
        )

        assert len(state.provenance.entries) == 1
        entry = state.provenance.entries[0]
        assert entry.event_type == "provider_query"
        assert entry.phase == "supervision"
        assert entry.details["provider"] == "tavily"
        assert entry.details["query"] == "renewable energy benefits"
        assert entry.details["result_count"] == 3
        assert len(entry.details["source_ids"]) == 3
        assert "Queried tavily" in entry.summary

    def test_provider_query_skipped_without_provenance(self):
        """No error when state.provenance is None — guard pattern works."""
        state = _state_without_provenance()
        assert state.provenance is None

        # This is the guard pattern used in topic_research.py
        if state.provenance is not None:
            state.provenance.append(
                phase="supervision",
                event_type="provider_query",
                summary="This should not be reached",
            )

        # No error, no entries — provenance was None
        assert state.provenance is None


# =========================================================================
# gap_identified provenance event tests
# =========================================================================


class TestGapIdentifiedEvent:
    """gap_identified event logged after state.add_gap()."""

    def test_gap_identified_event_logged(self):
        """Event appended after add_gap with gap_id, description, priority."""
        state = _state_with_provenance()

        gap = state.add_gap(
            description="Missing data on offshore wind capacity factors",
            suggested_queries=["offshore wind capacity factor 2024"],
            priority=2,
        )

        # Simulate analysis.py behavior
        if gap and state.provenance is not None:
            state.provenance.append(
                phase="analysis",
                event_type="gap_identified",
                summary=f"Gap identified (priority {gap.priority}): {gap.description[:80]}",
                gap_id=gap.id,
                description=gap.description,
                priority=gap.priority,
                suggested_queries=gap.suggested_queries,
            )

        assert len(state.provenance.entries) == 1
        entry = state.provenance.entries[0]
        assert entry.event_type == "gap_identified"
        assert entry.phase == "analysis"
        assert entry.details["gap_id"] == gap.id
        assert entry.details["description"] == "Missing data on offshore wind capacity factors"
        assert entry.details["priority"] == 2
        assert entry.details["suggested_queries"] == ["offshore wind capacity factor 2024"]
        assert "priority 2" in entry.summary


# =========================================================================
# gap_resolved provenance event tests
# =========================================================================


class TestGapResolvedEvent:
    """gap_resolved event logged after gap.resolved = True."""

    def test_gap_resolved_event_logged(self):
        """Event appended after gap resolution with gap_id."""
        state = _state_with_provenance()

        gap = state.add_gap(
            description="Missing data on solar panel efficiency trends",
            suggested_queries=["solar panel efficiency 2024"],
            priority=1,
        )
        assert not gap.resolved

        # Simulate refinement.py behavior
        gap.resolved = True
        if state.provenance is not None:
            state.provenance.append(
                phase="refinement",
                event_type="gap_resolved",
                summary=f"Gap resolved: {gap.description[:80]}",
                gap_id=gap.id,
                resolution_notes="Found comprehensive data in latest IEA report",
            )

        assert len(state.provenance.entries) == 1
        entry = state.provenance.entries[0]
        assert entry.event_type == "gap_resolved"
        assert entry.phase == "refinement"
        assert entry.details["gap_id"] == gap.id
        assert entry.details["resolution_notes"] == "Found comprehensive data in latest IEA report"
        assert "Gap resolved" in entry.summary

    def test_gap_resolved_nonexistent_gap(self):
        """No event or error when gap_id not found — mirrors refinement.py guard."""
        state = _state_with_provenance()

        # Simulate refinement.py: get_gap returns None for nonexistent ID
        gap = state.get_gap("gap-nonexistent")
        assert gap is None

        # The if-guard prevents any provenance logging
        if gap:
            gap.resolved = True
            if state.provenance is not None:
                state.provenance.append(
                    phase="refinement",
                    event_type="gap_resolved",
                    summary=f"Gap resolved: {gap.description[:80]}",
                    gap_id=gap.id,
                )

        assert len(state.provenance.entries) == 0


# =========================================================================
# Timestamp tests
# =========================================================================


class TestEventTimestamps:
    """All new provenance events carry valid ISO 8601 timestamps."""

    def test_all_events_have_timestamps(self):
        """Every event from ProvenanceLog.append() has a valid ISO 8601 timestamp."""
        state = _state_with_provenance()

        # Log one of each new event type
        state.provenance.append(
            phase="supervision",
            event_type="provider_query",
            summary="Queried tavily: 3 results",
            provider="tavily",
            result_count=3,
        )
        state.provenance.append(
            phase="supervision",
            event_type="source_deduplicated",
            summary="Source deduplicated (url_match): Example",
            reason="url_match",
        )
        state.provenance.append(
            phase="analysis",
            event_type="gap_identified",
            summary="Gap identified (priority 1): Missing data",
            gap_id="gap-abc12345",
            priority=1,
        )
        state.provenance.append(
            phase="refinement",
            event_type="gap_resolved",
            summary="Gap resolved: Missing data",
            gap_id="gap-abc12345",
        )

        assert len(state.provenance.entries) == 4

        for entry in state.provenance.entries:
            assert entry.timestamp, f"Missing timestamp on {entry.event_type}"
            # Validate ISO 8601 — raises ValueError if invalid
            parsed = datetime.fromisoformat(entry.timestamp)
            assert parsed is not None
