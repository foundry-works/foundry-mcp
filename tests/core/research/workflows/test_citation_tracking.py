"""Unit tests for end-to-end citation tracking in deep research.

Tests citation number assignment on sources, synthesis prompt formatting
with [N] markers, citation post-processing (dangling removal, sources
section generation), and citation stability across refinement iterations.
"""

from __future__ import annotations

import pytest

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchState,
)
from foundry_mcp.core.research.models.enums import ConfidenceLevel
from foundry_mcp.core.research.models.sources import (
    ResearchSource,
    SourceQuality,
    SourceType,
)
from foundry_mcp.core.research.workflows.deep_research.phases._citation_postprocess import (
    build_sources_section,
    extract_cited_numbers,
    postprocess_citations,
    remove_dangling_citations,
    strip_llm_sources_section,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def state() -> DeepResearchState:
    """Create a minimal DeepResearchState for testing."""
    return DeepResearchState(original_query="test query")


@pytest.fixture
def state_with_sources(state: DeepResearchState) -> DeepResearchState:
    """State with three sources added via add_source()."""
    state.add_source(title="Alpha Source", url="https://alpha.example.com")
    state.add_source(title="Beta Source", url="https://beta.example.com")
    state.add_source(title="Gamma Source", url="https://gamma.example.com")
    return state


# =============================================================================
# Citation Number Assignment
# =============================================================================


class TestCitationNumberAssignment:
    """Tests for sequential citation number assignment."""

    def test_add_source_assigns_citation_number(self, state: DeepResearchState):
        s1 = state.add_source(title="First")
        assert s1.citation_number == 1

    def test_sequential_numbering(self, state: DeepResearchState):
        s1 = state.add_source(title="First")
        s2 = state.add_source(title="Second")
        s3 = state.add_source(title="Third")
        assert s1.citation_number == 1
        assert s2.citation_number == 2
        assert s3.citation_number == 3

    def test_citation_numbers_are_stable_after_addition(self, state_with_sources: DeepResearchState):
        """Adding new sources doesn't change existing citation numbers."""
        original_numbers = [s.citation_number for s in state_with_sources.sources]
        state_with_sources.add_source(title="Delta Source")
        for i, s in enumerate(state_with_sources.sources[:3]):
            assert s.citation_number == original_numbers[i]
        assert state_with_sources.sources[3].citation_number == 4

    def test_citation_number_on_source_model(self):
        """ResearchSource model supports citation_number field."""
        src = ResearchSource(title="Test", citation_number=42)
        assert src.citation_number == 42

    def test_citation_number_defaults_to_none(self):
        """ResearchSource without explicit citation_number is None."""
        src = ResearchSource(title="Test")
        assert src.citation_number is None


# =============================================================================
# append_source (centralised citation assignment for pre-constructed sources)
# =============================================================================


class TestAppendSource:
    """Tests for DeepResearchState.append_source()."""

    def test_append_source_assigns_citation_number(self, state: DeepResearchState):
        """append_source() assigns the next citation number."""
        src = ResearchSource(title="Pre-built Source", url="https://example.com")
        result = state.append_source(src)
        assert result.citation_number == 1
        assert result is src  # Same object returned

    def test_append_source_sequential_numbering(self, state: DeepResearchState):
        """Multiple append_source calls produce sequential citation numbers."""
        s1 = state.append_source(ResearchSource(title="First"))
        s2 = state.append_source(ResearchSource(title="Second"))
        s3 = state.append_source(ResearchSource(title="Third"))
        assert s1.citation_number == 1
        assert s2.citation_number == 2
        assert s3.citation_number == 3

    def test_append_source_overwrites_existing_citation_number(self, state: DeepResearchState):
        """append_source() overwrites any pre-set citation_number."""
        src = ResearchSource(title="Pre-numbered", citation_number=999)
        result = state.append_source(src)
        assert result.citation_number == 1  # Overwritten to next in sequence

    def test_append_source_interleaves_with_add_source(self, state: DeepResearchState):
        """Mixing add_source() and append_source() preserves sequencing."""
        s1 = state.add_source(title="Via add_source")
        s2 = state.append_source(ResearchSource(title="Via append_source"))
        s3 = state.add_source(title="Via add_source again")
        assert s1.citation_number == 1
        assert s2.citation_number == 2
        assert s3.citation_number == 3

    def test_append_source_increments_total_sources(self, state: DeepResearchState):
        """append_source() increments total_sources_examined."""
        assert state.total_sources_examined == 0
        state.append_source(ResearchSource(title="Test"))
        assert state.total_sources_examined == 1


# =============================================================================
# State Helper Methods
# =============================================================================


class TestStateCitationHelpers:
    """Tests for get_citation_map() and source_id_to_citation()."""

    def test_get_citation_map(self, state_with_sources: DeepResearchState):
        cm = state_with_sources.get_citation_map()
        assert set(cm.keys()) == {1, 2, 3}
        assert cm[1].title == "Alpha Source"
        assert cm[2].title == "Beta Source"
        assert cm[3].title == "Gamma Source"

    def test_source_id_to_citation(self, state_with_sources: DeepResearchState):
        mapping = state_with_sources.source_id_to_citation()
        for s in state_with_sources.sources:
            assert mapping[s.id] == s.citation_number

    def test_citation_map_excludes_none(self, state: DeepResearchState):
        """Sources without citation numbers are excluded from the map."""
        # Manually add a source without citation number
        src = ResearchSource(title="No Citation")
        state.sources.append(src)
        cm = state.get_citation_map()
        assert len(cm) == 0

    def test_empty_state(self, state: DeepResearchState):
        assert state.get_citation_map() == {}
        assert state.source_id_to_citation() == {}


# =============================================================================
# extract_cited_numbers
# =============================================================================


class TestExtractCitedNumbers:
    def test_basic_extraction(self):
        report = "Finding supported by [1] and [3]."
        assert extract_cited_numbers(report) == {1, 3}

    def test_no_citations(self):
        report = "A plain report with no citations."
        assert extract_cited_numbers(report) == set()

    def test_duplicate_citations(self):
        report = "Mentioned [2] here and [2] again."
        assert extract_cited_numbers(report) == {2}

    def test_multi_digit_citations(self):
        report = "Source [12] and [345] are relevant."
        assert extract_cited_numbers(report) == {12, 345}

    def test_adjacent_citations(self):
        report = "Evidence [1][2][3] supports this."
        assert extract_cited_numbers(report) == {1, 2, 3}

    def test_markdown_links_not_matched(self):
        """[N](url) patterns should NOT be extracted as citations."""
        report = "See [1](https://example.com) and also [2]."
        assert extract_cited_numbers(report) == {2}

    def test_markdown_link_mixed_with_citations(self):
        """Markdown links and bare citations coexist correctly."""
        report = "Ref [1] and link [2](https://x.com) and [3]."
        assert extract_cited_numbers(report) == {1, 3}

    def test_inline_link_followed_by_citation(self):
        """[Title](URL) [N] pattern: the [N] is extracted as a citation."""
        report = "According to [MIT Tech Review](https://example.com) [1], the technology is mature."
        assert extract_cited_numbers(report) == {1}

    def test_inline_link_followed_by_citation_no_space(self):
        """[Title](URL)[N] pattern without space: [N] is still extracted."""
        report = "See [Alpha Source](https://alpha.example.com)[1] for details."
        assert extract_cited_numbers(report) == {1}

    def test_multiple_inline_links_with_citations(self):
        """Multiple [Title](URL) [N] patterns in one report."""
        report = (
            "Research by [Alpha](https://a.com) [1] and [Beta](https://b.com) [2] "
            "confirms the finding. Later, [1] was cited again."
        )
        assert extract_cited_numbers(report) == {1, 2}


# =============================================================================
# remove_dangling_citations
# =============================================================================


class TestRemoveDanglingCitations:
    def test_removes_invalid_numbers(self):
        report = "Finding [1] is supported but [99] is not."
        result = remove_dangling_citations(report, valid_numbers={1, 2, 3})
        assert "[1]" in result
        assert "[99]" not in result

    def test_keeps_valid_numbers(self):
        report = "Sources [1], [2], and [3] are valid."
        result = remove_dangling_citations(report, valid_numbers={1, 2, 3})
        assert result == report

    def test_empty_valid_set(self):
        report = "All dangling: [1] [2] [3]."
        result = remove_dangling_citations(report, valid_numbers=set())
        assert "[1]" not in result
        assert "[2]" not in result
        assert "[3]" not in result

    def test_preserves_inline_link_with_valid_citation(self):
        """[Title](URL) [N] pattern: valid [N] is preserved, markdown link untouched."""
        report = "According to [Alpha Source](https://alpha.example.com) [1], the data shows..."
        result = remove_dangling_citations(report, valid_numbers={1})
        assert "[Alpha Source](https://alpha.example.com) [1]" in result

    def test_removes_dangling_citation_after_inline_link(self):
        """[Title](URL) [N] pattern: dangling [N] is removed, markdown link untouched."""
        report = "According to [Alpha Source](https://alpha.example.com) [99], the data shows..."
        result = remove_dangling_citations(report, valid_numbers={1, 2})
        assert "[Alpha Source](https://alpha.example.com)" in result
        assert "[99]" not in result


# =============================================================================
# strip_llm_sources_section
# =============================================================================


class TestStripLlmSourcesSection:
    def test_strips_sources_heading(self):
        report = "# Report\n\nContent.\n\n## Sources\n\n- Source 1\n- Source 2\n"
        result = strip_llm_sources_section(report)
        assert "## Sources" not in result
        assert "- Source 1" not in result
        assert "Content." in result

    def test_strips_references_heading(self):
        report = "# Report\n\nContent.\n\n## References\n\n[1] Foo\n[2] Bar\n"
        result = strip_llm_sources_section(report)
        assert "## References" not in result

    def test_preserves_content_before_and_after(self):
        report = "# Report\n\nContent.\n\n## Sources\n\n- Src\n\n## Conclusions\n\nFinal."
        result = strip_llm_sources_section(report)
        assert "Content." in result
        assert "## Conclusions" in result
        assert "Final." in result

    def test_no_sources_section(self):
        report = "# Report\n\nJust content.\n"
        result = strip_llm_sources_section(report)
        assert result == report

    def test_case_insensitive(self):
        report = "# Report\n\n## SOURCES\n\n- Foo\n"
        result = strip_llm_sources_section(report)
        assert "SOURCES" not in result


# =============================================================================
# build_sources_section
# =============================================================================


class TestBuildSourcesSection:
    def test_builds_numbered_list(self, state_with_sources: DeepResearchState):
        section = build_sources_section(state_with_sources)
        assert "## Sources" in section
        assert "[1] [Alpha Source](https://alpha.example.com)" in section
        assert "[2] [Beta Source](https://beta.example.com)" in section
        assert "[3] [Gamma Source](https://gamma.example.com)" in section

    def test_sorted_by_citation_number(self, state_with_sources: DeepResearchState):
        section = build_sources_section(state_with_sources)
        lines = [line for line in section.strip().split("\n") if line.startswith("[")]
        assert len(lines) == 3
        # Verify order
        assert lines[0].startswith("[1]")
        assert lines[1].startswith("[2]")
        assert lines[2].startswith("[3]")

    def test_source_without_url(self, state: DeepResearchState):
        state.add_source(title="No URL Source")
        section = build_sources_section(state)
        assert "[1] No URL Source" in section
        assert "](http" not in section

    def test_cited_only_filter(self, state_with_sources: DeepResearchState):
        section = build_sources_section(state_with_sources, cited_only=True, cited_numbers={1, 3})
        assert "[1]" in section
        assert "[2]" not in section
        assert "[3]" in section

    def test_empty_sources(self, state: DeepResearchState):
        section = build_sources_section(state)
        assert section == ""


# =============================================================================
# postprocess_citations (integration)
# =============================================================================


class TestPostprocessCitations:
    def test_full_pipeline(self, state_with_sources: DeepResearchState):
        report = "# Report\n\nFinding [1] and [2] are important.\n\n## Sources\n\n- Old LLM sources\n"
        processed, meta = postprocess_citations(report, state_with_sources)

        # LLM sources section should be stripped
        assert "Old LLM sources" not in processed
        # Deterministic sources section should be appended
        assert "[1] [Alpha Source](https://alpha.example.com)" in processed
        assert "[2] [Beta Source](https://beta.example.com)" in processed
        assert "[3] [Gamma Source](https://gamma.example.com)" in processed
        # Valid citations should be preserved
        assert "[1]" in processed.split("## Sources")[0]
        assert "[2]" in processed.split("## Sources")[0]
        # Metadata
        assert meta["total_citations_in_report"] == 2
        assert meta["dangling_citations_removed"] == 0
        assert meta["unreferenced_sources"] == 1  # [3] not cited

    def test_dangling_citations_removed(self, state_with_sources: DeepResearchState):
        report = "Finding [1] and [99] are mentioned."
        processed, meta = postprocess_citations(report, state_with_sources)
        assert "[1]" in processed
        assert "[99]" not in processed
        assert meta["dangling_citations_removed"] == 1

    def test_no_citations(self, state_with_sources: DeepResearchState):
        report = "# Report\n\nNo citations at all."
        processed, meta = postprocess_citations(report, state_with_sources)
        assert "## Sources" in processed
        assert meta["total_citations_in_report"] == 0
        assert meta["unreferenced_sources"] == 3

    def test_no_sources(self, state: DeepResearchState):
        report = "# Report\n\nSome [1] citation."
        processed, meta = postprocess_citations(report, state)
        assert meta["dangling_citations_removed"] == 1
        assert "[1]" not in processed

    def test_inline_links_survive_postprocessing(self, state_with_sources: DeepResearchState):
        """[Title](URL) [N] inline links survive the full post-processing pipeline."""
        report = (
            "# Report\n\n"
            "According to [Alpha Source](https://alpha.example.com) [1], caffeine affects sleep. "
            "Research by [Beta Source](https://beta.example.com) [2] confirms this. "
            "As noted in [1], the effect is dose-dependent.\n"
        )
        processed, meta = postprocess_citations(report, state_with_sources)

        body = processed.split("## Sources")[0]

        # Inline markdown links preserved
        assert "[Alpha Source](https://alpha.example.com) [1]" in body
        assert "[Beta Source](https://beta.example.com) [2]" in body
        # Bare [N] subsequent reference preserved
        assert "As noted in [1]" in body
        # Citations counted correctly (first [1] + [2] + second [1] = {1, 2})
        assert meta["total_citations_in_report"] == 2
        assert meta["dangling_citations_removed"] == 0

    def test_inline_links_with_sources_section_correct(self, state_with_sources: DeepResearchState):
        """Auto-appended Sources section is correct when inline links are present."""
        report = (
            "# Report\n\n"
            "See [Alpha Source](https://alpha.example.com) [1] for details.\n"
        )
        processed, meta = postprocess_citations(report, state_with_sources)

        # Sources section is appended
        assert "## Sources" in processed
        # All sources listed in the appended section
        assert "[1] [Alpha Source](https://alpha.example.com)" in processed
        assert "[2] [Beta Source](https://beta.example.com)" in processed
        assert "[3] [Gamma Source](https://gamma.example.com)" in processed
        # Unreferenced sources counted
        assert meta["unreferenced_sources"] == 2  # [2] and [3] not cited

    def test_dangling_inline_link_citation_removed(self, state_with_sources: DeepResearchState):
        """Dangling [N] after an inline link is removed, but the link itself stays."""
        report = (
            "# Report\n\n"
            "According to [Unknown Source](https://unknown.example.com) [99], something happened.\n"
        )
        processed, meta = postprocess_citations(report, state_with_sources)

        body = processed.split("## Sources")[0]

        # The markdown link itself is preserved (it's not a citation)
        assert "[Unknown Source](https://unknown.example.com)" in body
        # The dangling [99] is removed
        assert "[99]" not in body
        assert meta["dangling_citations_removed"] == 1


# =============================================================================
# Citation Stability Across Refinement
# =============================================================================


class TestCitationStabilityAcrossRefinement:
    """Verify citation numbers remain stable when new sources are added
    during refinement iterations."""

    def test_refinement_preserves_citation_numbers(self, state_with_sources: DeepResearchState):
        """Simulates refinement: existing citations stay, new ones are sequential."""
        # Record original citation numbers
        original = {s.id: s.citation_number for s in state_with_sources.sources}

        # Simulate refinement adding new sources
        s4 = state_with_sources.add_source(title="Refinement Source 1", url="https://refine1.example.com")
        s5 = state_with_sources.add_source(title="Refinement Source 2", url="https://refine2.example.com")

        # Original citations unchanged
        for s in state_with_sources.sources[:3]:
            assert s.citation_number == original[s.id]

        # New sources get sequential numbers
        assert s4.citation_number == 4
        assert s5.citation_number == 5

    def test_citations_in_report_survive_resynthesis(self, state_with_sources: DeepResearchState):
        """Post-processing should work correctly with the same state on re-synthesis."""
        # First synthesis
        report1 = "Finding [1] is key."
        processed1, _ = postprocess_citations(report1, state_with_sources)

        # Add refinement source
        state_with_sources.add_source(title="New Source")

        # Re-synthesis references the new source too
        report2 = "Finding [1] is key. New insight from [4]."
        processed2, meta2 = postprocess_citations(report2, state_with_sources)

        assert "[1]" in processed2.split("## Sources")[0]
        assert "[4]" in processed2.split("## Sources")[0]
        assert meta2["total_citations_in_report"] == 2
        assert meta2["dangling_citations_removed"] == 0

    def test_add_finding_with_citation_references(self, state_with_sources: DeepResearchState):
        """Findings referencing source_ids can be mapped to citation numbers."""
        src_ids = [s.id for s in state_with_sources.sources[:2]]
        state_with_sources.add_finding(
            content="Test finding",
            confidence=ConfidenceLevel.HIGH,
            source_ids=src_ids,
        )
        id_to_cn = state_with_sources.source_id_to_citation()
        finding = state_with_sources.findings[0]
        citation_refs = [id_to_cn[sid] for sid in finding.source_ids]
        assert citation_refs == [1, 2]
