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
    format_source_apa,
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
        report = "# Report\n\nSee [Alpha Source](https://alpha.example.com) [1] for details.\n"
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
        report = "# Report\n\nAccording to [Unknown Source](https://unknown.example.com) [99], something happened.\n"
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


# =============================================================================
# APA Citation Formatting (PLAN-1 Item 4)
# =============================================================================


class TestFormatSourceApa:
    """Tests for format_source_apa() — APA 7th-edition reference formatting.

    Note: Semantic Scholar formats authors as comma-separated full names
    (e.g. "John Smith, Jane Jones") — not "Last, First" format.
    """

    def test_full_academic_metadata(self):
        """Full academic source: Authors (Year). Title. *Venue*. DOI_URL."""
        source = ResearchSource(
            title="Deep Learning for NLP",
            url="https://semanticscholar.org/paper/abc",
            metadata={
                "authors": "John Smith, Alice Jones, Chang Lee",
                "year": 2023,
                "venue": "Nature Machine Intelligence",
                "doi": "10.1038/s42256-023-00001-1",
                "citation_count": 142,
            },
        )
        result = format_source_apa(source)
        assert "John Smith, Alice Jones, & Chang Lee" in result
        assert "(2023)" in result
        assert "Deep Learning for NLP." in result
        assert "*Nature Machine Intelligence*." in result
        assert "https://doi.org/10.1038/s42256-023-00001-1" in result
        # DOI URL preferred over source.url
        assert "semanticscholar.org" not in result

    def test_partial_metadata_missing_venue(self):
        """Missing venue: Authors (Year). Title. URL."""
        source = ResearchSource(
            title="Attention Is All You Need",
            url="https://arxiv.org/abs/1706.03762",
            metadata={
                "authors": "Ashish Vaswani, Noam Shazeer",
                "year": 2017,
                "doi": None,
                "venue": None,
            },
        )
        result = format_source_apa(source)
        assert "Ashish Vaswani & Noam Shazeer" in result
        assert "(2017)" in result
        assert "Attention Is All You Need." in result
        assert "https://arxiv.org/abs/1706.03762" in result
        # No venue italics
        assert "*" not in result

    def test_partial_metadata_missing_doi(self):
        """Missing DOI: falls back to source.url."""
        source = ResearchSource(
            title="Some Paper",
            url="https://example.com/paper",
            metadata={
                "authors": "Jane Doe",
                "year": 2020,
                "venue": "ICML",
            },
        )
        result = format_source_apa(source)
        assert "Jane Doe (2020). Some Paper." in result
        assert "*ICML*." in result
        assert "https://example.com/paper" in result

    def test_web_source_no_academic_metadata(self):
        """Web source with no academic metadata: Title. URL."""
        source = ResearchSource(
            title="Getting Started with Python",
            url="https://python.org/getting-started",
            metadata={},
        )
        result = format_source_apa(source)
        # No author → title in author position
        assert result.startswith("Getting Started with Python (n.d.).")
        assert "https://python.org/getting-started" in result
        # No venue
        assert "*" not in result

    def test_et_al_more_than_five_authors(self):
        """>5 authors: first author et al."""
        source = ResearchSource(
            title="Large Collaboration Paper",
            url="https://example.com",
            metadata={
                "authors": "Alice Alpha, Bob Beta, Carol Gamma, Dan Delta, Eve Epsilon, Frank Zeta",
                "year": 2024,
            },
        )
        result = format_source_apa(source)
        assert "Alice Alpha et al." in result
        assert "(2024)" in result
        # Other authors should not appear
        assert "Bob Beta" not in result
        assert "Frank Zeta" not in result

    def test_missing_year_nd(self):
        """Missing year: (n.d.) per APA convention."""
        source = ResearchSource(
            title="Undated Document",
            url="https://example.com/undated",
            metadata={"authors": "Alex Unknown"},
        )
        result = format_source_apa(source)
        assert "(n.d.)" in result
        assert "Alex Unknown (n.d.). Undated Document." in result

    def test_no_url_and_no_doi(self):
        """Source with no URL and no DOI: just author, year, title."""
        source = ResearchSource(
            title="Offline Paper",
            metadata={"authors": "Jane Doe", "year": 2019},
        )
        result = format_source_apa(source)
        assert result == "Jane Doe (2019). Offline Paper."

    def test_empty_metadata(self):
        """Source with completely empty metadata: minimal fallback."""
        source = ResearchSource(title="Bare Title")
        result = format_source_apa(source)
        assert result == "Bare Title (n.d.)."

    def test_two_authors(self):
        """Two authors: joined with &."""
        source = ResearchSource(
            title="Dual Author Paper",
            metadata={"authors": "Alice Foo, Bob Bar", "year": 2021},
        )
        result = format_source_apa(source)
        assert "Alice Foo & Bob Bar" in result

    def test_single_author(self):
        """Single author: no & or comma joining."""
        source = ResearchSource(
            title="Solo Paper",
            metadata={"authors": "Sam Solo", "year": 2022},
        )
        result = format_source_apa(source)
        assert result.startswith("Sam Solo (2022). Solo Paper.")

    def test_five_authors_all_listed(self):
        """Exactly 5 authors: all listed with & before last."""
        source = ResearchSource(
            title="Five Authors Paper",
            metadata={
                "authors": "A One, B Two, C Three, D Four, E Five",
                "year": 2023,
            },
        )
        result = format_source_apa(source)
        assert "A One, B Two, C Three, D Four, & E Five" in result


class TestBuildSourcesSectionApa:
    """Tests for build_sources_section() with format_style='apa'."""

    @pytest.fixture
    def state_with_academic_sources(self) -> DeepResearchState:
        """State with sources that have academic metadata."""
        state = DeepResearchState(original_query="literature review on X")
        state.add_source(
            title="Paper Alpha",
            url="https://doi.org/10.1234/alpha",
            metadata={
                "authors": "John Smith, Kate Lee",
                "year": 2022,
                "venue": "Nature",
                "doi": "10.1234/alpha",
            },
        )
        state.add_source(
            title="Paper Beta",
            url="https://example.com/beta",
            metadata={
                "authors": "Alice Jones",
                "year": 2023,
            },
        )
        state.add_source(
            title="Web Source Gamma",
            url="https://gamma.example.com",
        )
        return state

    def test_apa_format_produces_references_heading(self, state_with_academic_sources):
        section = build_sources_section(state_with_academic_sources, format_style="apa")
        assert "## References" in section
        assert "## Sources" not in section

    def test_default_format_produces_sources_heading(self, state_with_academic_sources):
        section = build_sources_section(state_with_academic_sources, format_style="default")
        assert "## Sources" in section
        assert "## References" not in section

    def test_apa_entries_contain_apa_formatting(self, state_with_academic_sources):
        section = build_sources_section(state_with_academic_sources, format_style="apa")
        # First source: full academic
        assert "John Smith & Kate Lee (2022). Paper Alpha." in section
        assert "*Nature*." in section
        # Second source: partial
        assert "Alice Jones (2023). Paper Beta." in section
        # Third source: web/minimal
        assert "Web Source Gamma (n.d.)." in section

    def test_default_format_preserves_existing_behavior(self, state_with_academic_sources):
        section = build_sources_section(state_with_academic_sources, format_style="default")
        assert "[1] [Paper Alpha](https://doi.org/10.1234/alpha)" in section
        assert "[2] [Paper Beta](https://example.com/beta)" in section
        assert "[3] [Web Source Gamma](https://gamma.example.com)" in section

    def test_apa_entries_have_citation_numbers(self, state_with_academic_sources):
        """APA entries still include [N] prefix for cross-referencing."""
        section = build_sources_section(state_with_academic_sources, format_style="apa")
        assert "[1] " in section
        assert "[2] " in section
        assert "[3] " in section

    def test_empty_state_returns_empty(self):
        state = DeepResearchState(original_query="test")
        section = build_sources_section(state, format_style="apa")
        assert section == ""


class TestPostprocessCitationsWithProfile:
    """Integration tests: postprocess_citations() respects profile and query_type."""

    @pytest.fixture
    def academic_state(self) -> DeepResearchState:
        """State with academic profile and sources."""
        from foundry_mcp.core.research.models.deep_research import PROFILE_ACADEMIC

        state = DeepResearchState(original_query="literature review on attention mechanisms")
        state.extensions.research_profile = PROFILE_ACADEMIC
        state.add_source(
            title="Attention Is All You Need",
            url="https://arxiv.org/abs/1706.03762",
            metadata={
                "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar",
                "year": 2017,
                "venue": "NeurIPS",
                "doi": "10.5555/3295222.3295349",
            },
        )
        state.add_source(
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            url="https://arxiv.org/abs/1810.04805",
            metadata={
                "authors": "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova",
                "year": 2019,
                "venue": "NAACL",
            },
        )
        return state

    def test_literature_review_query_type_forces_apa(self, academic_state):
        """query_type='literature_review' forces APA regardless of profile."""
        report = "# Report\n\nFindings [1] and [2] support the thesis.\n"
        processed, meta = postprocess_citations(
            report, academic_state, query_type="literature_review",
        )
        assert "## References" in processed
        assert "## Sources" not in processed
        assert meta["format_style"] == "apa"

    def test_academic_profile_uses_apa(self, academic_state):
        """Academic profile (citation_style='apa') produces APA output."""
        report = "# Report\n\nSee [1] for details.\n"
        processed, meta = postprocess_citations(
            report, academic_state, query_type="explanation",
        )
        assert "## References" in processed
        assert meta["format_style"] == "apa"

    def test_general_profile_uses_default(self):
        """General profile uses default formatting."""
        state = DeepResearchState(original_query="test query")
        state.add_source(title="Source A", url="https://a.example.com")
        report = "# Report\n\nFinding [1].\n"
        processed, meta = postprocess_citations(
            report, state, query_type="explanation",
        )
        assert "## Sources" in processed
        assert "## References" not in processed
        assert meta["format_style"] == "default"

    def test_no_query_type_defaults_to_profile(self, academic_state):
        """When query_type is None, uses profile's citation_style."""
        report = "# Report\n\nFinding [1].\n"
        processed, meta = postprocess_citations(report, academic_state)
        assert "## References" in processed
        assert meta["format_style"] == "apa"

    def test_apa_entries_in_final_output(self, academic_state):
        """Verify APA-formatted entries appear in the final processed report."""
        report = "# Report\n\nAttention mechanisms [1] are foundational.\n"
        processed, _meta = postprocess_citations(
            report, academic_state, query_type="literature_review",
        )
        # Source 1: 3 authors with DOI URL
        assert "Ashish Vaswani, Noam Shazeer, & Niki Parmar (2017)" in processed
        assert "*NeurIPS*." in processed
        assert "https://doi.org/10.5555/3295222.3295349" in processed
        # Source 2: 4 authors without DOI
        assert "Jacob Devlin, Ming-Wei Chang, Kenton Lee, & Kristina Toutanova (2019)" in processed
        assert "*NAACL*." in processed

    def test_existing_tests_still_pass_with_default(self):
        """Backward compat: postprocess_citations without query_type produces default format."""
        state = DeepResearchState(original_query="test query")
        state.add_source(title="Alpha Source", url="https://alpha.example.com")
        state.add_source(title="Beta Source", url="https://beta.example.com")
        report = "# Report\n\nFinding [1] and [2].\n\n## Sources\n\n- Old\n"
        processed, meta = postprocess_citations(report, state)
        # Default format
        assert "[1] [Alpha Source](https://alpha.example.com)" in processed
        assert "[2] [Beta Source](https://beta.example.com)" in processed
        assert "## Sources" in processed
        assert meta["format_style"] == "default"
