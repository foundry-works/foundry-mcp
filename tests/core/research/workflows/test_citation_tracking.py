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
    cleanup_citations,
    extract_cited_numbers,
    finalize_citations,
    format_source_apa,
    needs_renumber,
    postprocess_citations,
    remove_dangling_citations,
    renumber_citations,
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

    # --- max_citation filtering ---

    def test_max_citation_filters_year_references(self):
        """Year references like [2025] and [2026] are excluded when max_citation is set."""
        report = "published in [2025] and updated [2026]"
        assert extract_cited_numbers(report, max_citation=61) == set()

    def test_max_citation_keeps_valid_citations(self):
        """Valid citations below max_citation are kept while years are filtered."""
        report = "[1] and [2] and [2025]"
        assert extract_cited_numbers(report, max_citation=61) == {1, 2}

    def test_max_citation_boundary_value(self):
        """Citation equal to max_citation is kept; one above is excluded."""
        report = "[5] and [6] and [7]"
        assert extract_cited_numbers(report, max_citation=6) == {5, 6}

    def test_max_citation_none_preserves_all(self):
        """Without max_citation, all non-year citations are preserved."""
        report = "[1] and [500]"
        assert extract_cited_numbers(report) == {1, 500}
        assert extract_cited_numbers(report, max_citation=None) == {1, 500}


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
        # Deterministic sources section should be appended — only cited sources
        assert "[1] [Alpha Source](https://alpha.example.com)" in processed
        assert "[2] [Beta Source](https://beta.example.com)" in processed
        # Uncited source [3] should NOT appear in bibliography
        assert "[3] [Gamma Source]" not in processed
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
        # With cited_only filtering, no sources section when nothing is cited
        assert "## Sources" not in processed
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

        # Sources section is appended with only cited sources
        assert "## Sources" in processed
        assert "[1] [Alpha Source](https://alpha.example.com)" in processed
        # Uncited sources should NOT appear in bibliography
        assert "[2] [Beta Source]" not in processed
        assert "[3] [Gamma Source]" not in processed
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

    def test_year_references_not_reported_as_dangling(self, state_with_sources: DeepResearchState):
        """Year references like [2025] are not treated as dangling citations."""
        report = "# Report\n\nFinding [1] is from [2025] and updated in [2026].\n"
        processed, meta = postprocess_citations(report, state_with_sources)

        body = processed.split("## Sources")[0]

        # Valid citation preserved
        assert "[1]" in body
        # Year references left intact (not stripped as dangling)
        assert "[2025]" in body
        assert "[2026]" in body
        # No dangling citations reported
        assert meta["dangling_citations_removed"] == 0
        # Only [1] counted as a citation
        assert meta["total_citations_in_report"] == 1


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

        body2 = processed2.split("## Sources")[0]
        # After renumbering: [1]→[1], [4]→[2] (reading order)
        assert "[1]" in body2
        assert "[2]" in body2
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
# APA Citation Formatting
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
        report = "# Report\n\nAttention mechanisms [1] and transformers [2] are foundational.\n"
        processed, _meta = postprocess_citations(
            report, academic_state, query_type="literature_review",
        )
        # Source 1: 3 authors with DOI URL
        assert "Ashish Vaswani, Noam Shazeer, & Niki Parmar (2017)" in processed
        assert "*NeurIPS*." in processed
        assert "https://doi.org/10.5555/3295222.3295349" in processed
        # Source 2: 4 authors without DOI (now explicitly cited in report)
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


# =============================================================================
# Phase 2: Bibliography filtered to cited-only sources
# =============================================================================


class TestBibliographyCitedOnly:
    """Tests verifying bibliography only includes sources actually cited in the report."""

    def test_bibliography_contains_only_cited_sources(self):
        """Bibliography includes cited sources and excludes uncited ones."""
        state = DeepResearchState(original_query="test query")
        state.add_source(title="Cited A", url="https://a.example.com")
        state.add_source(title="Uncited B", url="https://b.example.com")
        state.add_source(title="Cited C", url="https://c.example.com")
        state.add_source(title="Uncited D", url="https://d.example.com")
        state.add_source(title="Cited E", url="https://e.example.com")

        report = "# Report\n\nEvidence from [1], [3], and [5] supports the thesis.\n"
        processed, meta = postprocess_citations(report, state)

        # After renumbering: [1]→[1], [3]→[2], [5]→[3]
        # Cited sources appear in bibliography with renumbered citations
        assert "[1] [Cited A](https://a.example.com)" in processed
        assert "[2] [Cited C](https://c.example.com)" in processed
        assert "[3] [Cited E](https://e.example.com)" in processed
        # Uncited sources do NOT appear in bibliography
        assert "Uncited B" not in processed.split("## Sources")[1]
        assert "Uncited D" not in processed.split("## Sources")[1]
        # Metadata reflects correct counts
        assert meta["total_citations_in_report"] == 3
        assert meta["unreferenced_sources"] == 2

    def test_apa_format_with_cited_only(self):
        """APA-formatted bibliography respects cited_only filtering."""
        from foundry_mcp.core.research.models.deep_research import PROFILE_ACADEMIC

        state = DeepResearchState(original_query="literature review")
        state.extensions.research_profile = PROFILE_ACADEMIC
        state.add_source(
            title="Cited Paper",
            url="https://doi.org/10.1234/cited",
            metadata={"authors": "Alice Smith", "year": 2023, "venue": "Nature", "doi": "10.1234/cited"},
        )
        state.add_source(
            title="Uncited Paper",
            url="https://doi.org/10.1234/uncited",
            metadata={"authors": "Bob Jones", "year": 2022, "venue": "Science", "doi": "10.1234/uncited"},
        )

        report = "# Report\n\nKey finding from [1] confirms the hypothesis.\n"
        processed, meta = postprocess_citations(
            report, state, query_type="literature_review",
        )

        # APA heading used
        assert "## References" in processed
        assert meta["format_style"] == "apa"
        # Cited source appears in APA format
        assert "Alice Smith (2023). Cited Paper." in processed
        assert "*Nature*." in processed
        # Uncited source does NOT appear
        assert "Bob Jones" not in processed
        assert "Uncited Paper" not in processed

    def test_provenance_still_has_all_sources(self):
        """state.sources retains all sources regardless of bibliography filtering."""
        state = DeepResearchState(original_query="test query")
        state.add_source(title="Cited A", url="https://a.example.com")
        state.add_source(title="Uncited B", url="https://b.example.com")
        state.add_source(title="Cited C", url="https://c.example.com")

        report = "# Report\n\nEvidence [1] and [3].\n"
        processed, meta = postprocess_citations(report, state)

        # Bibliography only has cited sources (renumbered: [1]→[1], [3]→[2])
        assert "Uncited B" not in processed.split("## Sources")[1]
        # But state.sources still contains ALL sources (for provenance/export)
        assert len(state.sources) == 3
        assert state.sources[0].title == "Cited A"
        assert state.sources[1].title == "Uncited B"
        assert state.sources[2].title == "Cited C"
        # Citation map has renumbered keys: Cited A=1, Cited C=2, Uncited B=3
        citation_map = state.get_citation_map()
        assert len(citation_map) == 3
        assert citation_map[1].title == "Cited A"
        assert citation_map[2].title == "Cited C"
        assert citation_map[3].title == "Uncited B"


# =============================================================================
# renumber_citations (unit tests)
# =============================================================================


class TestRenumberCitations:
    """Tests for renumber_citations() — reading-order citation renumbering."""

    def test_out_of_order_citations(self):
        """Out-of-order citations [5] [2] [5] → [1] [2] [1]."""
        state = DeepResearchState(original_query="test")
        state.add_source(title="S1", url="https://1.example.com")
        state.add_source(title="S2", url="https://2.example.com")
        state.add_source(title="S3", url="https://3.example.com")
        state.add_source(title="S4", url="https://4.example.com")
        state.add_source(title="S5", url="https://5.example.com")

        report = "Finding [5] foo [2] bar [5] baz."
        result, rmap = renumber_citations(report, state, max_citation=999)

        assert result == "Finding [1] foo [2] bar [1] baz."
        assert rmap == {5: 1, 2: 2}

    def test_gaps_eliminated(self):
        """Gaps in citation numbers [1], [3], [7] → [1], [2], [3]."""
        state = DeepResearchState(original_query="test")
        for i in range(7):
            state.add_source(title=f"S{i+1}", url=f"https://{i+1}.example.com")

        report = "Sources [1], [3], and [7] support the claim."
        result, rmap = renumber_citations(report, state, max_citation=999)

        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result
        assert "[7]" not in result
        assert rmap == {1: 1, 3: 2, 7: 3}

    def test_state_sources_updated(self):
        """Source citation_number values are updated after renumbering."""
        state = DeepResearchState(original_query="test")
        state.add_source(title="S1")  # cn=1
        state.add_source(title="S2")  # cn=2
        state.add_source(title="S3")  # cn=3
        state.add_source(title="S4")  # cn=4
        state.add_source(title="S5")  # cn=5

        report = "See [5] and [3]."
        renumber_citations(report, state, max_citation=999)

        # S5 (originally cn=5) → cn=1, S3 (originally cn=3) → cn=2
        cn_map = {s.title: s.citation_number for s in state.sources}
        assert cn_map["S5"] == 1
        assert cn_map["S3"] == 2

    def test_next_citation_number_updated(self):
        """state.next_citation_number accounts for all sources (cited + uncited)."""
        state = DeepResearchState(original_query="test")
        for i in range(5):
            state.add_source(title=f"S{i+1}")
        assert state.next_citation_number == 6

        report = "Cite [5] then [3]."
        renumber_citations(report, state, max_citation=999)

        # 2 cited (cn 1,2) + 3 uncited (cn 3,4,5) → next is 6
        assert state.next_citation_number == 6

    def test_year_references_preserved(self):
        """Year references like [2025] are not renumbered."""
        state = DeepResearchState(original_query="test")
        state.add_source(title="S1")
        state.add_source(title="S2")

        report = "Finding [2] from [2025] and [1]."
        result, rmap = renumber_citations(report, state, max_citation=999)

        # [2] appears first → becomes [1], [1] appears second → becomes [2]
        assert "[2025]" in result
        assert rmap == {2: 1, 1: 2}

    def test_markdown_links_not_affected(self):
        """Markdown links [text](url) are not treated as citations."""
        state = DeepResearchState(original_query="test")
        state.add_source(title="S1")
        state.add_source(title="S2")
        state.add_source(title="S3")

        report = "See [Example](https://example.com) and [3] then [1]."
        result, rmap = renumber_citations(report, state, max_citation=999)

        assert "[Example](https://example.com)" in result
        assert rmap == {3: 1, 1: 2}

    def test_already_ordered_is_noop(self):
        """An already-ordered report returns empty map (no-op)."""
        state = DeepResearchState(original_query="test")
        state.add_source(title="S1")
        state.add_source(title="S2")
        state.add_source(title="S3")

        report = "Sources [1], [2], and [3]."
        result, rmap = renumber_citations(report, state, max_citation=999)

        assert result == report
        assert rmap == {}

    def test_no_citations_is_noop(self):
        """Report with no citations returns empty map."""
        state = DeepResearchState(original_query="test")
        state.add_source(title="S1")

        report = "No citations here."
        result, rmap = renumber_citations(report, state)

        assert result == report
        assert rmap == {}


# =============================================================================
# postprocess_citations with renumbering (integration)
# =============================================================================


class TestPostprocessCitationsRenumbering:
    """Integration tests for renumbering within the full pipeline."""

    def test_bibliography_uses_renumbered_order(self):
        """After renumbering, bibliography entries use new citation numbers."""
        state = DeepResearchState(original_query="test")
        state.add_source(title="Alpha", url="https://alpha.example.com")
        state.add_source(title="Beta", url="https://beta.example.com")
        state.add_source(title="Gamma", url="https://gamma.example.com")

        # Report cites [3] first, then [1] — should renumber to [1], [2]
        report = "# Report\n\nFirst [3] then [1]."
        processed, meta = postprocess_citations(report, state)

        body = processed.split("## Sources")[0]
        assert "[1]" in body
        assert "[2]" in body
        assert "[3]" not in body

        # Bibliography should list [1] Gamma, [2] Alpha (renumbered order)
        assert "[1] [Gamma](https://gamma.example.com)" in processed
        assert "[2] [Alpha](https://alpha.example.com)" in processed
        assert meta["renumbered_count"] == 2

    def test_renumbering_metadata_in_response(self):
        """Renumber count appears in metadata."""
        state = DeepResearchState(original_query="test")
        state.add_source(title="S1", url="https://1.example.com")
        state.add_source(title="S2", url="https://2.example.com")

        report = "Finding [2] then [1]."
        _, meta = postprocess_citations(report, state)

        assert "renumbered_count" in meta
        assert meta["renumbered_count"] == 2

    def test_no_renumbering_when_already_ordered(self):
        """Already-ordered citations produce renumbered_count=0."""
        state = DeepResearchState(original_query="test")
        state.add_source(title="S1", url="https://1.example.com")
        state.add_source(title="S2", url="https://2.example.com")

        report = "Finding [1] then [2]."
        _, meta = postprocess_citations(report, state)

        assert meta["renumbered_count"] == 0

    def test_year_refs_survive_full_pipeline(self):
        """Year references survive through renumbering in the full pipeline."""
        state = DeepResearchState(original_query="test")
        state.add_source(title="S1", url="https://1.example.com")

        report = "In [2025], finding [1] was published."
        processed, meta = postprocess_citations(report, state)

        body = processed.split("## Sources")[0]
        assert "[2025]" in body
        assert "[1]" in body
        assert meta["renumbered_count"] == 0


# =============================================================================
# cleanup_citations (Phase 5.1)
# =============================================================================


class TestCleanupCitations:
    """Tests for cleanup_citations — strips LLM sources, removes dangling, does NOT renumber."""

    def test_strips_llm_sources_section(self, state_with_sources: DeepResearchState):
        report = "# Report\n\nFinding [1].\n\n## Sources\n\n- LLM generated\n"
        cleaned, meta = cleanup_citations(report, state_with_sources)
        assert "## Sources" not in cleaned
        assert "LLM generated" not in cleaned
        assert "[1]" in cleaned

    def test_removes_dangling_citations(self, state_with_sources: DeepResearchState):
        report = "Finding [1] and [99] are mentioned."
        cleaned, meta = cleanup_citations(report, state_with_sources)
        assert "[1]" in cleaned
        assert "[99]" not in cleaned
        assert meta["dangling_citations_removed"] == 1

    def test_does_not_renumber(self, state_with_sources: DeepResearchState):
        """cleanup_citations must NOT renumber — citations stay as-is."""
        report = "First [3] then [1]."
        cleaned, meta = cleanup_citations(report, state_with_sources)
        assert "[3]" in cleaned
        assert "[1]" in cleaned
        # No renumber keys in metadata
        assert "renumbered_count" not in meta

    def test_does_not_append_bibliography(self, state_with_sources: DeepResearchState):
        """cleanup_citations must NOT append a Sources/References section."""
        report = "Finding [1] and [2]."
        cleaned, meta = cleanup_citations(report, state_with_sources)
        assert "## Sources" not in cleaned
        assert "## References" not in cleaned

    def test_metadata_keys(self, state_with_sources: DeepResearchState):
        report = "Finding [1] and [2]."
        _, meta = cleanup_citations(report, state_with_sources)
        assert "total_citations_in_report" in meta
        assert "total_sources_with_numbers" in meta
        assert "dangling_citations_removed" in meta
        assert meta["total_citations_in_report"] == 2
        assert meta["total_sources_with_numbers"] == 3
        assert meta["dangling_citations_removed"] == 0

    def test_year_references_preserved(self, state_with_sources: DeepResearchState):
        report = "Finding [1] from [2025]."
        cleaned, _ = cleanup_citations(report, state_with_sources)
        assert "[2025]" in cleaned
        assert "[1]" in cleaned


# =============================================================================
# finalize_citations (Phase 5.2)
# =============================================================================


class TestFinalizeCitations:
    """Tests for finalize_citations — renumbers to reading order, appends bibliography."""

    def test_renumbers_to_reading_order(self):
        state = DeepResearchState(original_query="test")
        state.add_source(title="Alpha", url="https://alpha.example.com")
        state.add_source(title="Beta", url="https://beta.example.com")
        state.add_source(title="Gamma", url="https://gamma.example.com")

        report = "First [3] then [1]."
        finalized, meta = finalize_citations(report, state)

        body = finalized.split("## Sources")[0]
        assert "First [1] then [2]." in body
        assert meta["renumbered_count"] == 2

    def test_appends_bibliography(self):
        state = DeepResearchState(original_query="test")
        state.add_source(title="Alpha", url="https://alpha.example.com")
        state.add_source(title="Beta", url="https://beta.example.com")

        report = "Finding [1] and [2]."
        finalized, meta = finalize_citations(report, state)

        assert "## Sources" in finalized
        assert "[1] [Alpha](https://alpha.example.com)" in finalized
        assert "[2] [Beta](https://beta.example.com)" in finalized

    def test_bibliography_only_cited_sources(self):
        state = DeepResearchState(original_query="test")
        state.add_source(title="Alpha", url="https://alpha.example.com")
        state.add_source(title="Beta", url="https://beta.example.com")
        state.add_source(title="Gamma", url="https://gamma.example.com")

        report = "Finding [1]."
        finalized, meta = finalize_citations(report, state)

        assert "[1] [Alpha](https://alpha.example.com)" in finalized
        assert "[2] [Beta]" not in finalized
        assert "[3] [Gamma]" not in finalized
        assert meta["unreferenced_sources"] == 2

    def test_metadata_keys(self):
        state = DeepResearchState(original_query="test")
        state.add_source(title="S1", url="https://1.example.com")
        report = "Finding [1]."
        _, meta = finalize_citations(report, state)

        assert "renumbered_count" in meta
        assert "unreferenced_sources" in meta
        assert "format_style" in meta
        assert "total_citations_in_report" in meta

    def test_apa_format_for_literature_review(self):
        state = DeepResearchState(original_query="test")
        state.add_source(title="Alpha", url="https://alpha.example.com")
        report = "Finding [1]."
        finalized, meta = finalize_citations(report, state, query_type="literature_review")

        assert "## References" in finalized
        assert meta["format_style"] == "apa"

    def test_year_references_preserved(self):
        state = DeepResearchState(original_query="test")
        state.add_source(title="S1", url="https://1.example.com")

        report = "In [2025], finding [1] was published."
        finalized, _ = finalize_citations(report, state)

        body = finalized.split("## Sources")[0]
        assert "[2025]" in body
        assert "[1]" in body


# =============================================================================
# needs_renumber (Phase 5.3)
# =============================================================================


class TestNeedsRenumber:
    """Tests for needs_renumber helper."""

    def test_out_of_order_returns_true(self):
        assert needs_renumber("Finding [3] then [1].") is True

    def test_gaps_returns_true(self):
        assert needs_renumber("Finding [1] and [3].") is True

    def test_sequential_returns_false(self):
        assert needs_renumber("Finding [1] then [2].") is False

    def test_single_citation_1_returns_false(self):
        assert needs_renumber("Finding [1].") is False

    def test_single_citation_not_1_returns_true(self):
        assert needs_renumber("Finding [5].") is True

    def test_no_citations_returns_false(self):
        assert needs_renumber("No citations here.") is False

    def test_empty_report_returns_false(self):
        assert needs_renumber("") is False

    def test_year_references_ignored_with_max_citation(self):
        # [2025] should be ignored; [1] [2] are sequential
        assert needs_renumber("Finding [1] from [2025] and [2].", max_citation=999) is False

    def test_markdown_links_not_counted(self):
        # [text](url) should not be counted as a citation
        assert needs_renumber("[Example](https://example.com) and [1] then [2].") is False


# =============================================================================
# Integration: cleanup → mutations → finalize (Phase 5.4)
# =============================================================================


class TestCleanupMutateFinalize:
    """Integration test: cleanup → simulate claim verification mutations → finalize.

    Verifies the full pipeline produces sequential [1,2,3,...] output even after
    claim verification removes and remaps citations.
    """

    def test_full_pipeline_with_mutations(self):
        """Simulate: cleanup → remove [2], remap [5]→[3] → finalize → sequential."""
        state = DeepResearchState(original_query="test")
        for i in range(5):
            state.add_source(title=f"Source {i+1}", url=f"https://s{i+1}.example.com")

        report = (
            "# Report\n\n"
            "Finding [1] supports claim A. "
            "Finding [2] supports claim B. "
            "Finding [3] is related. "
            "Finding [4] contradicts. "
            "Finding [5] confirms [1].\n\n"
            "## Sources\n\n"
            "- LLM generated sources\n"
        )

        # Step 1: cleanup (strips LLM sources, removes dangling)
        cleaned, cleanup_meta = cleanup_citations(report, state)
        assert "## Sources" not in cleaned  # LLM section stripped
        assert cleanup_meta["dangling_citations_removed"] == 0

        # Step 2: simulate claim verification mutations
        # Remove [2] (UNSUPPORTED claim)
        cleaned = cleaned.replace("[2]", "")
        # Remap [5] → [3] (verified with different source)
        cleaned = cleaned.replace("[5]", "[3]")

        # At this point, citations are: [1], [3], [4], [3], [1] — out of order with gaps
        assert needs_renumber(cleaned, max_citation=999) is True

        # Step 3: finalize (renumber + bibliography)
        finalized, finalize_meta = finalize_citations(cleaned, state)

        body = finalized.split("## Sources")[0]

        # Should be renumbered to sequential: [1], [2], [3], [2], [1]
        # First-appearance: [1]→1, [3]→2, [4]→3
        assert "[1]" in body
        assert "[2]" in body
        assert "[3]" in body
        # No gaps — [4] or [5] shouldn't appear in body after renumbering
        assert "[4]" not in body
        assert "[5]" not in body

        # Bibliography should only have cited sources
        assert "## Sources" in finalized
        assert finalize_meta["renumbered_count"] > 0

    def test_pipeline_no_mutations(self):
        """When claim verification makes no changes, finalize still renumbers."""
        state = DeepResearchState(original_query="test")
        state.add_source(title="S1", url="https://1.example.com")
        state.add_source(title="S2", url="https://2.example.com")
        state.add_source(title="S3", url="https://3.example.com")

        report = "Finding [3] then [1]."
        cleaned, _ = cleanup_citations(report, state)

        # No mutations — directly finalize
        finalized, meta = finalize_citations(cleaned, state)

        body = finalized.split("## Sources")[0]
        assert "Finding [1] then [2]." in body
        assert meta["renumbered_count"] == 2


# =============================================================================
# Backward compatibility: postprocess_citations (Phase 5.5)
# =============================================================================


class TestPostprocessCitationsBackwardCompat:
    """Verify postprocess_citations produces identical output to cleanup+finalize."""

    def test_combined_matches_split(self):
        """postprocess_citations output must equal cleanup→finalize sequence."""
        state1 = DeepResearchState(original_query="test")
        state2 = DeepResearchState(original_query="test")
        for i in range(5):
            state1.add_source(title=f"S{i+1}", url=f"https://s{i+1}.example.com")
            state2.add_source(title=f"S{i+1}", url=f"https://s{i+1}.example.com")

        report = (
            "# Report\n\n"
            "First [3] and [1] then [5].\n\n"
            "## Sources\n\n- LLM generated\n"
        )

        # Combined path
        combined_report, combined_meta = postprocess_citations(report, state1)

        # Split path
        cleaned, _ = cleanup_citations(report, state2)
        split_report, _ = finalize_citations(cleaned, state2)

        assert combined_report == split_report

    def test_existing_tests_still_valid(self, state_with_sources: DeepResearchState):
        """Smoke test: postprocess_citations still works as before."""
        report = "# Report\n\nFinding [1] and [2].\n\n## Sources\n\n- Old\n"
        processed, meta = postprocess_citations(report, state_with_sources)

        assert "Old" not in processed
        assert "## Sources" in processed
        assert "[1] [Alpha Source](https://alpha.example.com)" in processed
        assert meta["total_citations_in_report"] == 2
        assert meta["dangling_citations_removed"] == 0
