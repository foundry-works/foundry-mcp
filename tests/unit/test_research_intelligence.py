"""Tests for research intelligence features.

Covers:
- Item 1: Influence-aware source ranking
- Item 2: Research landscape metadata
- Item 3: Explicit research gaps section
- Item 4: Cross-study comparison tables (model + prompt injection)
- Item 5: BibTeX & RIS export
"""

from __future__ import annotations

import pytest

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchState,
    ResearchExtensions,
    ResearchLandscape,
    StudyComparison,
)
from foundry_mcp.core.research.models.sources import (
    ResearchGap,
    ResearchMode,
    ResearchSource,
    SourceType,
)


# =====================================================================
# Helpers — build test sources with metadata
# =====================================================================


def _make_source(
    title: str = "Test Paper",
    source_type: SourceType = SourceType.ACADEMIC,
    citation_count: int | None = None,
    influential_count: int | None = None,
    year: int | None = None,
    venue: str | None = None,
    authors: list[dict] | None = None,
    doi: str | None = None,
    fields: list[str] | None = None,
    url: str | None = None,
) -> ResearchSource:
    meta: dict = {}
    if citation_count is not None:
        meta["citation_count"] = citation_count
    if influential_count is not None:
        meta["influential_citation_count"] = influential_count
    if year is not None:
        meta["year"] = year
    if venue is not None:
        meta["venue"] = venue
    if authors is not None:
        meta["authors"] = authors
    if doi is not None:
        meta["doi"] = doi
    if fields is not None:
        meta["fields_of_study"] = fields
    return ResearchSource(
        title=title,
        source_type=source_type,
        metadata=meta,
        url=url or f"https://example.com/{title.lower().replace(' ', '-')}",
    )


def _make_state(**kwargs) -> DeepResearchState:
    defaults = {"original_query": "test query"}
    defaults.update(kwargs)
    return DeepResearchState(**defaults)


# =====================================================================
# Item 1: Influence-aware source ranking
# =====================================================================


class TestSourceInfluence:
    """Tests for compute_source_influence and academic coverage weights."""

    def test_general_mode_returns_neutral(self):
        from foundry_mcp.core.research.workflows.deep_research.phases.supervision_coverage import (
            compute_source_influence,
        )

        state = _make_state(research_mode=ResearchMode.GENERAL)
        state.sources = [_make_source(citation_count=500)]

        class FakeConfig:
            pass

        score = compute_source_influence(state, FakeConfig())
        assert score == 1.0, "General mode should return neutral 1.0"

    def test_academic_high_citation_sources(self):
        from foundry_mcp.core.research.workflows.deep_research.phases.supervision_coverage import (
            compute_source_influence,
        )

        state = _make_state(research_mode=ResearchMode.ACADEMIC)
        state.sources = [
            _make_source(citation_count=500, influential_count=10),
            _make_source(citation_count=200, influential_count=5),
        ]

        class FakeConfig:
            deep_research_influence_high_citation_threshold = 100
            deep_research_influence_medium_citation_threshold = 20
            deep_research_influence_low_citation_threshold = 5

        score = compute_source_influence(state, FakeConfig())
        assert score > 0.8, f"Two highly-cited papers should score high, got {score}"

    def test_academic_unknown_citations(self):
        from foundry_mcp.core.research.workflows.deep_research.phases.supervision_coverage import (
            compute_source_influence,
        )

        state = _make_state(research_mode=ResearchMode.ACADEMIC)
        state.sources = [
            _make_source(),  # No citation_count in metadata
            _make_source(),
        ]

        class FakeConfig:
            deep_research_influence_high_citation_threshold = 100
            deep_research_influence_medium_citation_threshold = 20
            deep_research_influence_low_citation_threshold = 5

        score = compute_source_influence(state, FakeConfig())
        assert score < 0.3, f"Unknown citations should score low, got {score}"

    def test_academic_mixed_citations(self):
        from foundry_mcp.core.research.workflows.deep_research.phases.supervision_coverage import (
            compute_source_influence,
        )

        state = _make_state(research_mode=ResearchMode.ACADEMIC)
        state.sources = [
            _make_source(citation_count=500),  # High
            _make_source(citation_count=2),  # Low
            _make_source(),  # Unknown
        ]

        class FakeConfig:
            deep_research_influence_high_citation_threshold = 100
            deep_research_influence_medium_citation_threshold = 20
            deep_research_influence_low_citation_threshold = 5

        score = compute_source_influence(state, FakeConfig())
        assert 0.3 < score < 0.8, f"Mixed citations should score moderate, got {score}"

    def test_academic_no_sources(self):
        from foundry_mcp.core.research.workflows.deep_research.phases.supervision_coverage import (
            compute_source_influence,
        )

        state = _make_state(research_mode=ResearchMode.ACADEMIC)
        state.sources = []

        class FakeConfig:
            pass

        score = compute_source_influence(state, FakeConfig())
        assert score == 0.5, "No sources in academic mode should return 0.5"

    def test_academic_coverage_weights_include_influence(self):
        from foundry_mcp.core.research.workflows.deep_research.phases.supervision_coverage import (
            assess_coverage_heuristic,
        )

        state = _make_state(research_mode=ResearchMode.ACADEMIC)
        state.sub_queries = [
            type("SQ", (), {"id": "sq-1", "status": "completed", "source_ids": ["s1"]})()
        ]
        state.sources = [
            _make_source(citation_count=500),
        ]
        state.sources[0].sub_query_id = "sq-1"
        state.sources[0].id = "s1"

        class FakeConfig:
            deep_research_coverage_confidence_weights = None
            deep_research_academic_coverage_weights = None
            deep_research_coverage_confidence_threshold = 0.75
            deep_research_influence_high_citation_threshold = 100
            deep_research_influence_medium_citation_threshold = 20
            deep_research_influence_low_citation_threshold = 5

        result = assess_coverage_heuristic(state, min_sources=1, config=FakeConfig())
        assert "source_influence" in result["confidence_dimensions"]

    def test_general_coverage_weights_exclude_influence_dimension(self):
        from foundry_mcp.core.research.workflows.deep_research.phases.supervision_coverage import (
            assess_coverage_heuristic,
        )

        state = _make_state(research_mode=ResearchMode.GENERAL)
        state.sub_queries = [
            type("SQ", (), {"id": "sq-1", "status": "completed", "source_ids": ["s1"]})()
        ]
        state.sources = [_make_source(source_type=SourceType.WEB)]
        state.sources[0].sub_query_id = "sq-1"
        state.sources[0].id = "s1"

        class FakeConfig:
            deep_research_coverage_confidence_weights = None
            deep_research_academic_coverage_weights = None
            deep_research_coverage_confidence_threshold = 0.75
            deep_research_influence_high_citation_threshold = 100
            deep_research_influence_medium_citation_threshold = 20
            deep_research_influence_low_citation_threshold = 5

        result = assess_coverage_heuristic(state, min_sources=1, config=FakeConfig())
        # source_influence is in dimensions but general weights don't weight it
        assert result["confidence_dimensions"]["source_influence"] == 1.0


# =====================================================================
# Item 2: Research landscape metadata
# =====================================================================


class TestResearchLandscape:
    """Tests for ResearchLandscape model and builder."""

    def test_empty_landscape_serializes_to_defaults(self):
        landscape = ResearchLandscape()
        data = landscape.model_dump()
        assert data["timeline"] == []
        assert data["venue_distribution"] == {}
        assert data["top_cited_papers"] == []
        assert data["study_comparisons"] == []

    def test_landscape_with_data(self):
        landscape = ResearchLandscape(
            timeline=[{"year": 2023, "count": 5, "key_papers": []}],
            venue_distribution={"Nature": 3},
            top_cited_papers=[{"title": "Test", "citation_count": 100}],
        )
        data = landscape.model_dump()
        assert len(data["timeline"]) == 1
        assert data["venue_distribution"]["Nature"] == 3

    def test_extensions_landscape_field(self):
        ext = ResearchExtensions(research_landscape=ResearchLandscape())
        assert ext.research_landscape is not None
        data = ext.model_dump()
        assert "research_landscape" in data

    def test_state_landscape_accessor(self):
        state = _make_state()
        assert state.research_landscape is None
        state.extensions.research_landscape = ResearchLandscape(
            venue_distribution={"Science": 2}
        )
        assert state.research_landscape is not None
        assert state.research_landscape.venue_distribution["Science"] == 2

    def test_extensions_default_serializes_empty(self):
        ext = ResearchExtensions()
        data = ext.model_dump()
        # exclude_none=True by default, so None fields are excluded
        assert "research_landscape" not in data
        assert "research_profile" not in data

    def test_state_with_extensions_backward_compatible(self):
        state = _make_state()
        data = state.model_dump()
        assert "extensions" in data

    def test_study_comparison_model(self):
        sc = StudyComparison(
            study_title="Test Study",
            authors="Smith et al.",
            year=2023,
            methodology="RCT",
            key_finding="Significant effect",
            source_id="src-abc",
        )
        assert sc.study_title == "Test Study"
        assert sc.year == 2023


# =====================================================================
# Item 3: Research gaps (model-level; prompt injection tested via synthesis)
# =====================================================================


class TestResearchGaps:
    """Tests for research gap handling in academic mode."""

    def test_unresolved_gaps_available(self):
        state = _make_state(research_mode=ResearchMode.ACADEMIC)
        state.gaps = [
            ResearchGap(description="Gap 1", priority=1),
            ResearchGap(description="Gap 2", priority=2, resolved=True, resolution_notes="Addressed by X"),
        ]
        unresolved = state.unresolved_gaps()
        assert len(unresolved) == 1
        assert unresolved[0].description == "Gap 1"

    def test_gap_resolution_notes(self):
        gap = ResearchGap(
            description="Missing longitudinal data",
            resolved=True,
            resolution_notes="Found 2 longitudinal studies",
        )
        assert gap.resolved is True
        assert "longitudinal" in gap.resolution_notes


# =====================================================================
# Item 5: BibTeX & RIS export
# =====================================================================


class TestBibTeXExport:
    """Tests for BibTeX generation."""

    def test_bibtex_full_metadata(self):
        from foundry_mcp.core.research.export.bibtex import sources_to_bibtex

        source = _make_source(
            title="Deep Learning for NLP",
            citation_count=500,
            year=2021,
            venue="Nature Machine Intelligence",
            authors=[{"name": "John Smith"}, {"name": "Jane Doe"}],
            doi="10.1234/test",
        )
        result = sources_to_bibtex([source])
        assert "@article{" in result
        assert "Deep Learning for NLP" in result
        assert "John Smith" in result
        assert "Jane Doe" in result
        assert "2021" in result
        assert "10.1234/test" in result

    def test_bibtex_minimal_metadata(self):
        from foundry_mcp.core.research.export.bibtex import sources_to_bibtex

        source = _make_source(
            title="Some Web Page",
            source_type=SourceType.WEB,
        )
        result = sources_to_bibtex([source])
        assert "@misc{" in result
        assert "Some Web Page" in result

    def test_bibtex_conference_type(self):
        from foundry_mcp.core.research.export.bibtex import sources_to_bibtex

        source = _make_source(
            title="A Conference Paper",
            venue="Proceedings of ACL 2023",
        )
        result = sources_to_bibtex([source])
        assert "@inproceedings{" in result
        assert "booktitle" in result

    def test_bibtex_special_character_escaping(self):
        from foundry_mcp.core.research.export.bibtex import sources_to_bibtex

        source = _make_source(title="Results & Methods: 50% improvement")
        result = sources_to_bibtex([source])
        assert r"\&" in result
        assert r"\%" in result

    def test_bibtex_citation_key_uniqueness(self):
        from foundry_mcp.core.research.export.bibtex import sources_to_bibtex

        sources = [
            _make_source(title="Same Title", year=2023, authors=[{"name": "Smith"}]),
            _make_source(title="Same Title", year=2023, authors=[{"name": "Smith"}]),
        ]
        result = sources_to_bibtex(sources)
        # Should have two distinct entries
        assert result.count("@") == 2

    def test_bibtex_empty_sources(self):
        from foundry_mcp.core.research.export.bibtex import sources_to_bibtex

        assert sources_to_bibtex([]) == ""


class TestRISExport:
    """Tests for RIS generation."""

    def test_ris_full_metadata(self):
        from foundry_mcp.core.research.export.ris import sources_to_ris

        source = _make_source(
            title="Machine Learning Survey",
            citation_count=100,
            year=2022,
            venue="ACM Computing Surveys",
            authors=[{"name": "Alice Brown"}],
            doi="10.5555/survey",
        )
        result = sources_to_ris([source])
        assert "TY  - JOUR" in result
        assert "TI  - Machine Learning Survey" in result
        assert "AU  - Alice Brown" in result
        assert "PY  - 2022" in result
        assert "JO  - ACM Computing Surveys" in result
        assert "DO  - 10.5555/survey" in result
        assert "ER  - " in result

    def test_ris_minimal_metadata(self):
        from foundry_mcp.core.research.export.ris import sources_to_ris

        source = _make_source(
            title="A Blog Post",
            source_type=SourceType.WEB,
        )
        result = sources_to_ris([source])
        assert "TY  - ELEC" in result
        assert "TI  - A Blog Post" in result
        assert "ER  - " in result

    def test_ris_conference_type(self):
        from foundry_mcp.core.research.export.ris import sources_to_ris

        source = _make_source(
            title="Workshop Paper",
            venue="Workshop on AI Safety",
        )
        result = sources_to_ris([source])
        assert "TY  - CONF" in result

    def test_ris_empty_sources(self):
        from foundry_mcp.core.research.export.ris import sources_to_ris

        assert sources_to_ris([]) == ""

    def test_ris_multiple_authors(self):
        from foundry_mcp.core.research.export.ris import sources_to_ris

        source = _make_source(
            title="Multi-author",
            authors=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
        )
        result = sources_to_ris([source])
        assert result.count("AU  - ") == 3

    def test_ris_page_range_split(self):
        """Page range '123-456' should produce separate SP and EP tags."""
        from foundry_mcp.core.research.export.ris import source_to_ris_entry

        source = _make_source(title="Page Range Paper", year=2023)
        source.metadata["pages"] = "123-456"
        result = source_to_ris_entry(source)
        assert "SP  - 123" in result
        assert "EP  - 456" in result
        # Should NOT have the old combined format
        assert "SP  - 123-456" not in result

    def test_ris_single_page(self):
        """Single page '42' should produce SP only, no EP."""
        from foundry_mcp.core.research.export.ris import source_to_ris_entry

        source = _make_source(title="Single Page Paper", year=2023)
        source.metadata["pages"] = "42"
        result = source_to_ris_entry(source)
        assert "SP  - 42" in result
        assert "EP  - " not in result

    def test_ris_no_pages(self):
        """No pages value should produce no SP/EP tags."""
        from foundry_mcp.core.research.export.ris import source_to_ris_entry

        source = _make_source(title="No Pages Paper", year=2023)
        result = source_to_ris_entry(source)
        assert "SP  - " not in result
        assert "EP  - " not in result
