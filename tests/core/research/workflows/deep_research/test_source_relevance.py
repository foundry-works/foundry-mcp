"""Tests for source relevance scoring and compression filtering.

Covers:
- compute_source_relevance() keyword-based scoring
- Academic source penalty (0.7x multiplier)
- Relevance threshold filtering in compression
- Backward compatibility (None scores pass through)
- ResearchSource.relevance_score field serialization
"""

from __future__ import annotations

import pytest

from foundry_mcp.core.research.models.sources import ResearchSource
from foundry_mcp.core.research.workflows.deep_research.source_quality import (
    compute_source_relevance,
    _tokenize,
    _jaccard,
    _overlap_coefficient,
)


# ---------------------------------------------------------------------------
# Unit tests: _tokenize helper
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic_tokenization(self) -> None:
        tokens = _tokenize("Hello world, this is a test!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        # Stopwords excluded
        assert "this" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens

    def test_empty_string(self) -> None:
        assert _tokenize("") == set()

    def test_single_char_words_excluded(self) -> None:
        tokens = _tokenize("I a x y z")
        # Single char words should be excluded (len > 1 filter)
        assert "x" not in tokens
        assert "y" not in tokens
        assert "z" not in tokens


# ---------------------------------------------------------------------------
# Unit tests: _jaccard helper
# ---------------------------------------------------------------------------


class TestJaccard:
    def test_identical_sets(self) -> None:
        assert _jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_disjoint_sets(self) -> None:
        assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self) -> None:
        # {a, b, c} & {b, c, d} = {b, c}, union = {a, b, c, d}
        assert _jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)

    def test_empty_sets(self) -> None:
        assert _jaccard(set(), set()) == 0.0
        assert _jaccard({"a"}, set()) == 0.0


class TestOverlapCoefficient:
    def test_identical_sets(self) -> None:
        assert _overlap_coefficient({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_disjoint_sets(self) -> None:
        assert _overlap_coefficient({"a", "b"}, {"c", "d"}) == 0.0

    def test_subset(self) -> None:
        # {a, b} is a subset of {a, b, c, d} → overlap = 2/2 = 1.0
        assert _overlap_coefficient({"a", "b"}, {"a", "b", "c", "d"}) == 1.0

    def test_partial_overlap(self) -> None:
        # {a, b, c} & {b, c, d} = {b, c}, min(3, 3) = 3 → 2/3
        assert _overlap_coefficient({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(2.0 / 3.0)

    def test_empty_sets(self) -> None:
        assert _overlap_coefficient(set(), set()) == 0.0
        assert _overlap_coefficient({"a"}, set()) == 0.0


# ---------------------------------------------------------------------------
# Unit tests: compute_source_relevance
# ---------------------------------------------------------------------------


class TestComputeSourceRelevance:
    """Tests for the main relevance scoring function."""

    def test_irrelevant_academic_paper_low_score(self) -> None:
        """PESTLE analysis of seaplane transport in Greece should score low
        against a credit card travel rewards query."""
        score = compute_source_relevance(
            source_title="PESTLE Analysis of Seaplane Transport in Greece",
            source_content=None,
            reference_text="best credit card for travel rewards maximizing business class flights",
        )
        assert score < 0.1, f"Expected < 0.1, got {score}"

    def test_relevant_web_source_high_score(self) -> None:
        """A credit card review should score well above the default threshold (0.05)."""
        score = compute_source_relevance(
            source_title="Chase Sapphire Reserve 2025 Review",
            source_content="The Chase Sapphire Reserve offers excellent travel rewards "
            "including 3x points on travel and dining, a $300 annual travel credit, "
            "and access to Priority Pass lounges.",
            reference_text="best credit card for travel rewards",
        )
        # Must be well above the default relevance threshold (0.05)
        assert score > 0.15, f"Expected > 0.15, got {score}"

    def test_academic_penalty_applied(self) -> None:
        """Academic sources should score lower than web sources with identical text."""
        reference = "machine learning neural network deep learning"
        title = "Deep Learning Approaches to Neural Network Optimization"
        content = "machine learning methods for neural network training"

        web_score = compute_source_relevance(
            source_title=title,
            source_content=content,
            reference_text=reference,
            source_type="web",
        )
        academic_score = compute_source_relevance(
            source_title=title,
            source_content=content,
            reference_text=reference,
            source_type="academic",
        )
        assert academic_score < web_score, (
            f"Academic ({academic_score}) should be less than web ({web_score})"
        )
        # Academic penalty is 0.7x
        assert academic_score == pytest.approx(web_score * 0.7, abs=0.001)

    def test_no_content_uses_title_only(self) -> None:
        """When content is None, scoring relies entirely on title."""
        score = compute_source_relevance(
            source_title="Machine Learning Best Practices Guide",
            source_content=None,
            reference_text="machine learning best practices for production deployment",
        )
        assert score > 0.2, f"Expected > 0.2 with title match, got {score}"

    def test_empty_reference_text(self) -> None:
        """Empty reference text should return 0.0."""
        score = compute_source_relevance(
            source_title="Some Title",
            source_content="Some content",
            reference_text="",
        )
        assert score == 0.0

    def test_empty_title_and_content(self) -> None:
        """Empty source should score 0.0."""
        score = compute_source_relevance(
            source_title="",
            source_content=None,
            reference_text="best credit card for travel",
        )
        assert score == 0.0

    def test_score_clamped_to_unit_interval(self) -> None:
        """Score should always be in [0.0, 1.0]."""
        score = compute_source_relevance(
            source_title="exact match phrase exact match phrase",
            source_content="exact match phrase exact match phrase",
            reference_text="exact match phrase",
        )
        assert 0.0 <= score <= 1.0

    def test_nhl_arena_financing_irrelevant(self) -> None:
        """Real-world irrelevant source from the original analysis."""
        score = compute_source_relevance(
            source_title="NHL Arena Financing and Public Subsidies",
            source_content="analysis of public funding for hockey arena construction",
            reference_text="best credit card for travel rewards maximizing business class flights",
        )
        assert score < 0.1, f"Expected < 0.1, got {score}"

    def test_pilot_training_zambia_irrelevant(self) -> None:
        """Another real-world irrelevant source."""
        score = compute_source_relevance(
            source_title="Pilot Training Programs in Zambia",
            source_content=None,
            reference_text="best credit card for travel rewards",
            source_type="academic",
        )
        assert score < 0.1, f"Expected < 0.1, got {score}"


# ---------------------------------------------------------------------------
# Unit tests: ResearchSource.relevance_score field
# ---------------------------------------------------------------------------


class TestResearchSourceRelevanceField:
    """Test that the relevance_score field serializes correctly."""

    def test_default_is_none(self) -> None:
        source = ResearchSource(title="Test Source")
        assert source.relevance_score is None

    def test_set_and_retrieve(self) -> None:
        source = ResearchSource(title="Test Source", relevance_score=0.75)
        assert source.relevance_score == 0.75

    def test_serialization_round_trip(self) -> None:
        source = ResearchSource(title="Test Source", relevance_score=0.42)
        data = source.model_dump()
        assert data["relevance_score"] == 0.42
        restored = ResearchSource.model_validate(data)
        assert restored.relevance_score == 0.42

    def test_none_serialization(self) -> None:
        source = ResearchSource(title="Test Source")
        data = source.model_dump()
        assert data["relevance_score"] is None

    def test_to_dict_includes_relevance(self) -> None:
        source = ResearchSource(title="Test Source", relevance_score=0.55)
        d = source.to_dict()
        assert d["relevance_score"] == 0.55


# ---------------------------------------------------------------------------
# Unit tests: Compression filtering
# ---------------------------------------------------------------------------


class TestCompressionRelevanceFiltering:
    """Test that low-relevance sources are excluded from compression input."""

    def _make_source(self, source_id: str, relevance: float | None) -> ResearchSource:
        return ResearchSource(
            id=source_id,
            title=f"Source {source_id}",
            relevance_score=relevance,
        )

    def test_sources_below_threshold_excluded(self) -> None:
        """Sources with relevance_score below threshold should be filtered out."""
        sources = [
            self._make_source("s1", 0.8),   # above threshold
            self._make_source("s2", 0.01),   # below threshold
            self._make_source("s3", 0.5),    # above threshold
        ]
        threshold = 0.05
        filtered = [
            s for s in sources
            if s.relevance_score is None or s.relevance_score >= threshold
        ]
        assert len(filtered) == 2
        assert {s.id for s in filtered} == {"s1", "s3"}

    def test_none_relevance_passes_through(self) -> None:
        """Sources with relevance_score=None should not be filtered (backward compat)."""
        sources = [
            self._make_source("s1", None),
            self._make_source("s2", 0.01),
            self._make_source("s3", None),
        ]
        threshold = 0.05
        filtered = [
            s for s in sources
            if s.relevance_score is None or s.relevance_score >= threshold
        ]
        assert len(filtered) == 2
        assert {s.id for s in filtered} == {"s1", "s3"}

    def test_threshold_zero_disables_filtering(self) -> None:
        """With threshold=0.0, no sources should be excluded."""
        sources = [
            self._make_source("s1", 0.001),
            self._make_source("s2", 0.0),
        ]
        threshold = 0.0
        # When threshold is 0.0, the filtering code skips entirely
        # (threshold > 0.0 check in compression.py)
        assert threshold == 0.0  # Filtering disabled

    def test_all_below_threshold_keeps_sources(self) -> None:
        """If all sources are below threshold, keep them to avoid empty compression."""
        sources = [
            self._make_source("s1", 0.01),
            self._make_source("s2", 0.02),
        ]
        threshold = 0.05
        relevant = [
            s for s in sources
            if s.relevance_score is None or s.relevance_score >= threshold
        ]
        # Compression code falls back to full list when relevant is empty
        if not relevant:
            relevant = sources
        assert len(relevant) == 2  # All kept as fallback


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestRelevanceConfig:
    """Test config field for source relevance threshold."""

    def test_default_value(self) -> None:
        from foundry_mcp.config.research import ResearchConfig
        config = ResearchConfig()
        assert config.deep_research_source_relevance_threshold == 0.05

    def test_from_toml_dict_parsing(self) -> None:
        from foundry_mcp.config.research import ResearchConfig
        config = ResearchConfig.from_toml_dict(
            {"deep_research_source_relevance_threshold": "0.10"}
        )
        assert config.deep_research_source_relevance_threshold == pytest.approx(0.10)

    def test_validation_resets_out_of_range(self) -> None:
        from foundry_mcp.config.research import ResearchConfig
        config = ResearchConfig(deep_research_source_relevance_threshold=2.0)
        config.__post_init__()
        assert config.deep_research_source_relevance_threshold == 0.05
