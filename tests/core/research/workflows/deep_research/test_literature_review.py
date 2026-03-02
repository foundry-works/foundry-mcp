"""Tests for literature_review query type.

Validates classification patterns, profile overrides, academic bias,
structure guidance injection, and no-regression for existing types.
"""

from __future__ import annotations

from foundry_mcp.core.research.models.deep_research import (
    PROFILE_ACADEMIC,
    PROFILE_GENERAL,
    ResearchProfile,
)
from foundry_mcp.core.research.models.sources import ResearchMode
from foundry_mcp.core.research.workflows.deep_research.phases.synthesis import (
    _STRUCTURE_GUIDANCE,
    _classify_query_type,
)


# =========================================================================
# Classification: literature review patterns
# =========================================================================


class TestLiteratureReviewClassification:
    """Tests for _classify_query_type detecting literature_review queries."""

    def test_literature_review_on(self) -> None:
        assert _classify_query_type("literature review on X") == "literature_review"

    def test_systematic_review(self) -> None:
        assert _classify_query_type("systematic review of cognitive behavioral therapy") == "literature_review"

    def test_meta_analysis(self) -> None:
        assert _classify_query_type("meta-analysis of remote work productivity") == "literature_review"

    def test_meta_analysis_no_hyphen(self) -> None:
        assert _classify_query_type("meta analysis of sleep interventions") == "literature_review"

    def test_survey_of(self) -> None:
        assert _classify_query_type("survey of prior work on neural scaling laws") == "literature_review"

    def test_state_of_the_art(self) -> None:
        assert _classify_query_type("state of the art in natural language processing") == "literature_review"

    def test_body_of_research(self) -> None:
        assert _classify_query_type("body of research on climate change mitigation") == "literature_review"

    def test_body_of_literature(self) -> None:
        assert _classify_query_type("body of literature on mindfulness") == "literature_review"

    def test_body_of_work(self) -> None:
        assert _classify_query_type("body of work on distributed systems") == "literature_review"

    def test_prior_work(self) -> None:
        assert _classify_query_type("prior work on attention mechanisms") == "literature_review"

    def test_prior_research(self) -> None:
        assert _classify_query_type("prior research on vaccine hesitancy") == "literature_review"

    def test_prior_studies(self) -> None:
        assert _classify_query_type("prior studies on sleep deprivation") == "literature_review"

    def test_existing_research(self) -> None:
        assert _classify_query_type("existing research on gene therapy") == "literature_review"

    def test_existing_literature(self) -> None:
        assert _classify_query_type("existing literature on microplastics") == "literature_review"

    def test_existing_studies(self) -> None:
        assert _classify_query_type("existing studies on screen time effects") == "literature_review"

    def test_review_of_the_literature(self) -> None:
        assert _classify_query_type("review of the literature on burnout") == "literature_review"

    def test_review_of_the_research(self) -> None:
        assert _classify_query_type("review of the research on intermittent fasting") == "literature_review"

    def test_what_does_the_research_say(self) -> None:
        assert _classify_query_type("what does the research say about meditation") == "literature_review"

    def test_what_does_the_research_show(self) -> None:
        assert _classify_query_type("what does the research show about bilingualism") == "literature_review"

    def test_what_does_the_evidence_suggest(self) -> None:
        assert _classify_query_type("what does the evidence suggest about probiotics") == "literature_review"

    def test_research_landscape(self) -> None:
        assert _classify_query_type("research landscape of quantum computing") == "literature_review"

    def test_scoping_review(self) -> None:
        assert _classify_query_type("scoping review of AI in education") == "literature_review"

    def test_case_insensitive(self) -> None:
        assert _classify_query_type("LITERATURE REVIEW on deep learning") == "literature_review"

    def test_mixed_case(self) -> None:
        assert _classify_query_type("Meta-Analysis of exercise interventions") == "literature_review"


# =========================================================================
# Profile override: synthesis_template forces type
# =========================================================================


class TestProfileSynthesisTemplateOverride:
    """Profile synthesis_template takes highest priority."""

    def test_profile_forces_literature_review(self) -> None:
        """synthesis_template='literature_review' overrides auto-detection."""
        profile = ResearchProfile(
            name="custom",
            synthesis_template="literature_review",
        )
        # This query would normally be classified as "comparison"
        result = _classify_query_type("compare X vs Y", profile=profile)
        assert result == "literature_review"

    def test_profile_forces_arbitrary_template(self) -> None:
        """synthesis_template can be any string, not just built-in types."""
        profile = ResearchProfile(
            name="custom",
            synthesis_template="comparison",
        )
        result = _classify_query_type("what does the research say about X", profile=profile)
        assert result == "comparison"

    def test_profile_none_template_uses_auto_detection(self) -> None:
        """synthesis_template=None falls through to auto-detection."""
        profile = ResearchProfile(name="general", synthesis_template=None)
        result = _classify_query_type("literature review on X", profile=profile)
        assert result == "literature_review"


# =========================================================================
# Academic bias: ambiguous queries default to literature_review
# =========================================================================


class TestAcademicBias:
    """Academic profiles bias ambiguous queries toward literature_review."""

    def test_ambiguous_query_academic_profile(self) -> None:
        """Ambiguous query with academic profile → literature_review."""
        result = _classify_query_type(
            "effects of caffeine on cognitive performance",
            profile=PROFILE_ACADEMIC,
        )
        assert result == "literature_review"

    def test_ambiguous_query_general_profile(self) -> None:
        """Ambiguous query with general profile → explanation (no bias)."""
        result = _classify_query_type(
            "effects of caffeine on cognitive performance",
            profile=PROFILE_GENERAL,
        )
        assert result == "explanation"

    def test_ambiguous_query_no_profile(self) -> None:
        """Ambiguous query with no profile → explanation (default)."""
        result = _classify_query_type("effects of caffeine on cognitive performance")
        assert result == "explanation"

    def test_explicit_pattern_beats_academic_bias(self) -> None:
        """Comparison pattern still wins even with academic profile."""
        result = _classify_query_type(
            "compare X vs Y",
            profile=PROFILE_ACADEMIC,
        )
        assert result == "comparison"

    def test_howto_not_overridden_by_academic(self) -> None:
        """How-to pattern still wins with academic profile."""
        result = _classify_query_type(
            "how to implement gradient descent",
            profile=PROFILE_ACADEMIC,
        )
        assert result == "howto"

    def test_enumeration_not_overridden_by_academic(self) -> None:
        """Enumeration pattern still wins with academic profile."""
        result = _classify_query_type(
            "top 5 machine learning algorithms",
            profile=PROFILE_ACADEMIC,
        )
        assert result == "enumeration"

    def test_custom_academic_profile(self) -> None:
        """Custom profile with ACADEMIC source_quality_mode triggers bias."""
        profile = ResearchProfile(
            name="custom-academic",
            source_quality_mode=ResearchMode.ACADEMIC,
        )
        result = _classify_query_type("quantum computing applications", profile=profile)
        assert result == "literature_review"


# =========================================================================
# No-regression: existing types still classify correctly
# =========================================================================


class TestNoRegression:
    """Existing query type classifications are unchanged."""

    def test_comparison_still_works(self) -> None:
        assert _classify_query_type("React vs Vue for web development") == "comparison"

    def test_comparison_versus(self) -> None:
        assert _classify_query_type("Python versus JavaScript") == "comparison"

    def test_enumeration_still_works(self) -> None:
        assert _classify_query_type("List the best Python frameworks") == "enumeration"

    def test_enumeration_top_n(self) -> None:
        assert _classify_query_type("Top 5 machine learning algorithms") == "enumeration"

    def test_howto_still_works(self) -> None:
        assert _classify_query_type("How to deploy a FastAPI application") == "howto"

    def test_explanation_still_works(self) -> None:
        assert _classify_query_type("Explain quantum computing") == "explanation"

    def test_empty_query(self) -> None:
        assert _classify_query_type("") == "explanation"

    def test_generic_overview(self) -> None:
        assert _classify_query_type("Overview of transformer architectures") == "explanation"


# =========================================================================
# Structure guidance: literature_review entry exists and is well-formed
# =========================================================================


class TestStructureGuidance:
    """Verify _STRUCTURE_GUIDANCE has a literature_review entry."""

    def test_literature_review_key_exists(self) -> None:
        assert "literature_review" in _STRUCTURE_GUIDANCE

    def test_contains_expected_sections(self) -> None:
        guidance = _STRUCTURE_GUIDANCE["literature_review"]
        expected_sections = [
            "Executive Summary",
            "Introduction & Scope",
            "Theoretical Foundations",
            "Thematic Analysis",
            "Methodological Approaches",
            "Key Debates & Contradictions",
            "Research Gaps & Future Directions",
            "Conclusions",
        ]
        for section in expected_sections:
            assert section in guidance, f"Missing section: {section}"

    def test_existing_types_unchanged(self) -> None:
        """Other structure guidance entries still exist."""
        for key in ("comparison", "enumeration", "howto", "explanation"):
            assert key in _STRUCTURE_GUIDANCE
