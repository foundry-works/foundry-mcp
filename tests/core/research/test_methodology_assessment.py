"""Tests for PLAN-4 Item 3: Methodology Quality Assessment.

Tests cover:
1. StudyDesign enum and MethodologyAssessment model serialization
2. LLM extraction prompt parsing with mocked responses
3. StudyDesign classification from various abstracts
4. Graceful handling for sources without sufficient content
5. Confidence forced to "low" for abstract-only content
6. Assessment data correctly injected into synthesis prompt
7. MethodologyAssessor filtering and assessment logic
"""

import json
from typing import Any, Optional

import pytest

from foundry_mcp.core.research.models.sources import (
    MethodologyAssessment,
    ResearchSource,
    SourceType,
    StudyDesign,
)
from foundry_mcp.core.research.workflows.deep_research.phases.methodology_assessment import (
    MIN_CONTENT_LENGTH,
    METHODOLOGY_EXTRACTION_SYSTEM_PROMPT,
    MethodologyAssessor,
    _build_extraction_user_prompt,
    _get_source_content,
    _parse_llm_response,
    format_methodology_context,
)


# =============================================================================
# Fixtures
# =============================================================================


def _make_academic_source(
    source_id: str = "src-1",
    title: str = "Test Paper",
    content: Optional[str] = None,
    snippet: Optional[str] = None,
    source_type: SourceType = SourceType.ACADEMIC,
) -> ResearchSource:
    """Helper to create an academic research source."""
    return ResearchSource(
        id=source_id,
        title=title,
        source_type=source_type,
        content=content,
        snippet=snippet,
        metadata={"authors": "Smith et al.", "year": 2023},
    )


def _make_llm_json_response(
    study_design: str = "randomized_controlled_trial",
    sample_size: Optional[int] = 450,
    sample_description: Optional[str] = "Adults aged 18-65 from urban hospitals",
    effect_size: Optional[str] = "d=0.45",
    statistical_significance: Optional[str] = "p<0.001",
    limitations_noted: Optional[list[str]] = None,
    potential_biases: Optional[list[str]] = None,
) -> str:
    """Build a mock LLM JSON response."""
    return json.dumps({
        "study_design": study_design,
        "sample_size": sample_size,
        "sample_description": sample_description,
        "effect_size": effect_size,
        "statistical_significance": statistical_significance,
        "limitations_noted": limitations_noted or ["Single institution"],
        "potential_biases": potential_biases or ["Self-report measures"],
    })


# Long content strings for testing
LONG_ABSTRACT = "A" * 300  # Exceeds MIN_CONTENT_LENGTH
SHORT_CONTENT = "Too short"
LONG_FULL_TEXT = "B" * 1000


# =============================================================================
# StudyDesign Enum Tests
# =============================================================================


class TestStudyDesign:
    """Tests for the StudyDesign enum."""

    def test_all_study_designs_are_strings(self) -> None:
        for design in StudyDesign:
            assert isinstance(design.value, str)

    def test_study_design_values(self) -> None:
        expected = {
            "meta_analysis", "systematic_review", "randomized_controlled_trial",
            "quasi_experimental", "cohort_study", "case_control",
            "cross_sectional", "qualitative", "case_study", "theoretical",
            "expert_opinion", "unknown",
        }
        actual = {d.value for d in StudyDesign}
        assert actual == expected

    def test_study_design_from_value(self) -> None:
        assert StudyDesign("randomized_controlled_trial") == StudyDesign.RCT
        assert StudyDesign("meta_analysis") == StudyDesign.META_ANALYSIS
        assert StudyDesign("unknown") == StudyDesign.UNKNOWN


# =============================================================================
# MethodologyAssessment Model Tests
# =============================================================================


class TestMethodologyAssessment:
    """Tests for the MethodologyAssessment Pydantic model."""

    def test_default_values(self) -> None:
        assessment = MethodologyAssessment(source_id="src-1")
        assert assessment.source_id == "src-1"
        assert assessment.study_design == StudyDesign.UNKNOWN
        assert assessment.sample_size is None
        assert assessment.sample_description is None
        assert assessment.effect_size is None
        assert assessment.statistical_significance is None
        assert assessment.limitations_noted == []
        assert assessment.potential_biases == []
        assert assessment.confidence == "low"
        assert assessment.content_basis == "abstract"

    def test_full_construction(self) -> None:
        assessment = MethodologyAssessment(
            source_id="src-1",
            study_design=StudyDesign.RCT,
            sample_size=450,
            sample_description="Adults aged 18-65",
            effect_size="d=0.45",
            statistical_significance="p<0.001",
            limitations_noted=["Single institution"],
            potential_biases=["Self-report measures"],
            confidence="high",
            content_basis="full_text",
        )
        assert assessment.study_design == StudyDesign.RCT
        assert assessment.sample_size == 450
        assert assessment.confidence == "high"
        assert assessment.content_basis == "full_text"

    def test_serialization_roundtrip(self) -> None:
        assessment = MethodologyAssessment(
            source_id="src-1",
            study_design=StudyDesign.COHORT,
            sample_size=1200,
            limitations_noted=["Short follow-up"],
            confidence="medium",
            content_basis="full_text",
        )
        data = assessment.model_dump()
        restored = MethodologyAssessment.model_validate(data)
        assert restored == assessment

    def test_json_roundtrip(self) -> None:
        assessment = MethodologyAssessment(
            source_id="src-2",
            study_design=StudyDesign.META_ANALYSIS,
            sample_size=5000,
        )
        json_str = assessment.model_dump_json()
        restored = MethodologyAssessment.model_validate_json(json_str)
        assert restored == assessment


# =============================================================================
# LLM Response Parsing Tests
# =============================================================================


class TestParseLLMResponse:
    """Tests for _parse_llm_response."""

    def test_parse_valid_rct(self) -> None:
        raw = _make_llm_json_response()
        result = _parse_llm_response(raw, "src-1", "full_text")
        assert result.source_id == "src-1"
        assert result.study_design == StudyDesign.RCT
        assert result.sample_size == 450
        assert result.effect_size == "d=0.45"
        assert result.statistical_significance == "p<0.001"
        assert result.content_basis == "full_text"

    def test_parse_meta_analysis(self) -> None:
        raw = _make_llm_json_response(
            study_design="meta_analysis",
            sample_size=5000,
            effect_size="OR=1.8",
        )
        result = _parse_llm_response(raw, "src-2", "full_text")
        assert result.study_design == StudyDesign.META_ANALYSIS
        assert result.sample_size == 5000

    def test_parse_qualitative(self) -> None:
        raw = _make_llm_json_response(
            study_design="qualitative",
            sample_size=12,
            effect_size=None,
            statistical_significance=None,
        )
        result = _parse_llm_response(raw, "src-3", "abstract")
        assert result.study_design == StudyDesign.QUALITATIVE
        assert result.sample_size == 12
        assert result.effect_size is None

    def test_confidence_forced_low_for_abstract(self) -> None:
        """Confidence MUST be forced to 'low' for abstract-only content."""
        raw = _make_llm_json_response()
        result = _parse_llm_response(raw, "src-1", "abstract")
        assert result.confidence == "low"
        assert result.content_basis == "abstract"

    def test_confidence_preserved_for_full_text(self) -> None:
        """Confidence from LLM preserved when content_basis is full_text."""
        # LLM doesn't return confidence in our prompt, so default is "medium"
        raw = _make_llm_json_response()
        result = _parse_llm_response(raw, "src-1", "full_text")
        assert result.confidence == "medium"

    def test_parse_unknown_study_design(self) -> None:
        raw = _make_llm_json_response(study_design="invalid_design_type")
        result = _parse_llm_response(raw, "src-1", "full_text")
        assert result.study_design == StudyDesign.UNKNOWN

    def test_parse_invalid_json(self) -> None:
        result = _parse_llm_response("not json at all", "src-1", "abstract")
        assert result.study_design == StudyDesign.UNKNOWN
        assert result.confidence == "low"
        assert result.source_id == "src-1"

    def test_parse_json_with_markdown_fences(self) -> None:
        raw = "```json\n" + _make_llm_json_response() + "\n```"
        result = _parse_llm_response(raw, "src-1", "full_text")
        assert result.study_design == StudyDesign.RCT
        assert result.sample_size == 450

    def test_parse_null_fields(self) -> None:
        raw = json.dumps({
            "study_design": "theoretical",
            "sample_size": None,
            "sample_description": None,
            "effect_size": None,
            "statistical_significance": None,
            "limitations_noted": [],
            "potential_biases": [],
        })
        result = _parse_llm_response(raw, "src-1", "full_text")
        assert result.study_design == StudyDesign.THEORETICAL
        assert result.sample_size is None
        assert result.effect_size is None
        assert result.limitations_noted == []

    def test_parse_invalid_sample_size(self) -> None:
        raw = json.dumps({
            "study_design": "cohort_study",
            "sample_size": "many participants",
        })
        result = _parse_llm_response(raw, "src-1", "full_text")
        assert result.sample_size is None


# =============================================================================
# Source Content Extraction Tests
# =============================================================================


class TestGetSourceContent:
    """Tests for _get_source_content."""

    def test_prefers_full_content(self) -> None:
        source = _make_academic_source(content=LONG_FULL_TEXT, snippet=LONG_ABSTRACT)
        text, basis = _get_source_content(source)
        assert text == LONG_FULL_TEXT
        assert basis == "full_text"

    def test_falls_back_to_snippet(self) -> None:
        source = _make_academic_source(content=None, snippet=LONG_ABSTRACT)
        text, basis = _get_source_content(source)
        assert text == LONG_ABSTRACT
        assert basis == "abstract"

    def test_short_content_falls_to_snippet(self) -> None:
        source = _make_academic_source(content=SHORT_CONTENT, snippet=LONG_ABSTRACT)
        text, basis = _get_source_content(source)
        assert text == LONG_ABSTRACT
        assert basis == "abstract"

    def test_no_usable_content(self) -> None:
        source = _make_academic_source(content=None, snippet=SHORT_CONTENT)
        text, basis = _get_source_content(source)
        assert text == ""
        assert basis == ""


# =============================================================================
# Extraction Prompt Tests
# =============================================================================


class TestBuildExtractionPrompt:
    """Tests for _build_extraction_user_prompt."""

    def test_abstract_prompt_contains_caveat(self) -> None:
        prompt = _build_extraction_user_prompt("Paper Title", "some content", "abstract")
        assert "abstract only" in prompt
        assert "Paper Title" in prompt

    def test_full_text_prompt_contains_basis(self) -> None:
        prompt = _build_extraction_user_prompt("Paper Title", "some content", "full_text")
        assert "full paper text" in prompt

    def test_content_capped_at_8000(self) -> None:
        long_content = "X" * 10000
        prompt = _build_extraction_user_prompt("Paper", long_content, "full_text")
        # The content portion should be capped
        assert len(prompt) < 10000

    def test_system_prompt_contains_all_study_designs(self) -> None:
        for design in StudyDesign:
            assert design.value in METHODOLOGY_EXTRACTION_SYSTEM_PROMPT


# =============================================================================
# MethodologyAssessor Tests
# =============================================================================


class TestMethodologyAssessor:
    """Tests for the MethodologyAssessor class."""

    def test_filter_only_academic_sources(self) -> None:
        assessor = MethodologyAssessor()
        sources = [
            _make_academic_source("src-1", content=LONG_FULL_TEXT, source_type=SourceType.ACADEMIC),
            _make_academic_source("src-2", content=LONG_FULL_TEXT, source_type=SourceType.WEB),
        ]
        eligible = assessor.filter_assessable_sources(sources)
        assert len(eligible) == 1
        assert eligible[0][0].id == "src-1"

    def test_filter_skips_short_content(self) -> None:
        assessor = MethodologyAssessor()
        sources = [
            _make_academic_source("src-1", content=SHORT_CONTENT, snippet=SHORT_CONTENT),
        ]
        eligible = assessor.filter_assessable_sources(sources)
        assert len(eligible) == 0

    def test_filter_accepts_long_snippet(self) -> None:
        assessor = MethodologyAssessor()
        sources = [
            _make_academic_source("src-1", content=None, snippet=LONG_ABSTRACT),
        ]
        eligible = assessor.filter_assessable_sources(sources)
        assert len(eligible) == 1
        assert eligible[0][2] == "abstract"

    def test_filter_custom_min_content_length(self) -> None:
        assessor = MethodologyAssessor(min_content_length=50)
        sources = [
            _make_academic_source("src-1", content="A" * 60),
        ]
        eligible = assessor.filter_assessable_sources(sources)
        assert len(eligible) == 1

    @pytest.mark.asyncio
    async def test_assess_sources_no_eligible(self) -> None:
        assessor = MethodologyAssessor()
        sources = [
            _make_academic_source("src-1", source_type=SourceType.WEB, content=LONG_FULL_TEXT),
        ]
        results = await assessor.assess_sources(sources)
        assert results == []

    @pytest.mark.asyncio
    async def test_assess_sources_no_workflow_returns_fallback(self) -> None:
        """Without workflow/state, assessor returns fallback UNKNOWN assessments."""
        assessor = MethodologyAssessor()
        sources = [
            _make_academic_source("src-1", content=LONG_FULL_TEXT),
        ]
        results = await assessor.assess_sources(sources, workflow=None, state=None)
        assert len(results) == 1
        assert results[0].source_id == "src-1"
        assert results[0].study_design == StudyDesign.UNKNOWN
        assert results[0].confidence == "low"


# =============================================================================
# Format Methodology Context Tests
# =============================================================================


class TestFormatMethodologyContext:
    """Tests for format_methodology_context synthesis injection."""

    def test_empty_assessments(self) -> None:
        result = format_methodology_context([], {}, [])
        assert result == ""

    def test_assessments_with_no_meaningful_data(self) -> None:
        """Assessments with only UNKNOWN design and no metrics are skipped."""
        assessments = [
            MethodologyAssessment(source_id="src-1"),
        ]
        sources = [_make_academic_source("src-1")]
        result = format_methodology_context(assessments, {"src-1": 1}, sources)
        assert result == ""

    def test_basic_rct_formatting(self) -> None:
        assessments = [
            MethodologyAssessment(
                source_id="src-1",
                study_design=StudyDesign.RCT,
                sample_size=450,
                effect_size="d=0.45",
                statistical_significance="p<0.001",
                limitations_noted=["Single institution", "Self-report measures"],
                confidence="high",
                content_basis="full_text",
            ),
        ]
        sources = [_make_academic_source("src-1", title="Smith et al. (2021)")]
        id_to_citation = {"src-1": 1}

        result = format_methodology_context(assessments, id_to_citation, sources)

        assert "## Methodology Context" in result
        assert "[1] Smith et al. (2021)" in result
        assert "Randomized Controlled Trial" in result
        assert "N=450" in result
        assert "d=0.45" in result
        assert "p<0.001" in result
        assert "Single institution" in result
        assert "full_text" in result

    def test_abstract_basis_caveat(self) -> None:
        assessments = [
            MethodologyAssessment(
                source_id="src-1",
                study_design=StudyDesign.QUALITATIVE,
                sample_size=12,
                confidence="low",
                content_basis="abstract",
            ),
        ]
        sources = [_make_academic_source("src-1")]
        result = format_methodology_context(assessments, {"src-1": 1}, sources)
        assert "abstract" in result
        assert "lower confidence" in result

    def test_multiple_assessments(self) -> None:
        assessments = [
            MethodologyAssessment(
                source_id="src-1",
                study_design=StudyDesign.RCT,
                sample_size=450,
                content_basis="full_text",
            ),
            MethodologyAssessment(
                source_id="src-2",
                study_design=StudyDesign.META_ANALYSIS,
                sample_size=5000,
                content_basis="full_text",
            ),
        ]
        sources = [
            _make_academic_source("src-1", title="Paper A"),
            _make_academic_source("src-2", title="Paper B"),
        ]
        id_to_citation = {"src-1": 1, "src-2": 2}

        result = format_methodology_context(assessments, id_to_citation, sources)
        assert "[1] Paper A" in result
        assert "[2] Paper B" in result
        assert "Randomized Controlled Trial" in result
        assert "Meta Analysis" in result

    def test_missing_source_skipped(self) -> None:
        """Assessment for a source not in the sources list is skipped."""
        assessments = [
            MethodologyAssessment(
                source_id="src-missing",
                study_design=StudyDesign.RCT,
                sample_size=100,
            ),
        ]
        result = format_methodology_context(assessments, {}, [])
        assert result == ""

    def test_context_section_framed_as_qualitative(self) -> None:
        """The header should frame methodology as context, not ground truth."""
        assessments = [
            MethodologyAssessment(
                source_id="src-1",
                study_design=StudyDesign.COHORT,
                sample_size=800,
                content_basis="full_text",
            ),
        ]
        sources = [_make_academic_source("src-1")]
        result = format_methodology_context(assessments, {"src-1": 1}, sources)
        assert "not as ground truth" in result
        assert "qualitative weighting" in result

    def test_sample_description_included(self) -> None:
        assessments = [
            MethodologyAssessment(
                source_id="src-1",
                study_design=StudyDesign.RCT,
                sample_size=200,
                sample_description="Elderly patients with diabetes",
                content_basis="full_text",
            ),
        ]
        sources = [_make_academic_source("src-1")]
        result = format_methodology_context(assessments, {"src-1": 1}, sources)
        assert "Elderly patients with diabetes" in result

    def test_biases_included(self) -> None:
        assessments = [
            MethodologyAssessment(
                source_id="src-1",
                study_design=StudyDesign.CROSS_SECTIONAL,
                sample_size=500,
                potential_biases=["Selection bias", "Recall bias"],
                content_basis="full_text",
            ),
        ]
        sources = [_make_academic_source("src-1")]
        result = format_methodology_context(assessments, {"src-1": 1}, sources)
        assert "Selection bias" in result
        assert "Recall bias" in result

    def test_source_id_fallback_when_no_citation_number(self) -> None:
        assessments = [
            MethodologyAssessment(
                source_id="src-1",
                study_design=StudyDesign.RCT,
                sample_size=100,
                content_basis="full_text",
            ),
        ]
        sources = [_make_academic_source("src-1")]
        result = format_methodology_context(assessments, {}, sources)
        assert "[src-1]" in result
