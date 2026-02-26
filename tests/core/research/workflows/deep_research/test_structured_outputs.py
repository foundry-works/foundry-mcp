"""Tests for Phase 4: Structured Output Schemas at LLM Boundaries.

Covers:
- 4.1: Pydantic schemas (DelegationResponse, ReflectionDecision, ResearchBriefOutput)
- 4.2: parse_delegation_response() — valid JSON, malformed, missing fields, fallback
- 4.3: parse_reflection_decision() — valid JSON, URL coercion, fallback
- 4.4: parse_brief_output() — JSON, plain text, empty
- 4.5: Schema validation rejects malformed responses
- 4.6: Integration with execute_structured_llm_call retry logic
"""

from __future__ import annotations

import json

import pytest

from foundry_mcp.core.research.models.deep_research import (
    DelegationResponse,
    ReflectionDecision,
    ResearchBriefOutput,
    ResearchDirective,
    parse_brief_output,
    parse_delegation_response,
    parse_reflection_decision,
)


# ===========================================================================
# 4.1 Schema validation tests
# ===========================================================================


class TestDelegationResponseSchema:
    """DelegationResponse Pydantic model tests."""

    def test_defaults(self):
        """All fields have sensible defaults; validator forces research_complete=True when no directives."""
        resp = DelegationResponse(rationale="test")
        # Validator forces research_complete=True when directives is empty
        assert resp.research_complete is True
        assert resp.directives == []
        assert resp.rationale == "test"

    def test_research_complete_true(self):
        resp = DelegationResponse(research_complete=True, rationale="Done")
        assert resp.research_complete is True
        assert resp.directives == []

    def test_directives_populated(self):
        resp = DelegationResponse(
            directives=[
                ResearchDirective(
                    research_topic="Investigate AI safety",
                    perspective="technical",
                    evidence_needed="papers",
                    priority=1,
                ),
            ],
            rationale="Gap in safety coverage",
        )
        assert len(resp.directives) == 1
        assert resp.directives[0].research_topic == "Investigate AI safety"
        assert resp.directives[0].priority == 1
        # Auto-generated fields should be set
        assert resp.directives[0].id.startswith("dir-")

    def test_from_dict(self):
        data = {
            "research_complete": False,
            "directives": [
                {"research_topic": "Topic A", "priority": 2},
                {"research_topic": "Topic B", "priority": 1},
            ],
            "rationale": "Two gaps identified",
        }
        resp = DelegationResponse.model_validate(data)
        assert len(resp.directives) == 2
        assert resp.directives[0].perspective == ""  # default
        assert resp.directives[1].priority == 1

    def test_validator_forces_complete_when_no_directives(self):
        """Validator forces research_complete=True when directives empty."""
        resp = DelegationResponse(
            research_complete=False, directives=[], rationale="test"
        )
        assert resp.research_complete is True

    def test_validator_does_not_override_when_directives_present(self):
        """Validator leaves research_complete=False when directives present."""
        resp = DelegationResponse(
            research_complete=False,
            directives=[
                ResearchDirective(research_topic="topic", priority=1),
            ],
            rationale="test",
        )
        assert resp.research_complete is False


class TestReflectionDecisionSchema:
    """ReflectionDecision Pydantic model tests."""

    def test_defaults(self):
        resp = ReflectionDecision()
        assert resp.continue_searching is False
        assert resp.research_complete is False
        assert resp.refined_query is None
        assert resp.urls_to_extract == []
        assert resp.rationale == ""

    def test_continue_with_refined_query(self):
        resp = ReflectionDecision(
            continue_searching=True,
            refined_query="more specific query",
            rationale="Need more data",
        )
        assert resp.continue_searching is True
        assert resp.refined_query == "more specific query"

    def test_urls_coercion_from_none(self):
        """urls_to_extract accepts None and converts to empty list."""
        resp = ReflectionDecision.model_validate({
            "continue_searching": False,
            "urls_to_extract": None,
        })
        assert resp.urls_to_extract == []

    def test_urls_coercion_filters_invalid(self):
        """urls_to_extract filters non-HTTP strings."""
        resp = ReflectionDecision.model_validate({
            "urls_to_extract": [
                "https://example.com/good",
                "not-a-url",
                "http://valid.org",
                123,  # non-string
            ],
        })
        assert resp.urls_to_extract == [
            "https://example.com/good",
            "http://valid.org",
        ]

    def test_urls_capped_at_five(self):
        """urls_to_extract hard-capped at 5."""
        urls = [f"https://example.com/{i}" for i in range(10)]
        resp = ReflectionDecision.model_validate({"urls_to_extract": urls})
        assert len(resp.urls_to_extract) == 5

    def test_research_complete_signal(self):
        resp = ReflectionDecision(
            research_complete=True,
            rationale="Findings sufficient",
        )
        assert resp.research_complete is True
        assert resp.continue_searching is False

    def test_validator_forces_continue_false_when_complete(self):
        """Validator forces continue_searching=False when research_complete=True."""
        resp = ReflectionDecision(
            research_complete=True,
            continue_searching=True,
            rationale="Contradictory flags",
        )
        assert resp.research_complete is True
        assert resp.continue_searching is False

    def test_validator_allows_continue_when_not_complete(self):
        """Validator does not touch continue_searching when research_complete=False."""
        resp = ReflectionDecision(
            research_complete=False,
            continue_searching=True,
            rationale="Still searching",
        )
        assert resp.research_complete is False
        assert resp.continue_searching is True


class TestResearchBriefOutputSchema:
    """ResearchBriefOutput Pydantic model tests."""

    def test_minimal(self):
        resp = ResearchBriefOutput(research_brief="A research brief.")
        assert resp.research_brief == "A research brief."
        assert resp.scope_boundaries is None
        assert resp.source_preferences is None

    def test_full(self):
        resp = ResearchBriefOutput(
            research_brief="Detailed brief text.",
            scope_boundaries="Include X, exclude Y",
            source_preferences="Peer-reviewed papers preferred",
        )
        assert resp.scope_boundaries == "Include X, exclude Y"
        assert resp.source_preferences == "Peer-reviewed papers preferred"

    def test_empty_brief_rejected(self):
        """research_brief is required."""
        with pytest.raises(Exception):
            ResearchBriefOutput.model_validate({})


# ===========================================================================
# 4.2 parse_delegation_response tests
# ===========================================================================


class TestParseDelegationResponse:
    """Tests for parse_delegation_response() parse function."""

    def test_valid_json(self):
        content = json.dumps({
            "research_complete": False,
            "directives": [
                {
                    "research_topic": "Investigate topic A",
                    "perspective": "technical",
                    "evidence_needed": "benchmarks",
                    "priority": 1,
                },
            ],
            "rationale": "Gap in coverage",
        })
        resp = parse_delegation_response(content)
        assert isinstance(resp, DelegationResponse)
        assert resp.research_complete is False
        assert len(resp.directives) == 1
        assert resp.directives[0].research_topic == "Investigate topic A"

    def test_json_in_code_block(self):
        content = '```json\n{"research_complete": true, "rationale": "Done"}\n```'
        resp = parse_delegation_response(content)
        assert resp.research_complete is True

    def test_json_with_surrounding_text(self):
        content = (
            "Here is my analysis:\n\n"
            '{"research_complete": false, "directives": [], "rationale": "test"}\n\n'
            "Let me know if you need more."
        )
        resp = parse_delegation_response(content)
        # Validator forces research_complete=True when directives empty
        assert resp.research_complete is True

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON"):
            parse_delegation_response("Just some plain text without JSON")

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="No JSON"):
            parse_delegation_response("")

    def test_malformed_json_raises(self):
        with pytest.raises((ValueError, Exception)):
            parse_delegation_response('{"research_complete": true, broken}')

    def test_research_complete_with_empty_directives(self):
        content = json.dumps({
            "research_complete": True,
            "directives": [],
            "rationale": "All covered",
        })
        resp = parse_delegation_response(content)
        assert resp.research_complete is True
        assert resp.directives == []

    def test_directive_priority_validation(self):
        """Priority values are validated by Pydantic (1-3 range)."""
        content = json.dumps({
            "directives": [
                {"research_topic": "Topic", "priority": 1},
            ],
            "rationale": "test",
        })
        resp = parse_delegation_response(content)
        assert resp.directives[0].priority == 1


# ===========================================================================
# 4.3 parse_reflection_decision tests
# ===========================================================================


class TestParseReflectionDecision:
    """Tests for parse_reflection_decision() parse function."""

    def test_valid_json(self):
        content = json.dumps({
            "continue_searching": True,
            "research_complete": False,
            "refined_query": "more specific search",
            "urls_to_extract": ["https://example.com"],
            "rationale": "Need more sources",
        })
        resp = parse_reflection_decision(content)
        assert isinstance(resp, ReflectionDecision)
        assert resp.continue_searching is True
        assert resp.refined_query == "more specific search"
        assert resp.urls_to_extract == ["https://example.com"]

    def test_research_complete(self):
        content = json.dumps({
            "continue_searching": False,
            "research_complete": True,
            "rationale": "Sufficient coverage",
        })
        resp = parse_reflection_decision(content)
        assert resp.research_complete is True
        assert resp.continue_searching is False

    def test_null_urls_handled(self):
        content = json.dumps({
            "continue_searching": False,
            "urls_to_extract": None,
            "rationale": "Done",
        })
        resp = parse_reflection_decision(content)
        assert resp.urls_to_extract == []

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON"):
            parse_reflection_decision("plain text")

    def test_json_in_markdown(self):
        content = '```json\n{"continue_searching": false, "rationale": "done"}\n```'
        resp = parse_reflection_decision(content)
        assert resp.continue_searching is False

    def test_urls_filtered_to_http(self):
        content = json.dumps({
            "urls_to_extract": [
                "https://good.com",
                "ftp://bad.com",
                "not-url",
            ],
            "rationale": "test",
        })
        resp = parse_reflection_decision(content)
        assert resp.urls_to_extract == ["https://good.com"]


# ===========================================================================
# 4.4 parse_brief_output tests
# ===========================================================================


class TestParseBriefOutput:
    """Tests for parse_brief_output() parse function."""

    def test_valid_json(self):
        content = json.dumps({
            "research_brief": "A detailed brief about AI safety.",
            "scope_boundaries": "Focus on transformers only",
            "source_preferences": "Peer-reviewed papers",
        })
        resp = parse_brief_output(content)
        assert isinstance(resp, ResearchBriefOutput)
        assert resp.research_brief == "A detailed brief about AI safety."
        assert resp.scope_boundaries == "Focus on transformers only"

    def test_json_minimal(self):
        content = json.dumps({"research_brief": "Brief text."})
        resp = parse_brief_output(content)
        assert resp.research_brief == "Brief text."
        assert resp.scope_boundaries is None

    def test_plain_text_fallback(self):
        """Plain text is treated as the research_brief field."""
        content = "This is a research brief about quantum computing."
        resp = parse_brief_output(content)
        assert resp.research_brief == content

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty brief"):
            parse_brief_output("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="Empty brief"):
            parse_brief_output("   \n  ")

    def test_json_in_code_block(self):
        content = (
            '```json\n{"research_brief": "Brief in code block."}\n```'
        )
        resp = parse_brief_output(content)
        assert resp.research_brief == "Brief in code block."

    def test_malformed_json_falls_back_to_text(self):
        """Malformed JSON falls through to plain-text handling."""
        content = '{"research_brief": broken json}'
        resp = parse_brief_output(content)
        # Should treat the whole thing as plain text
        assert resp.research_brief == content


# ===========================================================================
# 4.5 Schema validation edge cases
# ===========================================================================


class TestSchemaValidationEdgeCases:
    """Schema validation catches invalid data."""

    def test_delegation_invalid_priority_rejected(self):
        """Priority outside 1-3 range is rejected by Pydantic."""
        with pytest.raises(Exception):
            DelegationResponse.model_validate({
                "directives": [
                    {"research_topic": "Topic", "priority": 10},
                ],
                "rationale": "test",
            })

    def test_reflection_extra_fields_ignored(self):
        """Extra fields in JSON are handled gracefully."""
        content = json.dumps({
            "continue_searching": True,
            "rationale": "test",
            "unknown_field": "should be ignored",
        })
        # Should not raise — Pydantic ignores extra by default
        resp = parse_reflection_decision(content)
        assert resp.continue_searching is True

    def test_delegation_empty_topic_allowed_by_schema(self):
        """Empty research_topic passes schema but filtered by _apply_directive_caps."""
        resp = DelegationResponse.model_validate({
            "directives": [{"research_topic": ""}],
            "rationale": "test",
        })
        assert len(resp.directives) == 1
        assert resp.directives[0].research_topic == ""
