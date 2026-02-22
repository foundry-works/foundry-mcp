"""Tests for structured output parsing across ThinkDeep, Ideate, and DeepResearch.

Phase 1d: validates JSON parsing, fallback behavior, Pydantic validation,
and parse_method metadata for all three workflows.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.research.models.enums import ConfidenceLevel, IdeationPhase
from foundry_mcp.core.research.models.ideation import Idea, IdeaCluster, IdeationState
from foundry_mcp.core.research.models.thinkdeep import InvestigationStep, ThinkDeepState
from foundry_mcp.core.research.workflows.base import WorkflowResult

# ──────────────────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_config(mock_config):
    """Extend base mock_config for thinkdeep + ideate."""
    mock_config.thinkdeep_max_depth = 5
    mock_config.ideate_perspectives = ["technical", "creative", "practical"]
    mock_config.default_timeout = 30
    return mock_config


@pytest.fixture
def mock_memory(mock_memory):
    """Extend base mock_memory with save methods."""
    mock_memory.save_investigation = MagicMock()
    mock_memory.save_ideation = MagicMock()
    mock_memory.load_investigation = MagicMock(return_value=None)
    mock_memory.load_ideation = MagicMock(return_value=None)
    return mock_memory


@pytest.fixture
def thinkdeep_workflow(mock_config, mock_memory):
    from foundry_mcp.core.research.workflows.thinkdeep import ThinkDeepWorkflow

    return ThinkDeepWorkflow(mock_config, mock_memory)


@pytest.fixture
def ideate_workflow(mock_config, mock_memory):
    from foundry_mcp.core.research.workflows.ideate import IdeateWorkflow

    return IdeateWorkflow(mock_config, mock_memory)


@pytest.fixture
def thinkdeep_state():
    return ThinkDeepState(topic="test topic", max_depth=5)


@pytest.fixture
def ideation_state():
    state = IdeationState(topic="test topic")
    # Pre-populate some ideas for clustering/scoring tests
    for i in range(5):
        state.ideas.append(Idea(content=f"Idea {i + 1}", perspective="technical"))
    return state


# ──────────────────────────────────────────────────────────────────────
#  ThinkDeep Structured Output Tests
# ──────────────────────────────────────────────────────────────────────


class TestThinkDeepStructuredOutput:
    """Tests for ThinkDeep JSON parsing and fallback."""

    def test_valid_json_parsed_correctly(self, thinkdeep_workflow, thinkdeep_state):
        """Valid JSON response is parsed into hypotheses with evidence."""
        step = thinkdeep_state.add_step(query="test query", depth=0)
        response = json.dumps(
            {
                "hypotheses": [
                    {
                        "statement": "Solar energy is more cost-effective",
                        "evidence": [
                            {
                                "text": "Cost per watt has dropped 90%",
                                "strength": "strong",
                                "supporting": True,
                            }
                        ],
                        "is_new": True,
                    }
                ],
                "next_questions": ["What about storage costs?"],
                "key_insights": ["Solar costs declining rapidly"],
            }
        )

        parse_method = thinkdeep_workflow._update_hypotheses_from_response(thinkdeep_state, step, response)

        assert parse_method == "json"
        assert len(thinkdeep_state.hypotheses) == 1
        assert "Solar energy" in thinkdeep_state.hypotheses[0].statement
        assert len(thinkdeep_state.hypotheses[0].supporting_evidence) == 1
        assert step.hypotheses_generated

    def test_json_in_markdown_code_block(self, thinkdeep_workflow, thinkdeep_state):
        """JSON wrapped in markdown code blocks is extracted correctly."""
        step = thinkdeep_state.add_step(query="test query", depth=0)
        response = """Here's my analysis:

```json
{
    "hypotheses": [
        {
            "statement": "Test hypothesis",
            "evidence": [],
            "is_new": true
        }
    ],
    "next_questions": [],
    "key_insights": []
}
```"""

        parse_method = thinkdeep_workflow._update_hypotheses_from_response(thinkdeep_state, step, response)

        assert parse_method == "json"
        assert len(thinkdeep_state.hypotheses) == 1

    def test_malformed_json_triggers_fallback(self, thinkdeep_workflow, thinkdeep_state):
        """Malformed JSON falls back to keyword extraction."""
        step = thinkdeep_state.add_step(query="test query", depth=0)
        response = "This hypothesis suggests that the evidence supports our claim. {invalid json"

        parse_method = thinkdeep_workflow._update_hypotheses_from_response(thinkdeep_state, step, response)

        assert parse_method == "fallback_keyword"
        # Keyword fallback should have created a hypothesis (depth < 2, "hypothesis" keyword)
        assert len(thinkdeep_state.hypotheses) == 1

    def test_plain_text_triggers_fallback(self, thinkdeep_workflow, thinkdeep_state):
        """Plain text response without any JSON triggers keyword fallback."""
        step = thinkdeep_state.add_step(query="test query", depth=0)
        response = "No structured content here, just a plain response."

        parse_method = thinkdeep_workflow._update_hypotheses_from_response(thinkdeep_state, step, response)

        assert parse_method == "fallback_keyword"

    def test_empty_response_triggers_fallback(self, thinkdeep_workflow, thinkdeep_state):
        """Empty response triggers keyword fallback (which does nothing)."""
        step = thinkdeep_state.add_step(query="test query", depth=0)

        parse_method = thinkdeep_workflow._update_hypotheses_from_response(thinkdeep_state, step, "")

        assert parse_method == "fallback_keyword"
        assert len(thinkdeep_state.hypotheses) == 0

    def test_pydantic_validation_catches_missing_fields(self, thinkdeep_workflow, thinkdeep_state):
        """JSON missing required 'statement' field triggers fallback."""
        step = thinkdeep_state.add_step(query="test query", depth=0)
        response = json.dumps(
            {
                "hypotheses": [
                    {
                        # Missing 'statement' field
                        "evidence": [{"text": "something", "strength": "strong"}],
                        "is_new": True,
                    }
                ],
            }
        )

        parse_method = thinkdeep_workflow._update_hypotheses_from_response(thinkdeep_state, step, response)

        assert parse_method == "fallback_keyword"

    def test_evidence_strength_affects_confidence(self, thinkdeep_workflow, thinkdeep_state):
        """Strong evidence bumps confidence higher than moderate evidence."""
        step = thinkdeep_state.add_step(query="test query", depth=0)

        # Strong evidence
        response = json.dumps(
            {
                "hypotheses": [
                    {
                        "statement": "Strong evidence hypothesis",
                        "evidence": [{"text": "Strong proof", "strength": "strong", "supporting": True}],
                        "is_new": True,
                    }
                ],
            }
        )

        thinkdeep_workflow._update_hypotheses_from_response(thinkdeep_state, step, response)
        assert thinkdeep_state.hypotheses[0].confidence == ConfidenceLevel.MEDIUM

    def test_weak_evidence_keeps_low_confidence(self, thinkdeep_workflow, thinkdeep_state):
        """Weak evidence results in speculation-level confidence."""
        step = thinkdeep_state.add_step(query="test query", depth=0)
        response = json.dumps(
            {
                "hypotheses": [
                    {
                        "statement": "Weak evidence hypothesis",
                        "evidence": [{"text": "Weak hint", "strength": "weak", "supporting": True}],
                        "is_new": True,
                    }
                ],
            }
        )

        thinkdeep_workflow._update_hypotheses_from_response(thinkdeep_state, step, response)
        assert thinkdeep_state.hypotheses[0].confidence == ConfidenceLevel.SPECULATION

    def test_contradicting_evidence_recorded(self, thinkdeep_workflow, thinkdeep_state):
        """Contradicting evidence is recorded correctly."""
        step = thinkdeep_state.add_step(query="test query", depth=0)
        response = json.dumps(
            {
                "hypotheses": [
                    {
                        "statement": "Contradicted hypothesis",
                        "evidence": [
                            {"text": "Against it", "strength": "strong", "supporting": False},
                            {"text": "For it", "strength": "weak", "supporting": True},
                        ],
                        "is_new": True,
                    }
                ],
            }
        )

        thinkdeep_workflow._update_hypotheses_from_response(thinkdeep_state, step, response)
        hyp = thinkdeep_state.hypotheses[0]
        assert len(hyp.contradicting_evidence) == 1
        assert len(hyp.supporting_evidence) == 1

    def test_parse_method_in_execute_metadata(self, thinkdeep_workflow, mock_config, mock_memory):
        """parse_method is surfaced in execute() result metadata."""
        provider_response = json.dumps(
            {
                "hypotheses": [{"statement": "Test", "evidence": [], "is_new": True}],
                "next_questions": [],
                "key_insights": [],
            }
        )

        mock_result = WorkflowResult(
            success=True,
            content=provider_response,
            provider_id="test",
            model_used="test-model",
        )

        with patch.object(thinkdeep_workflow, "_execute_provider", return_value=mock_result):
            result = thinkdeep_workflow.execute(topic="Test topic")

        assert result.success
        assert result.metadata.get("parse_method") == "json"

    def test_false_positive_keyword_documented(self, thinkdeep_workflow, thinkdeep_state):
        """Words containing keywords (e.g., 'unsupported') trigger keyword fallback."""
        step = thinkdeep_state.add_step(query="test query", depth=0)
        # "unsupported" contains "support" - this is a known false positive in keyword matching
        response = "This feature is unsupported in the current version."

        parse_method = thinkdeep_workflow._update_hypotheses_from_response(thinkdeep_state, step, response)

        assert parse_method == "fallback_keyword"

    def test_existing_hypothesis_updated_not_duplicated(self, thinkdeep_workflow, thinkdeep_state):
        """Updating an existing hypothesis matches by statement, doesn't create duplicate."""
        # Add an existing hypothesis
        thinkdeep_state.add_hypothesis(
            statement="Solar is cheaper",
            confidence=ConfidenceLevel.SPECULATION,
        )

        step = thinkdeep_state.add_step(query="test query", depth=1)
        response = json.dumps(
            {
                "hypotheses": [
                    {
                        "statement": "Solar is cheaper",
                        "evidence": [{"text": "New data confirms", "strength": "strong", "supporting": True}],
                        "is_new": False,
                    }
                ],
            }
        )

        thinkdeep_workflow._update_hypotheses_from_response(thinkdeep_state, step, response)
        # Should update existing, not create new
        assert len(thinkdeep_state.hypotheses) == 1
        assert len(thinkdeep_state.hypotheses[0].supporting_evidence) == 1


# ──────────────────────────────────────────────────────────────────────
#  Ideate Structured Output Tests
# ──────────────────────────────────────────────────────────────────────


class TestIdeateStructuredOutput:
    """Tests for Ideate JSON parsing and fallback."""

    def test_parse_ideas_valid_json(self, ideate_workflow):
        """Valid JSON ideas response is parsed correctly."""
        response = json.dumps(
            {
                "ideas": [
                    {"content": "Use AI for scheduling"},
                    {"content": "Blockchain for supply chain"},
                ]
            }
        )

        ideas, method = ideate_workflow._parse_ideas(response, "technical", "test-provider", "test-model")

        assert method == "json"
        assert len(ideas) == 2
        assert ideas[0].content == "Use AI for scheduling"
        assert ideas[0].perspective == "technical"

    def test_parse_ideas_fallback_dash_format(self, ideate_workflow):
        """Line-based fallback parses dash-formatted ideas."""
        response = "- Use machine learning\n- Build a chatbot\n- Create analytics"

        ideas, method = ideate_workflow._parse_ideas(response, "creative", "test-provider", "test-model")

        assert method == "fallback_regex"
        assert len(ideas) == 3

    def test_parse_ideas_fallback_bullet_format(self, ideate_workflow):
        """Line-based fallback parses bullet-formatted ideas."""
        response = "• First idea\n• Second idea"

        ideas, method = ideate_workflow._parse_ideas(response, "practical", "test-provider", "test-model")

        assert method == "fallback_regex"
        assert len(ideas) == 2

    def test_parse_ideas_empty_json_array(self, ideate_workflow):
        """Empty ideas array returns empty list."""
        response = json.dumps({"ideas": []})

        ideas, method = ideate_workflow._parse_ideas(response, "technical", None, None)

        assert method == "json"
        assert len(ideas) == 0

    def test_parse_ideas_malformed_json(self, ideate_workflow):
        """Malformed JSON falls back to line parsing."""
        response = '{"ideas": [{"content": "partial...'

        ideas, method = ideate_workflow._parse_ideas(response, "technical", None, None)

        assert method == "fallback_regex"

    def test_parse_clusters_valid_json(self, ideate_workflow, ideation_state):
        """Valid JSON clusters response is parsed correctly."""
        response = json.dumps(
            {
                "clusters": [
                    {
                        "name": "AI Solutions",
                        "description": "Ideas involving artificial intelligence",
                        "idea_numbers": [1, 2],
                    },
                    {
                        "name": "Data Analytics",
                        "description": "Ideas about data analysis",
                        "idea_numbers": [3, 4, 5],
                    },
                ]
            }
        )

        clusters, method = ideate_workflow._parse_clusters(response, ideation_state)

        assert method == "json"
        assert len(clusters) == 2
        assert clusters[0].name == "AI Solutions"
        assert len(clusters[0].idea_ids) == 2
        assert len(clusters[1].idea_ids) == 3

    def test_parse_clusters_fallback_keyword_format(self, ideate_workflow, ideation_state):
        """Keyword-based fallback parses CLUSTER:/DESCRIPTION:/IDEAS: format."""
        response = """CLUSTER: AI Solutions
DESCRIPTION: Ideas involving AI
IDEAS: 1, 2

CLUSTER: Data Analytics
DESCRIPTION: Data analysis ideas
IDEAS: 3, 4, 5"""

        clusters, method = ideate_workflow._parse_clusters(response, ideation_state)

        assert method == "fallback_regex"
        assert len(clusters) == 2
        assert clusters[0].name == "AI Solutions"

    def test_parse_clusters_out_of_range_ideas_ignored(self, ideate_workflow, ideation_state):
        """Idea numbers out of range are silently ignored."""
        response = json.dumps({"clusters": [{"name": "Test", "description": "desc", "idea_numbers": [1, 99, 100]}]})

        clusters, method = ideate_workflow._parse_clusters(response, ideation_state)

        assert method == "json"
        assert len(clusters) == 1
        assert len(clusters[0].idea_ids) == 1  # Only idea 1 is valid

    def test_parse_scores_valid_json(self, ideate_workflow, ideation_state):
        """Valid JSON scores are applied to ideas."""
        response = json.dumps(
            {
                "scores": [
                    {"idea_number": 1, "score": 0.9, "justification": "Great idea"},
                    {"idea_number": 3, "score": 0.5, "justification": "Average"},
                ]
            }
        )

        method = ideate_workflow._parse_scores(response, ideation_state)

        assert method == "json"
        assert ideation_state.ideas[0].score == 0.9
        assert ideation_state.ideas[2].score == 0.5
        assert ideation_state.ideas[1].score is None  # Not scored

    def test_parse_scores_fallback_colon_format(self, ideate_workflow, ideation_state):
        """Fallback parses 'number: score - justification' format."""
        response = "1: 0.8 - Good\n2: 0.6 - Okay\n3: 0.9 - Excellent"

        method = ideate_workflow._parse_scores(response, ideation_state)

        assert method == "fallback_regex"
        assert ideation_state.ideas[0].score == 0.8
        assert ideation_state.ideas[1].score == 0.6
        assert ideation_state.ideas[2].score == 0.9

    def test_parse_scores_invalid_score_range(self, ideate_workflow, ideation_state):
        """Scores outside 0-1 range are rejected by Pydantic validation."""
        response = json.dumps(
            {
                "scores": [
                    {"idea_number": 1, "score": 1.5, "justification": "Too high"},
                ]
            }
        )

        # Pydantic validation should fail, triggering fallback
        method = ideate_workflow._parse_scores(response, ideation_state)

        assert method == "fallback_regex"

    def test_parse_scores_multi_digit_fallback(self, ideate_workflow, ideation_state):
        """Multi-digit idea numbers work in fallback mode."""
        # Need more ideas for multi-digit test
        for i in range(5, 12):
            ideation_state.ideas.append(Idea(content=f"Idea {i + 1}", perspective="technical"))

        response = "10: 0.7 - Decent\n11: 0.3 - Weak"

        method = ideate_workflow._parse_scores(response, ideation_state)

        assert method == "fallback_regex"
        assert ideation_state.ideas[9].score == 0.7
        assert ideation_state.ideas[10].score == 0.3

    def test_parse_ideas_json_in_code_block(self, ideate_workflow):
        """JSON in markdown code blocks is extracted correctly."""
        response = """Here are some ideas:

```json
{
    "ideas": [
        {"content": "Idea from code block"}
    ]
}
```"""

        ideas, method = ideate_workflow._parse_ideas(response, "technical", None, None)

        assert method == "json"
        assert len(ideas) == 1


# ──────────────────────────────────────────────────────────────────────
#  Deep Research Analysis Structured Output Tests
# ──────────────────────────────────────────────────────────────────────


class TestDeepResearchAnalysisOutput:
    """Tests for Deep Research analysis parsing with Pydantic validation."""

    @pytest.fixture
    def parser(self):
        from foundry_mcp.core.research.workflows.deep_research.phases._analysis_parsing import (
            AnalysisParsingMixin,
        )

        return AnalysisParsingMixin()

    @pytest.fixture
    def mock_state(self):
        return MagicMock()

    def test_valid_json_parsed_with_pydantic(self, parser, mock_state):
        """Valid JSON response is parsed via Pydantic validation."""
        content = json.dumps(
            {
                "findings": [
                    {
                        "content": "Solar panel costs decreased by 90%",
                        "confidence": "high",
                        "source_ids": ["src-001"],
                        "category": "economics",
                    }
                ],
                "gaps": [
                    {
                        "description": "Storage cost data missing",
                        "suggested_queries": ["battery storage costs 2025"],
                        "priority": 2,
                    }
                ],
                "quality_updates": [{"source_id": "src-001", "quality": "high"}],
            }
        )

        result = parser._parse_analysis_response(content, mock_state)

        assert result["success"] is True
        assert result["parse_method"] == "json"
        assert len(result["findings"]) == 1
        assert result["findings"][0]["content"] == "Solar panel costs decreased by 90%"
        assert result["findings"][0]["confidence"] == ConfidenceLevel.HIGH
        assert result["findings"][0]["source_ids"] == ["src-001"]
        assert len(result["gaps"]) == 1
        assert result["gaps"][0]["priority"] == 2
        assert len(result["quality_updates"]) == 1

    def test_empty_content_returns_failure(self, parser, mock_state):
        """Empty content returns failure without crashing."""
        result = parser._parse_analysis_response("", mock_state)

        assert result["success"] is False
        assert result["findings"] == []

    def test_no_json_triggers_markdown_fallback(self, parser, mock_state):
        """Non-JSON content triggers markdown fallback parsing."""
        content = """# Findings

- Solar costs have dropped significantly, making it competitive with fossil fuels
- Battery storage technology is improving rapidly year over year

# Gaps

More data needed on grid integration costs.
"""

        result = parser._parse_analysis_response(content, mock_state)

        # Markdown fallback should extract bullet-point findings
        assert result["parse_method"] == "fallback_markdown"
        assert len(result["findings"]) >= 1

    def test_malformed_json_triggers_fallback(self, parser, mock_state):
        """Malformed JSON triggers markdown fallback."""
        content = '{"findings": [{"content": "partial...'

        result = parser._parse_analysis_response(content, mock_state)

        # Should attempt markdown fallback
        assert result["parse_method"] in ("fallback_markdown", None)

    def test_pydantic_validation_failure_falls_back_to_dict(self, parser, mock_state):
        """When Pydantic validation fails, dict extraction fallback is used."""
        content = json.dumps(
            {
                "findings": [
                    {
                        "content": "Valid finding",
                        "confidence": "high",
                    }
                ],
                "gaps": [
                    {
                        "description": "Valid gap",
                    }
                ],
                # quality_updates has invalid format that would fail strict Pydantic
                "quality_updates": [
                    {"source_id": "src-001", "quality": "excellent"}  # invalid quality value
                ],
            }
        )

        result = parser._parse_analysis_response(content, mock_state)

        # Should fall back to dict extraction since Pydantic would fail on "excellent"
        assert result["success"] is True
        assert result["parse_method"] in ("json", "fallback_dict")
        assert len(result["findings"]) == 1

    def test_json_in_code_block(self, parser, mock_state):
        """JSON wrapped in markdown code block is extracted."""
        content = """Here's my analysis:

```json
{
    "findings": [
        {"content": "Finding from code block", "confidence": "medium"}
    ],
    "gaps": [],
    "quality_updates": []
}
```"""

        result = parser._parse_analysis_response(content, mock_state)

        assert result["success"] is True
        assert result["parse_method"] == "json"
        assert result["findings"][0]["content"] == "Finding from code block"

    def test_empty_findings_returns_failure(self, parser, mock_state):
        """JSON with empty findings list returns success=False."""
        content = json.dumps({"findings": [], "gaps": [], "quality_updates": []})

        result = parser._parse_analysis_response(content, mock_state)

        assert result["success"] is False
        assert result["parse_method"] == "json"

    def test_confidence_mapping(self, parser, mock_state):
        """All confidence levels are mapped correctly."""
        content = json.dumps(
            {
                "findings": [
                    {"content": "Low confidence finding", "confidence": "low"},
                    {"content": "High confidence finding", "confidence": "high"},
                    {"content": "Speculation finding", "confidence": "speculation"},
                    {"content": "Confirmed finding", "confidence": "confirmed"},
                    {"content": "Unknown confidence", "confidence": "unknown_value"},
                ],
                "gaps": [],
                "quality_updates": [],
            }
        )

        result = parser._parse_analysis_response(content, mock_state)

        assert result["success"] is True
        findings = result["findings"]
        assert findings[0]["confidence"] == ConfidenceLevel.LOW
        assert findings[1]["confidence"] == ConfidenceLevel.HIGH
        assert findings[2]["confidence"] == ConfidenceLevel.SPECULATION
        assert findings[3]["confidence"] == ConfidenceLevel.CONFIRMED
        # Unknown maps to MEDIUM (default) via dict fallback
        assert findings[4]["confidence"] == ConfidenceLevel.MEDIUM

    def test_truncated_json_triggers_fallback(self, parser, mock_state):
        """Truncated JSON content (e.g., from token limit) triggers fallback."""
        content = '{"findings": [{"content": "This is a finding that got truncated mid-sen'

        result = parser._parse_analysis_response(content, mock_state)

        assert result["success"] is False

    def test_dict_fallback_filters_empty_content(self, parser, mock_state):
        """Dict fallback skips findings with empty content strings."""
        content = json.dumps(
            {
                "findings": [
                    {"content": "  ", "confidence": "high"},  # whitespace-only
                    {"content": "Real finding", "confidence": "medium"},
                ],
                "gaps": [],
                "quality_updates": [],
            }
        )

        result = parser._parse_analysis_response(content, mock_state)

        assert result["success"] is True
        # Pydantic validator strips whitespace and rejects empty -> fallback to dict
        # Dict fallback also strips and skips empty
        assert all(f["content"].strip() for f in result["findings"])
