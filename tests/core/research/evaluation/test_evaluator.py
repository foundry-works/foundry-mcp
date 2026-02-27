"""Tests for the core evaluation logic (prompt building, parsing, LLM integration).

Tests cover:
1. Evaluation prompt construction
2. Response parsing (valid JSON, missing fields, malformed)
3. LLM-as-judge integration via evaluate_report()
4. Consistent scores for identical inputs (low variance proxy)
5. Poor reports score lower than comprehensive reports (qualitative)
6. Dimensions produce independent scores
7. Evaluation results persisted in session metadata
8. Config integration (provider/model resolution, role chain)
9. Action handler dispatch
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.core.research.evaluation.dimensions import DIMENSIONS
from foundry_mcp.core.research.evaluation.evaluator import (
    _build_evaluation_prompt,
    _parse_evaluation_response,
    evaluate_report,
)
from foundry_mcp.core.research.evaluation.scoring import EvaluationResult
from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.models.sources import SourceType
from foundry_mcp.core.research.workflows.base import WorkflowResult

# =============================================================================
# Helpers
# =============================================================================


def _make_state(
    report: str = "A comprehensive report about AI safety.",
    num_sources: int = 3,
) -> DeepResearchState:
    """Create a DeepResearchState with a report for evaluation tests."""
    state = DeepResearchState(
        id="deepres-eval-test",
        original_query="What are the key challenges in AI safety?",
        phase=DeepResearchPhase.SYNTHESIS,
        iteration=1,
        max_iterations=3,
    )
    state.report = report
    state.research_brief = "Investigating AI safety challenges"

    for i in range(num_sources):
        state.add_source(
            title=f"Source {i}: AI Safety Paper {i}",
            url=f"https://example.com/paper-{i}",
            source_type=SourceType.WEB,
            snippet=f"Key finding {i} about AI safety.",
        )

    return state


def _make_valid_eval_response(scores: dict[str, int] | None = None) -> str:
    """Build a valid JSON evaluation response."""
    if scores is None:
        # Build defaults dynamically from DIMENSIONS to prevent drift
        default_pattern = [4, 3, 4, 3, 4, 5, 4, 3]
        scores = {d.name: default_pattern[i % len(default_pattern)] for i, d in enumerate(DIMENSIONS)}
    response = {
        "scores": {name: {"score": score, "rationale": f"Score {score} for {name}"} for name, score in scores.items()}
    }
    return json.dumps(response)


def _make_workflow_mock(response_content: str, success: bool = True) -> MagicMock:
    """Create a mock workflow that returns the given LLM response."""
    workflow = MagicMock()
    workflow.config = MagicMock()
    workflow.config.deep_research_evaluation_provider = None
    workflow.config.deep_research_evaluation_model = None
    workflow.config.default_provider = "gemini"
    workflow.config.get_phase_fallback_providers.return_value = []
    workflow.config.deep_research_max_retries = 2
    workflow.config.deep_research_retry_delay = 0.1
    workflow.memory = MagicMock()
    workflow._write_audit_event = MagicMock()
    workflow._execute_provider_async = AsyncMock(
        return_value=WorkflowResult(
            success=success,
            content=response_content,
            provider_id="gemini",
            model_used="gemini-2.5-pro",
            duration_ms=1500.0,
            input_tokens=2000,
            output_tokens=500,
            tokens_used=2500,
        )
    )
    return workflow


# =============================================================================
# Prompt Construction
# =============================================================================


class TestBuildEvaluationPrompt:
    """Verify evaluation prompt structure and content."""

    def test_includes_query(self):
        prompt = _build_evaluation_prompt(
            query="AI safety challenges",
            report="Report content",
            sources=[],
        )
        assert "AI safety challenges" in prompt

    def test_includes_report(self):
        prompt = _build_evaluation_prompt(
            query="test",
            report="Detailed report about AI safety",
            sources=[],
        )
        assert "Detailed report about AI safety" in prompt

    def test_includes_sources(self):
        sources = [
            {"title": "Paper A", "url": "https://a.com", "quality": "HIGH"},
            {"title": "Paper B", "url": "https://b.com", "quality": "MEDIUM"},
        ]
        prompt = _build_evaluation_prompt(query="test", report="report", sources=sources)
        assert "Paper A" in prompt
        assert "https://a.com" in prompt
        assert "HIGH" in prompt
        assert "Paper B" in prompt

    def test_includes_all_dimension_rubrics(self):
        prompt = _build_evaluation_prompt(query="test", report="report", sources=[])
        for dim in DIMENSIONS:
            assert dim.display_name in prompt
            assert dim.name in prompt

    def test_includes_json_format_instruction(self):
        prompt = _build_evaluation_prompt(query="test", report="report", sources=[])
        assert "JSON" in prompt
        assert '"scores"' in prompt

    def test_truncates_long_reports(self):
        long_report = "x" * 100_000
        prompt = _build_evaluation_prompt(query="test", report=long_report, sources=[])
        assert "truncated" in prompt.lower()
        assert len(prompt) < 100_000

    def test_limits_source_count(self):
        sources = [{"title": f"Source {i}", "url": f"https://{i}.com"} for i in range(50)]
        prompt = _build_evaluation_prompt(query="test", report="report", sources=sources)
        # Should only include up to _MAX_SOURCES_IN_PROMPT (30)
        assert "Source 29" in prompt
        assert "Source 35" not in prompt

    def test_handles_empty_sources(self):
        prompt = _build_evaluation_prompt(query="test", report="report", sources=[])
        assert "no sources listed" in prompt.lower()


# =============================================================================
# Response Parsing
# =============================================================================


class TestParseEvaluationResponse:
    """Verify parsing of LLM evaluation JSON responses."""

    def test_valid_complete_response(self):
        content = _make_valid_eval_response()
        result = _parse_evaluation_response(content)

        assert isinstance(result, EvaluationResult)
        assert len(result.dimension_scores) == 8
        assert 0.0 <= result.composite_score <= 1.0

    def test_extracts_all_dimension_scores(self):
        scores = {
            "depth": 5,
            "source_quality": 4,
            "analytical_rigor": 3,
            "completeness": 2,
            "groundedness": 1,
            "structure": 4,
            "practical_value": 3,
            "balance": 4,
        }
        content = _make_valid_eval_response(scores)
        result = _parse_evaluation_response(content)

        score_map = {ds.name: ds for ds in result.dimension_scores}
        assert score_map["depth"].raw_score == 5
        assert score_map["depth"].normalized_score == 1.0
        assert score_map["source_quality"].raw_score == 4
        assert score_map["completeness"].raw_score == 2
        assert score_map["groundedness"].raw_score == 1
        assert score_map["groundedness"].normalized_score == 0.0

    def test_dimensions_produce_independent_scores(self):
        """Different raw scores should produce different normalized values."""
        scores = {
            "depth": 5,
            "source_quality": 1,
            "analytical_rigor": 3,
            "completeness": 4,
            "groundedness": 2,
            "structure": 5,
            "practical_value": 3,
            "balance": 4,
        }
        content = _make_valid_eval_response(scores)
        result = _parse_evaluation_response(content)

        normalized = [ds.normalized_score for ds in result.dimension_scores]
        # Not all identical
        assert len(set(normalized)) > 1

    def test_extracts_rationales(self):
        content = _make_valid_eval_response({"depth": 4})
        result = _parse_evaluation_response(content)
        depth_score = next(ds for ds in result.dimension_scores if ds.name == "depth")
        assert "Score 4 for depth" in depth_score.rationale

    def test_missing_dimension_gets_neutral_score(self):
        """Missing dimensions should get score 3 (neutral)."""
        content = json.dumps({"scores": {"depth": {"score": 5, "rationale": "Great"}}})
        result = _parse_evaluation_response(content)

        score_map = {ds.name: ds for ds in result.dimension_scores}
        assert score_map["depth"].raw_score == 5
        # Missing dimensions default to 3
        assert score_map["source_quality"].raw_score == 3
        assert score_map["source_quality"].rationale == "Not evaluated"

    def test_handles_json_in_code_block(self):
        content = "```json\n" + _make_valid_eval_response() + "\n```"
        result = _parse_evaluation_response(content)
        assert len(result.dimension_scores) == 8

    def test_handles_json_with_surrounding_text(self):
        content = "Here is my evaluation:\n" + _make_valid_eval_response() + "\nThat's my assessment."
        result = _parse_evaluation_response(content)
        assert len(result.dimension_scores) == 8

    def test_no_json_raises_value_error(self):
        with pytest.raises(ValueError, match="No JSON"):
            _parse_evaluation_response("This is just text with no JSON")

    def test_invalid_json_raises(self):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            _parse_evaluation_response('{"scores": broken}')

    def test_missing_scores_key_raises(self):
        with pytest.raises(ValueError, match="scores"):
            _parse_evaluation_response('{"not_scores": {}}')

    def test_composite_normalizes_to_0_1(self):
        """Composite score must always be in [0, 1] range."""
        # All 1s
        content = _make_valid_eval_response({d.name: 1 for d in DIMENSIONS})
        result = _parse_evaluation_response(content)
        assert result.composite_score == pytest.approx(0.0)

        # All 5s
        content = _make_valid_eval_response({d.name: 5 for d in DIMENSIONS})
        result = _parse_evaluation_response(content)
        assert result.composite_score == pytest.approx(1.0)

        # Mixed
        content = _make_valid_eval_response({d.name: 3 for d in DIMENSIONS})
        result = _parse_evaluation_response(content)
        assert 0.0 <= result.composite_score <= 1.0

    def test_non_integer_score_defaults_to_3(self):
        content = json.dumps(
            {
                "scores": {
                    "depth": {"score": "high", "rationale": "non-numeric"},
                    "source_quality": {"score": 4, "rationale": "ok"},
                }
            }
        )
        result = _parse_evaluation_response(content)
        depth = next(ds for ds in result.dimension_scores if ds.name == "depth")
        assert depth.raw_score == 3  # default for non-numeric


# =============================================================================
# LLM Integration (evaluate_report)
# =============================================================================


class TestEvaluateReport:
    """Integration tests for evaluate_report() with mocked LLM."""

    @pytest.mark.asyncio
    async def test_successful_evaluation(self):
        state = _make_state()
        response = _make_valid_eval_response()
        workflow = _make_workflow_mock(response)

        result = await evaluate_report(
            workflow=workflow,
            state=state,
            provider_id=None,
            model=None,
            timeout=360.0,
        )

        assert isinstance(result, EvaluationResult)
        assert len(result.dimension_scores) == 8
        assert 0.0 <= result.composite_score <= 1.0

    @pytest.mark.asyncio
    async def test_stores_in_state_metadata(self):
        state = _make_state()
        response = _make_valid_eval_response()
        workflow = _make_workflow_mock(response)

        result = await evaluate_report(
            workflow=workflow,
            state=state,
            provider_id=None,
            model=None,
            timeout=360.0,
        )

        assert isinstance(result, EvaluationResult)
        assert "evaluation" in state.metadata
        eval_data = state.metadata["evaluation"]
        assert eval_data["composite_score"] == result.composite_score
        assert len(eval_data["dimension_scores"]) == 8

    @pytest.mark.asyncio
    async def test_persists_state(self):
        state = _make_state()
        response = _make_valid_eval_response()
        workflow = _make_workflow_mock(response)

        await evaluate_report(
            workflow=workflow,
            state=state,
            provider_id=None,
            model=None,
            timeout=360.0,
        )

        workflow.memory.save_deep_research.assert_called()

    @pytest.mark.asyncio
    async def test_emits_audit_events(self):
        state = _make_state()
        response = _make_valid_eval_response()
        workflow = _make_workflow_mock(response)

        await evaluate_report(
            workflow=workflow,
            state=state,
            provider_id=None,
            model=None,
            timeout=360.0,
        )

        audit_calls = workflow._write_audit_event.call_args_list
        event_names = [call[0][1] for call in audit_calls]
        assert "evaluation.started" in event_names
        assert "evaluation.completed" in event_names

    @pytest.mark.asyncio
    async def test_no_report_returns_error(self):
        state = _make_state(report="")
        state.report = None
        workflow = _make_workflow_mock("")

        result = await evaluate_report(
            workflow=workflow,
            state=state,
            provider_id=None,
            model=None,
            timeout=360.0,
        )

        assert isinstance(result, WorkflowResult)
        assert not result.success
        assert "report" in result.error.lower()

    @pytest.mark.asyncio
    async def test_llm_failure_returns_error(self):
        state = _make_state()
        workflow = _make_workflow_mock("", success=False)
        workflow._execute_provider_async.return_value = WorkflowResult(
            success=False,
            content="",
            error="Provider timeout",
        )

        result = await evaluate_report(
            workflow=workflow,
            state=state,
            provider_id=None,
            model=None,
            timeout=360.0,
        )

        assert isinstance(result, WorkflowResult)
        assert not result.success

    @pytest.mark.asyncio
    async def test_parse_failure_returns_error(self):
        state = _make_state()
        workflow = _make_workflow_mock("This is not JSON at all, no braces anywhere")

        result = await evaluate_report(
            workflow=workflow,
            state=state,
            provider_id=None,
            model=None,
            timeout=360.0,
        )

        assert isinstance(result, WorkflowResult)
        assert not result.success
        assert "parse" in result.error.lower()

    @pytest.mark.asyncio
    async def test_evaluation_metadata_includes_provider(self):
        state = _make_state()
        response = _make_valid_eval_response()
        workflow = _make_workflow_mock(response)

        result = await evaluate_report(
            workflow=workflow,
            state=state,
            provider_id=None,
            model=None,
            timeout=360.0,
        )

        assert isinstance(result, EvaluationResult)
        assert result.metadata["provider_id"] == "gemini"
        assert result.metadata["model_used"] == "gemini-2.5-pro"
        assert result.metadata["research_id"] == "deepres-eval-test"

    @pytest.mark.asyncio
    async def test_evaluation_uses_low_temperature(self):
        """Evaluation should use low temperature for consistency."""
        state = _make_state()
        response = _make_valid_eval_response()
        workflow = _make_workflow_mock(response)

        await evaluate_report(
            workflow=workflow,
            state=state,
            provider_id=None,
            model=None,
            timeout=360.0,
        )

        # Check the temperature passed to the provider
        call_kwargs = workflow._execute_provider_async.call_args
        assert call_kwargs[1].get("temperature", call_kwargs.kwargs.get("temperature")) == 0.1


# =============================================================================
# Config Integration
# =============================================================================


class TestConfigIntegration:
    """Verify evaluation config keys and role resolution."""

    def test_evaluation_config_fields_exist(self):
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig()
        assert hasattr(config, "deep_research_evaluation_provider")
        assert hasattr(config, "deep_research_evaluation_model")
        assert hasattr(config, "deep_research_evaluation_timeout")
        assert config.deep_research_evaluation_provider is None
        assert config.deep_research_evaluation_model is None
        assert config.deep_research_evaluation_timeout == 360.0

    def test_evaluation_role_in_resolution_chain(self):
        from foundry_mcp.config.research import ResearchConfig

        assert "evaluation" in ResearchConfig._ROLE_RESOLUTION_CHAIN
        chain = ResearchConfig._ROLE_RESOLUTION_CHAIN["evaluation"]
        assert "evaluation" in chain
        assert "research" in chain

    def test_role_resolution_falls_back_to_research(self):
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig(
            deep_research_research_provider="claude",
            deep_research_research_model="opus",
        )
        provider, model = config.resolve_model_for_role("evaluation")
        # Should fall back to research provider/model
        assert provider == "claude"
        assert model == "opus"

    def test_role_resolution_prefers_evaluation_specific(self):
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig(
            deep_research_evaluation_provider="gemini",
            deep_research_evaluation_model="pro",
            deep_research_research_provider="claude",
        )
        provider, model = config.resolve_model_for_role("evaluation")
        assert provider == "gemini"
        assert model == "pro"

    def test_from_toml_dict_parses_evaluation_keys(self):
        from foundry_mcp.config.research import ResearchConfig

        data = {
            "deep_research_evaluation_provider": "openai",
            "deep_research_evaluation_model": "gpt-4o",
            "deep_research_evaluation_timeout": 120.0,
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_evaluation_provider == "openai"
        assert config.deep_research_evaluation_model == "gpt-4o"
        assert config.deep_research_evaluation_timeout == 120.0

    def test_deep_research_sub_config_includes_evaluation(self):
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig(
            deep_research_evaluation_provider="gemini",
            deep_research_evaluation_model="flash",
            deep_research_evaluation_timeout=120.0,
        )
        sub = config.deep_research_config
        assert sub.evaluation_provider == "gemini"
        assert sub.evaluation_model == "flash"
        assert sub.evaluation_timeout == 120.0


# =============================================================================
# Action Handler
# =============================================================================


class TestActionHandler:
    """Verify the evaluate action dispatches correctly via core.py."""

    def test_core_dispatch_unknown_action_includes_evaluate(self):
        """The error message for unknown actions should mention 'evaluate'."""
        from foundry_mcp.core.research.workflows.deep_research.core import (
            DeepResearchWorkflow,
        )

        config = MagicMock()
        config.deep_research_mode = "general"
        memory = MagicMock()

        workflow = DeepResearchWorkflow(config, memory)
        result = workflow.execute(action="nonexistent")
        assert result.error and "evaluate" in result.error

    def test_evaluate_action_requires_research_id(self):
        """Evaluate action without research_id should fail."""
        from foundry_mcp.core.research.workflows.deep_research.action_handlers import (
            ActionHandlersMixin,
        )

        handler = MagicMock()
        handler._evaluate_research = ActionHandlersMixin._evaluate_research.__get__(handler)

        result = handler._evaluate_research(research_id=None)
        assert not result.success
        assert result.error and "research_id" in result.error

    def test_evaluate_action_requires_report(self):
        """Evaluate action on session without report should fail."""
        from foundry_mcp.core.research.workflows.deep_research.action_handlers import (
            ActionHandlersMixin,
        )

        state = _make_state()
        state.report = None

        handler = MagicMock()
        handler._evaluate_research = ActionHandlersMixin._evaluate_research.__get__(handler)
        handler.memory.load_deep_research.return_value = state

        result = handler._evaluate_research(research_id="deepres-test")
        assert not result.success
        assert result.error and "report" in result.error.lower()

    def test_evaluate_action_session_not_found(self):
        """Evaluate action on nonexistent session should fail."""
        from foundry_mcp.core.research.workflows.deep_research.action_handlers import (
            ActionHandlersMixin,
        )

        handler = MagicMock()
        handler._evaluate_research = ActionHandlersMixin._evaluate_research.__get__(handler)
        handler.memory.load_deep_research.return_value = None

        result = handler._evaluate_research(research_id="nonexistent")
        assert not result.success
        assert result.error and "not found" in result.error


# =============================================================================
# Quality Differentiation (Qualitative Tests)
# =============================================================================


class TestQualityDifferentiation:
    """Verify the scoring system can differentiate quality levels."""

    def test_high_scores_produce_higher_composite_than_low(self):
        """Reports scored all-4s should composite higher than all-2s."""
        high_content = _make_valid_eval_response({d.name: 4 for d in DIMENSIONS})
        low_content = _make_valid_eval_response({d.name: 2 for d in DIMENSIONS})

        high_result = _parse_evaluation_response(high_content)
        low_result = _parse_evaluation_response(low_content)

        assert high_result.composite_score > low_result.composite_score

    def test_variance_zero_for_uniform_scores(self):
        content = _make_valid_eval_response({d.name: 3 for d in DIMENSIONS})
        result = _parse_evaluation_response(content)
        assert result.score_variance == pytest.approx(0.0)

    def test_variance_nonzero_for_varied_scores(self):
        scores = {
            "depth": 5,
            "source_quality": 1,
            "analytical_rigor": 3,
            "completeness": 4,
            "groundedness": 2,
            "structure": 5,
            "practical_value": 3,
            "balance": 4,
        }
        content = _make_valid_eval_response(scores)
        result = _parse_evaluation_response(content)
        assert result.score_variance > 0.0


# =============================================================================
# Phase 3c: Raw notes as groundedness context
# =============================================================================


class TestRawNotesGroundedness:
    """Tests for PLAN Phase 3c: raw notes as ground-truth context for groundedness."""

    def test_raw_notes_included_in_evaluation_prompt(self):
        """When raw_notes are provided, they appear in the evaluation prompt."""
        raw_notes = [
            "Source A reports pricing at $10/month",
            "Source B confirms feature X is available",
        ]
        prompt = _build_evaluation_prompt(
            query="test query",
            report="A report about pricing and features",
            sources=[],
            raw_notes=raw_notes,
        )
        assert "Raw Research Evidence" in prompt
        assert "ground truth" in prompt.lower()
        assert "pricing at $10/month" in prompt
        assert "feature X is available" in prompt

    def test_raw_notes_absent_when_none(self):
        """When raw_notes is None, no raw evidence section appears."""
        prompt = _build_evaluation_prompt(
            query="test query",
            report="A report",
            sources=[],
            raw_notes=None,
        )
        assert "Raw Research Evidence" not in prompt

    def test_raw_notes_absent_when_empty_list(self):
        """When raw_notes is an empty list, no raw evidence section appears."""
        prompt = _build_evaluation_prompt(
            query="test query",
            report="A report",
            sources=[],
            raw_notes=[],
        )
        assert "Raw Research Evidence" not in prompt

    def test_raw_notes_truncated_when_too_long(self):
        """Very long raw notes are truncated to _MAX_RAW_NOTES_CHARS."""
        long_note = "x" * 50_000
        prompt = _build_evaluation_prompt(
            query="test query",
            report="A report",
            sources=[],
            raw_notes=[long_note],
        )
        assert "Raw Research Evidence" in prompt
        assert "truncated" in prompt.lower()
        # Should not contain the full 50k characters
        assert len(prompt) < 100_000

    def test_raw_notes_references_groundedness_dimension(self):
        """Raw notes section explicitly references the Groundedness dimension."""
        prompt = _build_evaluation_prompt(
            query="test",
            report="report",
            sources=[],
            raw_notes=["some evidence"],
        )
        assert "Groundedness" in prompt
        # Check the section instructs the judge to use notes for groundedness
        evidence_section_start = prompt.index("Raw Research Evidence")
        evidence_section = prompt[evidence_section_start : evidence_section_start + 500]
        assert "groundedness" in evidence_section.lower()

    @pytest.mark.asyncio
    async def test_evaluate_report_passes_raw_notes(self):
        """evaluate_report() passes state.raw_notes to the prompt builder."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            LLMCallResult,
        )

        state = _make_state()
        state.raw_notes = ["Evidence from researcher 1", "Evidence from researcher 2"]
        response = _make_valid_eval_response()
        workflow = _make_workflow_mock(response)

        captured_prompts: list[str] = []

        async def mock_execute_llm_call(**kwargs):
            captured_prompts.append(kwargs.get("user_prompt", ""))
            return LLMCallResult(
                result=WorkflowResult(
                    success=True,
                    content=response,
                    provider_id="gemini",
                    model_used="gemini-2.5-pro",
                    duration_ms=1500.0,
                    input_tokens=2000,
                    output_tokens=500,
                    tokens_used=2500,
                ),
                llm_call_duration_ms=1500.0,
            )

        with patch(
            "foundry_mcp.core.research.evaluation.evaluator.execute_llm_call",
            side_effect=mock_execute_llm_call,
        ):
            result = await evaluate_report(
                workflow=workflow,
                state=state,
                provider_id=None,
                model=None,
                timeout=360.0,
            )

        assert isinstance(result, EvaluationResult)
        assert len(captured_prompts) == 1
        user_prompt = captured_prompts[0]
        # The prompt should contain the raw notes evidence
        assert "Evidence from researcher 1" in user_prompt
        assert "Evidence from researcher 2" in user_prompt

    @pytest.mark.asyncio
    async def test_evaluate_report_no_raw_notes(self):
        """evaluate_report() works correctly when state has no raw_notes."""
        state = _make_state()
        assert state.raw_notes == []
        response = _make_valid_eval_response()
        workflow = _make_workflow_mock(response)

        result = await evaluate_report(
            workflow=workflow,
            state=state,
            provider_id=None,
            model=None,
            timeout=360.0,
        )

        assert isinstance(result, EvaluationResult)
