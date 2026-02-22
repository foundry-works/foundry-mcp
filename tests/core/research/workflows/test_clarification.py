"""Unit tests for clarification phase parsing and integration.

Tests cover:
1. _parse_clarification_response() — valid JSON, needs_clarification true/false,
   malformed JSON, missing fields, empty response, edge cases
2. Integration: query → clarification → planning flow with constraints
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research.phases.clarification import (
    ClarificationPhaseMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    LLMCallResult,
)


# =============================================================================
# Helpers
# =============================================================================


class StubClarificationMixin(ClarificationPhaseMixin):
    """Concrete class that satisfies the mixin's requirements for testing."""

    def __init__(self) -> None:
        self.config = MagicMock()
        self.memory = MagicMock()
        self._audit_events: list[tuple[str, dict]] = []

    def _write_audit_event(self, state: Any, event: str, **kwargs: Any) -> None:
        self._audit_events.append((event, kwargs))

    def _check_cancellation(self, state: Any) -> None:
        pass


def _make_state(
    query: str = "How does AI work?",
    system_prompt: str | None = None,
) -> DeepResearchState:
    """Create a minimal DeepResearchState for testing."""
    return DeepResearchState(
        id="deepres-test-clarify",
        original_query=query,
        phase=DeepResearchPhase.CLARIFICATION,
        iteration=1,
        max_iterations=3,
        system_prompt=system_prompt,
    )


# =============================================================================
# Unit tests: _parse_clarification_response
# =============================================================================


class TestParseClarificationResponse:
    """Tests for ClarificationPhaseMixin._parse_clarification_response()."""

    def setup_method(self) -> None:
        self.mixin = StubClarificationMixin()

    def test_valid_json_needs_clarification_true(self) -> None:
        """Valid JSON with needs_clarification=true returns questions and constraints."""
        content = json.dumps(
            {
                "needs_clarification": True,
                "questions": [
                    "Are you interested in machine learning specifically?",
                    "What level of detail do you need?",
                ],
                "inferred_constraints": {
                    "scope": "machine learning and neural networks",
                    "depth": "overview",
                },
            }
        )
        result = self.mixin._parse_clarification_response(content)

        assert result["needs_clarification"] is True
        assert len(result["questions"]) == 2
        assert "machine learning" in result["questions"][0]
        assert result["inferred_constraints"]["scope"] == "machine learning and neural networks"
        assert result["inferred_constraints"]["depth"] == "overview"

    def test_valid_json_needs_clarification_false(self) -> None:
        """Valid JSON with needs_clarification=false proceeds without constraints."""
        content = json.dumps(
            {
                "needs_clarification": False,
                "questions": [],
                "inferred_constraints": {},
            }
        )
        result = self.mixin._parse_clarification_response(content)

        assert result["needs_clarification"] is False
        assert result["questions"] == []
        assert result["inferred_constraints"] == {}

    def test_empty_content_returns_defaults(self) -> None:
        """Empty string returns safe defaults (no clarification needed)."""
        result = self.mixin._parse_clarification_response("")

        assert result["needs_clarification"] is False
        assert result["questions"] == []
        assert result["inferred_constraints"] == {}

    def test_none_content_returns_defaults(self) -> None:
        """None content returns safe defaults (graceful handling at runtime)."""
        # The type hint says str but the code guards with `if not content`
        result = self.mixin._parse_clarification_response(None)  # type: ignore[arg-type]

        assert result["needs_clarification"] is False
        assert result["questions"] == []
        assert result["inferred_constraints"] == {}

    def test_malformed_json_returns_defaults(self) -> None:
        """Malformed JSON string returns safe defaults."""
        result = self.mixin._parse_clarification_response("{broken json!!}")

        assert result["needs_clarification"] is False
        assert result["questions"] == []
        assert result["inferred_constraints"] == {}

    def test_no_json_in_content_returns_defaults(self) -> None:
        """Plain text with no JSON returns safe defaults."""
        result = self.mixin._parse_clarification_response(
            "I think this query is fine, no changes needed."
        )

        assert result["needs_clarification"] is False
        assert result["questions"] == []
        assert result["inferred_constraints"] == {}

    def test_json_in_code_block(self) -> None:
        """JSON wrapped in markdown code block is extracted correctly."""
        content = """Here is my analysis:

```json
{
    "needs_clarification": true,
    "questions": ["What domain?"],
    "inferred_constraints": {"scope": "general AI overview"}
}
```
"""
        result = self.mixin._parse_clarification_response(content)

        assert result["needs_clarification"] is True
        assert result["questions"] == ["What domain?"]
        assert result["inferred_constraints"]["scope"] == "general AI overview"

    def test_json_with_surrounding_text(self) -> None:
        """JSON embedded in surrounding text is extracted correctly."""
        content = """After analyzing the query, here is my assessment:
{"needs_clarification": true, "questions": ["What scope?"], "inferred_constraints": {"depth": "detailed"}}
That concludes my analysis."""

        result = self.mixin._parse_clarification_response(content)

        assert result["needs_clarification"] is True
        assert result["questions"] == ["What scope?"]
        assert result["inferred_constraints"]["depth"] == "detailed"

    def test_missing_needs_clarification_defaults_false(self) -> None:
        """Missing needs_clarification key defaults to False."""
        content = json.dumps(
            {
                "questions": ["Something?"],
                "inferred_constraints": {"scope": "narrow"},
            }
        )
        result = self.mixin._parse_clarification_response(content)

        assert result["needs_clarification"] is False

    def test_missing_questions_defaults_empty_list(self) -> None:
        """Missing questions key defaults to empty list."""
        content = json.dumps(
            {
                "needs_clarification": True,
                "inferred_constraints": {"scope": "narrow"},
            }
        )
        result = self.mixin._parse_clarification_response(content)

        assert result["questions"] == []

    def test_missing_constraints_defaults_empty_dict(self) -> None:
        """Missing inferred_constraints key defaults to empty dict."""
        content = json.dumps(
            {
                "needs_clarification": True,
                "questions": ["What?"],
            }
        )
        result = self.mixin._parse_clarification_response(content)

        assert result["inferred_constraints"] == {}

    def test_questions_truncated_to_three(self) -> None:
        """More than 3 questions are truncated to first 3."""
        content = json.dumps(
            {
                "needs_clarification": True,
                "questions": ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"],
                "inferred_constraints": {},
            }
        )
        result = self.mixin._parse_clarification_response(content)

        assert len(result["questions"]) == 3
        assert result["questions"] == ["Q1?", "Q2?", "Q3?"]

    def test_empty_questions_filtered(self) -> None:
        """Empty/falsy question strings are filtered out (after truncation to 3)."""
        content = json.dumps(
            {
                "needs_clarification": True,
                "questions": ["Real question?", "", "Another?"],
                "inferred_constraints": {},
            }
        )
        result = self.mixin._parse_clarification_response(content)

        # Empty string is filtered, leaving only non-empty questions
        assert result["questions"] == ["Real question?", "Another?"]

    def test_empty_constraint_values_filtered(self) -> None:
        """Constraint values that are empty/falsy are filtered out."""
        content = json.dumps(
            {
                "needs_clarification": True,
                "questions": [],
                "inferred_constraints": {
                    "scope": "AI research",
                    "timeframe": "",
                    "domain": None,
                    "depth": "overview",
                },
            }
        )
        result = self.mixin._parse_clarification_response(content)

        assert "scope" in result["inferred_constraints"]
        assert "depth" in result["inferred_constraints"]
        assert "timeframe" not in result["inferred_constraints"]
        assert "domain" not in result["inferred_constraints"]

    def test_non_string_constraint_values_converted(self) -> None:
        """Non-string constraint values (int, float, bool) are converted to string."""
        content = json.dumps(
            {
                "needs_clarification": True,
                "questions": [],
                "inferred_constraints": {
                    "depth": "detailed",
                    "max_results": 10,
                    "include_images": True,
                    "score_threshold": 0.8,
                },
            }
        )
        result = self.mixin._parse_clarification_response(content)

        assert result["inferred_constraints"]["depth"] == "detailed"
        assert result["inferred_constraints"]["max_results"] == "10"
        assert result["inferred_constraints"]["include_images"] == "True"
        assert result["inferred_constraints"]["score_threshold"] == "0.8"

    def test_non_list_questions_ignored(self) -> None:
        """If questions is not a list, return empty list."""
        content = json.dumps(
            {
                "needs_clarification": True,
                "questions": "What is the scope?",
                "inferred_constraints": {},
            }
        )
        result = self.mixin._parse_clarification_response(content)

        assert result["questions"] == []

    def test_non_dict_constraints_ignored(self) -> None:
        """If inferred_constraints is not a dict, return empty dict."""
        content = json.dumps(
            {
                "needs_clarification": True,
                "questions": [],
                "inferred_constraints": ["scope=AI"],
            }
        )
        result = self.mixin._parse_clarification_response(content)

        assert result["inferred_constraints"] == {}

    def test_nested_dict_constraint_values_filtered(self) -> None:
        """Constraint values that are dicts/lists are filtered (only scalars kept)."""
        content = json.dumps(
            {
                "needs_clarification": True,
                "questions": [],
                "inferred_constraints": {
                    "scope": "narrow",
                    "nested_object": {"key": "value"},
                    "list_value": [1, 2, 3],
                },
            }
        )
        result = self.mixin._parse_clarification_response(content)

        assert result["inferred_constraints"] == {"scope": "narrow"}

    def test_needs_clarification_truthy_values(self) -> None:
        """Various truthy values for needs_clarification are coerced to True."""
        for truthy_val in [True, 1, "yes", "true"]:
            content = json.dumps(
                {
                    "needs_clarification": truthy_val,
                    "questions": [],
                    "inferred_constraints": {},
                }
            )
            result = self.mixin._parse_clarification_response(content)
            assert result["needs_clarification"] is True, f"Failed for {truthy_val!r}"

    def test_needs_clarification_falsy_values(self) -> None:
        """Falsy values for needs_clarification are coerced to False."""
        for falsy_val in [False, 0, "", None]:
            content = json.dumps(
                {
                    "needs_clarification": falsy_val,
                    "questions": [],
                    "inferred_constraints": {},
                }
            )
            result = self.mixin._parse_clarification_response(content)
            assert result["needs_clarification"] is False, f"Failed for {falsy_val!r}"

    def test_all_supported_constraint_keys(self) -> None:
        """All documented constraint keys are preserved."""
        constraints = {
            "scope": "machine learning",
            "timeframe": "2020-2024",
            "domain": "computer science",
            "depth": "comprehensive",
            "geographic_focus": "global",
        }
        content = json.dumps(
            {
                "needs_clarification": True,
                "questions": ["Any specifics?"],
                "inferred_constraints": constraints,
            }
        )
        result = self.mixin._parse_clarification_response(content)

        assert result["inferred_constraints"] == constraints


# =============================================================================
# Unit tests: prompt building
# =============================================================================


class TestClarificationPromptBuilding:
    """Tests for system and user prompt construction."""

    def setup_method(self) -> None:
        self.mixin = StubClarificationMixin()

    def test_system_prompt_contains_json_schema(self) -> None:
        """System prompt describes the expected JSON output format."""
        prompt = self.mixin._build_clarification_system_prompt()

        assert "needs_clarification" in prompt
        assert "questions" in prompt
        assert "inferred_constraints" in prompt
        assert "JSON" in prompt

    def test_system_prompt_lists_constraint_keys(self) -> None:
        """System prompt documents supported constraint keys."""
        prompt = self.mixin._build_clarification_system_prompt()

        for key in ["scope", "timeframe", "domain", "depth", "geographic_focus"]:
            assert key in prompt

    def test_user_prompt_contains_query(self) -> None:
        """User prompt includes the original research query."""
        state = _make_state(query="Compare PostgreSQL vs MySQL")
        prompt = self.mixin._build_clarification_user_prompt(state)

        assert "Compare PostgreSQL vs MySQL" in prompt

    def test_user_prompt_includes_system_context(self) -> None:
        """User prompt appends system_prompt context when available."""
        state = _make_state(
            query="How does caching work?",
            system_prompt="Focus on web application caching only",
        )
        prompt = self.mixin._build_clarification_user_prompt(state)

        assert "How does caching work?" in prompt
        assert "Focus on web application caching only" in prompt

    def test_user_prompt_no_system_context(self) -> None:
        """User prompt works without system_prompt."""
        state = _make_state(query="What is Rust?", system_prompt=None)
        prompt = self.mixin._build_clarification_user_prompt(state)

        assert "What is Rust?" in prompt
        assert "Additional context" not in prompt


# =============================================================================
# Integration test: clarification → planning flow
# =============================================================================


class TestClarificationToPlanningFlow:
    """Integration tests for the full clarification → planning flow."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        config = MagicMock()
        config.default_provider = "test-provider"
        config.deep_research_allow_clarification = True
        config.deep_research_clarification_provider = None
        config.get_phase_timeout = MagicMock(return_value=60.0)
        config.get_phase_fallback_providers = MagicMock(return_value=[])
        config.deep_research_max_retries = 2
        config.deep_research_retry_delay = 1.0
        return config

    @pytest.fixture
    def mock_memory(self) -> MagicMock:
        memory = MagicMock()
        memory.save_deep_research = MagicMock()
        return memory

    @pytest.mark.asyncio
    async def test_clarification_stores_constraints_in_state(
        self,
        mock_config: MagicMock,
        mock_memory: MagicMock,
    ) -> None:
        """Clarification phase stores inferred constraints in state."""
        state = _make_state(query="How does AI work?")

        mixin = StubClarificationMixin()
        mixin.config = mock_config
        mixin.memory = mock_memory

        llm_response = json.dumps(
            {
                "needs_clarification": True,
                "questions": ["What aspect of AI?"],
                "inferred_constraints": {
                    "scope": "machine learning fundamentals",
                    "depth": "overview",
                },
            }
        )

        mock_result = MagicMock()
        mock_result.content = llm_response
        mock_result.provider_id = "test-provider"
        mock_result.model_used = "test-model"
        mock_result.tokens_used = 100
        mock_result.duration_ms = 500.0
        mock_result.input_tokens = 50
        mock_result.output_tokens = 50
        mock_result.cached_tokens = 0
        mock_result.success = True

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.clarification.execute_llm_call",
            return_value=LLMCallResult(result=mock_result, llm_call_duration_ms=500.0),
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.clarification.finalize_phase",
        ):
            result = await mixin._execute_clarification_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        assert result.success is True
        assert state.clarification_constraints == {
            "scope": "machine learning fundamentals",
            "depth": "overview",
        }
        assert state.metadata["clarification_questions"] == ["What aspect of AI?"]

    @pytest.mark.asyncio
    async def test_specific_query_produces_no_constraints(
        self,
        mock_config: MagicMock,
        mock_memory: MagicMock,
    ) -> None:
        """Specific query gets needs_clarification=false, no constraints stored."""
        state = _make_state(query="Compare PostgreSQL vs MySQL for OLTP workloads in 2024")

        mixin = StubClarificationMixin()
        mixin.config = mock_config
        mixin.memory = mock_memory

        llm_response = json.dumps(
            {
                "needs_clarification": False,
                "questions": [],
                "inferred_constraints": {},
            }
        )

        mock_result = MagicMock()
        mock_result.content = llm_response
        mock_result.provider_id = "test-provider"
        mock_result.model_used = "test-model"
        mock_result.tokens_used = 80
        mock_result.duration_ms = 300.0
        mock_result.input_tokens = 40
        mock_result.output_tokens = 40
        mock_result.cached_tokens = 0
        mock_result.success = True

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.clarification.execute_llm_call",
            return_value=LLMCallResult(result=mock_result, llm_call_duration_ms=300.0),
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.clarification.finalize_phase",
        ):
            result = await mixin._execute_clarification_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        assert result.success is True
        assert state.clarification_constraints == {}
        assert "clarification_questions" not in state.metadata

    @pytest.mark.asyncio
    async def test_llm_error_returns_failure(
        self,
        mock_config: MagicMock,
        mock_memory: MagicMock,
    ) -> None:
        """LLM call failure returns WorkflowResult(success=False)."""
        state = _make_state(query="Something")

        mixin = StubClarificationMixin()
        mixin.config = mock_config
        mixin.memory = mock_memory

        error_result = WorkflowResult(
            success=False,
            content="",
            error="Provider timeout",
        )

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases.clarification.execute_llm_call",
            return_value=error_result,
        ):
            result = await mixin._execute_clarification_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        assert result.success is False
        assert state.clarification_constraints == {}

    @pytest.mark.asyncio
    async def test_constraints_flow_to_planning_prompt(self) -> None:
        """Verify that clarification constraints are included in planning prompt.

        This tests the integration point: clarification sets constraints on state,
        and the planning phase reads them.
        """
        from foundry_mcp.core.research.workflows.deep_research.phases.planning import (
            PlanningPhaseMixin,
        )

        state = _make_state(query="How does AI work?")
        state.clarification_constraints = {
            "scope": "deep learning and neural networks",
            "depth": "comprehensive",
            "timeframe": "2020-2025",
        }
        state.max_sub_queries = 5

        # Create a stub that has the planning mixin's prompt method
        class StubPlanning(PlanningPhaseMixin):
            def __init__(self) -> None:
                self.config = MagicMock()

        stub = StubPlanning()
        prompt = stub._build_planning_user_prompt(state)

        assert "deep learning and neural networks" in prompt
        assert "comprehensive" in prompt
        assert "2020-2025" in prompt
        assert "Clarification constraints" in prompt

    @pytest.mark.asyncio
    async def test_no_constraints_no_planning_section(self) -> None:
        """When no constraints are set, planning prompt omits constraint section."""
        from foundry_mcp.core.research.workflows.deep_research.phases.planning import (
            PlanningPhaseMixin,
        )

        state = _make_state(query="Compare PostgreSQL vs MySQL for OLTP workloads")
        state.clarification_constraints = {}
        state.max_sub_queries = 5

        class StubPlanning(PlanningPhaseMixin):
            def __init__(self) -> None:
                self.config = MagicMock()

        stub = StubPlanning()
        prompt = stub._build_planning_user_prompt(state)

        assert "Clarification constraints" not in prompt
