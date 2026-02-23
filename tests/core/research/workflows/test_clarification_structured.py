"""Unit tests for Phase 4: Structured Clarification Gate.

Tests cover:
1. ClarificationDecision dataclass — creation, defaults, serialization
2. parse_clarification_decision() — JSON extraction, fallback regex, edge cases
3. _strict_parse_clarification() — strict validation with raises
4. execute_structured_llm_call() — retry on parse failure, fallback
5. Updated clarification phase integration — verification stored in state,
   audit events, need_clarification flows
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
from foundry_mcp.core.research.workflows.deep_research._helpers import (
    ClarificationDecision,
    parse_clarification_decision,
)
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    LLMCallResult,
    StructuredLLMCallResult,
)
from foundry_mcp.core.research.workflows.deep_research.phases.clarification import (
    ClarificationPhaseMixin,
    _strict_parse_clarification,
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
        id="deepres-test-clarify-structured",
        original_query=query,
        phase=DeepResearchPhase.CLARIFICATION,
        iteration=1,
        max_iterations=3,
        system_prompt=system_prompt,
    )


# =============================================================================
# ClarificationDecision dataclass
# =============================================================================


class TestClarificationDecision:
    """Tests for the ClarificationDecision dataclass."""

    def test_default_values(self) -> None:
        """Default decision has no clarification needed and empty fields."""
        decision = ClarificationDecision()
        assert decision.need_clarification is False
        assert decision.question == ""
        assert decision.verification == ""

    def test_custom_values(self) -> None:
        """All fields can be set via constructor."""
        decision = ClarificationDecision(
            need_clarification=True,
            question="What aspect of AI?",
            verification="User wants to know about machine learning.",
        )
        assert decision.need_clarification is True
        assert decision.question == "What aspect of AI?"
        assert decision.verification == "User wants to know about machine learning."

    def test_to_dict(self) -> None:
        """Serialization includes all fields."""
        decision = ClarificationDecision(
            need_clarification=True,
            question="What scope?",
            verification="General AI overview",
        )
        d = decision.to_dict()
        assert d == {
            "need_clarification": True,
            "question": "What scope?",
            "verification": "General AI overview",
        }

    def test_to_dict_defaults(self) -> None:
        """Default decision serializes correctly."""
        d = ClarificationDecision().to_dict()
        assert d == {
            "need_clarification": False,
            "question": "",
            "verification": "",
        }


# =============================================================================
# parse_clarification_decision() — lenient parser
# =============================================================================


class TestParseClarificationDecision:
    """Tests for the lenient parse_clarification_decision() function."""

    def test_valid_json_need_clarification_true(self) -> None:
        text = json.dumps({
            "need_clarification": True,
            "question": "What specific area of AI interests you?",
            "verification": "User wants to learn about AI in general.",
        })
        decision = parse_clarification_decision(text)
        assert decision.need_clarification is True
        assert decision.question == "What specific area of AI interests you?"
        assert decision.verification == "User wants to learn about AI in general."

    def test_valid_json_need_clarification_false(self) -> None:
        text = json.dumps({
            "need_clarification": False,
            "question": "",
            "verification": "User wants to compare PostgreSQL vs MySQL for OLTP.",
        })
        decision = parse_clarification_decision(text)
        assert decision.need_clarification is False
        assert decision.question == ""
        assert decision.verification == "User wants to compare PostgreSQL vs MySQL for OLTP."

    def test_empty_content_returns_defaults(self) -> None:
        decision = parse_clarification_decision("")
        assert decision.need_clarification is False
        assert decision.question == ""
        assert decision.verification == ""

    def test_none_like_empty_returns_defaults(self) -> None:
        """None-ish content (empty string) returns safe defaults."""
        decision = parse_clarification_decision("")
        assert decision.need_clarification is False

    def test_json_in_code_block(self) -> None:
        text = """Here is my analysis:
```json
{
    "need_clarification": true,
    "question": "What domain?",
    "verification": "Broad query about AI."
}
```"""
        decision = parse_clarification_decision(text)
        assert decision.need_clarification is True
        assert decision.question == "What domain?"
        assert decision.verification == "Broad query about AI."

    def test_json_with_surrounding_text(self) -> None:
        text = 'After analysis: {"need_clarification": false, "question": "", "verification": "Query is specific"} end.'
        decision = parse_clarification_decision(text)
        assert decision.need_clarification is False
        assert decision.verification == "Query is specific"

    def test_malformed_json_fallback_regex(self) -> None:
        """When JSON is unparseable, regex extraction is used."""
        text = '"need_clarification": true, "question": "What scope?", "verification": "broad query"'
        decision = parse_clarification_decision(text)
        assert decision.need_clarification is True
        assert decision.question == "What scope?"
        assert decision.verification == "broad query"

    def test_no_json_no_regex_returns_defaults(self) -> None:
        text = "I think the query is fine, no changes needed."
        decision = parse_clarification_decision(text)
        assert decision.need_clarification is False
        # Fallback text is set when no fields extracted
        assert "fallback" in decision.verification.lower()

    def test_missing_fields_use_defaults(self) -> None:
        text = json.dumps({"need_clarification": True})
        decision = parse_clarification_decision(text)
        assert decision.need_clarification is True
        assert decision.question == ""
        assert decision.verification == ""

    def test_truthy_values_coerced(self) -> None:
        for val in [True, 1, "yes"]:
            text = json.dumps({
                "need_clarification": val,
                "question": "Q?",
                "verification": "V",
            })
            decision = parse_clarification_decision(text)
            assert decision.need_clarification is True, f"Failed for {val!r}"

    def test_falsy_values_coerced(self) -> None:
        for val in [False, 0, "", None]:
            text = json.dumps({
                "need_clarification": val,
                "question": "",
                "verification": "V",
            })
            decision = parse_clarification_decision(text)
            assert decision.need_clarification is False, f"Failed for {val!r}"


# =============================================================================
# _strict_parse_clarification() — strict parser (raises on failure)
# =============================================================================


class TestStrictParseClarification:
    """Tests for _strict_parse_clarification() used by execute_structured_llm_call."""

    def test_valid_json_returns_decision(self) -> None:
        content = json.dumps({
            "need_clarification": True,
            "question": "What?",
            "verification": "Broad query",
        })
        decision = _strict_parse_clarification(content)
        assert decision.need_clarification is True
        assert decision.question == "What?"
        assert decision.verification == "Broad query"

    def test_no_json_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="No JSON"):
            _strict_parse_clarification("Just plain text, no JSON here.")

    def test_malformed_json_raises(self) -> None:
        with pytest.raises((ValueError, json.JSONDecodeError)):
            _strict_parse_clarification("{broken json!!}")

    def test_missing_need_clarification_field_raises(self) -> None:
        content = json.dumps({"question": "Q?", "verification": "V"})
        with pytest.raises(ValueError, match="need_clarification"):
            _strict_parse_clarification(content)

    def test_empty_content_raises(self) -> None:
        with pytest.raises(ValueError, match="No JSON"):
            _strict_parse_clarification("")

    def test_json_in_code_block(self) -> None:
        content = '```json\n{"need_clarification": false, "question": "", "verification": "OK"}\n```'
        decision = _strict_parse_clarification(content)
        assert decision.need_clarification is False

    def test_missing_optional_fields_default(self) -> None:
        """question and verification default to empty string if missing."""
        content = json.dumps({"need_clarification": False})
        decision = _strict_parse_clarification(content)
        assert decision.question == ""
        assert decision.verification == ""


# =============================================================================
# Updated system prompt tests
# =============================================================================


class TestStructuredClarificationPrompts:
    """Tests for the updated system and user prompts."""

    def setup_method(self) -> None:
        self.mixin = StubClarificationMixin()

    def test_system_prompt_contains_structured_schema(self) -> None:
        """System prompt describes the new structured JSON schema."""
        prompt = self.mixin._build_clarification_system_prompt()

        assert "need_clarification" in prompt
        assert "question" in prompt
        assert "verification" in prompt
        assert "JSON" in prompt

    def test_system_prompt_describes_binary_decision(self) -> None:
        """System prompt frames the decision as binary (clarify vs. verify)."""
        prompt = self.mixin._build_clarification_system_prompt()

        assert "true" in prompt.lower()
        assert "false" in prompt.lower()
        # Should explain what to do for each branch
        assert "restate" in prompt.lower() or "restatement" in prompt.lower() or "understanding" in prompt.lower()

    def test_user_prompt_contains_query(self) -> None:
        state = _make_state(query="Compare PostgreSQL vs MySQL")
        prompt = self.mixin._build_clarification_user_prompt(state)
        assert "Compare PostgreSQL vs MySQL" in prompt

    def test_user_prompt_includes_system_context(self) -> None:
        state = _make_state(
            query="How does caching work?",
            system_prompt="Focus on web caching",
        )
        prompt = self.mixin._build_clarification_user_prompt(state)
        assert "Focus on web caching" in prompt


# =============================================================================
# Integration: structured clarification phase
# =============================================================================


class TestStructuredClarificationPhase:
    """Integration tests for the clarification phase with structured output."""

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

    def _make_structured_result(
        self,
        content: str,
        parsed: Any = None,
        parse_retries: int = 0,
    ) -> StructuredLLMCallResult:
        """Helper to create a StructuredLLMCallResult for testing."""
        mock_result = MagicMock()
        mock_result.content = content
        mock_result.provider_id = "test-provider"
        mock_result.model_used = "test-model"
        mock_result.tokens_used = 100
        mock_result.duration_ms = 500.0
        mock_result.input_tokens = 50
        mock_result.output_tokens = 50
        mock_result.cached_tokens = 0
        mock_result.success = True
        return StructuredLLMCallResult(
            result=mock_result,
            llm_call_duration_ms=500.0,
            parsed=parsed,
            parse_retries=parse_retries,
        )

    @pytest.mark.asyncio
    async def test_need_clarification_true_stores_question(
        self,
        mock_config: MagicMock,
        mock_memory: MagicMock,
    ) -> None:
        """When need_clarification=True, question is stored in state metadata."""
        state = _make_state(query="How does AI work?")

        mixin = StubClarificationMixin()
        mixin.config = mock_config
        mixin.memory = mock_memory

        decision = ClarificationDecision(
            need_clarification=True,
            question="What specific area of AI?",
            verification="User asking broadly about AI.",
        )

        # The LLM response content (used by legacy parser for backward-compat)
        content = json.dumps({
            "need_clarification": True,
            "question": "What specific area of AI?",
            "verification": "User asking broadly about AI.",
        })

        call_result = self._make_structured_result(content=content, parsed=decision)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.clarification.execute_structured_llm_call",
                return_value=call_result,
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.clarification.finalize_phase",
            ),
        ):
            result = await mixin._execute_clarification_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        assert result.success is True
        assert result.metadata["need_clarification"] is True
        assert state.metadata["clarification_questions"] == ["What specific area of AI?"]

    @pytest.mark.asyncio
    async def test_need_clarification_false_stores_verification(
        self,
        mock_config: MagicMock,
        mock_memory: MagicMock,
    ) -> None:
        """When need_clarification=False, verification is stored in state constraints."""
        state = _make_state(query="Compare PostgreSQL vs MySQL for OLTP in 2024")

        mixin = StubClarificationMixin()
        mixin.config = mock_config
        mixin.memory = mock_memory

        decision = ClarificationDecision(
            need_clarification=False,
            question="",
            verification="User wants a comparison of PostgreSQL and MySQL for high-write OLTP workloads, focusing on 2024 benchmarks.",
        )

        content = json.dumps(decision.to_dict())
        call_result = self._make_structured_result(content=content, parsed=decision)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.clarification.execute_structured_llm_call",
                return_value=call_result,
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.clarification.finalize_phase",
            ),
        ):
            result = await mixin._execute_clarification_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        assert result.success is True
        assert result.metadata["need_clarification"] is False
        assert state.clarification_constraints == {
            "verification": "User wants a comparison of PostgreSQL and MySQL for high-write OLTP workloads, focusing on 2024 benchmarks.",
        }

    @pytest.mark.asyncio
    async def test_verification_audit_event_logged(
        self,
        mock_config: MagicMock,
        mock_memory: MagicMock,
    ) -> None:
        """When need_clarification=False, a clarification_verification audit event is logged."""
        state = _make_state(query="Rust borrow checker")

        mixin = StubClarificationMixin()
        mixin.config = mock_config
        mixin.memory = mock_memory

        decision = ClarificationDecision(
            need_clarification=False,
            question="",
            verification="User wants to understand how the Rust borrow checker prevents data races.",
        )

        content = json.dumps(decision.to_dict())
        call_result = self._make_structured_result(content=content, parsed=decision)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.clarification.execute_structured_llm_call",
                return_value=call_result,
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.clarification.finalize_phase",
            ),
        ):
            await mixin._execute_clarification_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        # Find the verification audit event
        verification_events = [
            (event, data) for event, data in mixin._audit_events
            if event == "clarification_verification"
        ]
        assert len(verification_events) == 1
        event_data = verification_events[0][1]["data"]
        assert "Rust borrow checker" in event_data["verification"]

    @pytest.mark.asyncio
    async def test_no_verification_audit_when_clarification_needed(
        self,
        mock_config: MagicMock,
        mock_memory: MagicMock,
    ) -> None:
        """When need_clarification=True, no verification audit event is logged."""
        state = _make_state(query="How does AI work?")

        mixin = StubClarificationMixin()
        mixin.config = mock_config
        mixin.memory = mock_memory

        decision = ClarificationDecision(
            need_clarification=True,
            question="What aspect?",
            verification="Broad AI query.",
        )

        content = json.dumps(decision.to_dict())
        call_result = self._make_structured_result(content=content, parsed=decision)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.clarification.execute_structured_llm_call",
                return_value=call_result,
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.clarification.finalize_phase",
            ),
        ):
            await mixin._execute_clarification_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        verification_events = [
            (event, data) for event, data in mixin._audit_events
            if event == "clarification_verification"
        ]
        assert len(verification_events) == 0

    @pytest.mark.asyncio
    async def test_llm_error_returns_failure(
        self,
        mock_config: MagicMock,
        mock_memory: MagicMock,
    ) -> None:
        """LLM-level error propagates as WorkflowResult(success=False)."""
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
            "foundry_mcp.core.research.workflows.deep_research.phases.clarification.execute_structured_llm_call",
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
    async def test_parse_failure_fallback_no_clarification(
        self,
        mock_config: MagicMock,
        mock_memory: MagicMock,
    ) -> None:
        """When all parse retries fail (parsed=None), fallback to no clarification."""
        state = _make_state(query="Some query")

        mixin = StubClarificationMixin()
        mixin.config = mock_config
        mixin.memory = mock_memory

        # parsed=None means all structured parse attempts failed
        # Content is garbage — lenient parser also returns defaults (no clarification)
        call_result = self._make_structured_result(
            content="not json at all",
            parsed=None,
            parse_retries=3,
        )

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.clarification.execute_structured_llm_call",
                return_value=call_result,
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.clarification.finalize_phase",
            ),
        ):
            result = await mixin._execute_clarification_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        assert result.success is True
        assert result.metadata["need_clarification"] is False
        assert result.metadata["parse_retries"] == 3

    @pytest.mark.asyncio
    async def test_parse_retries_tracked_in_metadata(
        self,
        mock_config: MagicMock,
        mock_memory: MagicMock,
    ) -> None:
        """Parse retry count is included in result metadata."""
        state = _make_state(query="Test query")

        mixin = StubClarificationMixin()
        mixin.config = mock_config
        mixin.memory = mock_memory

        decision = ClarificationDecision(
            need_clarification=False,
            question="",
            verification="Understood the query.",
        )

        content = json.dumps(decision.to_dict())
        call_result = self._make_structured_result(
            content=content,
            parsed=decision,
            parse_retries=2,  # Succeeded on 3rd attempt
        )

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.clarification.execute_structured_llm_call",
                return_value=call_result,
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.clarification.finalize_phase",
            ),
        ):
            result = await mixin._execute_clarification_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        assert result.metadata["parse_retries"] == 2

    @pytest.mark.asyncio
    async def test_clarification_result_audit_event(
        self,
        mock_config: MagicMock,
        mock_memory: MagicMock,
    ) -> None:
        """clarification_result audit event includes structured decision fields."""
        state = _make_state(query="Test")

        mixin = StubClarificationMixin()
        mixin.config = mock_config
        mixin.memory = mock_memory

        decision = ClarificationDecision(
            need_clarification=True,
            question="What scope?",
            verification="Broad query.",
        )

        content = json.dumps(decision.to_dict())
        call_result = self._make_structured_result(content=content, parsed=decision)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.clarification.execute_structured_llm_call",
                return_value=call_result,
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.clarification.finalize_phase",
            ),
        ):
            await mixin._execute_clarification_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        result_events = [
            (event, data) for event, data in mixin._audit_events
            if event == "clarification_result"
        ]
        assert len(result_events) == 1
        event_data = result_events[0][1]["data"]
        assert event_data["need_clarification"] is True
        assert event_data["question"] == "What scope?"
        assert event_data["verification"] == "Broad query."
        assert "parse_retries" in event_data

    @pytest.mark.asyncio
    async def test_empty_verification_not_stored(
        self,
        mock_config: MagicMock,
        mock_memory: MagicMock,
    ) -> None:
        """When verification is empty, constraints are not set."""
        state = _make_state(query="Specific query")

        mixin = StubClarificationMixin()
        mixin.config = mock_config
        mixin.memory = mock_memory

        decision = ClarificationDecision(
            need_clarification=False,
            question="",
            verification="",
        )

        content = json.dumps(decision.to_dict())
        call_result = self._make_structured_result(content=content, parsed=decision)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.clarification.execute_structured_llm_call",
                return_value=call_result,
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.clarification.finalize_phase",
            ),
        ):
            result = await mixin._execute_clarification_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        assert result.success is True
        assert state.clarification_constraints == {}


# =============================================================================
# execute_structured_llm_call() retry logic
# =============================================================================


class TestExecuteStructuredLLMCallRetry:
    """Tests for the retry logic in execute_structured_llm_call."""

    @pytest.fixture
    def mock_workflow(self) -> MagicMock:
        workflow = MagicMock()
        workflow.config = MagicMock()
        workflow.config.get_phase_fallback_providers = MagicMock(return_value=[])
        workflow.config.deep_research_max_retries = 2
        workflow.config.deep_research_retry_delay = 0.1
        workflow.memory = MagicMock()
        workflow._write_audit_event = MagicMock()
        return workflow

    @pytest.mark.asyncio
    async def test_successful_parse_on_first_attempt(self, mock_workflow: MagicMock) -> None:
        """Successful parse on first attempt returns immediately."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            execute_structured_llm_call,
        )

        state = _make_state()
        mock_result = MagicMock()
        mock_result.content = '{"key": "value"}'
        mock_result.provider_id = "test"
        mock_result.model_used = "test-model"
        mock_result.tokens_used = 50
        mock_result.duration_ms = 200.0
        mock_result.input_tokens = 25
        mock_result.output_tokens = 25
        mock_result.cached_tokens = 0
        mock_result.success = True

        llm_call_result = LLMCallResult(result=mock_result, llm_call_duration_ms=200.0)

        def parse_fn(content: str) -> dict:
            return json.loads(content)

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            return_value=llm_call_result,
        ) as mock_execute:
            result = await execute_structured_llm_call(
                workflow=mock_workflow,
                state=state,
                phase_name="test",
                system_prompt="sys",
                user_prompt="user",
                provider_id="test",
                model=None,
                temperature=0.3,
                timeout=60.0,
                parse_fn=parse_fn,
            )

        assert not isinstance(result, WorkflowResult)
        assert result.parsed == {"key": "value"}
        assert result.parse_retries == 0
        assert mock_execute.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_parse_failure(self, mock_workflow: MagicMock) -> None:
        """Parse failure triggers retry with reinforced JSON instruction."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            execute_structured_llm_call,
        )

        state = _make_state()

        # First call returns unparseable, second returns valid JSON
        bad_result = MagicMock()
        bad_result.content = "not json"
        bad_result.provider_id = "test"
        bad_result.model_used = "test-model"
        bad_result.tokens_used = 50
        bad_result.duration_ms = 200.0
        bad_result.input_tokens = 25
        bad_result.output_tokens = 25
        bad_result.cached_tokens = 0
        bad_result.success = True

        good_result = MagicMock()
        good_result.content = '{"key": "value"}'
        good_result.provider_id = "test"
        good_result.model_used = "test-model"
        good_result.tokens_used = 50
        good_result.duration_ms = 200.0
        good_result.input_tokens = 25
        good_result.output_tokens = 25
        good_result.cached_tokens = 0
        good_result.success = True

        call_results = [
            LLMCallResult(result=bad_result, llm_call_duration_ms=200.0),
            LLMCallResult(result=good_result, llm_call_duration_ms=200.0),
        ]

        def parse_fn(content: str) -> dict:
            data = json.loads(content)  # Will raise on "not json"
            return data

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=call_results,
        ) as mock_execute:
            result = await execute_structured_llm_call(
                workflow=mock_workflow,
                state=state,
                phase_name="test",
                system_prompt="sys",
                user_prompt="user prompt",
                provider_id="test",
                model=None,
                temperature=0.3,
                timeout=60.0,
                parse_fn=parse_fn,
            )

        assert not isinstance(result, WorkflowResult)
        assert result.parsed == {"key": "value"}
        assert result.parse_retries == 1
        assert mock_execute.call_count == 2

        # Second call should have reinforced JSON instruction
        second_call_args = mock_execute.call_args_list[1]
        assert "IMPORTANT" in second_call_args.kwargs.get("user_prompt", "")

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_returns_none_parsed(self, mock_workflow: MagicMock) -> None:
        """When all parse retries fail, returns parsed=None."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            execute_structured_llm_call,
        )

        state = _make_state()

        bad_result = MagicMock()
        bad_result.content = "still not json"
        bad_result.provider_id = "test"
        bad_result.model_used = "test-model"
        bad_result.tokens_used = 50
        bad_result.duration_ms = 200.0
        bad_result.input_tokens = 25
        bad_result.output_tokens = 25
        bad_result.cached_tokens = 0
        bad_result.success = True

        # 4 calls: 1 initial + 3 retries
        call_results = [
            LLMCallResult(result=bad_result, llm_call_duration_ms=200.0)
            for _ in range(4)
        ]

        def parse_fn(content: str) -> dict:
            raise ValueError("Cannot parse")

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=call_results,
        ) as mock_execute:
            result = await execute_structured_llm_call(
                workflow=mock_workflow,
                state=state,
                phase_name="test",
                system_prompt="sys",
                user_prompt="user",
                provider_id="test",
                model=None,
                temperature=0.3,
                timeout=60.0,
                parse_fn=parse_fn,
            )

        assert not isinstance(result, WorkflowResult)
        assert result.parsed is None
        assert result.parse_retries == 3
        # 1 initial + 3 retries = 4 calls total
        assert mock_execute.call_count == 4

    @pytest.mark.asyncio
    async def test_llm_error_propagates_immediately(self, mock_workflow: MagicMock) -> None:
        """LLM-level error is returned immediately without retry."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            execute_structured_llm_call,
        )

        state = _make_state()

        error_result = WorkflowResult(
            success=False,
            content="",
            error="Provider down",
        )

        def parse_fn(content: str) -> dict:
            return json.loads(content)

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            return_value=error_result,
        ) as mock_execute:
            result = await execute_structured_llm_call(
                workflow=mock_workflow,
                state=state,
                phase_name="test",
                system_prompt="sys",
                user_prompt="user",
                provider_id="test",
                model=None,
                temperature=0.3,
                timeout=60.0,
                parse_fn=parse_fn,
            )

        assert isinstance(result, WorkflowResult)
        assert result.success is False
        assert mock_execute.call_count == 1

    @pytest.mark.asyncio
    async def test_system_prompt_never_modified(self, mock_workflow: MagicMock) -> None:
        """System prompt stays the same across all retry attempts."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            execute_structured_llm_call,
        )

        state = _make_state()

        bad_result = MagicMock()
        bad_result.content = "bad"
        bad_result.provider_id = "test"
        bad_result.model_used = "test-model"
        bad_result.tokens_used = 50
        bad_result.duration_ms = 100.0
        bad_result.input_tokens = 25
        bad_result.output_tokens = 25
        bad_result.cached_tokens = 0
        bad_result.success = True

        call_results = [
            LLMCallResult(result=bad_result, llm_call_duration_ms=100.0)
            for _ in range(4)
        ]

        def parse_fn(content: str) -> dict:
            raise ValueError("nope")

        original_system_prompt = "My system prompt"

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=call_results,
        ) as mock_execute:
            await execute_structured_llm_call(
                workflow=mock_workflow,
                state=state,
                phase_name="test",
                system_prompt=original_system_prompt,
                user_prompt="user",
                provider_id="test",
                model=None,
                temperature=0.3,
                timeout=60.0,
                parse_fn=parse_fn,
            )

        # All calls should have the same system prompt
        for call_args in mock_execute.call_args_list:
            assert call_args.kwargs["system_prompt"] == original_system_prompt


# =============================================================================
# Legacy _parse_clarification_response() edge cases (consolidated from
# test_clarification.py — PT.6).
#
# These test the backward-compat legacy parser that handles the old plural
# "needs_clarification" schema with "questions" list and "inferred_constraints"
# dict.  Kept as a safety net until Phase 3.3 removes the legacy parsing path.
# =============================================================================


class TestLegacyParseClarificationResponse:
    """Edge-case tests for ClarificationPhaseMixin._parse_clarification_response().

    The legacy parser is still used as a fallback to extract inferred_constraints
    from LLM responses.  These tests cover sanitization and normalization logic
    that the new structured parser (parse_clarification_decision) does not need
    because the new schema uses a single ``question`` string instead of a list.
    """

    def setup_method(self) -> None:
        self.mixin = StubClarificationMixin()

    def test_questions_truncated_to_three(self) -> None:
        """More than 3 questions are truncated to first 3."""
        content = json.dumps({
            "needs_clarification": True,
            "questions": ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"],
            "inferred_constraints": {},
        })
        result = self.mixin._parse_clarification_response(content)
        assert len(result["questions"]) == 3
        assert result["questions"] == ["Q1?", "Q2?", "Q3?"]

    def test_empty_questions_filtered(self) -> None:
        """Empty/falsy question strings are filtered out."""
        content = json.dumps({
            "needs_clarification": True,
            "questions": ["Real question?", "", "Another?"],
            "inferred_constraints": {},
        })
        result = self.mixin._parse_clarification_response(content)
        assert result["questions"] == ["Real question?", "Another?"]

    def test_empty_constraint_values_filtered(self) -> None:
        """Constraint values that are empty/falsy are filtered out."""
        content = json.dumps({
            "needs_clarification": True,
            "questions": [],
            "inferred_constraints": {
                "scope": "AI research",
                "timeframe": "",
                "domain": None,
                "depth": "overview",
            },
        })
        result = self.mixin._parse_clarification_response(content)
        assert "scope" in result["inferred_constraints"]
        assert "depth" in result["inferred_constraints"]
        assert "timeframe" not in result["inferred_constraints"]
        assert "domain" not in result["inferred_constraints"]

    def test_non_string_constraint_values_converted(self) -> None:
        """Non-string constraint values (int, float, bool) are converted to string."""
        content = json.dumps({
            "needs_clarification": True,
            "questions": [],
            "inferred_constraints": {
                "depth": "detailed",
                "max_results": 10,
                "include_images": True,
                "score_threshold": 0.8,
            },
        })
        result = self.mixin._parse_clarification_response(content)
        assert result["inferred_constraints"]["depth"] == "detailed"
        assert result["inferred_constraints"]["max_results"] == "10"
        assert result["inferred_constraints"]["include_images"] == "true"
        assert result["inferred_constraints"]["score_threshold"] == "0.8"

    def test_nested_dict_constraint_values_filtered(self) -> None:
        """Constraint values that are dicts/lists are filtered (only scalars kept)."""
        content = json.dumps({
            "needs_clarification": True,
            "questions": [],
            "inferred_constraints": {
                "scope": "narrow",
                "nested_object": {"key": "value"},
                "list_value": [1, 2, 3],
            },
        })
        result = self.mixin._parse_clarification_response(content)
        assert result["inferred_constraints"] == {"scope": "narrow"}

    def test_non_list_questions_ignored(self) -> None:
        """If questions is not a list, return empty list."""
        content = json.dumps({
            "needs_clarification": True,
            "questions": "What is the scope?",
            "inferred_constraints": {},
        })
        result = self.mixin._parse_clarification_response(content)
        assert result["questions"] == []

    def test_non_dict_constraints_ignored(self) -> None:
        """If inferred_constraints is not a dict, return empty dict."""
        content = json.dumps({
            "needs_clarification": True,
            "questions": [],
            "inferred_constraints": ["scope=AI"],
        })
        result = self.mixin._parse_clarification_response(content)
        assert result["inferred_constraints"] == {}

    def test_needs_clarification_truthy_values(self) -> None:
        """Various truthy values for needs_clarification are coerced to True."""
        for truthy_val in [True, 1, "yes", "true"]:
            content = json.dumps({
                "needs_clarification": truthy_val,
                "questions": [],
                "inferred_constraints": {},
            })
            result = self.mixin._parse_clarification_response(content)
            assert result["needs_clarification"] is True, f"Failed for {truthy_val!r}"

    def test_needs_clarification_falsy_values(self) -> None:
        """Falsy values for needs_clarification are coerced to False."""
        for falsy_val in [False, 0, "", None]:
            content = json.dumps({
                "needs_clarification": falsy_val,
                "questions": [],
                "inferred_constraints": {},
            })
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
        content = json.dumps({
            "needs_clarification": True,
            "questions": ["Any specifics?"],
            "inferred_constraints": constraints,
        })
        result = self.mixin._parse_clarification_response(content)
        assert result["inferred_constraints"] == constraints

    def test_empty_content_returns_defaults(self) -> None:
        """Empty string returns safe defaults."""
        result = self.mixin._parse_clarification_response("")
        assert result["needs_clarification"] is False
        assert result["questions"] == []
        assert result["inferred_constraints"] == {}

    def test_malformed_json_returns_defaults(self) -> None:
        """Malformed JSON string returns safe defaults."""
        result = self.mixin._parse_clarification_response("{broken json!!}")
        assert result["needs_clarification"] is False
        assert result["questions"] == []
        assert result["inferred_constraints"] == {}

    def test_json_in_code_block(self) -> None:
        """JSON wrapped in markdown code block is extracted correctly."""
        content = '```json\n{"needs_clarification": true, "questions": ["What domain?"], "inferred_constraints": {"scope": "AI"}}\n```'
        result = self.mixin._parse_clarification_response(content)
        assert result["needs_clarification"] is True
        assert result["questions"] == ["What domain?"]
        assert result["inferred_constraints"]["scope"] == "AI"
